"""How much does the renderer change what the policy sees and does?

Re-renders logged simulator states (steps/*.npz of a rollout) with the current GL backend and compares
against the frames stored in the log (rendered when the rollout ran, e.g. NVIDIA EGL on another machine).

    # 1. re-render with the backend of this process (MUJOCO_GL = egl | osmesa)
    python scripts/check_renderer.py render --steps <run>/steps --out scratch/frames_osmesa.npz
    # 2. compare any two frame sets through the policy ("log" = frames stored in the rollout log)
    python scripts/check_renderer.py compare --a log --b scratch/frames_osmesa.npz --frames scratch/frames_osmesa.npz --ckpt <student>
"""
import argparse
import glob
import os

import numpy as np


def render(args):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    files = sorted(glob.glob(os.path.join(args.steps, "*.npz")))
    by_task = {}
    for f in files:
        by_task.setdefault(int(os.path.basename(f)[1:3]), []).append(f)
    out = {"rgb": [], "log_rgb": [], "sim_state": [], "task_id": []}
    for task_id in sorted(by_task)[: args.tasks]:
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                                 camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        for f in by_task[task_id][: args.episodes]:
            z = np.load(f)
            for i in range(0, len(z["t"]), args.stride):
                obs = env.set_init_state(z["sim_state"][i])
                out["rgb"].append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                out["log_rgb"].append(z["rgb"][i])
                out["sim_state"].append(z["sim_state"][i])
                out["task_id"].append(task_id)
        env.close()
    np.savez_compressed(args.out, **{k: np.stack(v) for k, v in out.items()})
    d = np.abs(np.stack(out["rgb"]).astype(np.int16) - np.stack(out["log_rgb"]).astype(np.int16))
    print(f"{len(out['rgb'])} frames; vs logged frames: mean |pixel diff| {d.mean():.3f}, 99th pct {np.percentile(d, 99):.0f}, "
          f"pixels differing by > 8 levels {(d.max(-1) > 8).mean():.2%}")
    print("RENDER_DONE")


def compare(args):
    import torch
    import torch.nn.functional as F

    from libero.libero import benchmark
    from src.policy.token_policy import TokenPolicy

    z = np.load(args.frames)
    load = lambda spec: z["log_rgb"] if spec == "log" else np.load(spec)["rgb"]  # noqa: E731
    a, b = load(args.a), load(args.b)
    assert a.shape == b.shape
    suite = benchmark.get_benchmark_dict()[args.suite]()
    descs = [suite.get_task(int(t)).language for t in z["task_id"]]
    policy = TokenPolicy(args.ckpt, args.suite)

    def logits(frames, bs):
        out = []
        for s in range(0, len(frames), bs):
            out.append(policy.act(list(frames[s : s + bs]), descs[s : s + bs])["logits"])
        return torch.cat(out)

    la, lb, la2 = logits(a, 1), logits(b, 1), logits(a, 2)  # la2: same frames, other batch size = bf16 noise floor

    def stats(x, y):
        px, py = F.log_softmax(x, -1), F.log_softmax(y, -1)
        return {"mean|dlogit|": float((x - y).abs().mean()), "kl": float((px.exp() * (px - py)).sum(-1).mean()),
                "argmax_agree": float((x.argmax(-1) == y.argmax(-1)).float().mean())}

    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    print(f"{len(a)} frames; pixels: mean |diff| {d.mean():.3f}, > 8 levels {(d.max(-1) > 8).mean():.2%}")
    print("renderer A vs B     :", {k: round(v, 4) for k, v in stats(la, lb).items()})
    print("noise floor (A, bs 1 vs 2):", {k: round(v, 4) for k, v in stats(la, la2).items()})
    print("COMPARE_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--steps", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--suite", default="libero_object")
    r.add_argument("--tasks", type=int, default=4)
    r.add_argument("--episodes", type=int, default=2)
    r.add_argument("--stride", type=int, default=3)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--frames", required=True, help="any frame file of the same states (task ids and logged frames)")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--suite", default="libero_object")
    args = ap.parse_args()
    render(args) if args.cmd == "render" else compare(args)
