"""Gate for the pi0.5 wrapper: (1) one forward on logged frames (shapes, action ranges, chunk sanity), (2) a short
greedy rollout on LIBERO with the LeRobot / openpi protocol (10 of 50 chunk steps executed), whose success rate must
be at the level LeRobot reports (99% on Object).
    PY_ENV=pi05 python scripts/check_pi05.py --ckpt checkpoints/lerobot__pi05_libero_finetuned --run outputs/week1/plus_oft_steps
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", default="", help="a run with steps/*.npz holding rgb + wrist_rgb frames for the forward check")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--trials", type=int, default=0, help="> 0: also roll out this many init states per task")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--num_envs", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=280)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/week1/check_pi05")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from src.policy.pi05_policy import Pi05Policy

    policy = Pi05Policy(args.ckpt, device=args.device)
    print("loaded; chunk", policy.chunk, "n_action_steps", policy.n_action_steps, flush=True)
    if args.run:
        f = sorted(glob.glob(os.path.join(args.run, "steps", "*.npz")))[0]
        z = np.load(f)
        eps = {(e["task_id"], e["trial_id"]): e for e in map(json.loads, open(os.path.join(args.run, "episodes.jsonl")))}
        tid, nid = [int(x) for x in os.path.basename(f)[1:-4].replace("_n", " ").split()]
        task = eps[(tid, nid)]["task"]
        obs = [{"wrist_rgb": z["wrist_rgb"][i], "eef_pos": z["eef_pos"][i], "eef_quat": z["eef_quat"][i], "gripper_qpos": z["gripper_qpos"][i]} for i in (0, 1)]
        chunk = policy.chunk_env([z["rgb"][0], z["rgb"][1]], [task, task], obs)
        logged = z["actions"][:2]  # the OFT policy's (unnormalised, gripper in [0,1]) chunk on the same states
        print("chunk", chunk.shape, "range", chunk.min().round(3), chunk.max().round(3))
        print("pi05 first 3 steps (env units):", np.round(chunk[0, :3], 3).tolist())
        print("oft   first 3 steps (gripper in [0,1]):", np.round(logged[0, :3], 3).tolist())
        out = policy.act([z["rgb"][0]], [task], obs=obs[:1])["actions"]
        assert out.shape == (1, policy.n_action_steps, 7) and 0 <= out[..., -1].min() and out[..., -1].max() <= 1
        print("forward check ok", flush=True)
    if args.trials > 0:
        from src.rollout.collector import Collector
        from src.rollout.vec_env import LiberoVecEnv

        vec = LiberoVecEnv(args.suite, args.num_envs, args.max_steps, wrist=True)
        col = Collector(vec, policy, {}, sample=False, seed=7)
        task_ids = sorted(col.tasks) if args.tasks == "all" else [int(t) for t in args.tasks.split(",")]
        recs = col.run([(t, n) for t in task_ids for n in range(args.trials)], out_dir=args.out, save_steps=False)
        vec.close()
        per_task = {t: float(np.mean([r["success"] for r in recs if r["task_id"] == t])) for t in task_ids}
        sr = float(np.mean([r["success"] for r in recs]))
        json.dump({"success_rate": sr, "n_episodes": len(recs), "per_task": per_task, "timing": col.timing}, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
        print(json.dumps({"success_rate": sr, "n_episodes": len(recs), "per_task": per_task}), flush=True)
    print("CHECK_PI05_DONE", flush=True)


if __name__ == "__main__":
    main()
