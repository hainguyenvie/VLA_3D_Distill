"""Gate for the visual perturbations of training rollouts (`vec_env.VIEW_AUG`).

One env, the same random action chunks in every episode:
  (a) nominal episode;
  (b) perturbed episode with all magnitudes zero: the extra render must reproduce the nominal frame and depth
      exactly (same camera sensor path), and the simulator states must equal (a);
  (c) episodes with drawn perturbations: simulator states and the nominal frame must still equal (a) at every
      query (the perturbation only touches camera and lights at render time), and the perturbed frame must differ.
Also writes a contact sheet of perturbed first frames for a visual check.
    python scripts/check_view_aug.py --suite libero_object --task 3 --out outputs/week1/view_aug_check
"""
import argparse
import contextlib
import os

import numpy as np

from src.rollout.vec_env import VIEW_AUG, EnvRunner, sample_perturbation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=3)
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from libero.libero import benchmark, get_libero_path
    from PIL import Image

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init = np.asarray(suite.get_task_init_states(args.task))[0]
    cfg = dict(suite=args.suite, max_steps=512, num_steps_wait=10, resolution=256, depth=True, wrist=False, perturb=VIEW_AUG)
    runner = EnvRunner(bddl, cfg, contextlib.nullcontext())
    rng = np.random.default_rng(0)
    chunks = [np.concatenate([rng.uniform(-0.3, 0.3, (8, 6)), -np.ones((8, 1))], axis=1) for _ in range(args.chunks)]

    def episode(perturb):
        return [runner.reset(init, perturb)] + [runner.step(c) for c in chunks]

    ref = episode(False)
    assert all("rgb_clean" not in o for o in ref)
    zero = dict(camera=dict(azimuth=0.0, elevation=0.0, distance=1.0, aim=np.zeros(2)),
                light=dict(diffuse=1.0, ambient=1.0, tint=np.ones(3), shift=np.zeros(3)))
    same = episode(zero)
    d_state = max(float(np.abs(a["sim_state"] - b["sim_state"]).max()) for a, b in zip(ref, same))
    d_clean = max(float(np.abs(a["rgb"].astype(int) - b["rgb_clean"].astype(int)).max()) for a, b in zip(ref, same))
    d_zero = max(float(np.abs(b["rgb"].astype(int) - b["rgb_clean"].astype(int)).max()) for b in same)
    d_depth = max(float(np.abs(a["depth"] - b["depth"]).max()) for a, b in zip(ref, same))
    print(f"zero perturbation: |state| {d_state:.1e}, nominal frame {d_clean}, extra render vs nominal {d_zero}, depth {d_depth:.1e}")
    assert d_state == 0 and d_clean == 0 and d_zero == 0 and d_depth == 0

    draw_rng, frames, worst_state, worst_clean, n_diff = np.random.default_rng(1), [], 0.0, 0, 0
    for k in range(args.draws):
        pert = None
        while not pert:
            pert = sample_perturbation(draw_rng, VIEW_AUG)
        obs = episode(pert) if k < 3 else [runner.reset(init, pert)]
        worst_state = max(worst_state, max(float(np.abs(a["sim_state"] - b["sim_state"]).max()) for a, b in zip(ref, obs)))
        worst_clean = max(worst_clean, max(int(np.abs(a["rgb"].astype(int) - b["rgb_clean"].astype(int)).max()) for a, b in zip(ref, obs)))
        n_diff += int(np.abs(obs[0]["rgb"].astype(int) - obs[0]["rgb_clean"].astype(int)).mean() > 1.0)
        frames.append(obs[-1]["rgb"] if k < 3 else obs[0]["rgb"])
        assert np.isfinite(obs[0]["depth"]).all() and obs[0]["depth"].min() > 0
        print(k, {name: {a: np.round(v, 2).tolist() if hasattr(v, "__len__") else round(float(v), 2) for a, v in p.items() if a != "seed"}
                  for name, p in pert.items()})
    print(f"drawn perturbations: |state| {worst_state:.1e}, nominal frame {worst_clean}, {n_diff}/{args.draws} visibly perturbed")
    assert worst_state == 0 and worst_clean == 0 and n_diff >= args.draws - 1
    cols = 4
    rows = [np.concatenate(frames[i : i + cols], axis=1) for i in range(0, len(frames) - len(frames) % cols, cols)]
    sheet = np.concatenate([np.concatenate([ref[0]["rgb"]] + [np.zeros_like(ref[0]["rgb"])] * (cols - 1), axis=1)] + rows, axis=0)
    Image.fromarray(sheet).save(os.path.join(args.out, "view_aug_sheet.jpg"), quality=85)
    print("CHECK_VIEW_AUG_OK")


if __name__ == "__main__":
    main()
