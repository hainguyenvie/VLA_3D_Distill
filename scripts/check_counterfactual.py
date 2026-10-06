"""Gate for the counterfactual (object-displaced) frames of training rollouts (`vec_env.COUNTERFACTUAL`).

One env, the same random action chunks: an episode with counterfactual rendering must produce exactly the
simulator states and nominal frames of a plain episode (the displacement is undone before any physics step), while
the counterfactual frames differ from the nominal ones and `target_pos_cf` = `target_pos` + delta. Also writes a
contact sheet (nominal | counterfactual) for a visual check.
    python scripts/check_counterfactual.py --suite libero_object --task 3 --out outputs/week1/cf_check
"""
import argparse
import contextlib
import os

import numpy as np

from src.rollout.vec_env import COUNTERFACTUAL, EnvRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=3)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from libero.libero import benchmark, get_libero_path
    from PIL import Image

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init = np.asarray(suite.get_task_init_states(args.task))[0]
    cfg = dict(suite=args.suite, max_steps=512, num_steps_wait=10, resolution=256, depth=False, wrist=True, perturb=None,
               counterfactual=COUNTERFACTUAL)
    runner = EnvRunner(bddl, cfg, contextlib.nullcontext())
    rng = np.random.default_rng(0)
    chunks = [np.concatenate([rng.uniform(-0.3, 0.3, (8, 6)), -np.ones((8, 1))], axis=1) for _ in range(args.chunks)]
    ref = [runner.reset(init)] + [runner.step(c) for c in chunks]
    cf = [runner.reset(init, counterfactual=True)] + [runner.step(c) for c in chunks]
    d_state = max(float(np.abs(a["sim_state"] - b["sim_state"]).max()) for a, b in zip(ref, cf))
    d_rgb = max(int(np.abs(a["rgb"].astype(int) - b["rgb"].astype(int)).max()) for a, b in zip(ref, cf))
    d_wrist = max(int(np.abs(a["wrist_rgb"].astype(int) - b["wrist_rgb"].astype(int)).max()) for a, b in zip(ref, cf))
    diff_cf = [float(np.abs(b["rgb_cf"].astype(int) - b["rgb"].astype(int)).mean()) for b in cf]
    d_tp = max(float(np.abs(b["target_pos_cf"] - b["target_pos"] - np.r_[b["cf_delta"], 0.0]).max()) for b in cf)
    mags = [float(np.linalg.norm(b["cf_delta"])) for b in cf]
    print(f"|state| {d_state:.1e}, nominal rgb {d_rgb}, wrist {d_wrist}, mean |rgb_cf - rgb| {np.round(diff_cf, 2).tolist()}, "
          f"delta magnitudes {np.round(mags, 3).tolist()}, target_pos consistency {d_tp:.1e}")
    assert d_state < 1e-9 and d_rgb == 0 and d_wrist == 0 and min(diff_cf) > 0.2 and d_tp < 1e-5
    rows = [np.concatenate([b["rgb"], b["rgb_cf"], b["wrist_rgb"], b["wrist_rgb_cf"]], axis=1) for b in cf[:4]]
    Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(args.out, "cf_sheet.jpg"), quality=85)
    print("CHECK_COUNTERFACTUAL_OK")


if __name__ == "__main__":
    main()
