"""What a LIBERO-PRO position cell changes, per task: displacement of the target and of the container between the
standard initial states and the cell's, and which object now stands where the target used to be.
    LIBERO_VARIANT=pro python scripts/analyze_swap_layout.py --cell libero_object_swap [--base libero_object]
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="libero_object_swap")
    ap.add_argument("--base", default="libero_object")
    ap.add_argument("--success", default="", help="optional: comma-separated runs (summary.json) to print per-task success")
    args = ap.parse_args()
    import json

    from libero.libero import benchmark

    from scripts.analyze_failures import task_layout

    bm = benchmark.get_benchmark_dict()
    base, cell = bm[args.base](), bm[args.cell]()
    runs = [r for r in args.success.split(",") if r]
    sr = [json.load(open(os.path.join(r, "summary.json")))["per_task"] for r in runs]
    print(f"{'task':<4} {'target':<22} {'d_target':>8} {'d_cont':>7}  now at the target's place (dist)   " + " ".join(os.path.basename(r)[:14] for r in runs))
    for t in range(cell.n_tasks):
        addr, target, container = task_layout(cell, t)
        if target not in addr:  # the target is a fixture or a region (open a drawer, turn on the stove)
            cols = " ".join(f"{p[str(t)]['success_rate']:>14.2f}" for p in sr)
            print(f"{t:<4} {target:<22} {'(fixture)':>8}                                          {cols}")
            continue
        if container not in addr:
            container = None
        names = list(addr)
        s0 = np.asarray(base.get_task_init_states(t))
        s1 = np.asarray(cell.get_task_init_states(t))
        # init states are [time, qpos, qvel]; free-joint xy at qpos[addr], qpos[addr + 1]
        xy = lambda s, n: s[:, 1 + addr[n] : 1 + addr[n] + 2]  # noqa: E731
        d_t = float(np.linalg.norm(xy(s1, target) - xy(s0, target), axis=1).mean())
        d_c = float(np.linalg.norm(xy(s1, container) - xy(s0, container), axis=1).mean()) if container else float("nan")
        nominal = xy(s0, target).mean(0)
        occ = sorted(((float(np.linalg.norm(xy(s1, n).mean(0) - nominal)), n) for n in names), key=lambda x: x[0])[0]
        cols = " ".join(f"{p[str(t)]['success_rate']:>14.2f}" for p in sr)
        print(f"{t:<4} {target:<22} {d_t:8.3f} {d_c:7.3f}  {occ[1]:<24} ({occ[0]:.3f})  {cols}")


if __name__ == "__main__":
    main()
