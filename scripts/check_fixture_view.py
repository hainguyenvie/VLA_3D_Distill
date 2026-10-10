"""Look at the furniture-nudge counterfactual (EnvRunner._fixture_view, train_pi05_ocd.py --cf_fixture): the factual
agent / wrist frames next to the nudged ones at a few steps of an episode, with the shift drawn.
    python scripts/check_fixture_view.py --suite libero_goal --task 3 --out fixture_view.png
"""
import argparse
import contextlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task", type=int, default=3)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--jitter", default="0.03,0.03", help="min,max shift (m); the training default is 0.01,0.03")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from PIL import Image

    from src.rollout.vec_env import COUNTERFACTUAL, EnvRunner, LiberoVecEnv

    cf = dict(COUNTERFACTUAL, mode="full", p_fixture=1.0, fixture_jitter=tuple(float(x) for x in args.jitter.split(",")))
    vec = LiberoVecEnv(args.suite, 0, 300, wrist=True, counterfactual=cf)
    r = EnvRunner(vec._bddl(args.task), vec.cfg, contextlib.nullcontext())
    o = r.reset(vec.init_states(args.task)[args.trial], counterfactual=True)
    rows = []
    for k in range(3):
        print(f"step {k}: kind {int(o.get('cf_kind', -1))} valid {bool(o.get('cf_valid'))} shift {np.round(o['cf_delta'], 3)} "
              f"mean |rgb - rgb_cf| {np.abs(o['rgb'].astype(float) - o['rgb_cf']).mean():.2f}")
        rows.append(np.concatenate([o["rgb"], o["rgb_cf"], o["wrist_rgb"], o["wrist_rgb_cf"]], axis=1))
        for _ in range(4):  # move the arm down and forward a little between the looks
            o = r.step(np.array([[0.4, 0.0, -0.4, 0, 0, 0, -1.0]] * 10))
    Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8)).save(args.out)
    print("saved: rows = steps; columns = agent, agent nudged, wrist, wrist nudged ->", args.out)


if __name__ == "__main__":
    main()
