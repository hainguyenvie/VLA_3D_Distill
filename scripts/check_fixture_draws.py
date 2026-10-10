"""Where LIBERO puts the furniture at each reset of one env (seeded as upstream): the position of every body fixed to the
world, for consecutive episodes of a task run in the benchmark's order (initial states 0, 1, 2, ...).
    python scripts/check_fixture_draws.py --suite libero_goal --task 0 --n 20
"""
import argparse
import contextlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--reseed", default="", help="comma-separated trial order: seed the env per (task, trial) before each reset, "
                    "as eval_libero.py --fixture_seed does (the same trial must give the same placement in any order)")
    args = ap.parse_args()
    from src.rollout.vec_env import EnvRunner, LiberoVecEnv, world_fixed_bodies

    vec = LiberoVecEnv(args.suite, 0, 300, wrist=True)
    inits = vec.init_states(args.task)
    r = EnvRunner(vec._bddl(args.task), vec.cfg, contextlib.nullcontext())
    rows = []
    order = [int(x) for x in args.reseed.split(",")] if args.reseed else list(range(args.n))
    for k in order:
        if args.reseed:
            r.env.seed(1000 * args.task + k)
        r.reset(inits[k % len(inits)])
        robo = r.env.env
        m = robo.sim.model._model
        fixed = [b for b in world_fixed_bodies(robo) if not (robo.sim.model.body_id2name(b) or "").endswith("table")]
        if k == order[0]:
            names = [robo.sim.model.body_id2name(b) for b in fixed]
            print("episode " + "  ".join(f"{n[:22]:>22s}" for n in names))
        pos = np.array([m.body_pos[b][:2] for b in fixed])
        rows.append(pos)
        print(f"{k:7d} " + "  ".join(f"({p[0]:+.3f},{p[1]:+.3f})".rjust(22) for p in pos))
    rows = np.array(rows)
    print("spread (max - min, m): " + "  ".join(f"({s[0]:.3f},{s[1]:.3f})" for s in rows.max(0) - rows.min(0)))


if __name__ == "__main__":
    main()
