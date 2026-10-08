"""Upper bound for fixing the approach phase only: on a LIBERO(-PRO) suite, a privileged scripted controller brings the
hand to above the true target (src/rollout/scripted.approach_chunk, executed 10 steps per chunk like a policy), then
pi0.5 (optionally with a LoRA adapter) takes over for the rest of the episode. Compare with pi0.5 alone on the same
initial states.
    LIBERO_VARIANT=pro python scripts/oracle_approach.py --ckpt <pi0.5 dir> --suite libero_object_swap --trials 10 --out <dir>
"""
import argparse
import contextlib
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--suite", default="libero_object_swap")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=280)
    ap.add_argument("--max_approach_queries", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from libero.libero import benchmark, get_libero_path

    from src.rollout.scripted import approach_chunk
    from src.rollout.vec_env import COUNTERFACTUAL, DUMMY_ACTION, EnvRunner, postprocess_actions

    suite = benchmark.get_benchmark_dict()[args.suite]()
    from src.policy.pi05_policy import Pi05Policy

    pol = Pi05Policy(args.ckpt)
    if args.lora:
        pol.add_lora(32, adapter_path=args.lora)
    res = []
    for t in range(suite.n_tasks):
        task = suite.get_task(t)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        cfg = dict(suite=args.suite, max_steps=args.max_steps, num_steps_wait=10, resolution=256, depth=False, wrist=True,
                   perturb=None, counterfactual=dict(COUNTERFACTUAL, mode="coshift"))
        r = EnvRunner(bddl, cfg, contextlib.nullcontext())
        inits = np.asarray(suite.get_task_init_states(t))
        for n in range(min(args.trials, len(inits))):
            obs = r.reset(inits[n])
            robo = r.env.env
            r._coshift_setup(robo)
            d = robo.sim.data._data
            hovered, q = False, 0
            if not getattr(r, "_cs_ok", False):  # no movable target (open a drawer, turn on the stove): pi0.5 alone
                hovered = None
            while hovered is False and q < args.max_approach_queries and obs["active"]:  # scripted approach to the true target
                chunk, k = approach_chunk(d.site_xpos[r._cs_site].copy(), d.xpos[r._cs_body].copy())
                hovered = k <= 10
                obs = r.step(chunk[:10])
                d = r.env.env.sim.data._data
                q += 1
            while obs["active"] and not obs["done"]:  # pi0.5 for the rest of the episode
                out = pol.act([obs["rgb"]], [task.language], obs=[obs])
                obs = r.step(postprocess_actions(out["actions"])[0])
            res.append(dict(task_id=t, trial_id=n, success=bool(obs["done"]), approach_queries=q, hovered=hovered))
            print(f"task {t} trial {n}: hovered {hovered} after {q} queries, success {obs['done']}", flush=True)
        r.close()
    sr = float(np.mean([x["success"] for x in res]))
    per = {t: float(np.mean([x["success"] for x in res if x["task_id"] == t])) for t in range(suite.n_tasks)}
    json.dump(dict(success_rate=sr, per_task=per, episodes=res), open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"ORACLE_APPROACH success {sr:.3f}; per task {per}")


if __name__ == "__main__":
    main()
