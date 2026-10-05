"""Evaluate a policy on LIBERO-Plus (perturbed LIBERO tasks, one trial each) and report by perturbation type.

Run with LIBERO_VARIANT=plus so that `libero` resolves to the LIBERO-Plus checkout:
    LIBERO_VARIANT=plus bash scripts/server/run_py.sh 0 scripts/eval_libero_plus.py --ckpt <dir> [--lora <dir>] \
        --suite libero_object --per_category 60 --out outputs/week1/plus_<name>
Tasks are a seeded stratified sample (`--per_category` per perturbation type; 0 = all tasks of the suite).
Outputs in --out: episodes.jsonl (resumable), summary.json (success by category and by difficulty level).
"""
import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_category", type=int, default=60)
    ap.add_argument("--num_envs", type=int, default=7)
    ap.add_argument("--max_steps", type=int, default=512, help="openvla-oft uses 220 / 280 / 300 / 520 for spatial / object / goal / 10")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    assert os.environ.get("LIBERO_VARIANT") == "plus", "run with LIBERO_VARIANT=plus"
    import libero.libero as L

    from src.rollout.vec_env import LiberoVecEnv

    vec = LiberoVecEnv(args.suite, args.num_envs, args.max_steps, wrist=args.ckpt.startswith("oft:"))
    cls_path = os.path.join(os.path.dirname(L.__file__), "benchmark", "task_classification.json")
    name_to_id = {vec.suite.get_task(i).name: i for i in range(vec.suite.n_tasks)}
    entries = [e for e in json.load(open(cls_path))[args.suite] if e["name"] in name_to_id]
    rng = np.random.default_rng(args.seed)
    chosen = []
    for cat in sorted({e["category"] for e in entries}):
        pool = [e for e in entries if e["category"] == cat]
        k = len(pool) if args.per_category <= 0 else min(args.per_category, len(pool))
        chosen += [pool[i] for i in rng.permutation(len(pool))[:k]]
    meta = {name_to_id[e["name"]]: e for e in chosen}
    print(f"{len(entries)} classified tasks in {args.suite}, evaluating {len(meta)}", flush=True)

    import torch

    from src.policy.rebin import load_policy
    from src.rollout.collector import Collector

    torch.manual_seed(args.seed)
    policy = load_policy(args.ckpt, args.suite, "cuda:0")  # plain path, raw:<path> or oft:<path>
    if args.lora:
        policy.add_lora(adapter_path=args.lora)
    col = Collector(vec, policy, {}, sample=False, seed=args.seed, task_ids=sorted(meta))
    t0, state = time.time(), {"n": 0, "s": 0}

    def on_episode(rec):
        state["n"] += 1
        state["s"] += rec["success"]
        if state["n"] % 20 == 0:
            print(f"[{time.time() - t0:6.0f}s] {state['n']} episodes, running_sr={state['s'] / state['n']:.3f}", flush=True)

    results = col.run([(t, 0) for t in sorted(meta)], out_dir=args.out, save_steps=False, on_episode=on_episode)
    vec.close()
    by_id = {r["task_id"]: r for r in results}
    rows = [dict(meta[t], task_id=t, success=by_id[t]["success"]) for t in sorted(meta) if t in by_id]
    assert len(rows) == len(meta), f"{len(meta) - len(rows)} episodes missing"

    def rate(sel):
        v = [r["success"] for r in rows if sel(r)]
        return {"n": len(v), "success_rate": float(np.mean(v)) if v else None}

    summary = {
        "overall": rate(lambda r: True),
        "by_category": {c: rate(lambda r, c=c: r["category"] == c) for c in sorted({r["category"] for r in rows})},
        "by_difficulty": {str(d): rate(lambda r, d=d: r["difficulty_level"] == d) for d in sorted({r["difficulty_level"] for r in rows})},
        "config": vars(args), "wall_seconds": round(time.time() - t0),
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps({"overall": summary["overall"], "by_category": summary["by_category"]}, indent=1))
    print("EVAL_PLUS_DONE", flush=True)


if __name__ == "__main__":
    main()
