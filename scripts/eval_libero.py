"""Evaluate a discrete-token OpenVLA-OFT checkpoint on a LIBERO suite and log every queried state.

Example (on the server, through scripts/server/run_eval.sh):
    python scripts/eval_libero.py --ckpt <dir> --suite libero_object --out outputs/week1/b0_object \
        --label teacher=<teacher dir> --num_envs 10

Outputs in --out: episodes.jsonl (one line per episode, resumable), steps/*.npz (per-query arrays),
summary.json (success rates + config; written only when every requested episode is present).
"""
import argparse
import json
import os
import time

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lora", default="", help="LoRA adapter directory (from train_opd.py) to put on top of --ckpt")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", action="append", default=[],
                    help="name=checkpoint, extra frozen policy that labels every state; name=rebin:checkpoint if its "
                         "action normalisation differs from --ckpt (labels are then expressed in --ckpt's bins)")
    ap.add_argument("--tasks", default="all", help="comma-separated task ids, or 'all'")
    ap.add_argument("--trials", type=int, default=50, help="use the first N benchmark initial states of each task")
    ap.add_argument("--num_envs", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=512, help="SimpleVLA-RL uses 512 for every suite; OpenVLA-OFT uses 280 for Object")
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--sample", action="store_true", help="sample action tokens instead of argmax")
    ap.add_argument("--temperature", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--depth", action="store_true", help="also log metric depth of the agent view")
    ap.add_argument("--no_steps", action="store_true", help="do not write steps/*.npz")
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    # env workers are spawned before torch / TF touch the GPU in this process
    from src.rollout.vec_env import LiberoVecEnv

    vec = LiberoVecEnv(args.suite, args.num_envs, args.max_steps, args.num_steps_wait, depth=args.depth)

    import torch

    from src.policy.rebin import load_policy
    from src.policy.token_policy import TokenPolicy
    from src.rollout.collector import Collector

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy = load_policy(args.ckpt, args.suite, args.device)  # plain path, or raw:<path> (RLinf image pipeline)
    print("loading info:", {k: len(v) for k, v in policy.loading_info.items()}, flush=True)
    if args.lora:
        policy.add_lora(adapter_path=args.lora)
    labelers = {}
    for spec in args.label:
        name, path = spec.split("=", 1)
        labelers[name] = load_policy(path, args.suite, args.device, target=policy)
    col = Collector(vec, policy, labelers, sample=args.sample, temperature=args.temperature, seed=args.seed)
    task_ids = sorted(col.tasks) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]
    episodes = [(t, n) for t in task_ids for n in range(min(args.trials, col.tasks[t][1]))]

    t0, state = time.time(), {"n": 0, "s": 0}

    def on_episode(rec):
        state["n"] += 1
        state["s"] += rec["success"]
        print(f"[{time.time() - t0:7.0f}s] ep {state['n']:4d} task {rec['task_id']} trial {rec['trial_id']:2d} "
              f"success={int(rec['success'])} steps={rec['env_steps']:3d} running_sr={state['s'] / state['n']:.3f}", flush=True)

    results = col.run(episodes, out_dir=args.out, save_steps=not args.no_steps, on_episode=on_episode)
    vec.close()

    by_id = {(r["task_id"], r["trial_id"]): r for r in results}
    missing = [e for e in episodes if e not in by_id]
    assert not missing, f"{len(missing)} episodes missing, e.g. {missing[:5]}"
    recs = [by_id[e] for e in episodes]
    per_task = {t: float(np.mean([r["success"] for r in recs if r["task_id"] == t])) for t in task_ids}
    summary = {
        "success_rate": float(np.mean([r["success"] for r in recs])),
        "n_episodes": len(recs),
        "per_task": {str(t): {"task": col.tasks[t][0], "success_rate": per_task[t]} for t in task_ids},
        "mean_env_steps": float(np.mean([r["env_steps"] for r in recs])),
        "mean_entropy": float(np.mean([r["entropy"] for r in recs])),
        "config": vars(args),
        "unnorm_key": policy.unnorm_key,
        "revision": open(os.path.join(os.environ.get("REPO", "."), "REVISION")).read().strip()
        if os.path.exists(os.path.join(os.environ.get("REPO", "."), "REVISION")) else None,
        "wall_seconds": round(time.time() - t0),
        "timing_seconds": {k: round(v, 1) for k, v in col.timing.items()},
    }
    for name in labelers:
        summary[f"mean_kl_{name}"] = float(np.mean([r[f"kl_{name}"] for r in recs]))
        summary[f"mean_agree_{name}"] = float(np.mean([r[f"agree_{name}"] for r in recs]))
    for name, path in [("ckpt", args.ckpt)] + [s.split("=", 1) for s in args.label]:
        meta = os.path.join(path.removeprefix("rebin:").removeprefix("raw:"), ".hf_fetch.json")
        if os.path.exists(meta):
            m = json.load(open(meta))
            summary.setdefault("checkpoints", {})[name] = {"repo": m["repo"], "revision": m["revision"]}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("success_rate", "n_episodes", "mean_env_steps", "timing_seconds")}), flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
