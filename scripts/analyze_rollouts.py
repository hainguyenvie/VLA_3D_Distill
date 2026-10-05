"""Numeric diagnostics on a rollout log written by scripts/eval_libero.py (acting policy + one labeler).

Answers, per state and per episode: where does the acting policy disagree with the labeling policy, how does
that evolve along the episode, and does it separate successful from failed episodes?
    python scripts/analyze_rollouts.py --run outputs/week1/b0_object_student [--horizons 280,512]
Writes <run>/analysis.json and prints the tables.
"""
import argparse
import glob
import json
import os

import numpy as np


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def auc(pos, neg):
    """Probability that a random positive scores above a random negative (rank statistic)."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)) + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--horizons", default="280,512")
    ap.add_argument("--bins", type=int, default=5, help="number of episode-progress bins")
    args = ap.parse_args()
    eps = [json.loads(l) for l in open(os.path.join(args.run, "episodes.jsonl"))]
    out = {"n_episodes": len(eps), "success_rate": float(np.mean([e["success"] for e in eps]))}
    # success under shorter horizons, from the step at which each episode ended
    out["success_by_horizon"] = {h: float(np.mean([e["success"] and e["env_steps"] <= int(h) for e in eps]))
                                 for h in args.horizons.split(",")}
    steps_ok = [e["env_steps"] for e in eps if e["success"]]
    out["success_steps_quantiles"] = [float(q) for q in np.quantile(steps_ok, [0.1, 0.5, 0.9, 1.0])] if steps_ok else None
    tasks = sorted({e["task_id"] for e in eps})
    out["per_task"] = {t: {"task": next(e["task"] for e in eps if e["task_id"] == t),
                           "n": sum(e["task_id"] == t for e in eps),
                           "success_rate": float(np.mean([e["success"] for e in eps if e["task_id"] == t]))} for t in tasks}

    label = next((k[3:] for k in eps[0] if k.startswith("kl_")), None)
    files = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(args.run, "steps", "*.npz"))}
    if label and files:
        rows = []  # (success, progress in [0,1], kl, entropy_actor, entropy_labeler, agree)
        ep_first = []  # per episode: mean KL over the first 5 queries
        for e in eps:
            p = files.get(f"t{e['task_id']:02d}_n{e['trial_id']:02d}")
            if p is None:
                continue
            z = np.load(p)
            la, ll = log_softmax(z["logits"].astype(np.float32)), log_softmax(z[f"logits_{label}"].astype(np.float32))
            kl = (np.exp(la) * (la - ll)).sum(-1).mean(-1)
            ha, hl = -(np.exp(la) * la).sum(-1).mean(-1), -(np.exp(ll) * ll).sum(-1).mean(-1)
            agree = (la.argmax(-1) == ll.argmax(-1)).mean(-1)
            n = len(kl)
            for i in range(n):
                rows.append((e["success"], i / max(n - 1, 1), kl[i], ha[i], hl[i], agree[i]))
            ep_first.append((e["success"], kl[:5].mean(), kl.mean()))
        r = np.array(rows, dtype=np.float64)
        ok, bad = r[r[:, 0] == 1], r[r[:, 0] == 0]
        out["labeler"] = label
        out["states"] = {"n": len(r), "n_in_successful_episodes": int(len(ok)), "n_in_failed_episodes": int(len(bad))}
        cols = {"kl_actor_labeler": 2, "entropy_actor": 3, "entropy_labeler": 4, "token_agreement": 5}
        out["state_means"] = {k: {"success": float(ok[:, c].mean()) if len(ok) else None,
                                  "failure": float(bad[:, c].mean()) if len(bad) else None} for k, c in cols.items()}
        edges = np.linspace(0, 1, args.bins + 1)
        prog = {}
        for b in range(args.bins):
            sel = lambda a: a[(a[:, 1] >= edges[b]) & ((a[:, 1] < edges[b + 1]) | (b == args.bins - 1))]  # noqa: E731
            prog[f"{edges[b]:.1f}-{edges[b + 1]:.1f}"] = {
                "kl_success": float(sel(ok)[:, 2].mean()) if len(sel(ok)) else None,
                "kl_failure": float(sel(bad)[:, 2].mean()) if len(sel(bad)) else None}
        out["kl_by_episode_progress"] = prog
        ef = np.array(ep_first, dtype=np.float64)
        # can disagreement predict the outcome? AUC of "episode fails" from KL (early = first 5 queries only)
        out["auc_failure_from_kl"] = {"first_5_queries": auc(ef[ef[:, 0] == 0, 1], ef[ef[:, 0] == 1, 1]),
                                      "whole_episode": auc(ef[ef[:, 0] == 0, 2], ef[ef[:, 0] == 1, 2])}
    json.dump(out, open(os.path.join(args.run, "analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
