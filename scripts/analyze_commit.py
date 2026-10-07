"""When does a policy commit to the wrong object? Per-episode trajectory analysis on runs with per-query states.

For every episode the horizontal distance from the end-effector to the target and to every other movable object is
read from the logged simulator states at each query. The "lure" of a failed episode is the object nearest the gripper
when it first closes. The commit query is the first query from which the gripper stays nearer to the lure than to the
target (and keeps getting nearer). Reports where commitment happens (query index, distance travelled), so that an
early-state intervention can be targeted, and, for two runs on the same episodes, which episodes flipped.
    python scripts/analyze_commit.py --run <run with steps> [--compare <other run>] --suite libero_object_swap
"""
import argparse
import csv
import json
import os

import numpy as np


def episode_tracks(run, suite_name):
    from analyze_failures import task_layout  # sibling script
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    fails = {(int(r["task_id"]), int(r["trial_id"])): r for r in csv.DictReader(open(os.path.join(run, "failures.csv")))}
    layouts, out = {}, {}
    for e in map(json.loads, open(os.path.join(run, "episodes.jsonl"))):
        t, n = e["task_id"], e["trial_id"]
        if t not in layouts:
            layouts[t] = task_layout(suite, t)
        addr, target, container = layouts[t]
        z = np.load(os.path.join(run, "steps", f"t{t:02d}_n{n:02d}.npz"))
        eef = z["eef_pos"][:, :2]
        pos = {k: z["sim_state"][:, 1 + a : 1 + a + 2] for k, a in addr.items() if k != container}
        d = {k: np.linalg.norm(eef - p, axis=1) for k, p in pos.items()}
        close = z["actions"][:, :, -1].min(1) < 0.5
        q = int(np.argmax(close)) if close.any() else len(close) - 1
        lure = min((k for k in pos), key=lambda k: d[k][q])
        commit = -1
        if lure != target:
            nearer = d[lure] < d[target]
            for i in range(len(nearer)):
                if nearer[i:q + 1].all():
                    commit = i
                    break
        out[(t, n)] = {"success": int(e["success"]), "mode": fails[(t, n)]["mode"], "lure": lure, "target": target,
                       "first_close_query": q, "commit_query": commit,
                       "travel_at_commit_cm": round(float(np.linalg.norm(eef[commit] - eef[0])) * 100, 1) if commit >= 0 else None,
                       "d0_target_cm": round(float(d[target][0]) * 100, 1), "d0_lure_cm": round(float(d[lure][0]) * 100, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", default="")
    ap.add_argument("--suite", default="libero_object_swap")
    args = ap.parse_args()
    a = episode_tracks(args.run, args.suite)
    wrong = [v for v in a.values() if not v["success"] and v["lure"] != v["target"]]
    cq = np.array([v["commit_query"] for v in wrong if v["commit_query"] >= 0])
    print(f"{args.run}: {len(a)} episodes, {sum(v['success'] for v in a.values())} successes, {len(wrong)} failures that close on another object")
    if len(cq):
        print(f"  commit query: median {np.median(cq):.0f}, quartiles {np.quantile(cq, [0.25, 0.75]).tolist()}, "
              f"share committed at query 0: {np.mean(cq == 0):.2f}, within 2 queries: {np.mean(cq <= 2):.2f}")
        print(f"  first close query (median): {np.median([v['first_close_query'] for v in wrong]):.0f}")
    by_task = {}
    for (t, n), v in a.items():
        by_task.setdefault(t, []).append(v["success"])
    print("  per task:", {t: round(float(np.mean(s)), 2) for t, s in sorted(by_task.items())})
    if args.compare:
        b = episode_tracks(args.compare, args.suite)
        common = sorted(set(a) & set(b))
        gained = [k for k in common if b[k]["success"] and not a[k]["success"]]
        lost = [k for k in common if a[k]["success"] and not b[k]["success"]]
        print(f"{args.compare} vs {args.run} on {len(common)} common episodes: +{len(gained)} / -{len(lost)}")
        from collections import Counter

        print("  gained by task:", dict(Counter(t for t, _ in gained)), " lost by task:", dict(Counter(t for t, _ in lost)))
        modes_b = Counter(b[k]["mode"] for k in common if not b[k]["success"])
        print("  remaining failure modes:", dict(modes_b))
        wb = [b[k] for k in common if not b[k]["success"] and b[k]["lure"] != b[k]["target"]]
        cb = np.array([v["commit_query"] for v in wb if v["commit_query"] >= 0])
        if len(cb):
            print(f"  its commit query: median {np.median(cb):.0f}, at query 0: {np.mean(cb == 0):.2f}, within 2: {np.mean(cb <= 2):.2f}")
    print("COMMIT_DONE")


if __name__ == "__main__":
    main()
