"""Within-state probe of pi0.5's internals (output of scripts/probe_internals.py): for each probe state, the change of the
pooled hidden states when only the target (before the grasp) or the container (while carrying) was moved, regressed on
that displacement. This removes everything that varies across states (where the hand is, the episode's time) and asks
only: does this stream, at this layer, register where the thing was moved to? R^2 per layer and stream, group k-fold over
episodes, separately for small (<= 6 cm) and large (>= 12 cm) displacements and for the swap.
    python scripts/analyze_internals.py outputs/week1/internals_object_base [...]
"""
import json
import sys

import numpy as np

from probe_internals import STREAMS, ridge_r2  # sibling script


def main():
    for run in sys.argv[1:]:
        z = np.load(f"{run}/internals.npz")
        F = z["feats"]
        meta = json.load(open(f"{run}/meta.json"))
        d_pre = (F.shape[2] - 1024) // 3  # pooled prefix streams (PaliGemma width), then the action expert (width 1024)
        cuts = {"img": (0, d_pre), "task": (d_pre, 2 * d_pre), "last": (2 * d_pre, 3 * d_pre), "act": (3 * d_pre, F.shape[2])}
        base = {(m["task_id"], m["trial_id"], m["q"]): i for i, m in enumerate(meta) if m["variant"] == "base"}
        print(f"=== {run}  ({len(meta)} queries, {F.shape[1]} layers)")
        for phase, key in (("pre", "tgt_rel"), ("post", "con_rel")):
            for sizes, label in (((0.03, 0.06), "small (<= 6 cm)"), ((0.12, 0.2), "large (>= 12 cm)"), (None, "swap")):
                idx, ys, grp = [], [], []
                for i, m in enumerate(meta):
                    if m["phase"] != phase or m["variant"] == "base":
                        continue
                    if sizes is None and m["variant"] != "swap":
                        continue
                    if sizes is not None and (not m["variant"].startswith("img_") or float(m["variant"][4:]) not in sizes):
                        continue
                    b = base.get((m["task_id"], m["trial_id"], m["q"]))
                    if b is None:
                        continue
                    idx.append((i, b))
                    ys.append(np.array(m[key]) - np.array(meta[b][key]))
                    grp.append(f"{m['task_id']}_{m['trial_id']}")
                if len(idx) < 30:
                    continue
                Y, G = np.array(ys), np.array(grp)
                print(f"{phase} {key} {label}: n {len(idx)}")
                for st in STREAMS:
                    a, b_ = cuts[st]
                    r2 = [ridge_r2((F[[i for i, _ in idx], l, a:b_].astype(np.float32) - F[[b for _, b in idx], l, a:b_].astype(np.float32)),
                                   Y, G) for l in range(F.shape[1])]
                    print(f"  {st:5s} " + " ".join(f"{x:5.2f}" for x in r2))


if __name__ == "__main__":
    main()
