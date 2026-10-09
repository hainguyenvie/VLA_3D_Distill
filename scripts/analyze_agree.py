"""Audit of the teacher-expert agreement check (train_pi05_ocd.py --cf_agree): from the pairs a run dumped at iteration 1
(<run>/agree_pairs.npz: the scripted teacher's chunk in the factual world, pi0.5's chunk there, task ids, phase), the share
of pairs each candidate criterion keeps, per task and phase (before the grasp / while carrying).
    python scripts/analyze_agree.py outputs/week1/smoke_uni_gl2 [...]
"""
import sys

import numpy as np


def cos(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / (na * nb + 1e-8), na, nb


def criteria(nom, exp, eef=None):
    """name -> boolean keep per pair."""
    out = {}
    if eef is not None:  # where each chunk would bring the hand (kinematic model of src/rollout/scripted.py)
        from src.rollout.scripted import TRACK_GAIN

        for k, dmax in ((25, 0.05), (25, 0.08), (50, 0.08)):
            end_t = (TRACK_GAIN * nom[:, :k, :3] * 0.05).sum(1)
            end_e = (TRACK_GAIN * exp[:, :k, :3] * 0.05).sum(1)
            out[f"e{k}<{int(dmax * 100)}"] = np.linalg.norm(end_t - end_e, axis=1) < dmax
    for k in (10, 25, 50):
        for dims, tag in (((0, 1, 2), "xyz"), ((0, 1), "xy")):
            c, nt, ne = cos(nom[:, :k, dims].sum(1), exp[:, :k, dims].sum(1))
            small = (nt < 0.05 * k) & (ne < 0.05 * k)  # both nearly still: nothing to compare
            out[f"{tag}{k}"] = (c >= 0.5) | small
    return out


def main():
    for run in sys.argv[1:]:
        z = np.load(f"{run}/agree_pairs.npz")
        nom, exp, tids, post, ok = z["nom"], z["expert"], z["tids"], z["post"], z["cf_ok"]
        has = np.abs(nom).sum((1, 2)) > 0
        sel = ok & has
        crit = criteria(nom, exp, z["eef"] if "eef" in z else None)
        print(f"== {run}: {sel.sum()} pairs with a factual-world label")
        print("task phase  n   " + " ".join(f"{k:>6}" for k in crit))
        for t in np.unique(tids):
            for p, ph in ((False, "pre "), (True, "post")):
                m = sel & (tids == t) & (post == p)
                if m.sum() == 0:
                    continue
                print(f"{t:4d} {ph} {m.sum():4d}  " + " ".join(f"{crit[k][m].mean():6.2f}" for k in crit))


if __name__ == "__main__":
    main()
