"""Paper figure: factual and counterfactual agent views of kept pairs, one example per kind, from the pair_examples.npz
files that train_pi05_ocd.py --dump_segments writes (full mode). Frames are shown as the policy sees them (the agent view
looks at the robot across the table).
    python scripts/figure_cf_examples.py --out <dir> --pick 0:<npz>:<i> 2:<npz>:<i> ...     (kind:file:index of that kind)
    python scripts/figure_cf_examples.py --list <npz> ...                                    (what each file holds)
"""
import argparse
import os

import numpy as np
from PIL import Image

KINDS = {0: "retarget", 1: "coshift", 2: "relocate", 3: "furniture"}


def upright(img):
    return np.ascontiguousarray(np.asarray(img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", nargs="*", default=[])
    ap.add_argument("--pick", nargs="*", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--sheet", default="", help="also save a contact sheet per kind of every example of the listed files")
    args = ap.parse_args()
    for f in args.list:
        z = np.load(f)
        print(f)
        for i, (k, t, d, dc) in enumerate(zip(z["kind"], z["tid"], z["desc"], z["desc_cf"])):
            n = int((z["kind"][:i] == k).sum())
            print(f"  kind {int(k)} ({KINDS[int(k)]}) #{n} task {int(t)}: {d}" + (f"  ->  {dc}" if dc != d else ""))
        if args.sheet:
            for k in sorted(set(int(x) for x in z["kind"])):
                idx = np.flatnonzero(z["kind"] == k)
                rows = [np.concatenate([upright(z["rgb"][i]), upright(z["rgb_cf"][i])], axis=1) for i in idx]
                grid = np.concatenate([np.concatenate(rows[r : r + 3], axis=1) for r in range(0, len(rows) - len(rows) % 3 or 3, 3)
                                       if len(rows[r : r + 3]) == 3], axis=0) if len(rows) >= 3 else np.concatenate(rows, axis=1)
                name = os.path.join(args.sheet, f"{os.path.basename(os.path.dirname(f))}_{KINDS[k]}.png")
                Image.fromarray(grid).save(name)
                print("  sheet ->", name)
    for spec in args.pick:
        k, f, i = spec.split(":")
        z = np.load(f)
        idx = np.flatnonzero(z["kind"] == int(k))[int(i)]
        for key, tag in (("rgb", "fact"), ("rgb_cf", "cf")):
            Image.fromarray(upright(z[key][idx])).save(os.path.join(args.out, f"cf_{KINDS[int(k)]}_{tag}.png"))
        print("saved", KINDS[int(k)], "task", int(z["tid"][idx]), str(z["desc"][idx]))


if __name__ == "__main__":
    main()
