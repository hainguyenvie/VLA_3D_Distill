"""Paper figure: factual and counterfactual agent views of kept pairs, one example per kind, from the pair_examples.npz
files that train_pi05_ocd.py --dump_segments writes (full mode). Frames are stored in the policy's convention (rotated by
180 degrees); they are turned upright here.
    python scripts/figure_cf_examples.py --out <dir> --pick 0:<npz>:<i> 2:<npz>:<i> ...     (kind:file:index of that kind)
    python scripts/figure_cf_examples.py --list <npz> ...                                    (what each file holds)
"""
import argparse
import os

import numpy as np
from PIL import Image

KINDS = {0: "retarget", 1: "coshift", 2: "relocate", 3: "furniture"}


def upright(img):
    return np.ascontiguousarray(np.asarray(img)[::-1, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", nargs="*", default=[])
    ap.add_argument("--pick", nargs="*", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--sheet", default="", help="also save a contact sheet of every example of the listed files")
    args = ap.parse_args()
    for f in args.list:
        z = np.load(f)
        print(f)
        for i, (k, t, d, dc) in enumerate(zip(z["kind"], z["tid"], z["desc"], z["desc_cf"])):
            n = int((z["kind"][:i] == k).sum())
            print(f"  kind {int(k)} ({KINDS[int(k)]}) #{n} task {int(t)}: {d}" + (f"  ->  {dc}" if dc != d else ""))
        if args.sheet:
            rows = [np.concatenate([upright(a), upright(b)], axis=1) for a, b in zip(z["rgb"], z["rgb_cf"])]
            pad = 4 - len(rows) % 4 if len(rows) % 4 else 0
            rows += [np.zeros_like(rows[0])] * pad
            grid = np.concatenate([np.concatenate(rows[r : r + 4], axis=1) for r in range(0, len(rows), 4)], axis=0)
            name = os.path.join(args.sheet, os.path.basename(os.path.dirname(f)) + "_sheet.png")
            Image.fromarray(grid).resize((grid.shape[1] // 2, grid.shape[0] // 2)).save(name)
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
