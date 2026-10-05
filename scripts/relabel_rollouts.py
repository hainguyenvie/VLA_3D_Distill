"""Label the logged states of a rollout with another policy and write per-state disagreement statistics.

Used to ask, without touching the simulator: on the states the student actually visits, how confident is
a candidate teacher, how far is the student from it, and does that depend on whether the episode fails?
    python scripts/relabel_rollouts.py --run outputs/week1/b0_object_student --student <ckpt> \
        --teacher rebin:<full-SFT ckpt> --name fullsft
Writes <run>/relabel_<name>.csv (one row per state) and <run>/relabel_<name>.json (group means).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True, help="checkpoint, or rebin:checkpoint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F

    from src.policy.rebin import load_policy
    from src.policy.token_policy import TokenPolicy, preprocess_batch

    student = TokenPolicy(args.student, args.suite)
    teacher = load_policy(args.teacher, args.suite, "cuda:0", target=student)
    eps = {(e["task_id"], e["trial_id"]): e for e in map(json.loads, open(os.path.join(args.run, "episodes.jsonl")))}
    rows = []
    for path in sorted(glob.glob(os.path.join(args.run, "steps", "*.npz"))):
        base = os.path.basename(path)
        ep = eps.get((int(base[1:3]), int(base[5:7])))
        if ep is None:
            continue
        z = np.load(path)
        rgb, n = z["rgb"], len(z["t"])
        ls_all, lt_all = [], []
        for s in range(0, n, args.batch_size):
            imgs = list(rgb[s : s + args.batch_size])
            pils = preprocess_batch(imgs)
            descs = [ep["task"]] * len(imgs)
            ls_all.append(student.act(imgs, descs, pils=pils)["logits"])
            lt_all.append(teacher.act(imgs, descs, pils=pils)["logits"])
        ls, lt = F.log_softmax(torch.cat(ls_all), -1), F.log_softmax(torch.cat(lt_all), -1)
        bins = torch.arange(ls.shape[-1], dtype=torch.float32)
        stats = {
            "kl_student_teacher": (ls.exp() * (ls - lt)).sum(-1).mean(-1),
            "entropy_student": -(ls.exp() * ls).sum(-1).mean(-1),
            "entropy_teacher": -(lt.exp() * lt).sum(-1).mean(-1),
            "argmax_agree": (ls.argmax(-1) == lt.argmax(-1)).float().mean(-1),
            # distance between the two expected actions, in bins: separates near misses from different actions
            "mean_bin_dist": ((ls.exp() * bins).sum(-1) - (lt.exp() * bins).sum(-1)).abs().mean(-1),
            "teacher_top1_prob": lt.exp().max(-1).values.mean(-1),
        }
        for i in range(n):
            rows.append({"task_id": ep["task_id"], "trial_id": ep["trial_id"], "query": i, "frac": round(i / max(n - 1, 1), 4),
                         "success": int(ep["success"]), **{k: round(float(v[i]), 5) for k, v in stats.items()}})
    with open(os.path.join(args.run, f"relabel_{args.name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    keys = ["kl_student_teacher", "entropy_student", "entropy_teacher", "argmax_agree", "mean_bin_dist", "teacher_top1_prob"]
    groups = {"success": lambda r: r["success"] == 1, "failure": lambda r: r["success"] == 0,
              "failure_first_third": lambda r: r["success"] == 0 and r["frac"] < 1 / 3,
              "failure_last_third": lambda r: r["success"] == 0 and r["frac"] > 2 / 3}
    out = {g: {"n_states": sum(map(sel, rows)), **{k: float(np.mean([r[k] for r in rows if sel(r)])) for k in keys}}
           for g, sel in groups.items()}
    json.dump(out, open(os.path.join(args.run, f"relabel_{args.name}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("RELABEL_DONE")


if __name__ == "__main__":
    main()
