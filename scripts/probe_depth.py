"""Cheap diagnostic before any 3D distillation: how well do the frozen student's visual tokens encode depth,
and does that degrade on the states where the student disagrees with the teacher?

A light depth head is trained on the frozen student's features (one head per requested layer) using rollout
logs written by scripts/eval_libero.py with --depth. The probe is fitted on the training split of *all*
given datasets (so a gap between datasets is not just probe distribution shift) and evaluated per state on
the held-out trials. Output: <out>/states.csv (one row per held-out state) and <out>/summary.json.

    python scripts/probe_depth.py --ckpt <student> --data teacher=outputs/week1/teacher_object \
        --data student=outputs/week1/b0_object_student --out outputs/week1/probe_depth
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--data", action="append", required=True, help="name=rollout dir (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="8,16,24,32")
    ap.add_argument("--holdout_from", type=int, default=40, help="trials >= this id are held out")
    ap.add_argument("--stride", type=int, default=2, help="use every k-th queried state for fitting")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_episodes(name, root):
    meta = {(r["task_id"], r["trial_id"]): r for r in map(json.loads, open(os.path.join(root, "episodes.jsonl")))}
    eps = []
    for path in sorted(glob.glob(os.path.join(root, "steps", "*.npz"))):
        base = os.path.basename(path)
        key = (int(base[1:3]), int(base[5:7]))
        if key in meta:
            eps.append({"dataset": name, "path": path, **meta[key]})
    return eps


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    import torch
    import torch.nn.functional as F

    from src.distill.spatial_loss import DepthHead, depth_loss, depth_metrics, depth_target
    from src.policy.token_policy import TokenPolicy

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    layers = [int(x) for x in args.layers.split(",")]
    policy = TokenPolicy(args.ckpt, args.suite)
    dev = policy.device
    heads = {l: DepthHead().to(dev) for l in layers}
    opt = torch.optim.AdamW([p for h in heads.values() for p in h.parameters()], lr=args.lr, weight_decay=0.0)

    episodes = [e for spec in args.data for e in load_episodes(*spec.split("=", 1))]
    train = [e for e in episodes if e["trial_id"] < args.holdout_from]
    test = [e for e in episodes if e["trial_id"] >= args.holdout_from]
    print(f"episodes: {len(train)} train, {len(test)} held out", flush=True)

    def features(rgb, task):
        with torch.inference_mode():
            _, hid = policy.forward_logits(policy.build_inputs(list(rgb), [task] * len(rgb)), hidden_layers=layers)
        return {l: h.float() for l, h in hid.items()}

    step = 0
    for epoch in range(args.epochs):
        for ei in rng.permutation(len(train)):
            ep = train[ei]
            z = np.load(ep["path"])
            rgb, depth = z["rgb"], z["depth"]  # each access to an npz member decompresses it again
            idx = rng.permutation(np.arange(rng.integers(args.stride), len(rgb), args.stride))
            for s in range(0, len(idx), args.batch_size):
                j = np.sort(idx[s : s + args.batch_size])
                feats = features(rgb[j], ep["task"])
                target = depth_target(torch.from_numpy(depth[j]).to(dev))
                loss = sum(depth_loss(heads[l](feats[l].clone()), target) for l in layers)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                if step % 100 == 0:
                    print(f"epoch {epoch} step {step} loss/layer {float(loss) / len(layers):.4f}", flush=True)

    rows = []
    for ep in test:
        z = np.load(ep["path"])
        rgb, depth = z["rgb"], z["depth"]
        n = len(rgb)
        errs = {l: [] for l in layers}
        for s in range(0, n, args.batch_size):
            feats = features(rgb[s : s + args.batch_size], ep["task"])
            target = depth_target(torch.from_numpy(depth[s : s + args.batch_size]).to(dev))
            for l in layers:
                with torch.no_grad():
                    errs[l].append(depth_metrics(heads[l](feats[l]), target)["abs_rel"].cpu().numpy())
        ls = F.log_softmax(torch.from_numpy(z["logits"].astype(np.float32)), -1)
        other = [k for k in z.files if k.startswith("logits_")]
        if other:  # KL(actor || labeler) and its reverse, per state, mean over the 56 action tokens
            lo = F.log_softmax(torch.from_numpy(z[other[0]].astype(np.float32)), -1)
            kl_actor_label = (ls.exp() * (ls - lo)).sum(-1).mean(-1).numpy()
            kl_label_actor = (lo.exp() * (lo - ls)).sum(-1).mean(-1).numpy()
        for i in range(n):
            row = {"dataset": ep["dataset"], "task_id": ep["task_id"], "trial_id": ep["trial_id"], "query": i,
                   "t": int(z["t"][i]), "frac": i / max(n - 1, 1), "success": int(ep["success"])}
            if other:
                # always expressed as KL(student || teacher): the student is the actor in its own rollouts
                row["kl_student_teacher"] = float(kl_actor_label[i] if other[0] == "logits_teacher" else kl_label_actor[i])
            for l in layers:
                row[f"depth_err_l{l}"] = float(np.concatenate(errs[l])[i])
            rows.append(row)
    with open(os.path.join(args.out, "states.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def mean(sel, key):
        v = [r[key] for r in rows if sel(r) and key in r]
        return float(np.mean(v)) if v else None

    groups = {}
    for name in sorted({r["dataset"] for r in rows}):
        for tag, sel in (("all", lambda r: True), ("success", lambda r: r["success"] == 1), ("failure", lambda r: r["success"] == 0)):
            f = lambda r, n=name, s=sel: r["dataset"] == n and s(r)  # noqa: E731
            groups[f"{name}/{tag}"] = {"n_states": sum(1 for r in rows if f(r)), "kl_student_teacher": mean(f, "kl_student_teacher"),
                                       **{f"depth_err_l{l}": mean(f, f"depth_err_l{l}") for l in layers}}
    summary = {"groups": groups, "config": vars(args)}
    kl = np.array([r.get("kl_student_teacher", np.nan) for r in rows])
    ok = ~np.isnan(kl)
    if ok.sum() > 10:  # rank correlation between disagreement and depth-probe error, over all held-out states
        rk = lambda a: np.argsort(np.argsort(a)).astype(float)  # noqa: E731
        summary["spearman_kl_vs_depth_err"] = {
            f"l{l}": float(np.corrcoef(rk(kl[ok]), rk(np.array([r[f"depth_err_l{l}"] for r in rows])[ok]))[0, 1]) for l in layers}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary["groups"], indent=1))
    print("spearman", summary.get("spearman_kl_vs_depth_err"))
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
