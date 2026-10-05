"""Perception or control? Probe the frozen policy for the gripper-to-target offset during the approach.

The dominant student failure is closing the gripper a few centimetres off the target. This asks whether the
policy's own representation already "knows" where the target is relative to the gripper on those states:
a small probe is fitted on frozen features to regress the vector end-effector -> target object (metres, from
the logged simulator state), using only states before the first grasp attempt, and its error is compared
between episodes that go on to succeed and episodes that go on to fail.

Features (one probe each):
  action   last-layer hidden states at the 56 action positions (what the policy decodes its actions from)
  visual   visual tokens of one LLM layer; they precede the instruction, so the pooling queries are per task
Reference: `memorised` predicts the mean training offset of the same task and query index (no image at all).
If the policy acts on its own estimate, the gripper should miss where the probe mislocates the target, so the
error of the probe is also correlated with the actual miss vector at the first grasp attempt.
    python scripts/probe_offset.py --run outputs/week1/b0_object_student --ckpt <student> --out outputs/week1/probe_offset
Writes <out>/states.csv (held-out states) and <out>/summary.json. Needs <run>/failures.csv (analyze_failures.py).
"""
import argparse
import csv
import json
import os

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--lora", default="", help="adapter of a distilled student")
    ap.add_argument("--visual_layer", type=int, default=24)
    ap.add_argument("--queries", type=int, default=4, help="attention-pooling queries per probe")
    ap.add_argument("--holdout_from", type=int, default=40, help="trials >= this id are held out")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    import torch
    import torch.nn as nn
    from libero.libero import benchmark

    from scripts.analyze_failures import task_layout
    from src.policy.token_policy import TokenPolicy

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    fails = {(int(r["task_id"]), int(r["trial_id"])): r for r in csv.DictReader(open(os.path.join(args.run, "failures.csv")))}
    eps = [json.loads(l) for l in open(os.path.join(args.run, "episodes.jsonl"))]

    # approach states only: queries up to (and including) the one whose chunk first closes the gripper
    layouts, data = {}, []
    for e in eps:
        f = fails[(e["task_id"], e["trial_id"])]
        if e["task_id"] not in layouts:
            layouts[e["task_id"]] = task_layout(suite, e["task_id"])
        addr, target, _ = layouts[e["task_id"]]
        z = np.load(os.path.join(args.run, "steps", f"t{e['task_id']:02d}_n{e['trial_id']:02d}.npz"))
        close = z["actions"][:, :, -1].min(1) < 0.5
        last = int(np.argmax(close)) if close.any() else len(close) - 1
        a = addr[target]
        tgt = z["sim_state"][:, 1 + a : 1 + a + 3]
        at = min(last + 1, len(close) - 1)  # state after the chunk that closed the gripper, as in analyze_failures
        miss = (z["eef_pos"][at] - tgt[at]) if close.any() else np.full(3, np.nan)
        data.append({"task_id": e["task_id"], "trial_id": e["trial_id"], "task": e["task"], "success": int(e["success"]),
                     "mode": f["mode"], "rgb": z["rgb"][: last + 1], "miss": miss,
                     "offset": (tgt[: last + 1] - z["eef_pos"][: last + 1]).astype(np.float32), "t": z["t"][: last + 1]})
    train = [d for d in data if d["trial_id"] < args.holdout_from]
    test = [d for d in data if d["trial_id"] >= args.holdout_from]
    print(f"approach states: {sum(len(d['t']) for d in train)} train, {sum(len(d['t']) for d in test)} held out", flush=True)

    # no-image reference: mean training offset of the same task and query index (falls back to the task's last index)
    memo = {}
    for d in train:
        for i, o in enumerate(d["offset"]):
            memo.setdefault((d["task_id"], i), []).append(o)
    memo = {k: np.mean(v, 0) for k, v in memo.items()}
    memo_last = {t: max(i for (tt, i) in memo if tt == t) for t in {k[0] for k in memo}}

    policy = TokenPolicy(args.ckpt, args.suite)
    if args.lora:
        policy.add_lora(adapter_path=args.lora)
    dev, D, K = policy.device, policy.vla.language_model.config.hidden_size, args.queries
    n_tasks = max(d["task_id"] for d in data) + 1

    class Probe(nn.Module):  # attention pooling over tokens with K queries, then an MLP to the 3-d offset
        def __init__(self, query_sets):
            super().__init__()
            self.norm, self.query = nn.LayerNorm(D), nn.Parameter(torch.randn(query_sets, K, D) / D**0.5)
            self.head = nn.Sequential(nn.Linear(K * D, 512), nn.GELU(), nn.Linear(512, 3))

        def forward(self, tokens, task_ids):
            x = self.norm(tokens)
            w = torch.softmax(torch.einsum("bnd,bkd->bnk", x, self.query[task_ids % len(self.query)]), dim=1)
            return self.head(torch.einsum("bnk,bnd->bkd", w, x).flatten(1))

    probes = {"action": Probe(1).to(dev), "visual": Probe(n_tasks).to(dev)}
    opt = torch.optim.AdamW([p for m in probes.values() for p in m.parameters()], lr=args.lr, weight_decay=0.0)

    def batch(eps, pairs):
        """Frozen features, task ids and offsets (cm) of the states `pairs` = [(episode index, query index), ...]."""
        rgb = [eps[e]["rgb"][i] for e, i in pairs]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out, idx, P = policy._run_llm(policy.build_inputs(rgb, [eps[e]["task"] for e, _ in pairs]))
            act = out.hidden_states[-1][torch.arange(len(rgb), device=dev)[:, None], idx]
            vis = out.hidden_states[args.visual_layer][:, 1 : 1 + P]
        tid = torch.tensor([eps[e]["task_id"] for e, _ in pairs], device=dev)
        y = torch.from_numpy(np.stack([eps[e]["offset"][i] for e, i in pairs])).to(dev) * 100.0
        return {"action": act.float(), "visual": vis.float()}, tid, y

    def predict(eps):
        """Offsets (m) predicted for every state of `eps`, per probe: {kind: [array (n_i, 3) per episode]}."""
        pairs = [(e, i) for e, d in enumerate(eps) for i in range(len(d["t"]))]
        out = {k: [] for k in probes}
        for s in range(0, len(pairs), args.batch_size):
            feats, tid, _ = batch(eps, pairs[s : s + args.batch_size])
            with torch.no_grad():
                for k in probes:
                    out[k].append(probes[k](feats[k], tid).cpu().numpy() / 100.0)
        cuts = np.cumsum([len(d["t"]) for d in eps])[:-1]
        return {k: np.split(np.concatenate(v), cuts) for k, v in out.items()}

    train_pairs = [(e, i) for e, d in enumerate(train) for i in range(len(d["t"]))]
    total = args.epochs * ((len(train_pairs) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    curve = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_pairs))
        for s in range(0, len(order), args.batch_size):
            feats, tid, y = batch(train, [train_pairs[j] for j in order[s : s + args.batch_size]])
            loss = sum(nn.functional.smooth_l1_loss(probes[k](feats[k], tid), y) for k in probes)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        preds = predict(test)
        held = {k: round(float(np.median(np.concatenate(
            [np.linalg.norm((p - d["offset"])[:, :2], axis=1) for p, d in zip(preds[k], test)])) * 100), 2) for k in probes}
        curve.append(held)
        print(f"epoch {epoch}: last train loss {float(loss) / len(probes):.3f}, held-out median xy error (cm) {held}", flush=True)

    KINDS = ("action", "visual", "memorised")
    rows = []
    for e, d in enumerate(test):
        n = len(d["t"])
        pred = {k: preds[k][e] for k in probes}
        pred["memorised"] = np.stack([memo[(d["task_id"], min(i, memo_last[d["task_id"]]))] for i in range(n)])
        for i in range(n):
            row = {"task_id": d["task_id"], "trial_id": d["trial_id"], "success": d["success"], "mode": d["mode"], "t": int(d["t"][i]),
                   "to_first_close": n - 1 - i, "dist_m": round(float(np.linalg.norm(d["offset"][i])), 4),
                   "miss_x_cm": round(float(d["miss"][0]) * 100, 3), "miss_y_cm": round(float(d["miss"][1]) * 100, 3)}
            for k in KINDS:
                err = (pred[k][i] - d["offset"][i]) * 100
                row[f"err_{k}_cm"] = round(float(np.linalg.norm(err)), 3)
                row[f"err_{k}_xy_cm"] = round(float(np.linalg.norm(err[:2])), 3)
                row[f"err_{k}_x_cm"], row[f"err_{k}_y_cm"] = round(float(err[0]), 3), round(float(err[1]), 3)
            rows.append(row)
    with open(os.path.join(args.out, "states.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def stat(sel, key):
        v = [r[key] for r in rows if sel(r)]
        return {"n": len(v), "median": round(float(np.median(v)), 2), "mean": round(float(np.mean(v)), 2)} if v else None

    groups = {"success": lambda r: r["success"] == 1, "failure": lambda r: r["success"] == 0,
              "near_miss": lambda r: r["mode"] == "near_miss", "target_toppled": lambda r: r["mode"] == "target_toppled",
              "wrong_object_or_place": lambda r: r["mode"] in ("wrong_object", "wrong_place")}

    def miss_corr(sel, k):
        """Pearson r between the probe's xy error one query before the gripper closes and the gripper's xy miss."""
        v = [r for r in rows if sel(r) and r["to_first_close"] == 1 and np.isfinite(r["miss_x_cm"])]
        if len(v) < 8:
            return None
        e = np.array([[r[f"err_{k}_x_cm"], r[f"err_{k}_y_cm"]] for r in v]).ravel()
        m = np.array([[r["miss_x_cm"], r["miss_y_cm"]] for r in v]).ravel()
        return {"n_episodes": len(v), "r": round(float(np.corrcoef(e, m)[0, 1]), 3)}

    summary = {"all_approach_states": {g: {k: stat(sel, f"err_{k}_xy_cm") for k in KINDS} for g, sel in groups.items()},
               # the last two queries before the gripper closes: where the grasp position is decided
               "last_two_queries_before_close": {g: {k: stat(lambda r, s=sel: s(r) and r["to_first_close"] <= 1, f"err_{k}_xy_cm") for k in KINDS}
                                                 for g, sel in groups.items()},
               "probe_error_vs_miss_vector": {g: {k: miss_corr(groups[g], k) for k in KINDS} for g in ("success", "failure", "near_miss")},
               "held_out_median_xy_cm_by_epoch": curve, "config": vars(args)}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))
    print("PROBE_OFFSET_DONE", flush=True)


if __name__ == "__main__":
    main()
