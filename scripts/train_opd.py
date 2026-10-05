"""Distil a frozen teacher into a LoRA student on LIBERO, either on-policy or on teacher-visited states.

One iteration = collect fresh rollouts, label every queried state with the teacher's token distribution,
take gradient steps on a fixed number of those states, discard them (VLA-OPD's loop, without its GRPO stage).

    --state_source student   the student acts, the teacher only labels (on-policy distillation, B2)
    --state_source teacher   the teacher acts and labels (same loss, teacher-state distribution: the matched
                             off-policy control)
    --state_source mixed     alternate batches of episodes driven by the student and by the teacher
    --spatial depth          add the privileged depth loss on the same states (training only); with the two
                             state sources this gives the 2 x 2 design {student, teacher states} x {no 3D, 3D}
    --view_aug               re-render the training states under random camera / light / sensor perturbations: the
                             student is fed (and, when it drives, acts on) the perturbed view, the teacher labels and
                             drives from the nominal view of the same simulator state. Evaluation stays nominal.

Outputs in --out: train_log.jsonl (one line per iteration), adapter_last/ + state.pt (resume), and
adapter_iterXXXX/ at every evaluation.
"""
import argparse
import json
import os
import time

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True, help="checkpoint, or rebin:checkpoint when its action normalisation differs from the student's")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state_source", choices=["student", "teacher", "mixed"], default="student")
    ap.add_argument("--view_aug", action="store_true", help="visual perturbations of the training rollouts (vec_env.VIEW_AUG)")
    ap.add_argument("--mode", choices=["rkl", "rkl_pg", "fkl", "ce"], default="rkl")
    ap.add_argument("--spatial", choices=["none", "depth"], default="none")
    ap.add_argument("--spatial_layer", type=int, default=24, help="LLM layer whose visual tokens feed the depth head (1..32)")
    ap.add_argument("--lambda_spatial", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--states_per_iter", type=int, default=2048,
                    help="queried states trained on per iteration; episodes are collected until there are enough, so "
                         "arms with short episodes (teacher) and long ones (failing student) get the same updates")
    ap.add_argument("--episodes_per_batch", type=int, default=32, help="episodes collected at a time until states_per_iter is reached")
    ap.add_argument("--train_trials", type=int, default=50, help="rollouts start from the first N benchmark init states")
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=512)
    ap.add_argument("--rollout_temperature", type=float, default=1.6, help="0 = greedy rollouts")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=0.0,
                    help="> 0: cosine decay of the learning rate over the iterations, from --lr down to this value")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_checkpointing", action="store_true")
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_trials", type=int, default=10, help="greedy evaluation on the first N init states of each task")
    ap.add_argument("--student_device", default="cuda:0")
    ap.add_argument("--teacher_device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    from src.rollout.vec_env import VIEW_AUG, LiberoVecEnv

    vec = LiberoVecEnv(args.suite, args.num_envs, args.max_steps, depth=args.spatial != "none",
                       perturb=VIEW_AUG if args.view_aug else None)

    import torch

    from src.distill.opd_loss import opd_loss, token_metrics
    from src.distill.spatial_loss import DepthHead, depth_loss, depth_target
    from src.policy.rebin import load_policy
    from src.policy.token_policy import TokenPolicy
    from src.rollout.collector import Collector

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    student = TokenPolicy(args.student, args.suite, device=args.student_device)
    teacher = load_policy(args.teacher, args.suite, args.teacher_device, target=student)
    last = os.path.join(args.out, "adapter_last")
    state_path = os.path.join(args.out, "state.pt")
    resume = os.path.exists(state_path)
    params = student.add_lora(args.lora_rank, adapter_path=last if resume else None)
    if args.grad_checkpointing:
        student.vla.language_model.gradient_checkpointing_enable()
    head = DepthHead().to(student.device) if args.spatial == "depth" else None
    if head is not None:
        params = params + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    start_it, totals = 0, {"episodes": 0, "env_steps": 0, "states": 0, "grad_steps": 0}
    if resume:
        st = torch.load(state_path, map_location="cpu")
        opt.load_state_dict(st["opt"])
        if head is not None:
            head.load_state_dict(st["head"])
        start_it, totals = st["iter"], st["totals"]
        rng = np.random.default_rng([args.seed, start_it])
        print(f"resumed at iteration {start_it}", flush=True)
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

    sample = args.rollout_temperature > 0
    kw = dict(sample=sample, temperature=args.rollout_temperature, seed=args.seed, perturb=args.view_aug)
    cols = []  # (collector, key of the teacher logits in its records): one per driver of the rollouts
    if args.state_source in ("student", "mixed"):
        cols.append((Collector(vec, student, {"teacher": teacher}, **kw), "logits_teacher"))
    if args.state_source in ("teacher", "mixed"):
        cols.append((Collector(vec, teacher, {}, actor_view="rgb_clean", **kw), "logits"))
    col = cols[0][0]
    evalc = Collector(vec, student, {}, sample=False, seed=args.seed)
    task_ids = sorted(col.tasks)
    pool = [(t, n) for t in task_ids for n in range(min(args.train_trials, col.tasks[t][1]))]
    eval_eps = [(t, n) for t in task_ids for n in range(min(args.eval_trials, col.tasks[t][1]))]
    log_path = os.path.join(args.out, "train_log.jsonl")

    def evaluate():
        """Greedy success rate of the student, overall and per task (per task: only eval_trials episodes each)."""
        recs = evalc.run(eval_eps)
        per_task = [float(np.mean([r["success"] for r in recs if r["task_id"] == t])) for t in task_ids]
        return {"eval_sr": float(np.mean([r["success"] for r in recs])), "eval_per_task": per_task}

    def log(row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    if start_it == 0 and args.eval_every > 0:
        log({"iter": 0, **evaluate(), **totals})

    for it in range(start_it + 1, args.iters + 1):
        t0 = time.time()
        order, recs, n_batches = rng.permutation(len(pool)), [], 0
        while sum(r["n_queries"] for r in recs) < args.states_per_iter and len(recs) < len(pool):
            batch = order[len(recs) : len(recs) + args.episodes_per_batch]
            c, t_key = cols[n_batches % len(cols)]
            new = c.run([pool[i] for i in batch], keep_steps=True)
            for r in new:
                r["teacher_logits"], r["driver"] = r["arrays"][t_key], "student" if t_key == "logits_teacher" else "teacher"
            recs, n_batches = recs + new, n_batches + 1
        n_collected = sum(r["n_queries"] for r in recs)
        keep = np.sort(rng.choice(n_collected, size=min(args.states_per_iter, n_collected), replace=False))
        images = np.concatenate([r["arrays"]["rgb"] for r in recs])[keep]
        t_logits = np.concatenate([r["teacher_logits"] for r in recs])[keep]
        bins = np.concatenate([r["arrays"]["bins"] for r in recs])[keep]
        depth = np.concatenate([r["arrays"]["depth"] for r in recs])[keep] if head is not None else None
        all_descs = [r["task"] for r in recs for _ in range(r["n_queries"])]
        descs = [all_descs[k] for k in keep]
        t_collect = time.time() - t0
        tm = {k: round(sum(c.timing[k] for c, _ in cols), 1) for k in col.timing}
        for c, _ in cols:
            c.timing.update({k: 0 if k == "rounds" else 0.0 for k in c.timing})

        if args.lr_min > 0:  # one value per iteration; a function of the iteration only, so a resumed run continues it
            lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + np.cos(np.pi * (it - 1) / max(args.iters - 1, 1)))
            for g in opt.param_groups:
                g["lr"] = float(lr)
        losses, sp_losses, mets, n = [], [], [], len(images)
        opt.zero_grad(set_to_none=True)
        if args.grad_checkpointing:  # HF only checkpoints in train mode (the LLM has no dropout, so nothing else changes)
            student.vla.language_model.train()
        for _ in range(args.epochs):
            order = rng.permutation(n)
            for b, s in enumerate(range(0, n, args.batch_size)):
                j = order[s : s + args.batch_size]
                inputs = student.build_inputs([images[k] for k in j], [descs[k] for k in j])
                logits, hid = student.forward_logits(inputs, hidden_layers=[args.spatial_layer] if head is not None else None)
                tl = torch.from_numpy(t_logits[j].astype(np.float32)).to(logits.device)
                loss = opd_loss(logits, tl, args.mode, bins=torch.from_numpy(bins[j]).to(logits.device),
                                temperature=args.rollout_temperature if args.mode == "rkl_pg" and sample else 1.0)
                total = loss
                if head is not None:
                    sp = depth_loss(head(hid[args.spatial_layer]), depth_target(torch.from_numpy(depth[j]).to(logits.device)))
                    total = loss + args.lambda_spatial * sp
                    sp_losses.append(float(sp))
                (total / args.grad_accum).backward()
                if (b + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    totals["grad_steps"] += 1
                losses.append(float(loss))
                if b % 20 == 0:
                    mets.append(token_metrics(logits.detach(), tl))
        opt.zero_grad(set_to_none=True)
        student.vla.eval()

        totals["episodes"] += len(recs)
        totals["env_steps"] += int(sum(r["env_steps"] for r in recs))
        totals["states"] += n
        totals["states_collected"] = totals.get("states_collected", 0) + n_collected
        row = {"iter": it, "rollout_sr": float(np.mean([r["success"] for r in recs])), "n_states": n,
               "n_episodes": len(recs), "n_states_collected": n_collected,
               "loss": float(np.mean(losses)), **{f"train_{k}": float(np.mean([m[k] for m in mets])) for k in mets[0]},
               "sec_collect": round(t_collect), "sec_total": round(time.time() - t0), "collect_timing": tm,
               "gpu_peak_gb": round(torch.cuda.max_memory_allocated(student.device) / 1e9, 1),
               "lr": opt.param_groups[0]["lr"], **totals}
        if sp_losses:
            row["spatial_loss"] = float(np.mean(sp_losses))
        for who in ("student", "teacher"):  # success of the rollouts by who drove them (under view_aug: perturbed for the student)
            v = [r["success"] for r in recs if r["driver"] == who]
            if v and len(cols) > 1:
                row[f"rollout_sr_{who}"] = float(np.mean(v))
        kl = [r["kl_teacher"] for r in recs if "kl_teacher" in r]
        if kl:
            row["rollout_kl"] = float(np.mean(kl))
        student.save_lora(last)
        torch.save({"opt": opt.state_dict(), "iter": it, "totals": totals,
                    "head": head.state_dict() if head is not None else None}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        if args.eval_every > 0 and (it % args.eval_every == 0 or it == args.iters):
            row.update(evaluate())
            student.save_lora(os.path.join(args.out, f"adapter_iter{it:04d}"))
        log(row)
    vec.close()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
