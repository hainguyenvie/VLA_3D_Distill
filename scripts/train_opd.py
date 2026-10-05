"""Distil a frozen teacher into a LoRA student on LIBERO, either on-policy or on teacher-visited states.

One iteration = collect fresh rollouts, label every queried state with the teacher's token distribution,
take gradient steps on those states, discard them (VLA-OPD's loop, without its GRPO stage).

    --state_source student   the student acts, the teacher only labels (on-policy distillation, B2)
    --state_source teacher   the teacher acts and labels (same loss, teacher-state distribution: the matched
                             off-policy control)
    --spatial depth          add the privileged depth loss on the same states (training only); with the two
                             state sources this gives the 2 x 2 design {student, teacher states} x {no 3D, 3D}

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
    ap.add_argument("--state_source", choices=["student", "teacher"], default="student")
    ap.add_argument("--mode", choices=["rkl", "rkl_pg", "fkl", "ce"], default="rkl")
    ap.add_argument("--spatial", choices=["none", "depth"], default="none")
    ap.add_argument("--spatial_layer", type=int, default=24, help="LLM layer whose visual tokens feed the depth head (1..32)")
    ap.add_argument("--lambda_spatial", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--episodes_per_iter", type=int, default=64)
    ap.add_argument("--train_trials", type=int, default=50, help="rollouts start from the first N benchmark init states")
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=512)
    ap.add_argument("--rollout_temperature", type=float, default=1.6, help="0 = greedy rollouts")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
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
    from src.rollout.vec_env import LiberoVecEnv

    vec = LiberoVecEnv(args.suite, args.num_envs, args.max_steps, depth=args.spatial != "none")

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
    if args.state_source == "student":
        col = Collector(vec, student, {"teacher": teacher}, sample=sample, temperature=args.rollout_temperature, seed=args.seed)
    else:
        col = Collector(vec, teacher, {}, sample=sample, temperature=args.rollout_temperature, seed=args.seed)
    evalc = Collector(vec, student, {}, sample=False, seed=args.seed)
    task_ids = sorted(col.tasks)
    pool = [(t, n) for t in task_ids for n in range(min(args.train_trials, col.tasks[t][1]))]
    eval_eps = [(t, n) for t in task_ids for n in range(min(args.eval_trials, col.tasks[t][1]))]
    log_path = os.path.join(args.out, "train_log.jsonl")

    def evaluate():
        recs = evalc.run(eval_eps)
        return float(np.mean([r["success"] for r in recs]))

    def log(row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    if start_it == 0 and args.eval_every > 0:
        log({"iter": 0, "eval_sr": evaluate(), **totals})

    for it in range(start_it + 1, args.iters + 1):
        t0 = time.time()
        idx = rng.choice(len(pool), size=min(args.episodes_per_iter, len(pool)), replace=False)
        recs = col.run([pool[i] for i in idx], keep_steps=True)
        images = np.concatenate([r["arrays"]["rgb"] for r in recs])
        t_key = "logits_teacher" if args.state_source == "student" else "logits"
        t_logits = np.concatenate([r["arrays"][t_key] for r in recs])
        bins = np.concatenate([r["arrays"]["bins"] for r in recs])
        depth = np.concatenate([r["arrays"]["depth"] for r in recs]) if head is not None else None
        descs = [r["task"] for r in recs for _ in range(r["n_queries"])]
        t_collect = time.time() - t0
        tm = {k: round(v, 1) for k, v in col.timing.items()}
        col.timing.update({k: 0 if k == "rounds" else 0.0 for k in col.timing})

        losses, sp_losses, mets, n = [], [], [], len(images)
        opt.zero_grad(set_to_none=True)
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

        totals["episodes"] += len(recs)
        totals["env_steps"] += int(sum(r["env_steps"] for r in recs))
        totals["states"] += n
        row = {"iter": it, "rollout_sr": float(np.mean([r["success"] for r in recs])), "n_states": n,
               "loss": float(np.mean(losses)), **{f"train_{k}": float(np.mean([m[k] for m in mets])) for k in mets[0]},
               "sec_collect": round(t_collect), "sec_total": round(time.time() - t0), "collect_timing": tm, **totals}
        if sp_losses:
            row["spatial_loss"] = float(np.mean(sp_losses))
        if args.state_source == "student":
            row["rollout_kl"] = float(np.mean([r["kl_teacher"] for r in recs]))
        student.save_lora(last)
        torch.save({"opt": opt.state_dict(), "iter": it, "totals": totals,
                    "head": head.state_dict() if head is not None else None}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        if args.eval_every > 0 and (it % args.eval_every == 0 or it == args.iters):
            row["eval_sr"] = evaluate()
            student.save_lora(os.path.join(args.out, f"adapter_iter{it:04d}"))
        log(row)
    vec.close()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
