"""Self-distillation of the standard OpenVLA-OFT policy (two images, proprio, L1 head) under visual perturbations.

The frozen checkpoint is the teacher (it sees the nominal view of every state); the same checkpoint with LoRA
adapters is the student (it sees the perturbed view when --view_aug is on, and acts on it in the episodes it
drives). The loss is the L1 distance between the two normalised action chunks: the student learns the teacher's
actions for views the teacher itself cannot handle. Without --view_aug the loop is the matched control: the same
states, labels, budget and optimiser, and nothing to learn beyond the teacher.

    --state_source student | teacher | mixed    who drives the rollouts (mixed = batches of episodes alternate)
Outputs in --out: train_log.jsonl, adapter_last/ + state.pt (resume), adapter_iterXXXX/ at every evaluation.
"""
import argparse
import json
import os
import time

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="standard OpenVLA-OFT checkpoint folder (teacher and student init)")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state_source", choices=["student", "teacher", "mixed"], default="mixed")
    ap.add_argument("--view_aug", action="store_true", help="visual perturbations of the training rollouts (vec_env.VIEW_AUG)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--states_per_iter", type=int, default=2048)
    ap.add_argument("--episodes_per_batch", type=int, default=32)
    ap.add_argument("--train_trials", type=int, default=50)
    ap.add_argument("--num_envs", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=280, help="openvla-oft's horizon for LIBERO-Object")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=1e-5, help="> 0: cosine decay of the learning rate down to this value")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_checkpointing", action="store_true")
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    from src.rollout.vec_env import VIEW_AUG, LiberoVecEnv

    vec = LiberoVecEnv(args.suite, args.num_envs, args.max_steps, wrist=True, perturb=VIEW_AUG if args.view_aug else None)

    import torch

    from src.policy.continuous_policy import AdapterOff, ContinuousPolicy
    from src.rollout.collector import Collector

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    student = ContinuousPolicy(args.ckpt, args.suite)
    last = os.path.join(args.out, "adapter_last")
    state_path = os.path.join(args.out, "state.pt")
    resume = os.path.exists(state_path)
    params = student.add_lora(args.lora_rank, adapter_path=last if resume else None)
    if args.grad_checkpointing:
        student.vla.language_model.gradient_checkpointing_enable()
    teacher = AdapterOff(student)  # the frozen base model: same weights, adapters switched off
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    start_it, totals = 0, {"episodes": 0, "env_steps": 0, "states": 0, "grad_steps": 0}
    if resume:
        st = torch.load(state_path, map_location="cpu")
        opt.load_state_dict(st["opt"])
        start_it, totals = st["iter"], st["totals"]
        rng = np.random.default_rng([args.seed, start_it])
        print(f"resumed at iteration {start_it}", flush=True)
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

    kw = dict(sample=False, seed=args.seed, perturb=args.view_aug)
    cols = []  # (collector, key of the teacher's normalised actions in its records)
    if args.state_source in ("student", "mixed"):
        cols.append((Collector(vec, student, {"teacher": teacher}, **kw), "actions_teacher"))
    if args.state_source in ("teacher", "mixed"):
        cols.append((Collector(vec, teacher, {}, actor_view="rgb_clean", **kw), "actions_norm"))
    col = cols[0][0]
    evalc = Collector(vec, student, {}, sample=False, seed=args.seed)
    task_ids = sorted(col.tasks)
    pool = [(t, n) for t in task_ids for n in range(min(args.train_trials, col.tasks[t][1]))]
    eval_eps = [(t, n) for t in task_ids for n in range(min(args.eval_trials, col.tasks[t][1]))]
    log_path = os.path.join(args.out, "train_log.jsonl")

    def evaluate():
        recs = evalc.run(eval_eps)
        per_task = [float(np.mean([r["success"] for r in recs if r["task_id"] == t])) for t in task_ids]
        return {"eval_sr": float(np.mean([r["success"] for r in recs])), "eval_per_task": per_task}

    def log(row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def obs_of(arrays, k):
        return {"wrist_rgb": arrays["wrist_rgb"][k], "eef_pos": arrays["eef_pos"][k], "eef_quat": arrays["eef_quat"][k],
                "gripper_qpos": arrays["gripper_qpos"][k]}

    if start_it == 0 and args.eval_every > 0:
        log({"iter": 0, **evaluate(), **totals})

    checked = resume
    for it in range(start_it + 1, args.iters + 1):
        t0 = time.time()
        order, recs, n_batches = rng.permutation(len(pool)), [], 0
        while sum(r["n_queries"] for r in recs) < args.states_per_iter and len(recs) < len(pool):
            batch = order[len(recs) : len(recs) + args.episodes_per_batch]
            c, t_key = cols[n_batches % len(cols)]
            new = c.run([pool[i] for i in batch], keep_steps=True)
            for r in new:
                r["target"], r["driver"] = r["arrays"][t_key], "student" if t_key == "actions_teacher" else "teacher"
            recs, n_batches = recs + new, n_batches + 1
        n_collected = sum(r["n_queries"] for r in recs)
        keep = np.sort(rng.choice(n_collected, size=min(args.states_per_iter, n_collected), replace=False))
        images = np.concatenate([r["arrays"]["rgb"] for r in recs])[keep]
        targets = np.concatenate([r["target"] for r in recs])[keep]
        per_state_obs = [obs_of(r["arrays"], k) for r in recs for k in range(r["n_queries"])]
        obs_kept = [per_state_obs[k] for k in keep]
        all_descs = [r["task"] for r in recs for _ in range(r["n_queries"])]
        descs = [all_descs[k] for k in keep]
        t_collect = time.time() - t0
        tm = {k: round(sum(c.timing[k] for c, _ in cols), 1) for k in col.timing}
        for c, _ in cols:
            c.timing.update({k: 0 if k == "rounds" else 0.0 for k in c.timing})

        if not checked:  # the differentiable forward must reproduce the labels the frozen teacher gave at collection
            j = np.arange(min(8, len(images)))
            with torch.no_grad(), student.peft.disable_adapter():
                again = student.forward_norm([images[k] for k in j], [descs[k] for k in j], [obs_kept[k] for k in j]).cpu().numpy()
            diff = float(np.abs(again - targets[j]).max())  # the stored rgb is the student's (perturbed) view, so under
            print(f"forward check: max |forward_norm(stored view) - teacher label| = {diff:.3f} "  # view_aug a gap is expected
                  f"({'view_aug on, gap expected' if args.view_aug else 'must be ~0'})", flush=True)
            checked = True

        if args.lr_min > 0:
            lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + np.cos(np.pi * (it - 1) / max(args.iters - 1, 1)))
            for g in opt.param_groups:
                g["lr"] = float(lr)
        losses, n = [], len(images)
        opt.zero_grad(set_to_none=True)
        if args.grad_checkpointing:
            student.vla.language_model.train()
        for _ in range(args.epochs):
            order = rng.permutation(n)
            for b, s in enumerate(range(0, n, args.batch_size)):
                j = order[s : s + args.batch_size]
                pred = student.forward_norm([images[k] for k in j], [descs[k] for k in j], [obs_kept[k] for k in j])
                loss = torch.nn.functional.l1_loss(pred, torch.from_numpy(targets[j]).to(pred.device))
                (loss / args.grad_accum).backward()
                if (b + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    totals["grad_steps"] += 1
                losses.append(float(loss))
        opt.zero_grad(set_to_none=True)
        student.vla.eval()

        totals["episodes"] += len(recs)
        totals["env_steps"] += int(sum(r["env_steps"] for r in recs))
        totals["states"] += n
        row = {"iter": it, "rollout_sr": float(np.mean([r["success"] for r in recs])), "n_states": n, "n_episodes": len(recs),
               "n_states_collected": n_collected, "loss": float(np.mean(losses)), "sec_collect": round(t_collect),
               "sec_total": round(time.time() - t0), "collect_timing": tm,
               "gpu_peak_gb": round(torch.cuda.max_memory_allocated(student.device) / 1e9, 1), "lr": opt.param_groups[0]["lr"], **totals}
        for who in ("student", "teacher"):
            v = [r["success"] for r in recs if r["driver"] == who]
            if v and len(cols) > 1:
                row[f"rollout_sr_{who}"] = float(np.mean(v))
        l1 = [r["l1_teacher"] for r in recs if "l1_teacher" in r]
        if l1:
            row["rollout_l1"] = float(np.mean(l1))
        student.save_lora(last)
        torch.save({"opt": opt.state_dict(), "iter": it, "totals": totals}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        if args.eval_every > 0 and (it % args.eval_every == 0 or it == args.iters):
            row.update(evaluate())
            student.save_lora(os.path.join(args.out, f"adapter_iter{it:04d}"))
        log(row)
    vec.close()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
