"""On-policy counterfactual distillation (OCD) for pi0.5 (LeRobot port), LoRA on the LIBERO checkpoint.

Each iteration: the student drives fresh rollouts (on-policy); at every queried state the simulator also renders a
counterfactual world (target object displaced horizontally, everything else unchanged; `vec_env.COUNTERFACTUAL`);
the teacher (the same checkpoint with the adapters off) labels both worlds with its full 50-step chunk. The flow
student is trained with paired flow matching: one noise sample and one time per state, shared by the two worlds,
    L = |v(o, x_t) - u| + |v(o', x'_t) - u'| + lambda_cf |(v(o', x'_t) - v(o, x_t)) - (u' - u)|      (squared)
with x_t = t e + (1 - t) A, u = e - A (A, A': teacher chunks of the two worlds), so the last term supervises how
the action must change when only the object moved (u' - u = A - A'). --lambda_cf 0 --no_cf is the matched on-policy
distillation baseline (same states, labels, budget, optimiser; no counterfactual world).
Outputs in --out: train_log.jsonl, adapter_last/ + state.pt (resume), adapter_iterXXXX/ at every evaluation.
"""
import argparse
import json
import os
import time

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state_source", choices=["student", "teacher", "mixed"], default="student")
    ap.add_argument("--no_cf", action="store_true", help="no counterfactual world (baseline)")
    ap.add_argument("--cf_delta", default="0.02,0.08", help="displacement range of the target (metres)")
    ap.add_argument("--lambda_cf", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--states_per_iter", type=int, default=1024)
    ap.add_argument("--episodes_per_batch", type=int, default=20)
    ap.add_argument("--train_trials", type=int, default=50)
    ap.add_argument("--num_envs", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=280)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_min", type=float, default=5e-6)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    from src.rollout.vec_env import COUNTERFACTUAL, LiberoVecEnv

    cf_cfg = None if args.no_cf else dict(COUNTERFACTUAL, delta=tuple(float(x) for x in args.cf_delta.split(",")))
    vec = LiberoVecEnv(args.suite, args.num_envs, args.max_steps, wrist=True, counterfactual=cf_cfg)

    import torch

    from src.policy.continuous_policy import AdapterOff
    from src.policy.pi05_policy import Pi05Policy
    from src.rollout.collector import Collector

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    student = Pi05Policy(args.ckpt)
    last, state_path = os.path.join(args.out, "adapter_last"), os.path.join(args.out, "state.pt")
    resume = os.path.exists(state_path)
    params = student.add_lora(args.lora_rank, adapter_path=last if resume else None)
    student.vla.model.gradient_checkpointing_enable()
    teacher = AdapterOff(student)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    start_it, totals = 0, {"episodes": 0, "states": 0, "grad_steps": 0}
    if resume:
        st = torch.load(state_path, map_location="cpu")
        opt.load_state_dict(st["opt"])
        start_it, totals = st["iter"], st["totals"]
        rng = np.random.default_rng([args.seed, start_it])
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

    kw = dict(sample=False, seed=args.seed, counterfactual=not args.no_cf)
    cols = []
    if args.state_source in ("student", "mixed"):
        cols.append((Collector(vec, student, {"teacher": teacher}, **kw), "student"))
    if args.state_source in ("teacher", "mixed"):
        cols.append((Collector(vec, teacher, {"teacher": teacher}, **kw), "teacher"))
    evalc = Collector(vec, student, {}, sample=False, seed=args.seed)
    task_ids = sorted(cols[0][0].tasks)
    pool = [(t, n) for t in task_ids for n in range(min(args.train_trials, cols[0][0].tasks[t][1]))]
    eval_eps = [(t, n) for t in task_ids for n in range(min(args.eval_trials, cols[0][0].tasks[t][1]))]
    log_path = os.path.join(args.out, "train_log.jsonl")

    def evaluate():
        recs = evalc.run(eval_eps)
        per_task = [float(np.mean([r["success"] for r in recs if r["task_id"] == t])) for t in task_ids]
        return {"eval_sr": float(np.mean([r["success"] for r in recs])), "eval_per_task": per_task}

    def log(row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    if start_it == 0 and args.eval_every > 0:
        log({"iter": 0, **evaluate(), **totals})

    def obs_list(arrays, idx, cf=False):
        w = "wrist_rgb_cf" if cf else "wrist_rgb"
        return [{"wrist_rgb": arrays[w][k], "eef_pos": arrays["eef_pos"][k], "eef_quat": arrays["eef_quat"][k],
                 "gripper_qpos": arrays["gripper_qpos"][k]} for k in idx]

    for it in range(start_it + 1, args.iters + 1):
        t0 = time.time()
        order, recs, nb = rng.permutation(len(pool)), [], 0
        while sum(r["n_queries"] for r in recs) < args.states_per_iter and len(recs) < len(pool):
            c, who = cols[nb % len(cols)]
            new = c.run([pool[i] for i in order[len(recs) : len(recs) + args.episodes_per_batch]], keep_steps=True)
            for r in new:
                r["driver"] = who
            recs, nb = recs + new, nb + 1
        # flatten (episode, query) -> arrays
        cat = lambda k: np.concatenate([r["arrays"][k] for r in recs])  # noqa: E731
        A = {k: cat(k) for k in ("rgb", "wrist_rgb", "eef_pos", "eef_quat", "gripper_qpos", "actions_teacher")}
        if not args.no_cf:
            A.update({k: cat(k) for k in ("rgb_cf", "wrist_rgb_cf", "cf_teacher")})
        descs_all = [r["task"] for r in recs for _ in range(r["n_queries"])]
        n_all = len(descs_all)
        keep = np.sort(rng.choice(n_all, size=min(args.states_per_iter, n_all), replace=False))
        t_collect = time.time() - t0

        if args.lr_min > 0:
            lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + np.cos(np.pi * (it - 1) / max(args.iters - 1, 1)))
            for g in opt.param_groups:
                g["lr"] = float(lr)
        student.vla.train()
        m = student.vla.model
        losses, cf_losses = [], []
        keep = rng.permutation(keep)
        for s in range(0, len(keep), args.batch_size):
            j = keep[s : s + args.batch_size]
            B = len(j)
            a = torch.from_numpy(A["actions_teacher"][j]).to(student.device, torch.float32)
            noise = torch.randn_like(a)
            t = m.sample_time(B, student.device)
            te = t[:, None, None]
            descs = [descs_all[k] for k in j]
            v = student.velocity([A["rgb"][k] for k in j], descs, obs_list(A, j), te * noise + (1 - te) * a, t)
            u = noise - a
            loss = ((v - u) ** 2).mean()
            if not args.no_cf:
                a2 = torch.from_numpy(A["cf_teacher"][j]).to(student.device, torch.float32)
                v2 = student.velocity([A["rgb_cf"][k] for k in j], descs, obs_list(A, j, cf=True), te * noise + (1 - te) * a2, t)
                u2 = noise - a2
                loss = loss + ((v2 - u2) ** 2).mean()
                lcf = (((v2 - v) - (u2 - u)) ** 2).mean()
                loss = loss + args.lambda_cf * lcf
                cf_losses.append(float(lcf))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            totals["grad_steps"] += 1
            losses.append(float(loss))
        student.vla.eval()

        totals["episodes"] += len(recs)
        totals["states"] += len(keep)
        row = {"iter": it, "rollout_sr": float(np.mean([r["success"] for r in recs])), "n_episodes": len(recs),
               "n_states_collected": n_all, "loss": float(np.mean(losses)), "sec_collect": round(t_collect),
               "sec_total": round(time.time() - t0), "lr": opt.param_groups[0]["lr"],
               "gpu_peak_gb": round(torch.cuda.max_memory_allocated(student.device) / 1e9, 1), **totals}
        if cf_losses:
            row["cf_loss"] = float(np.mean(cf_losses))
            # how much the teacher's own chunk moves between the two worlds (normalised units), for reference
            row["teacher_cf_shift"] = float(np.abs(A["cf_teacher"][keep] - A["actions_teacher"][keep]).mean())
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
