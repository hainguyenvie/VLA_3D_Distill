"""On-policy counterfactual distillation (OCD) for pi0.5 (LeRobot port), LoRA on the LIBERO checkpoint.

Each iteration: the student drives fresh rollouts (on-policy); at every queried state the simulator also renders a
counterfactual world (target object displaced horizontally, everything else unchanged; `vec_env.COUNTERFACTUAL`);
the teacher (the same checkpoint with the adapters off) labels both worlds with its full 50-step chunk. The flow
student is trained with paired flow matching: one noise sample and one time per state, shared by the two worlds,
    L = |v(o, x_t) - u| + |v(o', x'_t) - u'| + lambda_cf |(v(o', x'_t) - v(o, x_t)) - (u' - u)|      (squared)
with x_t = t e + (1 - t) A, u = e - A (A, A': labels of the two worlds), so the last term supervises how the action
must change between the worlds (u' - u = A - A'). Swap mode: A' = A and the instruction names the object now at the
target's place, so the pair asks for the same motion to a different named object (pre-grasp states only). --lambda_cf 0 --no_cf is the matched on-policy
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
    ap.add_argument("--cf_mode", choices=["shift", "swap", "mirror", "rotate", "coshift", "mix", "relocate", "coreloc", "retarget", "rr"], default="swap",
                    help="swap: the target exchanges places with another object and the counterfactual instruction names "
                         "the object now at the target's place, so the label is the nominal chunk (exact); shift: the "
                         "target is displaced and the teacher labels the counterfactual world")
    ap.add_argument("--cf_delta", default="0.02,0.08", help="shift mode: displacement range of the target (metres)")
    ap.add_argument("--c7", type=float, default=0.7853981633974483, help="mirror mode: reflection centre of joint 7")
    ap.add_argument("--cf_theta", default="0.1,0.35", help="rotate mode: range of |rotation angle| (radians)")
    ap.add_argument("--cf_coshift", default="0.08,0.3", help="coshift mode: range of the target's (and hand's) shift (metres)")
    ap.add_argument("--p_swap", type=float, default=0.5, help="coshift mode: probability that the target takes another object's spot")
    ap.add_argument("--p_coshift", type=float, default=0.5, help="mix mode: share of queries rendered as coshift (else rotate)")
    ap.add_argument("--cf_coshift_post", default="0.08,0.4",
                    help="post-grasp coshift / relocate: range of the container's shift (metres)")
    ap.add_argument("--relocate_label", choices=["placer", "teacher"], default="placer",
                    help="relocate mode: label the relocated world with the scripted placer, or (ablation) with pi0.5 itself")
    ap.add_argument("--cf_tasks", default="",
                    help="comma-separated task ids whose counterfactual pairs are used (default: all); e.g. the tasks where "
                         "the scripted teacher passes its simulation gate (scripts/check_scripted_place.py)")
    ap.add_argument("--cf_frac", type=float, default=0.0,
                    help="> 0: keep counterfactual pairs on at most this share of the iteration's states (random subset)")
    ap.add_argument("--coshift_phase", choices=["pre", "post", "both"], default="pre",
                    help="coshift mode: before the grasp move hand + target (pre); after it move hand + held target + "
                         "container (post); or both")
    ap.add_argument("--coshift_margin", type=int, default=10,
                    help="coshift mode: chunk steps kept after the chunk's first gripper-close command (the rest is masked)")
    ap.add_argument("--cf_max_query", type=int, default=0,
                    help="> 0: counterfactual pairs only on the first N queries of each episode (early-state intervention)")
    ap.add_argument("--lambda_cf", type=float, default=1.0)
    ap.add_argument("--img_aug", action="store_true",
                    help="ordinary 2D augmentation of the student's training images (openpi LIBERO recipe: crop 95%% + resize "
                         "and rotation +-5 deg on the agent view, colour jitter on both views); labels unchanged")
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
    ap.add_argument("--lora_scope", choices=["all", "llm"], default="all", help="llm: keep the image encoder frozen")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    from src.rollout.vec_env import COUNTERFACTUAL, LiberoVecEnv

    cf_cfg = None if args.no_cf else dict(COUNTERFACTUAL, mode=args.cf_mode, delta=tuple(float(x) for x in args.cf_delta.split(",")),
                                          c7=args.c7, theta=tuple(float(x) for x in args.cf_theta.split(",")),
                                          coshift=tuple(float(x) for x in args.cf_coshift.split(",")), p_swap=args.p_swap,
                                          p_coshift=args.p_coshift, coshift_phase=args.coshift_phase,
                                          coshift_post=tuple(float(x) for x in args.cf_coshift_post.split(",")))
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
    params = student.add_lora(args.lora_rank, adapter_path=last if resume else None, scope=args.lora_scope)
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

    aug_rng = np.random.default_rng([args.seed, 99])

    def aug(img, geometric):
        """openpi-style training augmentation of one HWC uint8 frame (no-op unless --img_aug)."""
        if not args.img_aug:
            return img
        from PIL import Image, ImageEnhance

        im = Image.fromarray(img)
        h, w = img.shape[:2]
        if geometric:
            ch, cw = int(0.95 * h), int(0.95 * w)
            y0, x0 = aug_rng.integers(0, h - ch + 1), aug_rng.integers(0, w - cw + 1)
            im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
            im = im.rotate(float(aug_rng.uniform(-5, 5)), resample=Image.BILINEAR)
        im = ImageEnhance.Brightness(im).enhance(float(aug_rng.uniform(0.7, 1.3)))
        im = ImageEnhance.Contrast(im).enhance(float(aug_rng.uniform(0.6, 1.4)))
        im = ImageEnhance.Color(im).enhance(float(aug_rng.uniform(0.5, 1.5)))
        return np.asarray(im)

    def obs_list(arrays, idx, cf=False):
        w = "wrist_rgb_cf" if cf else "wrist_rgb"
        pos = "cf_eef_pos" if cf and "cf_eef_pos" in arrays else "eef_pos"
        quat = "cf_eef_quat" if cf and "cf_eef_quat" in arrays else "eef_quat"
        return [{"wrist_rgb": aug(arrays[w][k], False), "eef_pos": arrays[pos][k], "eef_quat": arrays[quat][k],
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
        descs_all = [r["task"] for r in recs for _ in range(r["n_queries"])]
        cf_ok = np.zeros(len(descs_all), dtype=bool)
        descs_cf = list(descs_all)
        if not args.no_cf:
            A.update({k: cat(k) for k in ("rgb_cf", "wrist_rgb_cf", "cf_teacher", "closed_before")})
            phrase = lambda n: str(n).rsplit("_", 1)[0].replace("_", " ")  # "alphabet_soup_1" -> "alphabet soup"  # noqa: E731
            if args.cf_mode in ("mirror", "rotate"):
                A.update({k: cat(k) for k in ("cf_eef_pos", "cf_eef_quat")})
                cf_ok[:] = True  # the transformed scene is a valid world at every state
                exact = (student.mirror_norm(A["actions_teacher"]) if args.cf_mode == "mirror"
                         else student.rotate_norm(A["actions_teacher"], cat("cf_theta").astype(np.float64)))
                # how far the teacher itself is from equivariant: its own chunk in the transformed world vs the exact one
                teacher_gap = float(np.abs(A["cf_teacher"] - exact).mean())
                A["cf_teacher"] = exact
            elif args.cf_mode == "coreloc":  # before the grasp: coshift (nominal chunk, masked); while carrying: relocate
                A.update({k: cat(k) for k in ("cf_eef_pos", "cf_eef_quat")})
                cf_ok[:] = cat("cf_valid").astype(bool)
                post = A["closed_before"].astype(bool)
                lab = cat("cf_label").astype(np.float64)
                exact = A["actions_teacher"].copy()
                exact[post] = ((lab[post] - student.act_mean) / (student.act_std + 1e-8)).astype(np.float32)
                rel = lab[..., 6] < 0
                rel_first = np.where(rel.any(1), rel.argmax(1), lab.shape[1])
                g = A["actions_teacher"][..., 6] * (student.act_std[6] + 1e-8) + student.act_mean[6]
                first = np.where((g > 0).any(1), (g > 0).argmax(1), g.shape[1])
                mask = (np.arange(g.shape[1])[None] <= first[:, None] + args.coshift_margin).astype(np.float32)
                mask[post] = (np.arange(lab.shape[1])[None] <= rel_first[post][:, None] + 5).astype(np.float32)
                A["cf_mask"] = mask
                ok = np.flatnonzero(cf_ok)
                teacher_gap = float((np.abs(A["cf_teacher"][ok] - exact[ok]) * mask[ok][..., None]).sum()
                                    / max(mask[ok].sum() * 7, 1))
                A["cf_teacher"] = exact
            elif args.cf_mode == "rr":  # before the grasp: retarget (scripted approach); while carrying: relocate (placer)
                cf_ok[:] = cat("cf_valid").astype(bool)
                post = A["closed_before"].astype(bool)
                lab = cat("cf_label").astype(np.float64)
                norm = ((lab - student.act_mean) / (student.act_std + 1e-8)).astype(np.float32)
                n = cat("cf_len").astype(int)
                rel = lab[..., 6] < 0
                rel_first = np.where(rel.any(1), rel.argmax(1), lab.shape[1])
                steps = np.arange(lab.shape[1])[None]
                mask = np.where(post[:, None], steps <= rel_first[:, None] + 5, steps < n[:, None]).astype(np.float32)
                A["cf_mask"] = mask
                ok = np.flatnonzero(cf_ok)
                diff = np.abs(A["cf_teacher"][ok] - norm[ok]) * mask[ok][..., None]
                teacher_gap = float(diff.sum() / max(mask[ok].sum() * 7, 1))
                A["cf_teacher"] = norm
            elif args.cf_mode == "retarget":  # target moved before the grasp; label = scripted approach (env convention)
                cf_ok[:] = cat("cf_valid").astype(bool)
                lab = cat("cf_label").astype(np.float64)
                norm = ((lab - student.act_mean) / (student.act_std + 1e-8)).astype(np.float32)
                n = cat("cf_len").astype(int)
                A["cf_mask"] = (np.arange(lab.shape[1])[None] < n[:, None]).astype(np.float32)
                ok = np.flatnonzero(cf_ok)
                diff = np.abs(A["cf_teacher"][ok] - norm[ok]) * A["cf_mask"][ok][..., None]
                teacher_gap = float(diff.sum() / max(A["cf_mask"][ok].sum() * 7, 1))
                A["cf_teacher"] = norm
            elif args.cf_mode == "relocate":  # container moved while carrying; label = scripted placer (env convention)
                cf_ok[:] = cat("cf_valid").astype(bool)
                lab = cat("cf_label").astype(np.float64)
                norm = ((lab - student.act_mean) / (student.act_std + 1e-8)).astype(np.float32)
                # the placer's chunk is only meaningful until shortly after it releases the object (multi-object tasks
                # go on to the next object, the placer would just go up): mask the rest
                rel = lab[..., 6] < 0
                first = np.where(rel.any(1), rel.argmax(1), lab.shape[1])
                A["cf_mask"] = (np.arange(lab.shape[1])[None] <= first[:, None] + 5).astype(np.float32)
                ok = np.flatnonzero(cf_ok)
                teacher_gap = float(np.abs(A["cf_teacher"][ok] - norm[ok]).mean()) if len(ok) else 0.0
                if args.relocate_label == "placer":
                    A["cf_teacher"] = norm
                else:  # ablation: pi0.5's own chunk in the relocated world (already in A["cf_teacher"]), whole chunk
                    A["cf_mask"][:] = 1.0
            elif args.cf_mode == "mix":  # per query: rotate (kind 0) or coshift (kind 1)
                A.update({k: cat(k) for k in ("cf_eef_pos", "cf_eef_quat")})
                kind = cat("cf_kind").astype(int)
                rot = kind == 0
                cf_ok[:] = rot | (cat("cf_valid").astype(bool) & ~A["closed_before"].astype(bool))
                exact = A["actions_teacher"].copy()
                exact[rot] = student.rotate_norm(A["actions_teacher"][rot], cat("cf_theta").astype(np.float64)[rot])
                g = A["actions_teacher"][..., 6] * (student.act_std[6] + 1e-8) + student.act_mean[6]
                first = np.where((g > 0).any(1), (g > 0).argmax(1), g.shape[1])
                mask = (np.arange(g.shape[1])[None] <= first[:, None] + args.coshift_margin).astype(np.float32)
                mask[rot] = 1.0
                A["cf_mask"] = mask
                ok = np.flatnonzero(cf_ok)
                diff = np.abs(A["cf_teacher"][ok] - exact[ok]) * mask[ok][..., None]
                teacher_gap = float(diff.sum() / max(mask[ok].sum() * 7, 1))
                A["cf_teacher"] = exact
            elif args.cf_mode == "coshift":
                A.update({k: cat(k) for k in ("cf_eef_pos", "cf_eef_quat")})
                cf_ok[:] = cat("cf_valid").astype(bool)  # the env marks only the phases asked for (--coshift_phase)
                g = A["actions_teacher"][..., 6] * (student.act_std[6] + 1e-8) + student.act_mean[6]  # env convention, +1 close
                first = np.where((g > 0).any(1), (g > 0).argmax(1), g.shape[1])
                A["cf_mask"] = (np.arange(g.shape[1])[None] <= first[:, None] + args.coshift_margin).astype(np.float32)
                A["cf_mask"][A["closed_before"].astype(bool)] = 1.0  # after the grasp: hand + held target + container moved
                ok = np.flatnonzero(cf_ok)  # the teacher's own chunk in the co-shifted world vs the exact one, on kept steps
                diff = np.abs(A["cf_teacher"][ok] - A["actions_teacher"][ok]) * A["cf_mask"][ok][..., None]
                teacher_gap = float(diff.sum() / max(A["cf_mask"][ok].sum() * 7, 1))
                A["cf_teacher"] = A["actions_teacher"]  # hand-target geometry unchanged: the nominal chunk, until it carries
            elif args.cf_mode == "swap":
                tgt, src = cat("cf_target_name"), cat("cf_source_name")
                for k, d in enumerate(descs_all):  # pre-grasp states whose instruction names the target explicitly
                    if not A["closed_before"][k] and phrase(src[k]) in d and phrase(tgt[k]) != phrase(src[k]):
                        descs_cf[k], cf_ok[k] = d.replace(phrase(src[k]), phrase(tgt[k])), True
                A["cf_teacher"] = A["actions_teacher"]  # the correct action of the swapped world is the nominal one
            else:
                cf_ok[:] = ~A["closed_before"].astype(bool)
            if args.cf_max_query > 0:  # early-state intervention: pairs only on the first queries of each episode
                qidx = np.concatenate([np.arange(r["n_queries"]) for r in recs])
                cf_ok &= qidx < args.cf_max_query
        n_all = len(descs_all)
        if args.cf_tasks:  # only the tasks whose teacher passed its gate
            allowed = {int(x) for x in args.cf_tasks.split(",")}
            tids = np.concatenate([np.full(r["n_queries"], r["task_id"]) for r in recs])
            cf_ok &= np.isin(tids, list(allowed))
        if args.cf_frac > 0 and cf_ok.sum() > args.cf_frac * n_all:  # cap the share of states that carry a pair
            on = np.flatnonzero(cf_ok)
            cf_ok[:] = False
            cf_ok[rng.choice(on, size=int(args.cf_frac * n_all), replace=False)] = True
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
            v = student.velocity([aug(A["rgb"][k], True) for k in j], descs, obs_list(A, j), te * noise + (1 - te) * a, t)
            u = noise - a
            loss = ((v - u) ** 2).mean()
            jc = np.array([k for k in j if cf_ok[k]], dtype=int)
            if not args.no_cf and len(jc):
                sel = torch.from_numpy(np.isin(j, jc)).to(student.device)
                a2 = torch.from_numpy(A["cf_teacher"][jc]).to(student.device, torch.float32)
                n2, t2, te2 = noise[sel], t[sel], te[sel]
                v2 = student.velocity([A["rgb_cf"][k] for k in jc], [descs_cf[k] for k in jc], obs_list(A, jc, cf=True),
                                      te2 * n2 + (1 - te2) * a2, t2)
                u2 = n2 - a2
                w = (torch.from_numpy(A["cf_mask"][jc]).to(student.device)[..., None] if "cf_mask" in A
                     else torch.ones_like(u2[..., :1]))  # chunk steps for which the counterfactual label holds
                wmean = lambda x: (x * w).sum() / (w.sum() * x.shape[-1])  # noqa: E731
                loss = loss + wmean((v2 - u2) ** 2)
                lcf = wmean(((v2 - v[sel]) - (u2 - u[sel])) ** 2)
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
            row["cf_states"] = int(cf_ok[keep].sum())
            if args.cf_mode in ("mirror", "rotate", "coshift", "mix", "relocate", "coreloc", "retarget", "rr"):
                row["teacher_cf_gap"] = teacher_gap
            if args.cf_mode == "shift":  # how much the teacher's chunk moves between the two worlds (normalised units)
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
