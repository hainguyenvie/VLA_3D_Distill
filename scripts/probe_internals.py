"""Inside pi0.5: where is the position of what the current phase is about, and what do the action tokens look at?

On the probe states of scripts/probe_mechanism.py (logged states of successful episodes, the target moved / swapped
before the grasp, the container moved while carrying), one joint forward of the VLM (PaliGemma) and the action expert at
flow time `--time` (1.0 = the first denoising step, pure noise; fixed noise) records, at every layer:
  - hidden states, pooled per stream: the agent-view image tokens, the instruction tokens, the last prefix token, and
    the action tokens (expert);
  - attention of the action tokens (queries) over the key segments: agent-view image, wrist image, instruction, proprio
    (the discretised state in the prompt), the rest of the prompt, the action tokens themselves.
Then, per layer and stream, a ridge probe (group k-fold over episodes) for the target's position relative to the hand
(before the grasp) and the container's position relative to the hand (while carrying): R^2 says whether that stream at
that layer knows where the thing is.
    python scripts/probe_internals.py --steps outputs/week1/pi05_object_steps --suite libero_object \
        --ckpt pi05:<checkpoint> [--lora <adapter>] --out outputs/week1/internals_object_base
"""
import argparse
import json
import os

import numpy as np

STREAMS = ("img", "task", "last", "act")
SEGMENTS = ("img_agent", "img_wrist", "task", "state", "prompt_rest", "actions")


def ridge_r2(X, Y, groups, lam=10.0, k=5):
    """Cross-validated R^2 (folds by group) of a ridge regression of Y (n, 2) on X (n, d), standardised."""
    ug = np.unique(groups)
    folds = np.array_split(np.random.default_rng(0).permutation(ug), min(k, len(ug)))
    pred = np.zeros_like(Y)
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = Y[tr].mean(0)
        # dual form: d >> n
        K = Xtr @ Xtr.T
        alpha = np.linalg.solve(K + lam * len(Xtr) * np.eye(len(Xtr)), Y[tr] - ym)
        pred[te] = Xte @ (Xtr.T @ alpha) + ym
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes_per_task", type=int, default=3)
    ap.add_argument("--states_per_episode", type=int, default=8)
    ap.add_argument("--deltas", default="0.06,0.12,0.2")
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from probe_mechanism import build_jobs  # sibling script

    jobs, _ = build_jobs(args.steps, args.suite, args.episodes_per_task, args.states_per_episode, args.deltas, args.seed)
    jobs = [j for j in jobs if j["variant"] == "base" or j["variant"].startswith("img_") or j["variant"] == "swap"]
    print(f"{len(jobs)} queries", flush=True)

    import torch
    import lerobot.policies.pi05.modeling_pi05 as M

    from src.policy.rebin import load_policy
    from src.rollout.vec_env import LiberoVecEnv

    vec = LiberoVecEnv(args.suite, args.num_envs, max_steps=600, wrist=True)
    policy = load_policy(args.ckpt, args.suite, "cuda:0")
    if args.lora:
        policy.add_lora(adapter_path=args.lora)
    m = policy.vla.model
    # the prompt's tokenizer (LeRobot's TokenizerProcessorStep keeps it as `input_tokenizer`)
    tok = next((v for st in getattr(policy.pre, "steps", []) for v in vars(st).values() if hasattr(v, "decode")), None)
    assert tok is not None, "no tokenizer found in the preprocessor"

    REC = {"h": [], "att": [], "seg": None}
    orig_layer, orig_attn = M.compute_layer_complete, M.modeling_gemma.eager_attention_forward

    def rec_layer(layer_idx, inputs_embeds, *a, **kw):
        out = orig_layer(layer_idx, inputs_embeds, *a, **kw)
        pre, suf = out[0].float(), out[1].float()
        REC["dims"] = (pre.shape[-1], suf.shape[-1])
        seg = REC["seg"]
        feats = []
        for b in range(pre.shape[0]):
            s = seg[b]
            feats.append(np.concatenate([pre[b, s["img_agent"][0] : s["img_agent"][1]].mean(0).cpu().numpy(),
                                         pre[b, s["task"][0] : s["task"][1]].mean(0).cpu().numpy(),
                                         pre[b, s["last"]].cpu().numpy(), suf[b].mean(0).cpu().numpy()]).astype(np.float16))
        REC["h"].append(np.stack(feats))
        return out

    def rec_attn(module, q, k, v, mask, scaling, *a, **kw):
        out, w = orig_attn(module, q, k, v, mask, scaling, *a, **kw)
        P = REC["P"]
        wq = w[:, :, P:, :].float().mean(1).mean(1)  # (B, K): action-token queries, averaged over heads and queries
        masses = []
        for b in range(wq.shape[0]):
            s = REC["seg"][b]
            masses.append([float(wq[b, s[g][0] : s[g][1]].sum()) for g in SEGMENTS])
        REC["att"].append(np.array(masses, dtype=np.float32))
        return out, w

    M.compute_layer_complete, M.modeling_gemma.eager_attention_forward = rec_layer, rec_attn

    def segments(tokens, masks, n_img, n_images):
        """Key ranges of each segment for every batch row: [img_agent, img_wrist, task, state, prompt_rest, actions)
        (images first, n_img tokens each, then the right-padded prompt)."""
        out = []
        for b in range(tokens.shape[0]):
            ids = tokens[b][masks[b].bool()].tolist()
            words = [tok.decode([i]) for i in ids] if tok is not None else [""] * len(ids)
            st = next((i for i, w_ in enumerate(words) if "State" in w_), len(ids))
            ac = next((i for i, w_ in enumerate(words) if "Action" in w_), len(ids))
            L0 = n_images * n_img
            out.append({"img_agent": (0, n_img), "img_wrist": (n_img, 2 * n_img), "task": (L0, L0 + st),
                        "state": (L0 + st, L0 + ac), "prompt_rest": (L0 + ac, L0 + len(ids)), "last": L0 + len(ids) - 1})
        return out

    feats, atts, meta = [], [], []
    pending = []
    for k, job in enumerate(jobs):
        vec.restore(len(pending), job["task_id"], job["state"], job["t0"], job["cmd"])
        pending.append(job)
        if len(pending) < args.num_envs and k < len(jobs) - 1:
            continue
        obs = [vec.recv(i) for i in range(len(pending))]
        with torch.no_grad():
            b = policy.batch([o["rgb"] for o in obs], [j["lang"] for j in pending], obs)
            images_, img_masks = policy.vla._preprocess_images(b)
            tokens, masks = b["observation.language.tokens"], b["observation.language.attention_mask"]
            x = M.pad_vector(policy._noise(len(pending))[..., : policy.vla.config.max_action_dim], policy.vla.config.max_action_dim)
            time = torch.full((len(pending),), args.time, device=x.device)
            prefix_embs, prefix_pad, prefix_att = m.embed_prefix(images_, img_masks, tokens, masks)
            suffix_embs, suffix_pad, suffix_att, adarms_cond = m.embed_suffix(x, time)
            if m.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                suffix_embs, prefix_embs = suffix_embs.to(torch.bfloat16), prefix_embs.to(torch.bfloat16)
            n_img = (prefix_embs.shape[1] - tokens.shape[1]) // len(images_)
            P = prefix_embs.shape[1]
            seg = segments(tokens, masks, n_img, len(images_))
            for s in seg:
                s["actions"] = (P, P + suffix_embs.shape[1])
            REC.update(h=[], att=[], seg=seg, P=P)
            pad = torch.cat([prefix_pad, suffix_pad], dim=1)
            att = torch.cat([prefix_att, suffix_att], dim=1)
            att_4d = m._prepare_attention_masks_4d(M.make_att_2d_masks(pad, att))
            pos = torch.cumsum(pad, dim=1) - 1
            m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
            m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
            m.paligemma_with_expert.forward(attention_mask=att_4d, position_ids=pos, past_key_values=None,
                                            inputs_embeds=[prefix_embs, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond])
        feats.append(np.stack(REC["h"], 1))  # (B, layers, D)
        atts.append(np.stack(REC["att"], 1))  # (B, layers, segments)
        for j in pending:
            s = j["state"]
            tgt = s[1 + j["a_tgt"] : 1 + j["a_tgt"] + 2] - j["eef_xy"]
            con = s[1 + j["a_con"] : 1 + j["a_con"] + 2] - j["eef_xy"] if j["a_con"] >= 0 else np.full(2, np.nan)
            meta.append(dict(task_id=j["task_id"], trial_id=j["trial_id"], q=j["q"], phase=j["phase"], variant=j["variant"],
                             dist_target_cm=j["dist_target_cm"], tgt_rel=tgt.tolist(), con_rel=con.tolist()))
        pending = []
        if len(meta) % 200 < args.num_envs:
            print(f"{len(meta)}/{len(jobs)} queries", flush=True)
    vec.close()
    F, A = np.concatenate(feats), np.concatenate(atts)
    np.savez_compressed(os.path.join(args.out, "internals.npz"), feats=F, atts=A)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"))

    # ---- analysis
    d_pre = REC["dims"][0]
    cuts = {"img": (0, d_pre), "task": (d_pre, 2 * d_pre), "last": (2 * d_pre, 3 * d_pre), "act": (3 * d_pre, F.shape[2])}
    groups = np.array([f"{r['task_id']}_{r['trial_id']}" for r in meta])
    out = {"probe_R2": {}, "attention": {}, "n": len(meta)}
    for phase, key in (("pre", "tgt_rel"), ("post", "con_rel")):
        sel = np.array([r["phase"] == phase and not np.isnan(r[key][0]) for r in meta])
        if sel.sum() < 30:
            continue
        Y = np.array([r[key] for r in meta])[sel]
        out["probe_R2"][f"{phase}: {key}"] = {
            st: [round(ridge_r2(F[sel, l, cuts[st][0] : cuts[st][1]].astype(np.float32), Y, groups[sel]), 3) for l in range(F.shape[1])]
            for st in STREAMS}
    bins = {"pre far (>15 cm)": lambda r: r["phase"] == "pre" and r["dist_target_cm"] > 15,
            "pre near (<5 cm)": lambda r: r["phase"] == "pre" and r["dist_target_cm"] < 5, "post": lambda r: r["phase"] == "post"}
    for name, f in bins.items():
        sel = np.array([f(r) and r["variant"] == "base" for r in meta])
        if sel.any():
            out["attention"][name] = {g: [round(float(x), 3) for x in A[sel, :, i].mean(0)] for i, g in enumerate(SEGMENTS)}
    json.dump(out, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    for k_, v in out["probe_R2"].items():
        print(f"probe R2 {k_}")
        for st, r in v.items():
            print(f"  {st:5s} " + " ".join(f"{x:5.2f}" for x in r))
    for k_, v in out["attention"].items():
        print(f"attention of action tokens, {k_} (per layer)")
        for g, r in v.items():
            print(f"  {g:12s} " + " ".join(f"{x:5.2f}" for x in r))
    print("PROBE_INTERNALS_DONE", flush=True)


if __name__ == "__main__":
    main()
