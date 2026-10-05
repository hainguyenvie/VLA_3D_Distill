"""Gate for the policy wrapper: batched, padded `TokenPolicy.act` must reproduce upstream `predict_action`.

Rolls two environments with different prompt lengths for a few chunks and compares, at every queried state,
(a) upstream OpenVLA-OFT `predict_action` (batch size 1, discrete tokens) against (b) our batched forward.
    python scripts/check_policy.py --ckpt <dir> --suite libero_object
"""
import argparse

import numpy as np


def check_oft(args):
    """Standard OpenVLA-OFT (two images, proprio, L1 head): batched `ContinuousPolicy.act` against upstream."""
    from src.rollout.vec_env import LiberoVecEnv, postprocess_actions

    vec = LiberoVecEnv(args.suite, 2, max_steps=512, wrist=True)
    info = vec.task_info()

    import torch

    from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
    from src.policy.continuous_policy import ContinuousPolicy, proprio_state
    from src.policy.token_policy import preprocess_image, prompt_for

    policy = ContinuousPolicy(args.ckpt, args.suite)
    print("unexpected keys:", policy.loading_info.get("unexpected_keys"), "missing:", policy.loading_info.get("missing_keys"))
    processor = PrismaticProcessor(image_processor=policy.image_processor, tokenizer=policy.tokenizer)
    lens = [len(policy._prompt_ids(lang)) for _, lang, _ in info]
    tasks = [int(np.argmin(lens)), int(np.argmax(lens))]
    descs = [info[t][1] for t in tasks]
    for i, t in enumerate(tasks):
        vec.reset(i, t, 0)
    obs = [vec.recv(i) for i in range(2)]
    d_ref, d_batch = 0.0, 0.0
    for q in range(args.queries):
        images = [o["rgb"] for o in obs]
        ours = policy.act(images, descs, obs=obs)["actions"]
        for i in range(2):
            inputs = processor(prompt_for(descs[i]), preprocess_image(images[i])).to(policy.device, dtype=torch.bfloat16)
            wrist = processor(prompt_for(descs[i]), preprocess_image(obs[i]["wrist_rgb"])).to(policy.device, dtype=torch.bfloat16)
            inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist["pixel_values"]], dim=1)
            proprio = policy._normalize_proprio(proprio_state(obs[i]))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                ref, _ = policy.vla.predict_action(**inputs, unnorm_key=policy.unnorm_key, do_sample=False, proprio=proprio,
                                                   proprio_projector=policy.proprio_projector, action_head=policy.action_head)
            single = policy.act([images[i]], [descs[i]], obs=[obs[i]])["actions"][0]
            d_ref = max(d_ref, float(np.abs(single - ref).max()))
            d_batch = max(d_batch, float(np.abs(ours[i] - single).max()))
        env_actions = postprocess_actions(ours)
        for i in range(2):
            vec.step(i, env_actions[i])
        obs = [vec.recv(i) for i in range(2)]
    vec.close()
    print(f"single-sample forward vs upstream predict_action: max |action diff| {d_ref:.5f} on {2 * args.queries} states")
    print(f"padded batch of 2 vs single: max |action diff| {d_batch:.5f} (bf16 batching noise)")
    assert d_ref < 1e-3, "batched re-implementation differs from upstream"
    print("CHECK_POLICY_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--queries", type=int, default=6)
    ap.add_argument("--oft", action="store_true", help="--ckpt is a standard OpenVLA-OFT checkpoint (two images, proprio, L1 head)")
    args = ap.parse_args()
    if args.oft:
        return check_oft(args)

    from src.rollout.vec_env import LiberoVecEnv, postprocess_actions

    vec = LiberoVecEnv(args.suite, 2, max_steps=512)
    info = vec.task_info()

    import torch

    from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
    from src.policy.token_policy import TokenPolicy, preprocess_image, prompt_for

    policy = TokenPolicy(args.ckpt, args.suite)
    print("unexpected keys:", policy.loading_info.get("unexpected_keys"), "missing:", policy.loading_info.get("missing_keys"))
    processor = PrismaticProcessor(image_processor=policy.image_processor, tokenizer=policy.tokenizer)
    lens = [len(policy._prompt_ids(lang)) for _, lang, _ in info]
    tasks = [int(np.argmin(lens)), int(np.argmax(lens))]
    print("tasks", tasks, "prompt lengths", [lens[t] for t in tasks])
    descs = [info[t][1] for t in tasks]
    for i, t in enumerate(tasks):
        vec.reset(i, t, 0)
    obs = [vec.recv(i) for i in range(2)]

    max_diff, n_tok, n_same, n_plain, n_plain_same = 0.0, 0, 0, 0, 0
    stats = [{"pad_flip": 0, "pad_diff": [], "twin_flip": 0, "twin_diff": []} for _ in range(2)]
    for q in range(args.queries):
        images = [o["rgb"] for o in obs]
        ours = policy.act(images, descs)
        for i in range(2):
            pil = preprocess_image(images[i])
            inputs = processor(prompt_for(descs[i]), pil).to(policy.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                plain, _ = policy.vla.predict_action(**inputs, unnorm_key=policy.unnorm_key, do_sample=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ref, _ = policy.vla.predict_action(**inputs, unnorm_key=policy.unnorm_key, do_sample=False)
            single = policy.act([images[i]], [descs[i]])
            assert np.array_equal(single["actions"][0], ref), f"query {q} env {i}: single-sample forward differs from upstream"
            n_plain += ref.size
            n_plain_same += int((plain == ref).sum())
            same = (ours["bins"][i] == single["bins"][0]).numpy()
            n_tok += same.size
            n_same += int(same.sum())
            # reference for pure batching noise in bf16: the same sample twice (no padding) against batch size 1
            twin = policy.act([images[i], images[i]], [descs[i], descs[i]])
            stats[i]["pad_flip"] += int((~same).sum())
            stats[i]["pad_diff"].append(float((ours["logits"][i] - single["logits"][0]).abs().mean()))
            stats[i]["twin_flip"] += int((twin["bins"][0] != single["bins"][0]).sum())
            stats[i]["twin_diff"].append(float((twin["logits"][0] - single["logits"][0]).abs().mean()))
            max_diff = max(max_diff, float((ours["logits"][i] - single["logits"][0]).abs().max()))
        env_actions = postprocess_actions(ours["actions"])
        for i in range(2):
            vec.step(i, env_actions[i])
        obs = [vec.recv(i) for i in range(2)]
    vec.close()
    print(f"single-sample forward == upstream predict_action (under autocast) on {2 * args.queries} states")
    print(f"upstream with vs without autocast: {n_plain_same}/{n_plain} action dims equal")
    print(f"padded batch vs single: {n_same}/{n_tok} argmax tokens equal, max |logit diff| = {max_diff:.4f}")
    for i, name in enumerate(("padded sample (short prompt)", "unpadded sample (long prompt)")):
        st = stats[i]
        print(f"{name}: mixed batch flips {st['pad_flip']} tokens, mean |dlogit| {np.mean(st['pad_diff']):.4f}; "
              f"same-sample batch of 2 flips {st['twin_flip']}, mean |dlogit| {np.mean(st['twin_diff']):.4f}")
    # padding must not add error beyond what batching alone does in bf16
    assert np.mean(stats[0]["pad_diff"]) < 2 * max(np.mean(stats[0]["twin_diff"]), np.mean(stats[1]["pad_diff"])) + 1e-3, "padding changes the prediction"
    print("CHECK_POLICY_OK")


if __name__ == "__main__":
    main()
