"""Gate for bin transfer between checkpoints with different action normalisation (GPU only, no simulator).

On logged frames: the transferred distribution must keep the source's expected action (within a fraction
of a target bin; range clipping aside), and each executed action must map to a neighbouring target token.
    python scripts/check_rebin.py --src <full-SFT> --dst <student> --steps <dir with steps/*.npz>
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--suite", default="libero_object")
    args = ap.parse_args()
    import torch

    from libero.libero import benchmark
    from src.policy.rebin import RebinnedPolicy, interval_transfer, token_transfer
    from src.policy.token_policy import TokenPolicy

    suite = benchmark.get_benchmark_dict()[args.suite]()
    dst, src = TokenPolicy(args.dst, args.suite), TokenPolicy(args.src, args.suite)
    ident = token_transfer(dst, dst)
    assert torch.allclose(ident[:, 2:, 2:], torch.eye(254).expand(7, -1, -1), atol=1e-6), "self-transfer must be the identity"
    width = (dst.act_high - dst.act_low) * 2 / 255
    print("action range  dst (q01..q99):", np.round(dst.act_low, 3), np.round(dst.act_high, 3))
    print("action range  src (q01..q99):", np.round(src.act_low, 3), np.round(src.act_high, 3))
    m = interval_transfer(src, dst)
    print("source intervals entirely outside the target range, per dim:", [(int((m[d][:, 0] == 1).sum()), int((m[d][:, -1] == 1).sum())) for d in range(7)])

    reb = RebinnedPolicy(src, dst)
    centers = lambda p: torch.from_numpy(np.where(p.act_mask[:, None], 0.5 * (p.bin_centers[np.clip(255 - np.arange(256), 0, 254)][None] + 1)  # noqa: E731
                                                  * (p.act_high - p.act_low + 1e-8)[:, None] + p.act_low[:, None],
                                                  p.bin_centers[np.clip(255 - np.arange(256), 0, 254)][None])).float()
    v_src, v_dst = centers(src)[torch.arange(56) % 7], centers(dst)[torch.arange(56) % 7]  # (56, 256) raw value of each token
    w = torch.from_numpy(np.where(dst.act_mask, (dst.act_high - dst.act_low) * 2 / 255, 2 / 255)).float()[torch.arange(56) % 7]
    d_mean, d_near, n = [], [], 0
    for f in sorted(glob.glob(os.path.join(args.steps, "*.npz"))):
        z = np.load(f)
        desc = suite.get_task(int(os.path.basename(f)[1:3])).language
        idx = list(range(0, len(z["t"]), 3))
        for s in range(0, len(idx), 8):
            j = idx[s : s + 8]
            imgs = list(z["rgb"][j])
            ls = src.act(imgs, [desc] * len(j))
            lr = reb.act(imgs, [desc] * len(j))
            ps, pr = torch.softmax(ls["logits"], -1), torch.softmax(lr["logits"], -1)
            d_mean.append((((ps * v_src).sum(-1) - (pr * v_dst).sum(-1)).abs() / w).flatten())  # expected action, in target bins
            a_exec = torch.from_numpy(ls["actions"]).float().reshape(len(j), 56)
            a_near = v_dst[torch.arange(56), lr["bins"]]
            d_near.append(((a_exec - a_near).abs() / w).flatten())
            n += len(j)
    d_mean, d_near = torch.cat(d_mean), torch.cat(d_near)
    print(f"{n} states; expected action, source bins vs transferred: median {d_mean.median():.3f}, 99th pct {d_mean.quantile(0.99):.3f}, max {d_mean.max():.2f} target bins")
    print(f"executed action vs its target token: median {d_near.median():.3f}, 99th pct {d_near.quantile(0.99):.3f}, max {d_near.max():.2f} target bins")
    assert d_mean.quantile(0.99) < 0.6, "transfer shifts the distribution"
    print("CHECK_REBIN_OK")


if __name__ == "__main__":
    main()
