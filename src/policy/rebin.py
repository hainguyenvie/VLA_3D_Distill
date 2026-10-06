"""Express one policy's action-token distribution in another policy's bins.

Two checkpoints can only be compared token by token if they share the action normalisation: token k means
"bin k of the range [q01, q99] of that checkpoint's training set", and the full-data SFT model (454 demos) and
the 1-trajectory student (10 demos) have different ranges. Each token is an interval of raw action values, so
the source distribution is pushed onto the target bins by interval overlap, per action dimension; mass that
falls outside the target range goes to the target's extreme bin (the target cannot represent it).
"""
from typing import Optional

import numpy as np
import torch

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from src.policy.token_policy import N_BINS, TokenPolicy

N_INT = N_BINS - 1  # 255 intervals between the 256 uniform edges on [-1, 1]; tokens 0 and 1 both decode to the last one


def _edges_raw(policy: TokenPolicy) -> np.ndarray:
    """(7, 256) raw-action values of the interval edges, per action dimension (ascending)."""
    e = np.linspace(-1, 1, N_BINS)[None].repeat(ACTION_DIM, 0)
    lo, hi = policy.act_low[:, None], policy.act_high[:, None]
    return np.where(policy.act_mask[:, None], 0.5 * (e + 1) * (hi - lo + 1e-8) + lo, e)


def interval_transfer(src: TokenPolicy, dst: TokenPolicy) -> np.ndarray:
    """(7, 255, 255) row-stochastic matrices: mass of source interval i -> target interval j."""
    es, ed = _edges_raw(src), _edges_raw(dst)
    out = np.zeros((ACTION_DIM, N_INT, N_INT))
    for d in range(ACTION_DIM):
        a0, a1 = np.clip(es[d, :-1], ed[d, 0], ed[d, -1]), np.clip(es[d, 1:], ed[d, 0], ed[d, -1])
        ov = np.clip(np.minimum(a1[:, None], ed[d, None, 1:]) - np.maximum(a0[:, None], ed[d, None, :-1]), 0, None)
        tot = ov.sum(1)
        outside = tot <= 0  # the whole source interval lies beyond the target range
        ov[outside & (es[d, 1:] <= ed[d, 0]), 0] = 1.0
        ov[outside & (es[d, :-1] >= ed[d, -1]), -1] = 1.0
        out[d] = ov / ov.sum(1, keepdims=True)
    return out


def token_transfer(src: TokenPolicy, dst: TokenPolicy) -> torch.Tensor:
    """(7, 256, 256) matrices on token probabilities (token k decodes to interval clip(255 - k, 0, 254))."""
    m = interval_transfer(src, dst)
    interval = np.clip(N_BINS - np.arange(N_BINS) - 1, 0, N_INT - 1)  # token -> interval
    share = 1.0 / np.bincount(interval, minlength=N_INT)[interval]  # tokens 0 and 1 share one interval
    t = m[:, interval][:, :, interval] * share[None, None, :]
    return torch.from_numpy(t).float()


class RebinnedPolicy:
    """`source` policy with its token distribution expressed in `target`'s bins.

    Acting is done in the source's own bins (its behaviour is unchanged); the returned `logits` are the
    transferred distribution, usable as a token-level teacher for `target`, and `bins` is the target token
    closest to each executed action.
    """

    def __init__(self, source: TokenPolicy, target: TokenPolicy):
        self.source, self.target = source, target
        self.device, self.center_crop, self.unnorm_key = source.device, source.center_crop, target.unnorm_key
        t = token_transfer(source, target).to(source.device)
        self.transfer = t[torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK) % ACTION_DIM]  # (56, 256, 256), token j is dim j % 7
        self.nearest = self.transfer.argmax(-1)  # (56, 256) source token -> target token holding most of its mass

    def transfer_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.einsum("bjk,jkl->bjl", torch.softmax(logits.float(), -1), self.transfer)
        return torch.log(probs.clamp_min(1e-12))

    @torch.inference_mode()
    def act(self, images, task_descriptions, sample: bool = False, temperature: float = 1.0, generator=None, pils=None,
            obs=None):  # `obs` is part of the common policy interface; a token policy does not use it
        logits, _ = self.source.forward_logits(self.source.build_inputs(images, task_descriptions, pils))
        if sample:
            p = torch.softmax(logits / temperature, -1)
            src_bins = torch.multinomial(p.reshape(-1, N_BINS), 1, generator=generator).reshape(p.shape[:2])
        else:
            src_bins = logits.argmax(-1)
        bins = self.nearest.gather(1, src_bins.T).T.cpu()
        return {"logits": self.transfer_logits(logits).cpu(), "bins": bins, "actions": self.source.decode(src_bins.cpu().numpy())}


def load_policy(spec: str, suite: str, device: str, target: Optional[TokenPolicy] = None):
    """`spec` is a checkpoint path, optionally prefixed:

    rebin:<path>   express that checkpoint's distribution in `target`'s bins (different action normalisation)
    raw:<path>     feed that checkpoint raw 256 x 256 frames (RLinf's image pipeline, see `preprocess_image`)
    oft:<path>     standard OpenVLA-OFT checkpoint (two images, proprio, L1 head): `ContinuousPolicy`
    pi05:<path>    LeRobot pi0.5 checkpoint: `Pi05Policy` (pi05 environment)
    """
    if spec.startswith("oft:"):
        from src.policy.continuous_policy import ContinuousPolicy

        return ContinuousPolicy(spec[len("oft:"):], suite, device=device)
    if spec.startswith("pi05:"):  # LeRobot pi0.5 checkpoint (needs the pi05 environment)
        from src.policy.pi05_policy import Pi05Policy

        return Pi05Policy(spec[len("pi05:"):], suite, device=device)
    if spec.startswith("rebin:"):
        assert target is not None, "rebin: needs a target policy"
        return RebinnedPolicy(TokenPolicy(spec[len("rebin:"):], suite, device=device), target)
    if spec.startswith("raw:"):
        return TokenPolicy(spec[len("raw:"):], suite, device=device, raw_images=True)
    return TokenPolicy(spec, suite, device=device)
