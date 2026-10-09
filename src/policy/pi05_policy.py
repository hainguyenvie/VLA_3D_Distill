"""pi0.5 (LeRobot PyTorch port) behind the policy interface of this repo (see `Collector`).

The LIBERO-finetuned checkpoint `lerobot/pi05_libero_finetuned` consumes the two 256 x 256 frames (third-person
and wrist, rotated 180 degrees as this repo already does), the 8-d proprio (eef position, axis-angle, gripper) and
the instruction; it predicts a 50-step chunk of 7-d actions, of which the first `n_action_steps` (10 in the LeRobot /
openpi LIBERO protocol) are executed. Actions come out in the environment's own convention (gripper -1 open, +1
close); `act` hands the gripper back in this repo's [0, 1] convention (1 = open) so that `postprocess_actions` of
the collector restores the environment one.
"""
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.policy.continuous_policy import proprio_state
from src.policy.token_policy import preprocess_batch  # noqa: F401  (kept importable for callers that share preprocessing)

TOKENIZER_FALLBACK = "paligemma_tokenizer_fallback"  # folder under checkpoints/ with an ungated copy of the tokenizer


class Pi05Policy:
    needs_obs = True
    raw_images = True  # the collector must not run the OpenVLA image pipeline for this policy
    center_crop = False

    def __init__(self, checkpoint: str, unnorm_key: str = "", device: str = "cuda:0", n_action_steps: int = 10,
                 tokenizer: Optional[str] = None):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        from lerobot.configs.policies import PreTrainedConfig

        self.checkpoint, self.device, self.n_action_steps = checkpoint, torch.device(device), n_action_steps
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.compile_model = False  # batches of varying size; torch.compile would recompile and slows the first calls
        cfg.device = str(self.device)
        self.vla = PI05Policy.from_pretrained(checkpoint, config=cfg)
        self.vla.to(self.device).eval()
        self.loading_info = {}  # LeRobot reports mismatches itself (a warning about remapped keys is expected)
        self.unnorm_key = unnorm_key or "libero"  # normalisation lives in the checkpoint's processors
        overrides = {"device_processor": {"device": str(self.device)}}
        tok = tokenizer or (os.path.join(os.path.dirname(checkpoint.rstrip("/")), TOKENIZER_FALLBACK)
                            if not os.environ.get("HF_TOKEN") else None)
        if tok and os.path.isdir(tok):
            overrides["tokenizer_processor"] = {"tokenizer_name": tok}
        self.pre, self.post = make_pre_post_processors(self.vla.config, pretrained_path=checkpoint, preprocessor_overrides=overrides)
        self.chunk = self.vla.config.chunk_size
        from safetensors.torch import load_file

        stats = load_file(os.path.join(checkpoint, "policy_preprocessor_step_2_normalizer_processor.safetensors"))
        self.act_mean, self.act_std = stats["action.mean"].numpy(), stats["action.std"].numpy()
        self.fixed_noise = True  # deterministic sampling (see `_noise`); the LeRobot eval draws fresh noise per call
        # `state_blind`: the proprio given to the model is a constant (the dataset mean), so it carries no information
        # (ablation of the proprio route; the teacher of a distillation still sees the true state, see AdapterOff)
        self.state_blind = False
        self.state_const = (stats["observation.state.mean"].numpy().astype(np.float32)
                            if "observation.state.mean" in stats else None)

    # ---------------------------------------------------------------- inputs
    @staticmethod
    def _images(frames: Sequence[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W) uint8
        return x.float() / 255.0

    def batch(self, images, task_descriptions, obs) -> Dict:
        raw = {
            "observation.images.image": self._images(images),
            "observation.images.image2": self._images([o["wrist_rgb"] for o in obs]),
            "observation.state": torch.from_numpy(np.stack([self.state_const if self.state_blind else proprio_state(o)
                                                            for o in obs])).float(),
            "task": list(task_descriptions),
        }
        return self.pre(raw)

    # --------------------------------------------------------------- forward
    def _noise(self, bsize: int) -> torch.Tensor:
        """Sampling noise. With `fixed_noise` every call draws the same rows (row i depends only on i), so two calls on
        two versions of the same batch (e.g. a scene and its counterfactual) differ only through the observation."""
        shape = (bsize, self.chunk, self.vla.config.max_action_dim)
        g = torch.Generator(device="cpu").manual_seed(1234) if self.fixed_noise else None
        return torch.randn(shape, generator=g).to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def chunk_norm(self, images, task_descriptions, obs) -> torch.Tensor:
        """Normalised 50-step chunk (B, 50, 7), as the flow head predicts it (before un-normalisation)."""
        b = self.batch(images, task_descriptions, obs)
        return self.vla.predict_action_chunk(b, noise=self._noise(len(images)))

    @torch.no_grad()
    def chunk_env(self, images, task_descriptions, obs) -> np.ndarray:
        """Full 50-step chunk in environment units (B, 50, 7); gripper -1 open, +1 close."""
        return self.post(self.chunk_norm(images, task_descriptions, obs)).float().cpu().numpy()

    def act(self, images, task_descriptions, sample: bool = False, temperature: float = 1.0, generator=None, pils=None,
            obs=None):
        """Returns dict(actions (B, n_action_steps, 7) with the gripper in [0, 1] (1 = open), as the other policies;
        actions_norm (B, 50, 7): the full normalised chunk, the distillation target)."""
        norm = self.chunk_norm(images, task_descriptions, obs)
        chunk = self.post(norm).float().cpu().numpy()[:, : self.n_action_steps]
        out = chunk.copy()
        out[..., -1] = 0.5 * (1.0 - np.clip(chunk[..., -1], -1.0, 1.0))  # env convention -> [0, 1], 1 = open
        return {"actions": out, "actions_env": chunk, "actions_norm": norm.float().cpu().numpy()}

    def mirror_norm(self, a_norm: np.ndarray) -> np.ndarray:
        """Reflect normalised chunks through the robot's vertical plane: un-normalise, negate dy, rx, rz, normalise."""
        a = a_norm * (self.act_std + 1e-8) + self.act_mean
        a = a * np.array([1, -1, 1, -1, 1, -1, 1], dtype=a.dtype)
        return ((a - self.act_mean) / (self.act_std + 1e-8)).astype(np.float32)

    def rotate_norm(self, a_norm: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Rotate normalised chunks (B, T, 7) about the vertical axis by `theta` (B,): un-normalise, rotate (dx, dy) and
        (rx, ry) (world-frame deltas), normalise."""
        a = a_norm * (self.act_std + 1e-8) + self.act_mean
        c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
        r = a.copy()
        for i, j in ((0, 1), (3, 4)):
            r[..., i], r[..., j] = c * a[..., i] - s * a[..., j], s * a[..., i] + c * a[..., j]
        return ((r - self.act_mean) / (self.act_std + 1e-8)).astype(np.float32)

    def velocity(self, images, task_descriptions, obs, x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Flow velocity v(o, x_t, t) of the action expert (B, 50, 7); differentiable (the training forward of
        PI05Pytorch.forward without its loss). x_t: (B, 50, 7) normalised noisy chunk; time: (B,)."""
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, pad_vector

        b = self.batch(images, task_descriptions, obs)
        m = self.vla.model
        images_, img_masks = self.vla._preprocess_images(b)
        tokens, masks = b["observation.language.tokens"], b["observation.language.attention_mask"]
        x = pad_vector(x_t, self.vla.config.max_action_dim)
        prefix_embs, prefix_pad, prefix_att = m.embed_prefix(images_, img_masks, tokens, masks)
        suffix_embs, suffix_pad, suffix_att, adarms_cond = m.embed_suffix(x, time)
        if m.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs, prefix_embs = suffix_embs.to(torch.bfloat16), prefix_embs.to(torch.bfloat16)
        pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        att = torch.cat([prefix_att, suffix_att], dim=1)
        att_4d = m._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        pos = torch.cumsum(pad, dim=1) - 1

        def fwd(prefix_embs, suffix_embs, att_4d, pos, adarms_cond):
            (_, suffix_out), _ = m.paligemma_with_expert.forward(attention_mask=att_4d, position_ids=pos, past_key_values=None,
                                                                 inputs_embeds=[prefix_embs, suffix_embs], use_cache=False,
                                                                 adarms_cond=[None, adarms_cond])
            return suffix_out

        out = m._apply_checkpoint(fwd, prefix_embs, suffix_embs, att_4d, pos, adarms_cond)
        v = m.action_out_proj(out[:, -self.chunk :].to(torch.float32))
        return v[..., : x_t.shape[-1]]

    # ------------------------------------------------------------------ LoRA
    def add_lora(self, rank: int = 32, adapter_path: Optional[str] = None, scope: str = "all"):
        """LoRA on the attention / MLP projections. scope "all": every such projection (vision tower included);
        "llm": only the PaliGemma language model and the action expert (the image encoder stays frozen);
        "readout": only the action expert's output projection (hidden state -> action velocity);
        "expert_late": the projections of the action expert's last 6 layers (12-17) and its output projection."""
        from peft import LoraConfig, PeftModel, get_peft_model

        if adapter_path:
            self.peft = PeftModel.from_pretrained(self.vla.model, adapter_path, is_trainable=True)
            self.vla.model = self.peft
        else:
            proj = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            if scope == "all":
                targets = proj
            elif scope == "llm":
                targets = r".*(language_model|gemma_expert).*\.(" + "|".join(proj) + ")"
            elif scope == "readout":
                targets = r"(.*\.)?action_out_proj"
            elif scope == "expert_late":
                targets = r"(.*gemma_expert.*layers\.1[2-7]\..*\.(" + "|".join(proj) + r")|(.*\.)?action_out_proj)"
            else:
                raise ValueError(scope)
            cfg = LoraConfig(r=rank, lora_alpha=min(rank, 16), lora_dropout=0.0, target_modules=targets,
                             init_lora_weights="gaussian")
            self.peft = get_peft_model(self.vla.model, cfg)
            self.vla.model = self.peft  # PI05Policy calls self.model.<...>: route it through the adapters
        params = [p for p in self.peft.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()
        return params

    def save_lora(self, path: str) -> None:
        """Adapter weights + config in the PEFT layout (PeftModel.from_pretrained loads it). PEFT's own
        save_pretrained fails on the model card of a non-transformers config, so it is written directly."""
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        os.makedirs(path, exist_ok=True)
        self.peft.peft_config["default"].save_pretrained(path)
        state = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(self.peft).items()}
        save_file(state, os.path.join(path, "adapter_model.safetensors"))
