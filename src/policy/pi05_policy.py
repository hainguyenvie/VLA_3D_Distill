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
        overrides = {"device_processor": {"device": str(self.device)}}
        tok = tokenizer or (os.path.join(os.path.dirname(checkpoint.rstrip("/")), TOKENIZER_FALLBACK)
                            if not os.environ.get("HF_TOKEN") else None)
        if tok and os.path.isdir(tok):
            overrides["tokenizer_processor"] = {"tokenizer_name": tok}
        self.pre, self.post = make_pre_post_processors(self.vla.config, pretrained_path=checkpoint, preprocessor_overrides=overrides)
        self.chunk = self.vla.config.chunk_size

    # ---------------------------------------------------------------- inputs
    @staticmethod
    def _images(frames: Sequence[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W) uint8
        return x.float() / 255.0

    def batch(self, images, task_descriptions, obs) -> Dict:
        raw = {
            "observation.images.image": self._images(images),
            "observation.images.image2": self._images([o["wrist_rgb"] for o in obs]),
            "observation.state": torch.from_numpy(np.stack([proprio_state(o) for o in obs])).float(),
            "task": list(task_descriptions),
        }
        return self.pre(raw)

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def chunk_env(self, images, task_descriptions, obs) -> np.ndarray:
        """Full 50-step chunk in environment units (B, 50, 7); gripper -1 open, +1 close."""
        b = self.batch(images, task_descriptions, obs)
        norm = self.vla.predict_action_chunk(b)
        return self.post(norm).float().cpu().numpy()

    def act(self, images, task_descriptions, sample: bool = False, temperature: float = 1.0, generator=None, pils=None,
            obs=None):
        """Returns dict(actions (B, n_action_steps, 7)) with the gripper in [0, 1] (1 = open), as the other policies."""
        chunk = self.chunk_env(images, task_descriptions, obs)[:, : self.n_action_steps]
        out = chunk.copy()
        out[..., -1] = 0.5 * (1.0 - np.clip(chunk[..., -1], -1.0, 1.0))  # env convention -> [0, 1], 1 = open
        return {"actions": out, "actions_env": chunk}

    # ------------------------------------------------------------------ LoRA
    def add_lora(self, rank: int = 32, adapter_path: Optional[str] = None):
        """LoRA on the attention / MLP projections of the PaliGemma language model and the action expert."""
        from peft import LoraConfig, PeftModel, get_peft_model

        if adapter_path:
            self.peft = PeftModel.from_pretrained(self.vla.model, adapter_path, is_trainable=True)
        else:
            cfg = LoraConfig(r=rank, lora_alpha=min(rank, 16), lora_dropout=0.0,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                             init_lora_weights="gaussian")
            self.peft = get_peft_model(self.vla.model, cfg)
        params = [p for p in self.peft.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()
        return params

    def save_lora(self, path: str) -> None:
        self.peft.save_pretrained(path)
