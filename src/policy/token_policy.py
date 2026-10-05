"""Discrete-token OpenVLA-OFT policy (the SimpleVLA-RL / VLA-OPD variant).

One third-person image, no proprio, parallel decoding of an 8-step action chunk as 7 x 8 = 56 action tokens,
each a categorical over 256 bins. Unlike upstream `predict_action` (batch size 1, argmax only) this wrapper is
batched over mixed prompts and exposes the per-token logits, which on-policy distillation needs.

The forward pass mirrors `OpenVLAForActionPrediction.predict_action` + `_regression_or_discrete_prediction`
(openvla-oft) and the 256-bin slicing of SimpleVLA-RL's `_verl_discrete_prediction`; `scripts/check_policy.py`
asserts that greedy actions are identical to upstream on real observations.

Numerical protocol (fixed for rollout, training and evaluation): bf16 base weights, forward under
`torch.autocast(bfloat16)` as in SimpleVLA-RL rollouts and OpenVLA-OFT training; LoRA adapters stay fp32.
"""
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # TF is only used for CPU image preprocessing

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from experiments.robot.openvla_utils import center_crop_image, resize_image_for_policy  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, STOP_INDEX  # noqa: E402

N_BINS = 256
N_ACT_TOKENS = ACTION_DIM * NUM_ACTIONS_CHUNK
EMPTY_TOKEN = 29871  # the '' token that follows "Out:" at training time
IMAGE_SIZE = 224


def prompt_for(task_description: str) -> str:
    return f"In: What action should the robot take to {task_description.lower()}?\nOut:"


def preprocess_image(img: np.ndarray, center_crop: bool = True) -> Image.Image:
    """uint8 HxWx3 (already rotated to the training orientation) -> PIL image ready for the image processor."""
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        img = resize_image_for_policy(img, IMAGE_SIZE)
    pil = Image.fromarray(img).convert("RGB")
    return center_crop_image(pil) if center_crop else pil


class TokenPolicy:
    def __init__(self, checkpoint: str, unnorm_key: str, device: str = "cuda:0", center_crop: bool = True):
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.center_crop = center_crop
        vla, info = OpenVLAForActionPrediction.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, output_loading_info=True
        )
        self.loading_info = {k: sorted(v) if isinstance(v, (list, set)) else v for k, v in info.items()}
        vla.vision_backbone.set_num_images_in_input(1)
        self.vla = vla.eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.image_processor = PrismaticImageProcessor.from_pretrained(checkpoint)
        self.n_patches = vla.vision_backbone.get_num_patches()
        self.vocab_size = vla.vocab_size  # 32000: text vocab without the pad-to-multiple rows
        self.bin_centers = vla.bin_centers  # 255 centres of 256 uniform bin edges on [-1, 1]

        stats_path = os.path.join(checkpoint, "dataset_statistics.json")
        norm_stats = json.load(open(stats_path)) if os.path.isfile(stats_path) else vla.norm_stats
        if unnorm_key not in norm_stats and f"{unnorm_key}_no_noops" in norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in norm_stats, f"unnorm key {unnorm_key} not in {list(norm_stats)}"
        self.unnorm_key = unnorm_key
        vla.norm_stats = norm_stats  # as upstream `_load_dataset_stats`, so `predict_action` un-normalises alike
        a = norm_stats[unnorm_key]["action"]
        self.act_low, self.act_high = np.array(a["q01"]), np.array(a["q99"])
        self.act_mask = np.array(a.get("mask", np.ones_like(a["q01"], dtype=bool)))
        self._prompt_cache: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ LoRA
    def add_lora(self, rank: int = 32, adapter_path: Optional[str] = None):
        """Same adapter recipe as OpenVLA-OFT fine-tuning (all linear layers, alpha = min(rank, 16)).

        PEFT injects the adapters into `self.vla` in place, so every forward below goes through them.
        """
        from peft import LoraConfig, PeftModel, get_peft_model

        if adapter_path:
            self.peft = PeftModel.from_pretrained(self.vla, adapter_path, is_trainable=True)
        else:
            cfg = LoraConfig(r=rank, lora_alpha=min(rank, 16), lora_dropout=0.0, target_modules="all-linear",
                             init_lora_weights="gaussian")
            self.peft = get_peft_model(self.vla, cfg)
        params = [p for p in self.peft.parameters() if p.requires_grad]
        for p in params:  # bf16 master weights lose Adam updates of ~1e-4 to rounding; keep adapters in fp32
            p.data = p.data.float()
        return params

    def save_lora(self, path: str) -> None:
        self.peft.save_pretrained(path)

    # ---------------------------------------------------------------- inputs
    def _prompt_ids(self, task_description: str) -> torch.Tensor:
        if task_description not in self._prompt_cache:
            ids = self.tokenizer(prompt_for(task_description), return_tensors="pt").input_ids[0]
            if ids[-1] != EMPTY_TOKEN:
                ids = torch.cat([ids, torch.tensor([EMPTY_TOKEN])])
            self._prompt_cache[task_description] = ids
        return self._prompt_cache[task_description]

    def build_inputs(self, images: Sequence[np.ndarray], task_descriptions: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Right-padded batch: [BOS, prompt, '', 56 placeholders, </s>, pad...]; `prompt_len` counts BOS..''."""
        pils = [preprocess_image(im, self.center_crop) for im in images]
        pixel_values = self.image_processor.preprocess(pils, return_tensors="pt")["pixel_values"]
        prompts = [self._prompt_ids(t) for t in task_descriptions]
        prompt_len = torch.tensor([len(p) for p in prompts])
        total = int(prompt_len.max()) + N_ACT_TOKENS + 1
        input_ids = torch.full((len(prompts), total), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), total), dtype=torch.bool)
        for i, p in enumerate(prompts):
            n = len(p)
            input_ids[i, :n] = p
            input_ids[i, n : n + N_ACT_TOKENS] = 1  # placeholders; their embeddings are zeroed in the forward
            input_ids[i, n + N_ACT_TOKENS] = STOP_INDEX
            attention_mask[i, : n + N_ACT_TOKENS + 1] = True
        return {
            "pixel_values": pixel_values.to(self.device, dtype=torch.bfloat16),
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "prompt_len": prompt_len.to(self.device),
        }

    # --------------------------------------------------------------- forward
    def forward_logits(self, inputs: Dict[str, torch.Tensor], hidden_layers: Optional[List[int]] = None):
        """Returns action logits (B, 56, 256) and, if requested, {layer: visual-token hidden states (B, 256, D)}."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self._forward_logits(inputs, hidden_layers)

    def _forward_logits(self, inputs, hidden_layers):
        vla = self.vla
        input_ids, prompt_len = inputs["input_ids"], inputs["prompt_len"]
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device)[None]
        act_mask = (pos >= prompt_len[:, None]) & (pos < prompt_len[:, None] + N_ACT_TOKENS)
        emb = vla.get_input_embeddings()(input_ids) * ~act_mask[..., None]
        patches = vla.projector(vla.vision_backbone(inputs["pixel_values"]))  # (B, 256, D)
        P = patches.shape[1]
        mm = torch.cat([emb[:, :1], patches, emb[:, 1:]], dim=1)
        am = inputs["attention_mask"]
        mm_mask = torch.cat([am[:, :1], torch.ones((B, P), dtype=am.dtype, device=am.device), am[:, 1:]], dim=1)
        out = vla.language_model(
            input_ids=None,
            attention_mask=mm_mask,
            inputs_embeds=mm,
            use_cache=None,
            output_hidden_states=True,
            return_dict=True,
        )
        # logits at position j predict token j+1, so the 56 action tokens are read from the '' token onwards
        idx = (P + prompt_len - 1)[:, None] + torch.arange(N_ACT_TOKENS, device=input_ids.device)[None]
        act_logits = out.logits[torch.arange(B, device=idx.device)[:, None], idx]
        act_logits = act_logits[..., self.vocab_size - N_BINS : self.vocab_size].float()
        if hidden_layers is None:
            return act_logits, None
        return act_logits, {l: out.hidden_states[l][:, 1 : 1 + P] for l in hidden_layers}

    @torch.inference_mode()
    def act(self, images, task_descriptions, sample: bool = False, temperature: float = 1.0, generator=None):
        """Returns dict(logits (B,56,256) float32 cpu, bins (B,56) int64 cpu, actions (B,8,7) unnormalised)."""
        logits, _ = self.forward_logits(self.build_inputs(images, task_descriptions))
        if sample:
            probs = torch.softmax(logits / temperature, dim=-1)
            bins = torch.multinomial(probs.reshape(-1, N_BINS), 1, generator=generator).reshape(logits.shape[:2])
        else:
            bins = logits.argmax(dim=-1)
        bins = bins.cpu()
        return {"logits": logits.cpu(), "bins": bins, "actions": self.decode(bins.numpy())}

    # ---------------------------------------------------------------- decode
    def decode(self, bins: np.ndarray) -> np.ndarray:
        """Bin index k in [0, 256) is token id vocab-256+k; upstream maps token -> bin centre -> q01/q99 range."""
        disc = np.clip(N_BINS - bins - 1, 0, self.bin_centers.shape[0] - 1)
        norm = self.bin_centers[disc].reshape(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)
        return np.where(self.act_mask, 0.5 * (norm + 1) * (self.act_high - self.act_low + 1e-8) + self.act_low, norm)
