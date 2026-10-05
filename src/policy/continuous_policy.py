"""Standard OpenVLA-OFT policy: third-person + wrist image, proprio token, L1-regression head over an 8-step chunk.

This is the configuration of the official LIBERO checkpoints (and of Spatial Forcing / ROCKET, which add
nothing at inference). Batched re-implementation of upstream `predict_action` with an action head;
`scripts/check_policy.py --oft` asserts it reproduces upstream on real observations.
"""
import glob
import json
import os

import numpy as np
import torch

from experiments.robot.libero.libero_utils import quat2axisangle
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from src.policy.token_policy import TokenPolicy, preprocess_batch


def _load_component(module, checkpoint_dir, pattern):
    files = sorted(glob.glob(os.path.join(checkpoint_dir, f"{pattern}*")))
    assert len(files) == 1, f"expected one {pattern}* file in {checkpoint_dir}, found {files}"
    state = torch.load(files[0], map_location="cpu", weights_only=True)
    module.load_state_dict({k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()})


def proprio_state(obs) -> np.ndarray:
    """8-d proprio of the LIBERO datasets: end-effector position, axis-angle orientation, gripper joints."""
    return np.concatenate((obs["eef_pos"], quat2axisangle(obs["eef_quat"].copy()), obs["gripper_qpos"]))


class ContinuousPolicy(TokenPolicy):
    needs_obs = True  # `act` needs the full observation (wrist image, proprio), not only the third-person frame

    def __init__(self, checkpoint: str, unnorm_key: str, device: str = "cuda:0", center_crop: bool = True):
        super().__init__(checkpoint, unnorm_key, device, center_crop)
        self.vla.vision_backbone.set_num_images_in_input(2)
        d = self.vla.llm_dim
        self.action_head = L1RegressionActionHead(input_dim=d, hidden_dim=d, action_dim=ACTION_DIM)
        self.proprio_projector = ProprioProjector(llm_dim=d, proprio_dim=PROPRIO_DIM)
        _load_component(self.action_head, checkpoint, "action_head")
        _load_component(self.proprio_projector, checkpoint, "proprio_projector")
        self.action_head = self.action_head.to(torch.bfloat16).to(self.device).eval()
        self.proprio_projector = self.proprio_projector.to(torch.bfloat16).to(self.device).eval()
        p = json.load(open(os.path.join(checkpoint, "dataset_statistics.json")))[self.unnorm_key]["proprio"]
        self.prop_low, self.prop_high = np.array(p["q01"]), np.array(p["q99"])
        self.prop_mask = np.array(p.get("mask", np.ones_like(p["q01"], dtype=bool)))

    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        scaled = 2 * (proprio - self.prop_low) / (self.prop_high - self.prop_low + 1e-8) - 1
        return np.clip(np.where(self.prop_mask, scaled, proprio), -1.0, 1.0)

    @torch.inference_mode()
    def act(self, images, task_descriptions, sample: bool = False, temperature: float = 1.0, generator=None, pils=None,
            obs=None):
        """Returns dict(actions (B, 8, 7) unnormalised). Deterministic: `sample` is ignored (regression head)."""
        inputs = self.build_inputs(images, task_descriptions, pils)
        wrist = preprocess_batch([o["wrist_rgb"] for o in obs], self.center_crop)
        wrist_pv = self.image_processor.preprocess(wrist, return_tensors="pt")["pixel_values"]
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_pv.to(self.device, dtype=torch.bfloat16)], dim=1)
        proprio = self._normalize_proprio(np.stack([proprio_state(o) for o in obs]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prop = self.proprio_projector(torch.from_numpy(proprio).to(self.device, dtype=torch.bfloat16)).unsqueeze(1)
            out, idx, _ = self._run_llm(inputs, extra_tokens=prop)
            h = out.hidden_states[-1][torch.arange(idx.shape[0], device=idx.device)[:, None], idx]
            norm = self.action_head.predict_action(h)  # (B, 8, 7) in [-1, 1]
        norm = norm.float().cpu().numpy().reshape(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)
        actions = np.where(self.act_mask, 0.5 * (norm + 1) * (self.act_high - self.act_low + 1e-8) + self.act_low, norm)
        return {"actions": actions}
