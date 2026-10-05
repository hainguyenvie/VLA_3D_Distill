"""Privileged approach oracle: a diagnostic upper bound, not a method.

Until the wrapped policy first closes the gripper in an episode, the horizontal translation of its action chunk is
replaced by a proportional servo on the true gripper -> target offset (simulator state). Height, rotation, gripper
timing and everything after the first grasp attempt stay the policy's own. The success rate with this wrapper is
what the policy would reach if the only thing fixed were where the gripper is, horizontally, when it first closes.
"""
import numpy as np

MAX_DELTA = 0.05  # metres of commanded end-effector translation per control step at action 1.0 (robosuite OSC_POSE)


class ApproachOracle:
    needs_obs = True

    def __init__(self, policy, gain: float = 0.5):
        self.policy, self.gain = policy, gain

    def __getattr__(self, name):  # device, center_crop, raw_images, ... of the wrapped policy
        return getattr(self.policy, name)

    def act(self, images, task_descriptions, sample: bool = False, temperature: float = 1.0, generator=None, pils=None,
            obs=None):
        out = self.policy.act(images, task_descriptions, sample=sample, temperature=temperature, generator=generator,
                              pils=pils, obs=obs if getattr(self.policy, "needs_obs", False) else None)
        actions = np.array(out["actions"], dtype=np.float32)  # (B, chunk, 7); gripper in [0, 1], 1 = open
        for i, o in enumerate(obs):
            if o["closed_before"] or not np.isfinite(o["target_pos"]).all():
                continue
            offset = (o["target_pos"][:2] - o["eef_pos"][:2]).astype(np.float64)
            for j in range(actions.shape[1]):
                if actions[i, j, -1] < 0.5:  # the policy closes the gripper from this step on: hands off
                    break
                a = np.clip(self.gain * offset / MAX_DELTA, -1.0, 1.0)
                actions[i, j, :2] = a
                offset = offset - a * MAX_DELTA  # open-loop within the chunk; re-measured at the next query
        return dict(out, actions=actions)
