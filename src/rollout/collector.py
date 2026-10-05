"""Batched rollout collection with per-query logging.

The acting policy is queried once per action chunk for all active environments. Every queried state is
logged with its simulator state (so depth / other views / teacher labels can be regenerated later), the
acting policy's logits, and the logits of any extra `labelers` (e.g. the frozen teacher on student states).
"""
import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.policy.token_policy import TokenPolicy
from src.rollout.vec_env import LiberoVecEnv, postprocess_actions

STEP_KEYS = ("rgb", "depth", "wrist_rgb", "eef_pos", "eef_quat", "gripper_qpos", "sim_state")


def token_stats(logits: torch.Tensor, ref_logits: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Mean per-token entropy of `logits` and, if given, mean per-token KL(logits || ref_logits)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    out = {"entropy": float(-(logp.exp() * logp).sum(-1).mean())}
    if ref_logits is not None:
        logq = torch.log_softmax(ref_logits.float(), dim=-1)
        out["kl"] = float((logp.exp() * (logp - logq)).sum(-1).mean())
        out["agree"] = float((logits.argmax(-1) == ref_logits.argmax(-1)).float().mean())
    return out


class Collector:
    def __init__(self, vec: LiberoVecEnv, policy: TokenPolicy, labelers: Optional[Dict[str, TokenPolicy]] = None,
                 sample: bool = False, temperature: float = 1.0, seed: int = 7):
        self.vec, self.policy, self.labelers = vec, policy, labelers or {}
        self.sample, self.temperature = sample, temperature
        self.gen = torch.Generator(device=policy.device).manual_seed(seed)
        self.tasks = {i: (lang, n) for i, lang, n in vec.task_info()}

    def run(self, episodes: Sequence[Tuple[int, int]], out_dir: Optional[str] = None, save_steps: bool = True,
            on_episode: Optional[Callable[[dict], None]] = None) -> List[dict]:
        """Run `episodes` = [(task_id, trial_id), ...]; returns one summary dict per episode.

        With `out_dir`, a summary line is appended to episodes.jsonl per finished episode (already present
        episodes are skipped on restart) and, if `save_steps`, the per-query arrays go to steps/<id>.npz.
        """
        done_ids, results = set(), []
        if out_dir:
            os.makedirs(os.path.join(out_dir, "steps"), exist_ok=True)
            path = os.path.join(out_dir, "episodes.jsonl")
            if os.path.exists(path):
                results = [json.loads(l) for l in open(path) if l.strip()]
                done_ids = {(r["task_id"], r["trial_id"]) for r in results}
        # keep each worker on one task as long as possible: rebuilding an env costs several seconds
        by_task: Dict[int, List[Tuple[int, int]]] = {}
        for t, n in sorted(tuple(e) for e in episodes if tuple(e) not in done_ids):
            by_task.setdefault(t, []).append((t, n))
        slots: List[Optional[dict]] = [None] * self.vec.num_envs
        last_task = [None] * self.vec.num_envs

        def next_episode(i):
            if not by_task:
                return None
            t = last_task[i] if last_task[i] in by_task else None
            if t is None:  # take the task with the most remaining episodes that no other worker is on
                busy = {s["task_id"] for s in slots if s is not None}
                free = [k for k in by_task if k not in busy] or list(by_task)
                t = max(free, key=lambda k: len(by_task[k]))
            ep = by_task[t].pop(0)
            if not by_task[t]:
                del by_task[t]
            last_task[i] = t
            return ep

        def start(i):
            ep = next_episode(i)
            if ep is None:
                slots[i] = None
                return
            self.vec.send(i, "reset", ep)
            slots[i] = {"task_id": ep[0], "trial_id": ep[1], "steps": {k: [] for k in STEP_KEYS}, "t": [],
                        "bins": [], "logits": [], "actions": [], "label_logits": {k: [] for k in self.labelers},
                        "obs": None, "t0": time.time()}

        for i in range(self.vec.num_envs):
            start(i)
        for i, s in enumerate(slots):
            if s is not None:
                s["obs"] = self.vec.recv(i)

        while any(s is not None for s in slots):
            act_idx = [i for i, s in enumerate(slots) if s is not None]
            images = [slots[i]["obs"]["rgb"] for i in act_idx]
            descs = [self.tasks[slots[i]["task_id"]][0] for i in act_idx]
            out = self.policy.act(images, descs, sample=self.sample, temperature=self.temperature, generator=self.gen)
            label_logits = {k: p.act(images, descs)["logits"] for k, p in self.labelers.items()}
            env_actions = postprocess_actions(out["actions"])
            for j, i in enumerate(act_idx):
                s, obs = slots[i], slots[i]["obs"]
                for k in STEP_KEYS:
                    if k in obs:
                        s["steps"][k].append(obs[k])
                s["t"].append(obs["t"])
                s["bins"].append(out["bins"][j].numpy().astype(np.uint8))
                s["logits"].append(out["logits"][j].numpy().astype(np.float16))
                s["actions"].append(out["actions"][j].astype(np.float32))
                for k in self.labelers:
                    s["label_logits"][k].append(label_logits[k][j].numpy().astype(np.float16))
                self.vec.send(i, "step", env_actions[j])
            finished = []
            for i in act_idx:
                obs = self.vec.recv(i)
                slots[i]["obs"] = obs
                if not obs["active"]:
                    finished.append(i)
            for i in finished:
                rec = self._finish(slots[i], out_dir, save_steps)
                results.append(rec)
                if on_episode:
                    on_episode(rec)
                start(i)
                if slots[i] is not None:
                    slots[i]["obs"] = self.vec.recv(i)
        return results

    def _finish(self, s: dict, out_dir: Optional[str], save_steps: bool) -> dict:
        final = s["obs"]
        logits = torch.from_numpy(np.stack(s["logits"]).astype(np.float32))
        rec = {
            "task_id": s["task_id"], "trial_id": s["trial_id"], "task": self.tasks[s["task_id"]][0],
            "success": bool(final["done"]), "env_steps": int(final["t"]), "n_queries": len(s["t"]),
            "seconds": round(time.time() - s["t0"], 1), "entropy": token_stats(logits)["entropy"],
        }
        for k, v in s["label_logits"].items():
            st = token_stats(logits, torch.from_numpy(np.stack(v).astype(np.float32)))
            rec[f"kl_{k}"], rec[f"agree_{k}"] = st["kl"], st["agree"]
        if out_dir:
            eid = f"t{s['task_id']:02d}_n{s['trial_id']:02d}"
            if save_steps:
                arrays = {k: np.stack(v) for k, v in s["steps"].items() if v}
                arrays.update(t=np.array(s["t"]), bins=np.stack(s["bins"]), logits=np.stack(s["logits"]),
                              actions=np.stack(s["actions"]), final_sim_state=final["sim_state"])
                arrays.update({f"logits_{k}": np.stack(v) for k, v in s["label_logits"].items()})
                tmp = os.path.join(out_dir, "steps", f".{eid}.{os.getpid()}.npz")
                np.savez_compressed(tmp, **arrays)
                os.replace(tmp, os.path.join(out_dir, "steps", f"{eid}.npz"))
            with open(os.path.join(out_dir, "episodes.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec
