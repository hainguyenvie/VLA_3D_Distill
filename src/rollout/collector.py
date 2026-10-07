"""Batched rollout collection with per-query logging.

The acting policy is queried once per action chunk for all active environments. Every queried state is
logged with its simulator state (so depth / other views / teacher labels can be regenerated later), the
acting policy's logits, and the logits of any extra `labelers` (e.g. the frozen teacher on student states).
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.policy.token_policy import TokenPolicy, preprocess_batch
from src.rollout.vec_env import LiberoVecEnv, mem_available_gb, postprocess_actions

STEP_KEYS = ("rgb", "depth", "wrist_rgb", "eef_pos", "eef_quat", "gripper_qpos", "sim_state", "target_pos",
             "rgb_cf", "wrist_rgb_cf", "cf_delta", "target_pos_cf", "cf_target_name", "cf_source_name", "closed_before",
             "cf_eef_pos", "cf_eef_quat", "cf_theta", "cf_valid")


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
                 sample: bool = False, temperature: float = 1.0, seed: int = 7, task_ids: Optional[Sequence[int]] = None,
                 perturb: bool = False, actor_view: str = "rgb", counterfactual: bool = False):
        """`perturb` asks the envs for visually perturbed episodes (see `vec_env.VIEW_AUG`): `rgb` is then the perturbed
        frame and `rgb_clean` the nominal one. The acting policy is fed `actor_view`; labelers always get the nominal
        frame (a privileged teacher). The logged `rgb` stays the perturbed frame either way."""
        self.vec, self.policy, self.labelers = vec, policy, labelers or {}
        self.sample, self.temperature = sample, temperature
        self.perturb, self.actor_view, self.counterfactual = perturb, actor_view, counterfactual
        self.min_free_gb = float(os.environ.get("MIN_FREE_GB_RUN", 4))
        # the default-pipeline preprocessing is shared with labelers only when the actor uses it too
        assert not getattr(policy, "raw_images", False) or all(getattr(p, "raw_images", False) for p in self.labelers.values()), \
            "shared preprocessing follows the actor"
        self.timing = {"preprocess": 0.0, "act": 0.0, "label": 0.0, "env": 0.0, "other": 0.0, "rounds": 0}
        self._saver = ThreadPoolExecutor(2)  # npz compression off the rollout loop (zlib releases the GIL)
        self._jsonl_lock = threading.Lock()
        self.gen = torch.Generator(device=policy.device).manual_seed(seed)
        if task_ids is None:  # (instruction, number of initial states) per task
            self.tasks = {i: (lang, n) for i, lang, n in vec.task_info()}
        else:  # only these tasks (LIBERO-Plus has thousands; avoids loading every initial-state file)
            self.tasks = {i: (vec.task_language(i), len(vec.init_states(i))) for i in task_ids}

    def run(self, episodes: Sequence[Tuple[int, int]], out_dir: Optional[str] = None, save_steps: bool = True,
            on_episode: Optional[Callable[[dict], None]] = None, keep_steps: bool = False) -> List[dict]:
        """Run `episodes` = [(task_id, trial_id), ...]; returns one summary dict per episode.

        With `out_dir`, a summary line is appended to episodes.jsonl per finished episode (already present
        episodes are skipped on restart) and, if `save_steps`, the per-query arrays go to steps/<id>.npz.
        With `keep_steps`, each returned dict also carries the per-query arrays under "arrays" (in memory).
        """
        self._keep_steps = keep_steps
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
        pending_saves = []
        self._pending_saves = pending_saves
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
            self.vec.reset(i, ep[0], ep[1], perturb=self.perturb, counterfactual=self.counterfactual)
            slots[i] = {"task_id": ep[0], "trial_id": ep[1], "steps": {k: [] for k in STEP_KEYS}, "t": [],
                        "bins": [], "logits": [], "actions": [], "actions_norm": [],
                        "label_logits": {k: [] for k in self.labelers}, "label_actions": {k: [] for k in self.labelers},
                        "label_cf": {k: [] for k in self.labelers},
                        "obs": None, "t0": time.time()}

        for i in range(self.vec.num_envs):
            start(i)

        rounds = 0
        while any(s is not None for s in slots):
            rounds += 1
            # Resets (simulator rebuild, about 2 s) run in the background: an env joins the batch again once its
            # first observation is there. Only when no env at all has an observation do we wait for one.
            for i, s in enumerate(slots):
                if s is not None and s["obs"] is None:
                    if self.vec.ready(i) or not any(x is not None and x["obs"] is not None for x in slots):
                        s["obs"] = self.vec.recv(i)
            if rounds % 20 == 0 and mem_available_gb() < self.min_free_gb:
                # shared machine without swap: stop before we push it into thrashing; finished episodes are saved
                raise MemoryError(f"only {mem_available_gb():.1f} GB RAM available on the machine, aborting the rollout")
            tm, t0 = self.timing, time.perf_counter()
            act_idx = [i for i, s in enumerate(slots) if s is not None and s["obs"] is not None]
            cur = [slots[i]["obs"] for i in act_idx]
            images = [o.get(self.actor_view, o["rgb"]) for o in cur]
            descs = [self.tasks[slots[i]["task_id"]][0] for i in act_idx]
            # one default-pipeline preprocessing shared by the actor and the labelers (a raw-image policy redoes its own)
            pils = None if getattr(self.policy, "raw_images", False) else preprocess_batch(images, self.policy.center_crop)
            clean, clean_pils = images, pils
            if self.labelers and any("rgb_clean" in o for o in cur) and self.actor_view != "rgb_clean":
                clean = [o.get("rgb_clean", o["rgb"]) for o in cur]  # perturbed episodes: labelers see the nominal frame
                clean_pils = preprocess_batch(clean, self.policy.center_crop)
            t1 = time.perf_counter()
            out = self.policy.act(images, descs, sample=self.sample, temperature=self.temperature, generator=self.gen, pils=pils,
                                  obs=cur if getattr(self.policy, "needs_obs", False) else None)
            t2 = time.perf_counter()
            labels = {k: p.act(clean, descs, pils=clean_pils, obs=cur if getattr(p, "needs_obs", False) else None)
                      for k, p in self.labelers.items()}
            label_logits = {k: v["logits"] for k, v in labels.items() if "logits" in v}
            label_actions = {k: v["actions_norm"] for k, v in labels.items() if "actions_norm" in v}
            label_cf = {}
            if self.labelers and all("rgb_cf" in o for o in cur):  # counterfactual world: the labelers also see it
                cf_obs = [dict(o, wrist_rgb=o.get("wrist_rgb_cf", o.get("wrist_rgb")), eef_pos=o.get("cf_eef_pos", o["eef_pos"]),
                               eef_quat=o.get("cf_eef_quat", o["eef_quat"])) for o in cur]
                cf_imgs = [o["rgb_cf"] for o in cur]
                cf_pils = None if getattr(self.policy, "raw_images", False) else preprocess_batch(cf_imgs, self.policy.center_crop)
                for k, p in self.labelers.items():
                    v = p.act(cf_imgs, descs, pils=cf_pils, obs=cf_obs if getattr(p, "needs_obs", False) else None)
                    label_cf[k] = v["actions_norm"] if "actions_norm" in v else v["logits"]
            t3 = time.perf_counter()
            env_actions = postprocess_actions(out["actions"])
            for j, i in enumerate(act_idx):
                s, obs = slots[i], slots[i]["obs"]
                for k in STEP_KEYS:
                    if k in obs:
                        s["steps"][k].append(obs[k])
                s["t"].append(obs["t"])
                if "logits" in out:  # token policies only; a regression-head policy has no distribution to log
                    s["bins"].append(out["bins"][j].numpy().astype(np.uint8))
                    s["logits"].append(out["logits"][j].numpy().astype(np.float16))
                s["actions"].append(out["actions"][j].astype(np.float32))
                if "actions_norm" in out:  # regression-head policies: the normalised chunk is the distillation target
                    s["actions_norm"].append(out["actions_norm"][j].astype(np.float32))
                for k in label_logits:
                    s["label_logits"][k].append(label_logits[k][j].numpy().astype(np.float16))
                for k in label_actions:
                    s["label_actions"][k].append(label_actions[k][j].astype(np.float32))
                for k in label_cf:
                    s["label_cf"][k].append(np.asarray(label_cf[k][j], dtype=np.float32))
                self.vec.step(i, env_actions[j])
            finished = []
            t4 = time.perf_counter()
            for i in act_idx:
                obs = self.vec.recv(i)
                slots[i]["obs"] = obs
                if not obs["active"]:
                    finished.append(i)
            t5 = time.perf_counter()
            for i in finished:
                rec = self._finish(slots[i], out_dir, save_steps)
                results.append(rec)
                if on_episode:
                    on_episode(rec)
                start(i)
            t6 = time.perf_counter()
            for k, v in (("preprocess", t1 - t0), ("act", t2 - t1), ("label", t3 - t2), ("env", t5 - t4 + t6 - t5),
                         ("other", t4 - t3)):
                tm[k] += v
            tm["rounds"] += 1
        for f in pending_saves:
            f.result()
        return results

    def _finish(self, s: dict, out_dir: Optional[str], save_steps: bool) -> dict:
        final = s["obs"]
        logits = torch.from_numpy(np.stack(s["logits"]).astype(np.float32)) if s["logits"] else None
        rec = {
            "task_id": s["task_id"], "trial_id": s["trial_id"], "task": self.tasks[s["task_id"]][0],
            "success": bool(final["done"]), "env_steps": int(final["t"]), "n_queries": len(s["t"]),
            "seconds": round(time.time() - s["t0"], 1), "entropy": token_stats(logits)["entropy"] if logits is not None else None,
        }
        for k, v in s["label_logits"].items():
            if v and logits is not None:
                st = token_stats(logits, torch.from_numpy(np.stack(v).astype(np.float32)))
                rec[f"kl_{k}"], rec[f"agree_{k}"] = st["kl"], st["agree"]
        for k, v in s["label_actions"].items():
            if v and s["actions_norm"]:  # mean |student - teacher| over the chunk, normalised action units
                rec[f"l1_{k}"] = float(np.abs(np.stack(v) - np.stack(s["actions_norm"])).mean())
        arrays = None
        if (out_dir and save_steps) or self._keep_steps:
            arrays = {k: np.stack(v) for k, v in s["steps"].items() if v}
            arrays.update(t=np.array(s["t"]), actions=np.stack(s["actions"]), final_sim_state=final["sim_state"])
            if s["logits"]:
                arrays.update(bins=np.stack(s["bins"]), logits=np.stack(s["logits"]))
            if s["actions_norm"]:
                arrays["actions_norm"] = np.stack(s["actions_norm"])
            arrays.update({f"logits_{k}": np.stack(v) for k, v in s["label_logits"].items() if v})
            arrays.update({f"actions_{k}": np.stack(v) for k, v in s["label_actions"].items() if v})
            arrays.update({f"cf_{k}": np.stack(v) for k, v in s["label_cf"].items() if v})  # labels on the counterfactual world
        if out_dir:
            eid = f"t{s['task_id']:02d}_n{s['trial_id']:02d}"

            def save(rec=rec, arrays=arrays):  # the summary line is written only after the arrays are on disk
                if save_steps:
                    tmp = os.path.join(out_dir, "steps", f".{eid}.{os.getpid()}.npz")
                    np.savez_compressed(tmp, **arrays)
                    os.replace(tmp, os.path.join(out_dir, "steps", f"{eid}.npz"))
                with self._jsonl_lock, open(os.path.join(out_dir, "episodes.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")

            self._pending_saves.append(self._saver.submit(save))
        if self._keep_steps:
            rec = dict(rec, arrays=arrays)
        return rec
