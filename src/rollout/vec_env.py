"""Subprocess-vectorised LIBERO environments.

Each worker owns one OffScreenRenderEnv and follows the SimpleVLA-RL / OpenVLA-OFT episode protocol:
fixed benchmark initial state, `num_steps_wait` no-op steps, then open-loop execution of action chunks.
Workers never import TF or CUDA; rendering goes through EGL on the GPU named by MUJOCO_EGL_DEVICE_ID.
"""
import multiprocessing as mp
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def _pack_obs(env, obs, cfg) -> Dict[str, Any]:
    out = {
        # rotate 180 degrees to match the training-data preprocessing
        "rgb": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "eef_pos": obs["robot0_eef_pos"].astype(np.float32),
        "eef_quat": obs["robot0_eef_quat"].astype(np.float32),
        "gripper_qpos": obs["robot0_gripper_qpos"].astype(np.float32),
        "sim_state": env.get_sim_state().astype(np.float64),
    }
    if cfg.get("wrist"):
        out["wrist_rgb"] = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    if cfg.get("depth"):
        from robosuite.utils.camera_utils import get_real_depth_map

        d = get_real_depth_map(env.sim, obs["agentview_depth"])[..., 0]  # metres
        out["depth"] = np.ascontiguousarray(d[::-1, ::-1]).astype(np.float32)
    return out


def _worker(conn, cfg: Dict[str, Any]) -> None:
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[cfg["suite"]]()
        env, cur_task, init_states, t, done = None, None, None, 0, False
        while True:
            cmd, arg = conn.recv()
            if cmd == "reset":
                task_id, trial_id = arg
                if task_id != cur_task:
                    if env is not None:
                        env.close()
                    task = suite.get_task(task_id)
                    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
                    env = OffScreenRenderEnv(
                        bddl_file_name=bddl,
                        camera_heights=cfg["resolution"],
                        camera_widths=cfg["resolution"],
                        camera_depths=bool(cfg.get("depth")),
                    )
                    env.seed(0)  # upstream: the seed affects object positions even with a fixed initial state
                    init_states = suite.get_task_init_states(task_id)
                    cur_task = task_id
                env.reset()
                obs = env.set_init_state(init_states[trial_id])
                for _ in range(cfg["num_steps_wait"]):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                t, done = 0, False
                conn.send(("ok", dict(_pack_obs(env, obs, cfg), t=0, done=False, active=True)))
            elif cmd == "step":
                for a in arg:  # (k, 7) actions, already gripper-post-processed
                    obs, _, done, _ = env.step(a.tolist())
                    t += 1
                    if done or t >= cfg["max_steps"]:
                        break
                active = not (done or t >= cfg["max_steps"])
                conn.send(("ok", dict(_pack_obs(env, obs, cfg), t=t, done=bool(done), active=active)))
            elif cmd == "task_info":
                conn.send(("ok", [(i, suite.get_task(i).language, len(suite.get_task_init_states(i))) for i in range(suite.n_tasks)]))
            elif cmd == "close":
                if env is not None:
                    env.close()
                conn.send(("ok", None))
                return
    except Exception:  # surface the traceback in the parent instead of dying silently
        conn.send(("error", traceback.format_exc()))


class LiberoVecEnv:
    def __init__(self, suite: str, num_envs: int, max_steps: int, num_steps_wait: int = 10, resolution: int = 256,
                 depth: bool = False, wrist: bool = False):
        self.cfg = dict(suite=suite, max_steps=max_steps, num_steps_wait=num_steps_wait, resolution=resolution,
                        depth=depth, wrist=wrist)
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(num_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, self.cfg), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.num_envs = num_envs

    def send(self, i: int, cmd: str, arg: Any = None) -> None:
        self.conns[i].send((cmd, arg))

    def recv(self, i: int, timeout: float = 600.0) -> Any:
        if not self.conns[i].poll(timeout):
            raise TimeoutError(f"env worker {i} did not answer within {timeout}s")
        status, payload = self.conns[i].recv()
        if status != "ok":
            raise RuntimeError(f"env worker {i} failed:\n{payload}")
        return payload

    def task_info(self) -> List[Tuple[int, str, int]]:
        self.send(0, "task_info")
        return self.recv(0)

    def close(self) -> None:
        for i, p in enumerate(self.procs):
            try:
                self.send(i, "close")
                self.recv(i, timeout=20)
            except Exception:
                pass
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def postprocess_actions(actions: np.ndarray) -> np.ndarray:
    """Policy gripper output in [0, 1] (1 = open) -> env convention {-1 = open, +1 = close}, binarised."""
    a = actions.copy()
    g = 2 * a[..., -1] - 1
    a[..., -1] = -np.sign(g)
    return a
