"""Vectorised LIBERO environments, one env per worker process.

Every env follows the SimpleVLA-RL / OpenVLA-OFT episode protocol: fixed benchmark initial state,
`num_steps_wait` no-op steps, then open-loop execution of action chunks. Workers never import TF or CUDA.
Rendering goes through MUJOCO_GL (EGL on the GPU named by MUJOCO_EGL_DEVICE_ID, or OSMesa on the CPU) and
happens once per action chunk, at exactly the simulator instant at which upstream stepping renders.

One env per process is required: robosuite renders garbage when a process holds two EGL contexts.
"""
import contextlib
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


class EnvRunner:
    """One LIBERO env that renders only the frames the policy consumes.

    robosuite renders every camera at every control step, but the policy only looks at the observation
    returned by the last step of each action chunk. Camera observables are therefore disabled for all other
    steps (physics is unaffected) and enabled just for that step, with their sampling timer restored, so
    the frame is taken at the same physics substep as in upstream stepping (the 24th of 25, a consequence of
    the two forced observation updates in `reset` + `set_init_state`).
    The steps that render run under `lock`, which serialises GPU rendering across worker processes
    (`lock` is a no-op context for the CPU renderer, where workers render in parallel).
    """

    def __init__(self, bddl: str, cfg: Dict[str, Any], lock):
        from libero.libero.envs import OffScreenRenderEnv

        self.cfg, self.lock = cfg, lock
        with lock:
            self.env = OffScreenRenderEnv(
                bddl_file_name=bddl,
                camera_heights=cfg["resolution"],
                camera_widths=cfg["resolution"],
                camera_depths=bool(cfg.get("depth")),
                camera_names=["agentview"] + (["robot0_eye_in_hand"] if cfg.get("wrist") else []),
            )
            self.env.seed(0)  # upstream: the seed affects object positions even with a fixed initial state
        self.t, self.done = 0, False

    def _cameras(self, enabled: bool) -> None:
        robo = self.env.env
        for name, ob in robo._observables.items():
            if name.endswith(("_image", "_depth")):
                robo.modify_observable(name, "enabled", enabled)
                if enabled:  # enabling zeroes the sampling timer; put it back in the phase upstream stepping has
                    ob._time_since_last_sample, ob._sampled = self._phase[name]

    def _run(self, actions, stop_at_end: bool = True) -> Dict[str, Any]:
        """Execute actions until the episode ends; only the last one renders."""
        obs, rendered = None, False
        for k, a in enumerate(actions):
            if k == len(actions) - 1:
                with self.lock:
                    self._cameras(True)
                    obs, _, self.done, _ = self.env.step(a)
                    self._cameras(False)
                rendered = True
            else:
                obs, _, self.done, _ = self.env.step(a)
            self.t += 1
            if stop_at_end and (self.done or self.t >= self.cfg["max_steps"]):
                break
        if not rendered:  # the episode ended mid-chunk: render its final state (only used for logging)
            with self.lock:
                self._cameras(True)
                obs = self.env.env._get_observations(force_update=True)
                self._cameras(False)
        return dict(_pack_obs(self.env, obs, self.cfg), t=self.t, done=bool(self.done),
                    active=not (self.done or self.t >= self.cfg["max_steps"]))

    def reset(self, init_state) -> Dict[str, Any]:
        robo = self.env.env
        cams = [name for name in robo._observables if name.endswith(("_image", "_depth"))]
        with self.lock:  # a (hard) reset rebuilds the simulator and its render context
            # cameras must be enabled across reset + set_init_state, exactly as upstream: these two calls reset the
            # sampling timers and force two observation updates, which is what puts the timers in their phase
            for name in cams:
                robo.modify_observable(name, "enabled", True)
            self.env.reset()
            self.env.set_init_state(init_state)
            # sampling-timer state of each camera at the start of a control step (identical at every step)
            self._phase = {name: (robo._observables[name]._time_since_last_sample, robo._observables[name]._sampled)
                           for name in cams}
            self._cameras(False)
        self.t = -self.cfg["num_steps_wait"]
        out = self._run([DUMMY_ACTION] * self.cfg["num_steps_wait"], stop_at_end=False)  # upstream ignores `done` here
        self.t, self.done = 0, False
        return dict(out, t=0, done=False, active=True)

    def step(self, actions) -> Dict[str, Any]:
        return self._run([a.tolist() for a in actions])  # (k, 7) actions, already gripper-post-processed

    def close(self) -> None:
        with self.lock:
            self.env.close()


def _worker(conn, cfg: Dict[str, Any], lock) -> None:
    import faulthandler

    faulthandler.enable()  # a crash inside MuJoCo / EGL otherwise kills the worker without any message
    try:
        runner, cur_task = None, None
        lock = lock if lock is not None else contextlib.nullcontext()
        while True:
            cmd, arg = conn.recv()
            if cmd == "reset":
                task_id, bddl, init_state = arg
                if task_id != cur_task:
                    if runner is not None:
                        runner.close()
                    runner, cur_task = EnvRunner(bddl, cfg, lock), task_id  # imports only the env stack, never torch
                conn.send(("ok", runner.reset(init_state)))
            elif cmd == "step":
                conn.send(("ok", runner.step(arg)))
            elif cmd == "close":
                conn.send(("ok", None))
                conn.close()
                os._exit(0)  # skip the slow env / EGL teardown
    except Exception:  # surface the traceback in the parent instead of dying silently
        conn.send(("error", traceback.format_exc()))


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return float("inf")


class LiberoVecEnv:
    def __init__(self, suite: str, num_envs: int, max_steps: int, num_steps_wait: int = 10, resolution: int = 256,
                 depth: bool = False, wrist: bool = False):
        from libero.libero import benchmark, get_libero_path

        self.cfg = dict(suite=suite, max_steps=max_steps, num_steps_wait=num_steps_wait, resolution=resolution,
                        depth=depth, wrist=wrist)
        self.suite = benchmark.get_benchmark_dict()[suite]()
        self._bddl, self._init = {}, {}
        for i in range(self.suite.n_tasks):
            task = self.suite.get_task(i)
            self._bddl[i] = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            self._init[i] = np.asarray(self.suite.get_task_init_states(i))
        self.num_envs = num_envs
        ctx = mp.get_context("spawn")
        # GPU rendering is serialised across workers: concurrent EGL rendering from several processes gets them
        # killed by the driver on the H200 machine (NVRM Xid 31). The CPU renderer needs no lock.
        lock = ctx.Lock() if os.environ.get("MUJOCO_GL", "egl") == "egl" else None
        self.conns, self.procs = [], []
        for _ in range(num_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, self.cfg, lock), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    def reset(self, i: int, task_id: int, trial_id: int) -> None:
        """Ask env i to start an episode from benchmark initial state `trial_id` of task `task_id`."""
        self.conns[i].send(("reset", (task_id, self._bddl[task_id], self._init[task_id][trial_id])))

    def step(self, i: int, actions: np.ndarray) -> None:
        self.conns[i].send(("step", actions))

    def ready(self, i: int) -> bool:
        """True if env i has answered its pending request (so `recv` will not block)."""
        return self.conns[i].poll(0)

    def recv(self, i: int, timeout: float = 900.0) -> Any:
        if not self.conns[i].poll(timeout):
            raise TimeoutError(f"env {i} did not answer within {timeout}s")
        status, payload = self.conns[i].recv()
        if status != "ok":
            raise RuntimeError(f"env worker {i} failed:\n{payload}")
        return payload

    def task_info(self) -> List[Tuple[int, str, int]]:
        return [(i, self.suite.get_task(i).language, len(self._init[i])) for i in range(self.suite.n_tasks)]

    def close(self) -> None:
        for conn, p in zip(self.conns, self.procs):
            try:
                conn.send(("close", None))
                if conn.poll(20):
                    conn.recv()
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
