"""Gate for the env runner: rendering once per chunk must not change the rollout.

Drives one env with a fixed sequence of random action chunks twice: (a) plain upstream stepping, with every
camera rendered at every step, (b) `EnvRunner` (cameras enabled only for the last step of a chunk).
Simulator states and frames must be identical.
    python scripts/check_env_stepping.py --suite libero_object --task 3
"""
import argparse
import multiprocessing as mp
import os

import numpy as np

from src.rollout.vec_env import DUMMY_ACTION, EnvRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=3)
    ap.add_argument("--chunks", type=int, default=12)
    args = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init = np.asarray(suite.get_task_init_states(args.task))[0]
    rng = np.random.default_rng(0)
    chunks = np.clip(rng.normal(0, 0.4, (args.chunks, 8, 7)), -1, 1)
    chunks[..., -1] = np.sign(chunks[..., -1])

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, camera_depths=True)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(init)
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    ref = [(env.get_sim_state().copy(), obs["agentview_image"][::-1, ::-1].copy())]
    for c in chunks:
        for a in c:
            obs, _, _, _ = env.step(a.tolist())
        ref.append((env.get_sim_state().copy(), obs["agentview_image"][::-1, ::-1].copy()))
    env.close()

    cfg = dict(resolution=256, depth=True, wrist=False, num_steps_wait=10, max_steps=10**6)
    runner = EnvRunner(bddl, cfg, mp.get_context("spawn").Lock())
    out = [runner.reset(init)] + [runner.step(c) for c in chunks]
    state_diff = max(float(np.abs(o["sim_state"] - r[0]).max()) for o, r in zip(out, ref))
    img_diff = [float(np.abs(o["rgb"].astype(np.int16) - r[1].astype(np.int16)).mean()) for o, r in zip(out, ref)]
    print(f"max |sim state diff| over {len(ref)} checkpoints: {state_diff:.3e}")
    print(f"mean |pixel diff| per checkpoint (0..255): max {max(img_diff):.4f}; all: {[round(d, 3) for d in img_diff]}")
    big = [float((np.abs(o["rgb"].astype(np.int16) - r[1].astype(np.int16)).max(-1) > 8).mean()) for o, r in zip(out, ref)]
    print(f"fraction of pixels differing by > 8 levels: {[round(b, 4) for b in big]}")
    print(f"depth range at the end: {out[-1]['depth'].min():.3f}..{out[-1]['depth'].max():.3f} m")
    assert state_diff == 0.0, "skipping intermediate renders changed the physics"
    assert max(img_diff) == 0.0, "frames differ: not sampled at the upstream instant"
    print("CHECK_ENV_STEPPING_OK")


if __name__ == "__main__":
    main()
