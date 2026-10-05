"""Gate for the env runner: rendering once per chunk must not change the rollout.

Drives one env through several consecutive episodes of random action chunks twice: (a) plain upstream
stepping, with every camera rendered at every step, (b) `EnvRunner` (cameras enabled only for the last step of a
chunk), including episodes that end mid-chunk. Simulator states and frames must be identical in every episode
(a first version passed on the first episode only and returned stale or black frames from the second on).
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
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    inits = np.asarray(suite.get_task_init_states(args.task))[: args.episodes]
    rng = np.random.default_rng(0)
    chunks = np.clip(rng.normal(0, 0.4, (args.episodes, args.chunks, 8, 7)), -1, 1)
    chunks[..., -1] = np.sign(chunks[..., -1])

    # (a) upstream stepping: every camera rendered at every step, several episodes in one env
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, camera_depths=True)
    env.seed(0)
    ref = []
    for ep in range(args.episodes):
        env.reset()
        obs = env.set_init_state(inits[ep])
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        ref.append([(env.get_sim_state().copy(), obs["agentview_image"][::-1, ::-1].copy())])
        for c in chunks[ep]:
            for a in c:
                obs, _, _, _ = env.step(a.tolist())
            ref[ep].append((env.get_sim_state().copy(), obs["agentview_image"][::-1, ::-1].copy()))
    env.close()

    # (b) EnvRunner, same episodes in one runner. Odd episodes are cut 3 steps into their last chunk (max_steps),
    # so the next reset follows an episode that ended mid-chunk, as after a success.
    full = 8 * args.chunks
    cfg = dict(resolution=256, depth=True, wrist=False, num_steps_wait=10, max_steps=full)
    runner = EnvRunner(bddl, cfg, mp.get_context("spawn").Lock())
    state_diff, img_diff, n = 0.0, 0.0, 0
    for ep in range(args.episodes):
        cfg["max_steps"] = full - 5 if ep % 2 else full
        out = [runner.reset(inits[ep])] + [runner.step(c) for c in chunks[ep]]
        last = len(out) - 1 if ep % 2 else len(out)  # the truncated chunk has no upstream counterpart
        for o, r in zip(out[:last], ref[ep][:last]):
            state_diff = max(state_diff, float(np.abs(o["sim_state"] - r[0]).max()))
            img_diff = max(img_diff, float(np.abs(o["rgb"].astype(np.int16) - r[1].astype(np.int16)).mean()))
            n += 1
        assert out[-1]["active"] is False and out[-1]["rgb"].mean() > 1, "final frame of an episode must be a real render"
    print(f"{args.episodes} episodes, {n} checkpoints: max |sim state diff| {state_diff:.3e}, max mean |pixel diff| {img_diff:.4f}")
    assert state_diff == 0.0, "skipping intermediate renders changed the physics"
    assert img_diff == 0.0, "frames differ: not sampled at the upstream instant"
    print("CHECK_ENV_STEPPING_OK")


if __name__ == "__main__":
    main()
