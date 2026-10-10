"""Does an env carry state from one episode into the next? Compares, for one initial state, an EnvRunner that just ran
another episode (reset, then set_init_state, as the collector and the official LIBERO loop do) with a freshly built one:
the first observation (simulator state, proprio, images) and the internal state the simulator state does not hold
(arm controller goal, gripper command, actuator activations), then both trajectories under the same action sequence.
    python scripts/check_env_reuse.py --suite libero_goal --task 0 --prev 0 --trial 1
"""
import argparse
import contextlib

import numpy as np


def internals(r):
    robo = r.env.env
    robot = robo.robots[0]
    c = robot.controller
    d = robo.sim.data._data
    out = {
        "gripper.current_action": np.array(robot.gripper.current_action, dtype=float),
        "act": np.array(d.act, dtype=float),
        "qacc_warmstart": np.array(d.qacc_warmstart, dtype=float),
        "ctrl": np.array(d.ctrl, dtype=float),
        "timestep": np.array([getattr(robo, "timestep", 0), getattr(robo, "cur_time", 0.0)], dtype=float),
    }
    for k in ("goal_pos", "goal_ori", "goal_qpos", "ee_pos", "initial_ee_pos", "initial_ee_ori_mat", "ori_ref",
              "relative_ori", "torques"):
        if hasattr(c, k) and getattr(c, k) is not None:
            out["controller." + k] = np.array(getattr(c, k), dtype=float)
    if hasattr(c, "interpolator_pos") and c.interpolator_pos is not None:
        out["controller.interp_pos_goal"] = np.array(getattr(c.interpolator_pos, "goal", np.zeros(1)), dtype=float)
    m = robo.sim.model._model
    import mujoco

    for arr in ("body_pos", "body_quat", "geom_pos", "geom_quat", "geom_size", "geom_rgba", "site_pos", "cam_pos", "cam_quat",
                "cam_fovy", "light_pos", "light_dir", "light_diffuse", "mat_rgba", "tex_data" if hasattr(m, "tex_data") else "tex_rgb"):
        if hasattr(m, arr):
            out["model." + arr] = np.array(getattr(m, arr), dtype=float)
    out["_body_names"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    for name, ob in robo._observables.items():
        if name.endswith("_image"):
            out["obs." + name + ".timer"] = np.array([ob._time_since_last_sample, float(ob._sampled)])
    return out


def diff(a, b, label):
    keys = sorted(set(a) | set(b))
    names = a.get("_body_names")
    for k in keys:
        if k.startswith("_"):
            continue
        if k == "model.body_pos" and k in a and k in b and np.asarray(a[k]).shape == np.asarray(b[k]).shape:
            dd = np.abs(np.asarray(a[k]) - np.asarray(b[k])).max(1)
            for i in np.flatnonzero(dd > 1e-9):
                print(f"  {label} body_pos[{names[i] if names else i}]: reused {np.round(a[k][i], 4)} fresh {np.round(b[k][i], 4)}")
        if k not in a or k not in b:
            print(f"  {label} {k}: only in {'reused' if k in a else 'fresh'}")
            continue
        x, y = np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float)
        if x.shape != y.shape:
            print(f"  {label} {k}: shape {x.shape} vs {y.shape}")
        elif x.size and np.abs(x - y).max() > 1e-9:
            print(f"  {label} {k}: max |diff| {np.abs(x - y).max():.3g}  reused {np.round(x.ravel()[:7], 4)}  fresh {np.round(y.ravel()[:7], 4)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--prev", type=int, default=0, help="initial state of the episode run first in the reused env")
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--prev_steps", type=int, default=150)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--png", default="", help="save the first frames of both envs (and of the previous episode's end) here")
    args = ap.parse_args()
    from src.rollout.vec_env import EnvRunner, LiberoVecEnv

    vec = LiberoVecEnv(args.suite, 0, 300, wrist=True)
    bddl, inits = vec._bddl(args.task), vec.init_states(args.task)
    lock = contextlib.nullcontext()
    rng = np.random.default_rng(0)
    # a previous episode that moves the arm around and works the gripper (ends closed)
    prev_actions = [np.r_[rng.uniform(-0.6, 0.6, 3), rng.uniform(-0.1, 0.1, 3), 1.0 if (i // 30) % 2 else -1.0]
                    for i in range(args.prev_steps)]
    test_actions = [np.r_[0.3 * np.sin(i / 10), 0.3 * np.cos(i / 13), -0.2, 0, 0, 0, -1.0 if i < 60 else 1.0]
                    for i in range(args.steps)]

    reused = EnvRunner(bddl, vec.cfg, lock)
    reused.reset(inits[args.prev])
    for a in prev_actions:
        reused.step(np.array([a]))
    print("prev episode ran; internals before the next reset:")
    for k, v in internals(reused).items():
        if k.startswith(("gripper", "controller.goal_pos")):
            print("  ", k, np.round(v.ravel()[:7], 4))
    o_prev = reused.step(np.array([prev_actions[-1]]))
    o_r = reused.reset(inits[args.trial])
    i_r = internals(reused)
    fresh = EnvRunner(bddl, vec.cfg, lock)
    o_f = fresh.reset(inits[args.trial])
    i_f = internals(fresh)

    print("first observation (reused vs fresh):")
    diff({k: v for k, v in o_r.items() if isinstance(v, np.ndarray) and v.dtype != object},
         {k: v for k, v in o_f.items() if isinstance(v, np.ndarray) and v.dtype != object}, "obs")
    if args.png:
        from PIL import Image

        rows = [np.concatenate([o[k] for k in ("rgb", "wrist_rgb")], axis=1) for o in (o_prev, o_r, o_f)]
        Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8)).save(args.png)
        print("saved rows: previous episode end / reused first obs / fresh first obs ->", args.png)
        for k in ("rgb", "wrist_rgb"):
            print(f"  {k}: |reused - prev end| {np.abs(o_r[k].astype(float) - o_prev[k]).mean():.2f}  "
                  f"|reused - fresh| {np.abs(o_r[k].astype(float) - o_f[k]).mean():.2f}")
    print("internal state after reset:")
    diff(i_r, i_f, "int")
    print("trajectories under the same actions (max |eef diff| so far, gripper qpos diff):")
    worst = 0.0
    for t, a in enumerate(test_actions):
        s_r, s_f = reused.step(np.array([a])), fresh.step(np.array([a]))
        e = float(np.abs(np.asarray(s_r["eef_pos"]) - np.asarray(s_f["eef_pos"])).max())
        g = float(np.abs(np.asarray(s_r["gripper_qpos"]) - np.asarray(s_f["gripper_qpos"])).max())
        worst = max(worst, e)
        im = float(np.abs(s_r["rgb"].astype(float) - s_f["rgb"]).mean())
        wr = float(np.abs(s_r["wrist_rgb"].astype(float) - s_f["wrist_rgb"]).mean())
        if t in (0, 1, 2, 5, 10, 20, 40, 60, 61, 62, 65, 70, 90, args.steps - 1):
            print(f"  step {t:3d}: eef diff {e:.2e}  grip diff {g:.2e}  (worst {worst:.2e})  mean |rgb diff| {im:.2f}  wrist {wr:.2f}")
    print("internal state at the end:")
    diff(internals(reused), internals(fresh), "end")


if __name__ == "__main__":
    main()
