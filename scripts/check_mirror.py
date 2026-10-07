"""Gate for the mirror counterfactual (`vec_env.COUNTERFACTUAL`, mode "mirror").

1. Rendering it must not disturb the nominal episode (states and frames identical to a plain run).
2. The reflected robot must be the reflection of the robot: for each candidate reflection centre c7 of joint 7, the
   end-effector pose rendered in the mirror world is compared with the analytic reflection of the nominal pose.
3. The world must be dynamically symmetric enough for the reflected action to be the right label: the episode is
   replayed from the reflected initial state with reflected actions, and its end-effector track compared with the
   reflection of the nominal track. Writes a contact sheet (nominal | mirror, agent and wrist views).
    python scripts/check_mirror.py --suite libero_object --task 3 --out outputs/verify/mirror_check
"""
import argparse
import contextlib
import os

import numpy as np

from src.rollout.vec_env import COUNTERFACTUAL, EnvRunner, mirror_quat


def quat_mirror_xyzw(q):
    x, y, z, w = q
    return np.array([-x, y, -z, w])


def quat_dist(a, b):  # rotation angle between two unit quaternions (xyzw), sign-invariant
    return 2 * np.degrees(np.arccos(np.clip(abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))), -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=3)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from libero.libero import benchmark, get_libero_path
    from PIL import Image

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init = np.asarray(suite.get_task_init_states(args.task))[0]
    rng = np.random.default_rng(0)
    chunks = [np.concatenate([rng.uniform(-0.4, 0.4, (8, 3)), rng.uniform(-0.05, 0.05, (8, 3)), -np.ones((8, 1))], axis=1)
              for _ in range(args.chunks)]

    def make(c7):
        cfg = dict(suite=args.suite, max_steps=512, num_steps_wait=10, resolution=256, depth=False, wrist=True, perturb=None,
                   counterfactual=dict(COUNTERFACTUAL, mode="mirror", c7=c7))
        return EnvRunner(bddl, cfg, contextlib.nullcontext())

    runner = make(np.pi / 4)
    ref = [runner.reset(init)] + [runner.step(c) for c in chunks]
    cf = [runner.reset(init, counterfactual=True)] + [runner.step(c) for c in chunks]
    d_state = max(float(np.abs(a["sim_state"] - b["sim_state"]).max()) for a, b in zip(ref, cf))
    d_rgb = max(int(np.abs(a["rgb"].astype(int) - b["rgb"].astype(int)).max()) for a, b in zip(ref, cf))
    d_wrist = max(int(np.abs(a["wrist_rgb"].astype(int) - b["wrist_rgb"].astype(int)).max()) for a, b in zip(ref, cf))
    print(f"[1] nominal untouched: |state| {d_state:.1e}, rgb {d_rgb}, wrist {d_wrist}")
    y0 = runner._mirror_y0
    print(f"    mirror plane y0 = {y0:.4f}")

    # [2] kinematic reflection for candidate c7
    best = None
    for c7 in (0.0, np.pi / 4, np.pi / 2, -np.pi / 4, np.pi):
        r = make(c7)
        obs = [r.reset(init, counterfactual=True)] + [r.step(c) for c in chunks]
        dp = max(float(np.linalg.norm(o["cf_eef_pos"] - o["eef_pos"] * np.array([1, -1, 1]) - np.array([0, 2 * y0, 0]))) for o in obs)
        dq = max(quat_dist(o["cf_eef_quat"], quat_mirror_xyzw(o["eef_quat"])) for o in obs)
        print(f"[2] c7 = {c7:+.3f}: max |eef pos - reflected| {dp * 1000:.2f} mm, max orientation error {dq:.2f} deg")
        if best is None or dq < best[1]:
            best = (c7, dq, dp, obs)
        r.close()
    c7, dq, dp, obs = best
    print(f"    best c7 = {c7:+.4f} (pos {dp * 1000:.2f} mm, orientation {dq:.2f} deg)")

    # [3] dynamic symmetry: replay from the reflected initial state with reflected actions
    r = make(c7)
    o0 = r.reset(init, counterfactual=True)
    robo = r.env.env
    m = robo.sim.model._model
    nq = m.nq
    st = np.array(r.env.get_sim_state(), dtype=np.float64)  # [time, qpos, qvel]
    qpos, qvel = st[1 : 1 + nq].copy(), st[1 + nq :].copy()
    import mujoco

    worst_up = 0.0
    for a in r._mirror_obj_addrs:
        qpos[a + 1] = 2 * y0 - qpos[a + 1]
        q0 = qpos[a + 3 : a + 7].copy()
        qpos[a + 3 : a + 7] = mirror_quat(q0)
        R0, R1 = np.zeros(9), np.zeros(9)
        mujoco.mju_quat2Mat(R0, q0)
        mujoco.mju_quat2Mat(R1, qpos[a + 3 : a + 7])
        R0, R1 = R0.reshape(3, 3), R1.reshape(3, 3)
        k = int(np.argmax(np.abs(R0[2])))  # body axis that is vertical in the world
        want = np.diag([1, -1, 1]) @ R0[:, k]  # its reflected world direction
        worst_up = max(worst_up, np.degrees(np.arccos(np.clip(abs(float(want @ R1[:, k])), -1, 1))))
        va = int(m.jnt_dofadr[[j for j in range(m.njnt) if m.jnt_qposadr[j] == a][0]])
        qvel[va + 1] *= -1  # vy
        qvel[va + 3] *= -1  # wx
        qvel[va + 5] *= -1  # wz
    arm = r._mirror_arm_addrs
    arm_dof = [int(m.jnt_dofadr[[j for j in range(m.njnt) if m.jnt_qposadr[j] == a][0]]) for a in arm]
    for i in (0, 2, 4):
        qpos[arm[i]] *= -1
        qvel[arm_dof[i]] *= -1
    qpos[arm[6]] = 2 * c7 - qpos[arm[6]]
    qvel[arm_dof[6]] *= -1
    mst = np.concatenate([st[:1], qpos, qvel])
    nominal = [o0] + [r.step(c) for c in chunks]
    r.restore(mst, nominal[0]["t"], -1.0)
    flip = np.array([1, -1, 1, -1, 1, -1, 1])
    mirrored = [r.step(c * flip) for c in chunks]
    errs = [float(np.linalg.norm(b["eef_pos"] - (a["eef_pos"] * np.array([1, -1, 1]) + np.array([0, 2 * y0, 0]))))
            for a, b in zip(nominal[1:], mirrored)]
    print(f"[3] reflected replay: eef track error per chunk (mm) {np.round(np.array(errs) * 1000, 2).tolist()}")
    print(f"[4] objects: worst angle between the reflected vertical axis and the mirror object's one {worst_up:.2f} deg")
    r.close()

    rows = [np.concatenate([o["rgb"], o["rgb_cf"], o["wrist_rgb"], o["wrist_rgb_cf"]], axis=1) for o in obs[:4]]
    Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(args.out, "mirror_sheet.jpg"), quality=85)
    ok = d_state < 1e-9 and d_rgb == 0 and d_wrist == 0 and dp < 0.005 and dq < 2.0 and max(errs) < 0.02 and worst_up < 1.0
    print("CHECK_MIRROR_OK" if ok else "CHECK_MIRROR_FAILED")


if __name__ == "__main__":
    main()
