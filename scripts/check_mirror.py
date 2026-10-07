"""Gate for the mirror counterfactual (`vec_env.COUNTERFACTUAL`, mode "mirror").

1. Rendering it must not disturb the nominal episode (states and frames identical to a plain run).
2. The reflected robot must be the reflection of the robot: for each candidate reflection centre c7 of joint 7, the
   end-effector pose rendered in the mirror world is compared with the analytic reflection of the nominal pose.
3. The world must be dynamically symmetric enough for the reflected action to be the right label: the episode is
   replayed from the reflected initial state with reflected actions, and its end-effector track compared with the
   reflection of the nominal track. Writes a contact sheet (nominal | mirror, agent and wrist views).
    python scripts/check_mirror.py --suite libero_object --task 3 --out outputs/verify/mirror_check
With --mode rotate the same checks are run for the rotation counterfactual (mode "rotate", angle --theta): the world
is rotated about the first joint's vertical axis, and the label is the nominal action rotated by the same angle.
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
    ap.add_argument("--mode", choices=["mirror", "rotate"], default="mirror")
    ap.add_argument("--theta", type=float, default=0.3, help="rotate mode: angle (radians)")
    ap.add_argument("--steps", default="", help="rotate mode: eval run dir with steps/*.npz and episodes.jsonl to replay")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    if args.mode == "rotate":
        return check_rotate(args)
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


def quat_mul_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def check_rotate(args):
    """Gate for the rotation counterfactual: [1] nominal untouched, [2] rendered robot = rotated robot, [3] replay of
    rotated random chunks from the rotated initial state (objects, furniture, tables and arm rotated in the physics)
    follows the rotated nominal track, [4] with --steps: logged successful episodes replayed in the rotated world with
    rotated actions still succeed (the label is right for the whole task, contacts included). Contact sheet."""
    import glob

    import mujoco
    from libero.libero import benchmark, get_libero_path
    from PIL import Image

    from src.rollout.vec_env import postprocess_actions, rotate_world, world_fixed_bodies

    os.makedirs(args.out, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    th = args.theta

    def make(task):
        t = suite.get_task(task)
        bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        cfg = dict(suite=args.suite, max_steps=600, num_steps_wait=10, resolution=256, depth=False, wrist=True, perturb=None,
                   counterfactual=dict(COUNTERFACTUAL, mode="rotate", theta=(th, th)))
        return EnvRunner(bddl, cfg, contextlib.nullcontext()), np.asarray(suite.get_task_init_states(task))

    r, inits = make(args.task)
    init = inits[0]
    rng = np.random.default_rng(0)
    chunks = [np.concatenate([rng.uniform(-0.4, 0.4, (8, 3)), rng.uniform(-0.05, 0.05, (8, 3)), -np.ones((8, 1))], axis=1)
              for _ in range(args.chunks)]
    ref = [r.reset(init)] + [r.step(c) for c in chunks]
    cf = [r.reset(init, counterfactual=True)] + [r.step(c) for c in chunks]
    d_state = max(float(np.abs(a["sim_state"] - b["sim_state"]).max()) for a, b in zip(ref, cf))
    d_rgb = max(int(np.abs(a["rgb"].astype(int) - b["rgb"].astype(int)).max()) for a, b in zip(ref, cf))
    d_wrist = max(int(np.abs(a["wrist_rgb"].astype(int) - b["wrist_rgb"].astype(int)).max()) for a, b in zip(ref, cf))
    print(f"[1] nominal untouched: |state| {d_state:.1e}, rgb {d_rgb}, wrist {d_wrist}")
    robo = r.env.env
    ctr = r._rot_center
    m, dd = robo.sim.model._model, robo.sim.data._data
    j1 = robo.sim.model.joint_name2id(robo.robots[0].robot_joints[0])
    print(f"    rotation centre {np.round(ctr, 4).tolist()}, first joint axis {np.round(dd.xaxis[j1], 4).tolist()}, "
          f"fixed bodies rotated: {[robo.sim.model.body_id2name(b) for b in r._rot_fixed]}")

    def rot_pos(p, t):
        c, s = np.cos(t), np.sin(t)
        x, y = p[0] - ctr[0], p[1] - ctr[1]
        return np.array([ctr[0] + c * x - s * y, ctr[1] + s * x + c * y, p[2]])

    qz = lambda t: np.array([0.0, 0.0, np.sin(t / 2), np.cos(t / 2)])  # noqa: E731  (xyzw)
    ths = [float(o["cf_theta"]) for o in cf]  # the sign is drawn per query
    dp = max(float(np.linalg.norm(o["cf_eef_pos"] - rot_pos(o["eef_pos"], t))) for o, t in zip(cf, ths))
    dq = max(quat_dist(o["cf_eef_quat"], quat_mul_xyzw(qz(t), o["eef_quat"])) for o, t in zip(cf, ths))
    print(f"[2] max |eef pos - rotated| {dp * 1000:.2f} mm, max orientation error {dq:.2f} deg (angles {np.round(ths, 3).tolist()})")
    rows = [np.concatenate([o["rgb"], o["rgb_cf"], o["wrist_rgb"], o["wrist_rgb_cf"]], axis=1) for o in cf[:4]]
    Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(args.out, "rotate_sheet.jpg"), quality=85)

    c, s = np.cos(th), np.sin(th)

    def rot_act(ch):
        out = ch.copy()
        for i, j in ((0, 1), (3, 4)):
            out[..., i], out[..., j] = c * ch[..., i] - s * ch[..., j], s * ch[..., i] + c * ch[..., j]
        return out

    def rotated_restore(r, st, t0):
        """Restore `st` in a world rotated by th (furniture moved in the model, kept for the rest of the episode)."""
        robo = r.env.env
        r.restore(st, t0, -1.0)  # rebuilds the simulator; the model is fresh
        m, d = robo.sim.model._model, robo.sim.data._data
        nq = m.nq
        d.qpos[:], d.qvel[:] = st[1 : 1 + nq], st[1 + nq :]
        addrs = []
        for n in robo.objects_dict:
            try:
                a = robo.sim.model.get_joint_qpos_addr(f"{n}_joint0")
            except Exception:
                continue
            addrs.append(int(a[0] if isinstance(a, (tuple, list, np.ndarray)) else a))
        arm_q1 = int(robo.robots[0]._ref_joint_pos_indexes[0])
        rotate_world(m, d, ctr, th, addrs, arm_q1, world_fixed_bodies(robo), qvel_too=True)
        st2 = np.concatenate([st[:1], d.qpos.copy(), d.qvel.copy()])
        robot = robo.robots[0]
        robo.sim.set_state_from_flattened(st2)
        robo.sim.forward()
        robot.controller.update(force=True)
        robot.controller.reset_goal()

    # [3] random chunks from the rotated initial state
    o0 = r.reset(init)
    st = np.array(r.env.get_sim_state(), dtype=np.float64)
    nominal = [o0] + [r.step(ch) for ch in chunks]
    rotated_restore(r, st, nominal[0]["t"])
    rotated = [r.step(rot_act(ch)) for ch in chunks]
    errs = [float(np.linalg.norm(b["eef_pos"] - rot_pos(a["eef_pos"], th))) for a, b in zip(nominal[1:], rotated)]
    qerrs = [quat_dist(b["eef_quat"], quat_mul_xyzw(qz(th), a["eef_quat"])) for a, b in zip(nominal[1:], rotated)]
    print(f"[3] rotated replay: eef track error per chunk (mm) {np.round(np.array(errs) * 1000, 2).tolist()}")
    print(f"    orientation error per chunk (deg) {np.round(np.array(qerrs), 2).tolist()}")
    r.close()
    ok = d_state < 1e-9 and d_rgb == 0 and d_wrist == 0 and dp < 0.005 and dq < 2.0 and max(errs) < 0.02 and max(qerrs) < 3.0

    # [4] whole logged successful episodes, nominal vs rotated replay
    if args.steps:
        import json

        succ = {}
        for line in open(os.path.join(args.steps, "episodes.jsonl")):
            e = json.loads(line)
            succ[(int(e["task_id"]), int(e["trial_id"]))] = bool(e["success"])
        files = sorted(glob.glob(os.path.join(args.steps, "steps", "t*_n*.npz")),
                       key=lambda f: (os.path.basename(f)[4:6], os.path.basename(f)[1:3]))
        files = [f for f in files if succ.get((int(os.path.basename(f)[1:3]), int(os.path.basename(f)[5:7])))][: args.episodes]
        res = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            task = int(os.path.basename(f)[1:3])
            r, _ = make(task)
            acts = postprocess_actions(z["actions"])
            st0 = np.array(z["sim_state"][0], dtype=np.float64)
            r.restore(st0, int(z["t"][0]), -1.0)
            nom = any(r.step(a)["done"] for a in acts)
            rotated_restore(r, st0, int(z["t"][0]))
            rot = any(r.step(rot_act(a))["done"] for a in acts)
            res.append((nom, rot))
            r.close()
        res = np.array(res)
        print(f"[4] logged successes replayed open-loop: nominal {res[:, 0].sum()}/{len(res)}, rotated {res[:, 1].sum()}/{len(res)}")
        ok = ok and res[:, 1].sum() >= 0.8 * res[:, 0].sum()
    print("CHECK_ROTATE_OK" if ok else "CHECK_ROTATE_FAILED")


if __name__ == "__main__":
    main()
