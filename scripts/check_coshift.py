"""Gate for the coshift counterfactual (`vec_env.COUNTERFACTUAL`, mode "coshift": target and hand moved together).

1. Rendering it must not disturb the nominal episode (states and frames identical to a plain run).
2. The hand of the counterfactual world must be the nominal hand displaced by the target's shift (IK accuracy), and
   most pre-grasp queries must find a valid placement.
3. The label must be right: successful logged pi0.5 episodes are replayed from a co-shifted initial state (target and
   hand moved, nothing else) with the logged actions; the hand must follow the nominal track displaced by the shift
   until the first grasp, and the target must be lifted. The arm is in another configuration, so the operational-space
   controller tracks a chunk slightly differently: the gate asks for per-chunk hand errors below 1 cm (90%) - small
   against the 8-30 cm the counterfactual teaches - and for most open-loop replays to still lift the target. Writes a contact sheet (nominal | coshift, agent and wrist).
    python scripts/check_coshift.py --steps outputs/week1/pi05_object_steps --out outputs/verify/coshift_check
"""
import argparse
import contextlib
import glob
import os

import numpy as np

import src.rollout.vec_env as V
from src.rollout.vec_env import COUNTERFACTUAL, EnvRunner, postprocess_actions


def quat_dist(a, b):  # rotation angle between two unit quaternions (xyzw), sign-invariant
    return 2 * np.degrees(np.arccos(np.clip(abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))), -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--steps", required=True, help="eval run dir with steps/t??_n??.npz of a pi0.5 run")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["pre", "post"], default="pre",
                    help="post: check the after-grasp variant (hand + held target + container moved)")
    args = ap.parse_args()
    if args.phase == "post":
        return check_post(args)
    os.makedirs(args.out, exist_ok=True)
    import mujoco
    from libero.libero import benchmark, get_libero_path
    from PIL import Image

    suite = benchmark.get_benchmark_dict()[args.suite]()

    def make(task):
        t = suite.get_task(task)
        bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        cfg = dict(suite=args.suite, max_steps=512, num_steps_wait=10, resolution=256, depth=False, wrist=True, perturb=None,
                   counterfactual=dict(COUNTERFACTUAL, mode="coshift"))
        return EnvRunner(bddl, cfg, contextlib.nullcontext()), np.asarray(suite.get_task_init_states(task))

    # [1] + [2] on random open-gripper chunks
    r, inits = make(3)
    rng = np.random.default_rng(0)
    chunks = [np.concatenate([rng.uniform(-0.4, 0.4, (8, 3)), rng.uniform(-0.05, 0.05, (8, 3)), -np.ones((8, 1))], axis=1)
              for _ in range(args.chunks)]
    ref = [r.reset(inits[0])] + [r.step(c) for c in chunks]
    cf = [r.reset(inits[0], counterfactual=True)] + [r.step(c) for c in chunks]
    d_state = max(float(np.abs(a["sim_state"] - b["sim_state"]).max()) for a, b in zip(ref, cf))
    d_rgb = max(int(np.abs(a["rgb"].astype(int) - b["rgb"].astype(int)).max()) for a, b in zip(ref, cf))
    d_wrist = max(int(np.abs(a["wrist_rgb"].astype(int) - b["wrist_rgb"].astype(int)).max()) for a, b in zip(ref, cf))
    print(f"[1] nominal untouched: |state| {d_state:.1e}, rgb {d_rgb}, wrist {d_wrist}")
    valid = [bool(o["cf_valid"]) for o in cf]
    dp = max(float(np.linalg.norm(o["cf_eef_pos"][:2] - o["eef_pos"][:2] - o["cf_delta"])) for o in cf if o["cf_valid"])
    dz = max(abs(float(o["cf_eef_pos"][2] - o["eef_pos"][2])) for o in cf if o["cf_valid"])
    dq = max(quat_dist(o["cf_eef_quat"], o["eef_quat"]) for o in cf if o["cf_valid"])
    print(f"[2] valid {sum(valid)}/{len(valid)}; |hand xy - (nominal + shift)| {dp * 1000:.2f} mm, |dz| {dz * 1000:.2f} mm, "
          f"orientation {dq:.2f} deg; shifts (cm) {[round(float(np.linalg.norm(o['cf_delta'])) * 100, 1) for o in cf]}; "
          f"swapped with {[str(o['cf_target_name']) for o in cf]}")
    rows = [np.concatenate([o["rgb"], o["rgb_cf"], o["wrist_rgb"], o["wrist_rgb_cf"]], axis=1) for o in cf[:4]]
    Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(args.out, "coshift_sheet.jpg"), quality=85)
    r.close()

    # [3] the label is per chunk (the policy re-plans after every chunk): from each logged pre-grasp state, co-shifted,
    # the logged chunk must move the hand as it does in the nominal world. Secondary: open-loop replay of the whole
    # episode from the co-shifted first state (no re-planning, errors accumulate), does the target still get lifted?
    files = sorted(glob.glob(os.path.join(args.steps, "steps", "t*_n*.npz")),  # one episode of every task, then the next
                   key=lambda f: (os.path.basename(f)[5:7], os.path.basename(f)[1:3]))
    V._PERT_RNG = np.random.default_rng(1)
    lifted, chunk_err, n = [], [], 0

    def coshifted(r, st, place):
        m, d = r.env.env.sim.model._model, r.env.env.sim.data._data
        shift, swap = place
        nq = m.nq
        d.qpos[:] = st[1 : 1 + nq]
        ta = r._cs_target
        if swap is not None:
            a = r._cs_others[swap]
            d.qpos[a : a + 2] = d.qpos[ta : ta + 2]
        d.qpos[ta : ta + 2] += shift
        err = r._ik_shift(m, d, shift)
        st2 = np.array(st, dtype=np.float64).copy()
        st2[1 : 1 + nq] = d.qpos
        st2[1 + nq :][r._cs_arm_v] = 0.0  # the arm's joint velocities do not transfer to another configuration
        return st2, err

    for f in files:
        if n >= args.episodes:
            break
        z = np.load(f, allow_pickle=True)
        task = int(os.path.basename(f)[1:3])
        tz = z["target_pos"][:, 2]
        if not (np.isfinite(tz).all() and tz.max() - tz[0] > 0.05):  # use episodes in which the target was lifted
            continue
        n += 1
        r, _ = make(task)
        robo = r.env.env
        acts = postprocess_actions(z["actions"])
        k_close = next((k for k, a in enumerate(acts) if (a[:, -1] > 0).any()), len(acts))
        r.restore(z["sim_state"][0], 0, -1.0)
        r._coshift_setup(robo)
        r.closed = False
        place = r._coshift_place(robo.sim.data._data)
        if place is None:
            print(f"    {os.path.basename(f)}: no placement found")
            r.close()
            continue
        errs = []
        for k in range(k_close + 1):
            st = np.array(z["sim_state"][k], dtype=np.float64)
            st[1 + robo.sim.model._model.nq :][r._cs_arm_v] = 0.0  # same start velocity in both worlds
            o0 = r.restore(st, int(z["t"][k]), -1.0)
            o1 = r.step(acts[k])
            st2, err = coshifted(r, st, place)
            c0 = r.restore(st2, int(z["t"][k]), -1.0)
            c1 = r.step(acts[k])
            errs.append(float(np.linalg.norm((c1["eef_pos"] - c0["eef_pos"]) - (o1["eef_pos"] - o0["eef_pos"]))))
        chunk_err += errs
        st2, err = coshifted(r, np.array(z["sim_state"][0], dtype=np.float64), place)
        cs = [r.restore(st2, 0, -1.0)] + [r.step(a) for a in acts]
        tz2 = np.array([o["target_pos"][2] for o in cs])
        ok = bool(tz2.max() - tz2[0] > 0.05)
        lifted.append(ok)
        print(f"    {os.path.basename(f)}: shift {np.round(place[0] * 100, 1).tolist()} cm (swap {place[1]}), IK err {err:.1e}, "
              f"per-chunk hand error to first grasp (mm) {np.round(np.array(errs) * 1000, 1).tolist()}, open-loop lift {ok}")
        r.close()
    ce = np.array(chunk_err) * 1000
    print(f"[3] per-chunk hand error before the grasp: median {np.median(ce):.1f} mm, 90% {np.percentile(ce, 90):.1f} mm, "
          f"max {ce.max():.1f} mm; open-loop replay lifts the target in {sum(lifted)}/{len(lifted)}")
    ok = d_state < 1e-9 and d_rgb == 0 and d_wrist == 0 and dp < 0.003 and dq < 1.0 and np.percentile(ce, 90) < 10.0 and np.mean(lifted) >= 0.7
    print("CHECK_COSHIFT_OK" if ok else "CHECK_COSHIFT_FAILED")


def check_post(args):
    """Gate for the after-grasp coshift: from the first logged state of each successful episode in which the target is
    held, move hand + held target + container by a drawn shift and replay the rest of the logged actions open-loop; the
    episode must still succeed about as often as the same replay in the nominal world."""
    import mujoco
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[args.suite]()
    succ = {}
    import json

    for line in open(os.path.join(args.steps, "episodes.jsonl")):
        e = json.loads(line)
        succ[(int(e["task_id"]), int(e["trial_id"]))] = bool(e["success"])
    files = sorted(glob.glob(os.path.join(args.steps, "steps", "t*_n*.npz")),
                   key=lambda f: (os.path.basename(f)[5:7], os.path.basename(f)[1:3]))
    files = [f for f in files if succ.get((int(os.path.basename(f)[1:3]), int(os.path.basename(f)[5:7])))]
    V._PERT_RNG = np.random.default_rng(2)
    res, n = [], 0
    for f in files:
        if n >= args.episodes:
            break
        z = np.load(f, allow_pickle=True)
        task = int(os.path.basename(f)[1:3])
        t = suite.get_task(task)
        bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        cfg = dict(suite=args.suite, max_steps=600, num_steps_wait=10, resolution=128, depth=False, wrist=True, perturb=None,
                   counterfactual=dict(COUNTERFACTUAL, mode="coshift", coshift_phase="post"))
        r = EnvRunner(bddl, cfg, contextlib.nullcontext())
        robo = r.env.env
        acts = postprocess_actions(z["actions"])
        tz = z["target_pos"][:, 2]
        held = [k for k in range(len(acts)) if tz[k] - tz[0] > 0.03 and (acts[: max(k, 1)][..., -1] > 0).any()]
        if not held:
            r.close()
            continue
        k = held[0]
        st = np.array(z["sim_state"][k], dtype=np.float64)
        st[1 + robo.sim.model._model.nq :] = 0.0  # same (zero) start velocity in both worlds
        r.restore(st, int(z["t"][k]), 1.0)
        nom = any(r.step(a)["done"] for a in acts[k:])
        r.restore(st, int(z["t"][k]), 1.0)
        m, d = robo.sim.model._model, robo.sim.data._data  # a reset rebuilds the simulator: fetch after it
        r._coshift_setup(robo)
        r._cs_z0 = float(z["sim_state"][0][1 + r._cs_target + 2])
        # resting heights must come from the start of the episode, not from this (lifted) state
        r._cs_carry = {n: (a, float(z["sim_state"][0][1 + a + 2])) for n, (a, _) in r._cs_carry.items()}
        sh = r._coshift_place_post(d)
        if sh is None:
            print(f"    {os.path.basename(f)}: {getattr(r, '_cs_why', '?')} (query {k})")
            r.close()
            continue
        d.qpos[:] = st[1 : 1 + m.nq]
        for a in (r._cs_held, r._cs_others[r._cs_container]):
            d.qpos[a : a + 2] += sh
        err = r._ik_shift(m, d, sh)
        st2 = st.copy()
        st2[1 : 1 + m.nq] = d.qpos
        if err >= 1e-3:  # the hand cannot reach the shifted pose: the training marks such queries invalid too
            print(f"    {os.path.basename(f)}: shift {np.round(sh * 100, 1).tolist()} cm unreachable (IK err {err:.1e}), skipped")
            r.close()
            continue
        r.restore(st2, int(z["t"][k]), 1.0)
        cs = any(r.step(a)["done"] for a in acts[k:])
        n += 1
        res.append((nom, cs))
        print(f"    {os.path.basename(f)}: from query {k}, shift {np.round(sh * 100, 1).tolist()} cm, IK err {err:.1e}, "
              f"nominal replay success {nom}, co-shifted replay success {cs}")
        r.close()
    res = np.array(res).reshape(-1, 2)
    print(f"[post] successes replayed from the first held state: nominal {res[:, 0].sum()}/{len(res)}, "
          f"co-shifted {res[:, 1].sum()}/{len(res)}")
    ok = len(res) >= 5 and res[:, 1].sum() >= 0.8 * res[:, 0].sum()
    print("CHECK_COSHIFT_POST_OK" if ok else "CHECK_COSHIFT_POST_FAILED")


if __name__ == "__main__":
    main()
