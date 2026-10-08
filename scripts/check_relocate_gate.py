"""Gate for the goal-driven relocate counterfactual (src/rollout/vec_env.py, mode "relocate"): uses the environment's own
machinery to find what is held and its place target (BDDL goal on / in), from the first logged state in which an object
of the task is held in successful episodes. The place target is moved by the shift the training would draw (in the
physics: a movable container with what lies in it, or the whole piece of furniture), then the scripted placer finishes,
executed like the student would learn it (10 steps of each 50-step kinematic chunk, then re-plan). Scored per object:
the carried object lies in / on its place target and the hand has let go. Compared with the same replay unmoved.
    LIBERO_VARIANT=pro python scripts/check_relocate_gate.py --suite libero_10 --steps outputs/week1/pi05_10_std_steps
"""
import argparse
import contextlib
import glob
import json
import os

import numpy as np

import src.rollout.vec_env as V
from src.rollout.scripted import placer_chunk
from src.rollout.vec_env import COUNTERFACTUAL, EnvRunner, postprocess_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--pass_rate", type=float, default=0.6,
                    help="a task passes when the placer succeeds in at least this share of its relocated tries (and n >= 5)")
    args = ap.parse_args()
    import mujoco
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[args.suite]()
    succ = {(e["task_id"], e["trial_id"]): e["success"] for e in map(json.loads, open(os.path.join(args.steps, "episodes.jsonl")))}
    files = sorted(glob.glob(os.path.join(args.steps, "steps", "t*_n*.npz")),
                   key=lambda f: (os.path.basename(f)[5:7], os.path.basename(f)[1:3]))
    V._PERT_RNG = np.random.default_rng(0)
    per = {}
    for f in files:
        if sum(len(v) for v in per.values()) >= args.episodes:
            break
        t, n = int(os.path.basename(f)[1:3]), int(os.path.basename(f)[5:7])
        if not succ.get((t, n)):
            continue
        z = np.load(f, allow_pickle=True)
        task = suite.get_task(t)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        cfg = dict(suite=args.suite, max_steps=600, num_steps_wait=10, resolution=64, depth=False, wrist=False, perturb=None,
                   counterfactual=dict(COUNTERFACTUAL, mode="relocate", coshift_post=(0.15, 0.4), unique=True))
        r = EnvRunner(bddl, cfg, contextlib.nullcontext())
        acts = postprocess_actions(z["actions"])
        st0 = np.array(z["sim_state"][0], dtype=np.float64)
        out = []
        for k in range(len(acts)):  # first logged state in which the env finds something of the task held
            st = np.array(z["sim_state"][k], dtype=np.float64)
            r.restore(st, int(z["t"][k]), 1.0)
            robo = r.env.env
            r._coshift_setup(robo)
            r.cf = None
            if not r._cs_ok:
                break
            nq = robo.sim.model._model.nq
            r._cs_carry = {nm: (a, float(st0[1 + a + 2])) for nm, (a, _) in r._cs_carry.items()}  # resting heights
            r.closed, r.grip_cmd = True, 1.0
            sh = r._coshift_place_post(robo.sim.data._data)
            if sh is None:
                continue
            spec, ta, riders = r._cs_spec, r._cs_held, list(r._cs_riders)
            for moved in (False, True):
                r.restore(st, int(z["t"][k]), 1.0)
                robo = r.env.env
                m, d = robo.sim.model._model, robo.sim.data._data
                if moved:
                    if spec[0] == "fixture":
                        m.body_pos[spec[1]][:2] += sh
                    else:
                        for a in riders + [spec[1]]:
                            d.qpos[a : a + 2] += sh
                    mujoco.mj_forward(m, d)
                site = robo.robots[0].eef_site_id
                top = 0.0 if spec[0] == "fixture" else r._cs_rims.get(spec[3], 0.0)

                def goal():
                    if spec[0] == "fixture":
                        return d.site_xpos[spec[2]].copy()
                    g = d.qpos[spec[1] : spec[1] + 3].copy()
                    if spec[2] is not None:
                        g[:2] = d.site_xpos[spec[2]][:2]
                    return g

                plan = []
                for _ in range(args.max_steps):
                    if not plan:
                        plan = list(placer_chunk(d.site_xpos[site].copy(), d.qpos[ta : ta + 3].copy(), goal(), top=top)[:10])
                    robo.step(plan.pop(0))
                obj, g = d.qpos[ta : ta + 3], goal()
                tol = (m.site_size[spec[2]][:2] + 0.02) if spec[2] is not None else np.array([0.05, 0.05])
                placed = bool((np.abs(obj[:2] - g[:2]) < tol).all() and obj[2] - g[2] < top + 0.08
                              and np.linalg.norm(obj - d.site_xpos[site]) > 0.05)
                out.append(placed)
            print(f"    t{t:02d}_n{n:02d} q{k} {spec[0]} shift {np.round(sh * 100, 1).tolist()} cm: unmoved {out[0]}, moved {out[1]}", flush=True)
            break
        r.close()
        if len(out) == 2:
            per.setdefault(t, []).append(out)
    allr = np.array([x for v in per.values() for x in v])
    for t in sorted(per):
        a = np.array(per[t])
        print(f"task {t}: n {len(a)}, unmoved {a[:, 0].sum()}, moved {a[:, 1].sum()}")
    print(f"[relocate gate] unmoved {allr[:, 0].sum()}/{len(allr)}, moved {allr[:, 1].sum()}/{len(allr)}")
    ok = [t for t in sorted(per) if len(per[t]) >= 5 and np.mean([x[1] for x in per[t]]) >= args.pass_rate]
    print(f"[relocate gate] pass (moved >= {args.pass_rate:.0%}, n >= 5): --relocate_tasks {','.join(map(str, ok))}")


if __name__ == "__main__":
    main()
