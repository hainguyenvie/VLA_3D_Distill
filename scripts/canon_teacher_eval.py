"""Executed upper bound of the canonical-frame teacher on a LIBERO-PRO swap cell: before the grasp, pi0.5 is queried in
the canonical world W'' — the target put back on its usual spot (from the standard suite's initial state of the same
trial), the object now standing there put on the target's spot, and the hand displaced by the opposite of the target's
displacement (inverse kinematics, orientation kept) — so the hand-target relation is that of the real world W' while the
layout is the familiar one; its chunk is executed in the real world. After the grasp pi0.5 acts on the real world.
Compare with pi0.5 alone (same initial states) and with the scripted oracle approach (scripts/oracle_approach.py).
    LIBERO_VARIANT=pro python scripts/canon_teacher_eval.py --ckpt <pi0.5 dir> --cell libero_object_swap --base libero_object --out <dir>
"""
import argparse
import contextlib
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--cell", default="libero_object_swap")
    ap.add_argument("--base", default="libero_object")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=280)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import mujoco
    import robosuite.macros as macros
    from libero.libero import benchmark, get_libero_path
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    from src.policy.pi05_policy import Pi05Policy
    from src.rollout.vec_env import EnvRunner, postprocess_actions

    conv = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    bm = benchmark.get_benchmark_dict()
    cell, base = bm[args.cell](), bm[args.base]()
    pol = Pi05Policy(args.ckpt)
    if args.lora:
        pol.add_lora(32, adapter_path=args.lora)
    res = []
    for t in range(cell.n_tasks):
        task = cell.get_task(t)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        cfg = dict(suite=args.cell, max_steps=args.max_steps, num_steps_wait=10, resolution=256, depth=False, wrist=True,
                   perturb=None, counterfactual=None)
        r = EnvRunner(bddl, cfg, contextlib.nullcontext())
        inits, std = np.asarray(cell.get_task_init_states(t)), np.asarray(base.get_task_init_states(t))
        for n in range(min(args.trials, len(inits))):
            obs = r.reset(inits[n])
            robo = r.env.env
            r._coshift_setup(robo)
            if not r._cs_ok:
                r.close()
                break
            ta = r._cs_target
            home = std[n][1 + ta : 1 + ta + 2]  # the target's usual spot in this trial
            canon_q, ik_fail = 0, 0
            while obs["active"] and not obs["done"]:
                o = obs
                if not r.closed:
                    m, d = robo.sim.model._model, robo.sim.data._data
                    saved = {k: getattr(d, k).copy() for k in r._KIN + r._COM}
                    try:
                        xp = d.qpos[ta : ta + 2].copy()
                        delta = xp - home
                        if np.linalg.norm(delta) > 0.03:
                            lure = min(r._cs_others.items(), key=lambda kv: np.linalg.norm(d.qpos[kv[1] : kv[1] + 2] - home))
                            if np.linalg.norm(d.qpos[lure[1] : lure[1] + 2] - home) < 0.05:
                                d.qpos[lure[1] : lure[1] + 2] = xp
                            d.qpos[ta : ta + 2] = home
                            err = r._ik_shift(m, d, -delta)
                            if err < 1e-3:
                                mujoco.mj_kinematics(m, d)
                                mujoco.mj_camlight(m, d)
                                out = {}
                                r._render_cf(robo, 256, conv, out)
                                o = dict(obs, rgb=out["rgb_cf"], wrist_rgb=out["wrist_rgb_cf"], eef_pos=out["cf_eef_pos"],
                                         eef_quat=out["cf_eef_quat"])
                                canon_q += 1
                            else:
                                ik_fail += 1
                    finally:
                        for k, v in saved.items():
                            getattr(d, k)[:] = v
                out = pol.act([o["rgb"]], [task.language], obs=[o])
                obs = r.step(postprocess_actions(out["actions"])[0])
                robo = r.env.env
            res.append(dict(task_id=t, trial_id=n, success=bool(obs["done"]), canon_queries=canon_q, ik_fail=ik_fail))
            print(f"task {t} trial {n}: canonical queries {canon_q} (IK failed {ik_fail}), success {obs['done']}", flush=True)
        r.close()
    sr = float(np.mean([x["success"] for x in res]))
    per = {t: float(np.mean([x["success"] for x in res if x["task_id"] == t])) for t in sorted({x["task_id"] for x in res})}
    json.dump(dict(success_rate=sr, per_task=per, episodes=res), open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"CANON_TEACHER success {sr:.3f}; per task {per}")


if __name__ == "__main__":
    main()
