"""Gate for the privileged scripted placer (src/rollout/scripted.py): from the first logged state of successful
episodes in which the target is held, move the container by a random horizontal vector (in the physics) and let the
placer finish the episode; it must succeed about as often as from the unmoved state.
    LIBERO_VARIANT=pro python scripts/check_scripted_place.py --suite libero_spatial_swap --steps outputs/week1/pi05_spatial_pro_swap_steps
"""
import argparse
import contextlib
import glob
import json
import os

import mujoco
import numpy as np

from src.rollout.scripted import ScriptedPlacer, placer_chunk
from src.rollout.vec_env import EnvRunner, postprocess_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial_swap")
    ap.add_argument("--steps", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--shift", default="0.1,0.4")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--chunked", action="store_true",
                    help="execute the placer as the student would learn it: the first 10 actions of the 50-step chunk "
                         "rolled out on the kinematic model (src/rollout/scripted.placer_chunk), then re-plan")
    args = ap.parse_args()
    from libero.libero import benchmark, get_libero_path

    from scripts.analyze_failures import task_layout

    suite = benchmark.get_benchmark_dict()[args.suite]()
    succ = {(e["task_id"], e["trial_id"]): e["success"] for e in map(json.loads, open(os.path.join(args.steps, "episodes.jsonl")))}
    files = sorted(glob.glob(os.path.join(args.steps, "steps", "t*_n*.npz")),
                   key=lambda f: (os.path.basename(f)[5:7], os.path.basename(f)[1:3]))
    rng = np.random.default_rng(0)
    lo, hi = (float(x) for x in args.shift.split(","))
    res, lay = [], {}
    for f in files:
        if len(res) >= args.episodes:
            break
        t, n = int(os.path.basename(f)[1:3]), int(os.path.basename(f)[5:7])
        if not succ.get((t, n)):
            continue
        if t not in lay:
            lay[t] = task_layout(suite, t)
        addr, target, cont = lay[t]
        if target not in addr:
            continue
        z = np.load(f, allow_pickle=True)
        acts = postprocess_actions(z["actions"])
        tz = z["target_pos"][:, 2]
        held = [k for k in range(len(acts)) if tz[k] - tz[0] > 0.03 and (acts[: max(k, 1)][..., -1] > 0).any()]
        if not held:
            continue
        k = held[0]
        task = suite.get_task(t)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        cfg = dict(suite=args.suite, max_steps=600, num_steps_wait=10, resolution=64, depth=False, wrist=False, perturb=None,
                   counterfactual=None)
        r = EnvRunner(bddl, cfg, contextlib.nullcontext())
        out = []
        for moved in (False, True):
            st = np.array(z["sim_state"][k], dtype=np.float64)
            r.restore(st, int(z["t"][k]), 1.0)
            robo = r.env.env
            m, d = robo.sim.model._model, robo.sim.data._data
            ta = addr[target]
            fix = None  # place target that is a region of a fixture: (site id, root body)
            if cont not in addr:
                try:
                    sid = robo.sim.model.site_name2id(cont)
                except Exception:
                    break
                root = int(m.site_bodyid[sid])
                while m.body_parentid[root] != 0:
                    root = int(m.body_parentid[root])
                fix = (sid, root)
            ca = addr.get(cont)
            if fix is not None:  # carry onto the region: move the fixture in the model (it stays moved for this replay)
                if moved:
                    rngm = rng.uniform(0, 2 * np.pi), rng.uniform(lo, hi)
                    sh = rngm[1] * np.array([np.cos(rngm[0]), np.sin(rngm[0])])
                    m.body_pos[fix[1]][:2] += sh
                    mujoco.mj_forward(m, d)
                site = robo.robots[0].eef_site_id
                placer = ScriptedPlacer(top=0.0)
                done, plan = False, []
                for step in range(args.max_steps):
                    goal = d.site_xpos[fix[0]].copy()
                    if args.chunked:
                        if not plan:
                            plan = list(placer_chunk(d.site_xpos[site].copy(), d.qpos[ta : ta + 3].copy(), goal)[:10])
                        a = plan.pop(0)
                    else:
                        a = placer.act(d.site_xpos[site].copy(), d.qpos[ta : ta + 3].copy(), goal)
                    robo.step(a)
                    if robo._check_success():
                        done = True
                        break
                obj, g, half = d.qpos[ta : ta + 3], d.site_xpos[fix[0]], m.site_size[fix[0]][:2]
                placed = bool((np.abs(obj[:2] - g[:2]) < half + 0.02).all() and obj[2] - g[2] < 0.08
                              and np.linalg.norm(obj - d.site_xpos[site]) > 0.05)
                out.append(bool(done or placed))
                continue
            if moved:
                from src.rollout.vec_env import world_fixed_bodies

                fixed = [d.xpos[b][:2].copy() for b in world_fixed_bodies(robo)
                         if not (robo.sim.model.body_id2name(b) or "").endswith("table")]
                frees = np.array([d.qpos[a : a + 2] for a in addr.values()])
                box_lo, box_hi = frees.min(0) - 0.05, frees.max(0) + 0.05  # the area the objects occupy
                ok = False
                for _ in range(200):
                    ang, mag = rng.uniform(0, 2 * np.pi), rng.uniform(lo, hi)
                    new = d.qpos[ca : ca + 2] + mag * np.array([np.cos(ang), np.sin(ang)])
                    others = [d.qpos[a : a + 2] for nme, a in addr.items() if nme not in (cont, target)]
                    if ((new >= box_lo).all() and (new <= box_hi).all() and all(np.linalg.norm(new - o) > 0.12 for o in others)
                            and all(np.linalg.norm(new - fx) > 0.2 for fx in fixed)):
                        ok = True
                        break
                if not ok:
                    out.append(None)
                    continue
                st2 = st.copy()
                st2[1 + ca : 1 + ca + 2] = new
                r.restore(st2, int(z["t"][k]), 1.0)
                robo = r.env.env
                m, d = robo.sim.model._model, robo.sim.data._data
            from src.rollout.scripted import rim_height
            from src.rollout.vec_env import EnvRunner as _E

            top = rim_height(m, d, robo.obj_body_id[cont], _E._is_descendant)
            placer = ScriptedPlacer(top=top)
            done = False
            site = robo.robots[0].eef_site_id
            plan = []
            for step in range(args.max_steps):
                if args.chunked:
                    if not plan:
                        plan = list(placer_chunk(d.site_xpos[site].copy(), d.qpos[ta : ta + 3].copy(), d.qpos[ca : ca + 3].copy(),
                                                 top=top)[:10])
                    a = plan.pop(0)
                else:
                    a = placer.act(d.site_xpos[site].copy(), d.qpos[ta : ta + 3].copy(), d.qpos[ca : ca + 3].copy())
                _, _, done, _ = robo.step(a)
                if robo._check_success():
                    done = True
                    break
            # per object (multi-object tasks need the other object too for the env's success): the carried object lies
            # in / on the container and the hand has let go of it
            obj, c = d.qpos[ta : ta + 3], d.qpos[ca : ca + 3]
            placed = (np.linalg.norm(obj[:2] - c[:2]) < 0.05 and obj[2] - c[2] < top + 0.03
                      and np.linalg.norm(obj - d.site_xpos[site]) > 0.05)
            done = bool(done or placed)
            out.append(bool(done))
            if moved and not done:  # where did it end?
                obj, c = d.qpos[ta : ta + 3], d.qpos[ca : ca + 3]
                print(f"      moved fail: obj-container xy {np.linalg.norm(obj[:2] - c[:2]) * 100:.1f} cm, dz {(obj[2] - c[2]) * 100:.1f} cm, "
                      f"container now {np.round(c[:2], 3).tolist()} (asked {np.round(new, 3).tolist()}), opened {placer.opened}")
        r.close()
        if len(out) < 2:  # no usable place target
            continue
        if None in out:
            print(f"    t{t:02d}_n{n:02d}: no free spot for the container")
            continue
        res.append(out)
        print(f"    t{t:02d}_n{n:02d} from query {k}: unmoved {out[0]}, container moved {out[1]}")
    res = np.array(res)
    print(f"[placer] success: unmoved {res[:, 0].sum()}/{len(res)}, container moved {res[:, 1].sum()}/{len(res)}")


if __name__ == "__main__":
    main()
