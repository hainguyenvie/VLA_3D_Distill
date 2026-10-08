"""Automatic failure taxonomy from logged simulator states (no policy, no re-simulation).

Every queried state of a rollout stores the full MuJoCo state, so object and gripper trajectories can be read
back directly. For each episode this computes what happened to the target object, to the other objects and to
the gripper, assigns one coarse failure mode, and estimates when the episode went wrong.
    python scripts/analyze_failures.py --run outputs/week1/b0_object_student [--suite libero_object]
Writes <run>/failures.csv (one row per episode, raw features + label) and <run>/failures.json (counts).
With LIBERO_VARIANT=plus (runs of eval_libero_plus.py --save_steps) modes are also counted per perturbation type.

Modes (first match wins; thresholds are heuristics, features are kept so they can be re-cut):
  success
  wrong_object      another object was lifted (> LIFT) while the target never was
  lost_in_transit   the target was lifted but the episode did not succeed (dropped, or placed outside the basket)
  target_toppled    the target was knocked over or pushed away without being held
  wrong_place       the first grasp attempt closed nearer to another object than to the target
  near_miss         the gripper closed next to the target and came up empty
  no_attempt        the gripper never closed
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter

import numpy as np

LIFT, PUSH, DROP = 0.05, 0.04, 0.015  # metres: lifted, pushed sideways, sunk (lying on its side)
TILT = 30.0  # degrees between the object's initial and current up axis
EMPTY = 0.01  # finger opening (m, both fingers) below which the closed gripper holds nothing


def up_axis(quat):
    """World direction of the object's local z axis; MuJoCo free-joint quaternions are (w, x, y, z)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=-1)


def task_layout(suite, task_id):
    """qpos addresses of every movable object, and which one is the target / the container."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                             camera_heights=64, camera_widths=64)
    robo = env.env
    addr = {}
    for name in robo.objects_dict:
        try:
            a = robo.sim.model.get_joint_qpos_addr(f"{name}_joint0")
        except Exception:  # an object without a free joint cannot move
            continue
        addr[name] = int(a[0] if isinstance(a, (tuple, list, np.ndarray)) else a)
    interest = list(robo.obj_of_interest)
    env.close()
    target = interest[0]
    container = interest[-1] if len(interest) > 1 else None  # the place target is named last (Long names two objects)
    return addr, target, container


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--suite", default="libero_object")
    args = ap.parse_args()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    eps = [json.loads(l) for l in open(os.path.join(args.run, "episodes.jsonl"))]
    meta = {}  # LIBERO-Plus: perturbation type and difficulty of each task
    if os.environ.get("LIBERO_VARIANT") == "plus":
        import libero.libero as L

        cls = json.load(open(os.path.join(os.path.dirname(L.__file__), "benchmark", "task_classification.json")))[args.suite]
        by_name = {c["name"]: c for c in cls}
        meta = {t: by_name[suite.get_task(t).name] for t in {e["task_id"] for e in eps}}
    layouts, rows = {}, []
    for e in sorted(eps, key=lambda e: (e["task_id"], e["trial_id"])):
        path = os.path.join(args.run, "steps", f"t{e['task_id']:02d}_n{e['trial_id']:02d}.npz")
        if not os.path.exists(path):
            continue
        if e["task_id"] not in layouts:
            layouts[e["task_id"]] = task_layout(suite, e["task_id"])
        addr, target, container = layouts[e["task_id"]]
        z = np.load(path)
        st = np.concatenate([z["sim_state"], z["final_sim_state"][None]])  # (T+1, 1 + nq + nv); column 0 is time
        pos = {k: st[:, 1 + a : 1 + a + 3] for k, a in addr.items()}
        up = {k: up_axis(st[:, 1 + a + 3 : 1 + a + 7]) for k, a in addr.items()}
        eef = z["eef_pos"].astype(np.float64)
        opening = np.abs(z["gripper_qpos"]).sum(-1)
        close_cmd = z["actions"][:, :, -1].min(1) < 0.5  # the policy's gripper output: 1 = open, 0 = close
        t_steps, n = z["t"], len(eef)

        lift = {k: float((p[:, 2] - p[0, 2]).max()) for k, p in pos.items()}
        push = {k: float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max()) for k, p in pos.items()}
        tilt = {k: np.degrees(np.arccos(np.clip((u * u[0]).sum(-1), -1, 1))) for k, u in up.items()}
        others = [k for k in pos if k not in (target, container)]
        tp = pos[target]
        q = int(np.argmax(close_cmd)) if close_cmd.any() else -1  # first grasp attempt
        at = min(q + 1, n - 1) if q >= 0 else -1  # state after the chunk that closed the gripper
        xy_err = float(np.linalg.norm(eef[at, :2] - tp[at, :2])) if q >= 0 else float("nan")
        dz_err = float(eef[at, 2] - tp[at, 2]) if q >= 0 else float("nan")
        nearest = min((k for k in pos if k != container), key=lambda k: np.linalg.norm(eef[at, :2] - pos[k][at, :2])) if q >= 0 else ""
        came_up_empty = bool(q >= 0 and opening[min(q + 2, n - 1)] < EMPTY)
        lifted_other = max(others, key=lambda k: lift[k]) if others else None
        # the target counts as knocked over when it tilts, sinks, or slides while never lifted
        toppled = (tilt[target] > TILT) | ((tp[0, 2] - tp[:, 2]) > DROP) | \
                  ((np.linalg.norm(tp[:, :2] - tp[0, :2], axis=1) > PUSH) & ((tp[:, 2] - tp[0, 2]) < 0.01))

        if e["success"]:
            mode = "success"
        elif lifted_other is not None and lift[lifted_other] > LIFT and lift[target] < LIFT:
            mode = "wrong_object"
        elif lift[target] > LIFT:
            mode = "lost_in_transit"
        elif toppled.any():
            mode = "target_toppled"
        elif q >= 0 and nearest != target:
            mode = "wrong_place"
        elif q >= 0:
            mode = "near_miss"
        else:
            mode = "no_attempt"

        bad = toppled.copy()
        for k in others:
            bad |= (pos[k][:, 2] - pos[k][0, 2]) > LIFT
        first_bad = int(np.argmax(bad)) if bad.any() else -1
        rows.append({
            "task_id": e["task_id"], "trial_id": e["trial_id"], "success": int(e["success"]), "mode": mode,
            "n_queries": n, "target": target,
            "target_lift": round(lift[target], 3), "target_push": round(push[target], 3),
            "target_max_tilt_deg": round(float(tilt[target].max()), 1),
            "first_close_step": int(t_steps[q]) if q >= 0 else -1,
            "xy_err_at_first_close": round(xy_err, 3), "dz_at_first_close": round(dz_err, 3),
            "nearest_at_first_close": nearest, "first_close_on_target": int(nearest == target),
            "first_close_came_up_empty": int(came_up_empty), "n_close_attempts": int((np.diff(close_cmd.astype(int)) == 1).sum() + int(close_cmd[0])),
            "other_lifted": lifted_other if lifted_other and lift[lifted_other] > LIFT else "",
            "max_other_lift": round(max((lift[k] for k in others), default=0.0), 3),
            "max_other_push": round(max((push[k] for k in others), default=0.0), 3),
            "first_bad_step": int(t_steps[min(first_bad, n - 1)]) if first_bad >= 0 else -1,
        })
        if meta:
            rows[-1].update(category=meta[e["task_id"]]["category"], difficulty_level=meta[e["task_id"]]["difficulty_level"])
    with open(os.path.join(args.run, "failures.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fails = [r for r in rows if not r["success"]]
    wins = [r for r in rows if r["success"]]
    q3 = lambda v: [round(float(x), 3) for x in np.quantile(v, [0.25, 0.5, 0.75])] if len(v) else None  # noqa: E731
    att = lambda rs: [r for r in rs if r["first_close_step"] >= 0]  # noqa: E731
    summary = {
        "n_episodes": len(rows), "n_failures": len(fails),
        "modes": dict(Counter(r["mode"] for r in fails).most_common()),
        "modes_by_task": {} if meta else {str(t): dict(Counter(r["mode"] for r in fails if r["task_id"] == t)) for t in sorted(layouts)},
        "modes_by_category": {c: dict(Counter(r["mode"] for r in rows if r["category"] == c).most_common())
                              for c in sorted({r["category"] for r in rows})} if meta else {},
        "xy_err_at_first_close_median_by_category": {
            c: {k: (round(float(np.median(v)), 3) if v else None) for k, v in (
                ("success", [r["xy_err_at_first_close"] for r in att(wins) if r["category"] == c]),
                ("failure", [r["xy_err_at_first_close"] for r in att(fails) if r["category"] == c]))}
            for c in sorted({r["category"] for r in rows})} if meta else {},
        # how far from the target (horizontal, metres) the gripper was when it first closed
        "xy_err_at_first_close_quartiles": {"success": q3([r["xy_err_at_first_close"] for r in att(wins)]),
                                            "failure": q3([r["xy_err_at_first_close"] for r in att(fails)])},
        "first_close_on_target": {"success": float(np.mean([r["first_close_on_target"] for r in att(wins)])),
                                  "failure": float(np.mean([r["first_close_on_target"] for r in att(fails)]))},
        "first_close_came_up_empty": {"success": float(np.mean([r["first_close_came_up_empty"] for r in att(wins)])),
                                      "failure": float(np.mean([r["first_close_came_up_empty"] for r in att(fails)]))},
        "first_close_step_quartiles": {"success": q3([r["first_close_step"] for r in att(wins)]),
                                       "failure": q3([r["first_close_step"] for r in att(fails)])},
        "first_bad_step_quartiles_failures": q3([r["first_bad_step"] for r in fails if r["first_bad_step"] >= 0]),
        "failures_with_irreversible_event": float(np.mean([r["first_bad_step"] >= 0 for r in fails])),
        "thresholds": {"lift_m": LIFT, "push_m": PUSH, "drop_m": DROP, "tilt_deg": TILT, "empty_opening_m": EMPTY},
    }
    json.dump(summary, open(os.path.join(args.run, "failures.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("FAILURES_DONE")


if __name__ == "__main__":
    main()
