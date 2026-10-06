"""Does the policy's action track the target, or replay a trajectory? Counterfactual object-displacement probe.

For logged pre-grasp states of a rollout run, the simulator is restored to the state, the target object is moved
by a small horizontal displacement d (the robot, the other objects and the instruction are unchanged), the frame is
re-rendered and the policy is queried. If the policy acts on where the object is, the horizontal motion of its
action chunk shifts with the object; if it replays a memorised motion, it does not. The reported number is the
projection p = <motion(d) - motion(0), d> / |d|^2: 1 = the chunk's motion follows the object fully, 0 = no reaction.
    python scripts/probe_counterfactual.py --run outputs/week1/b0_object_student --ckpt <policy> [--lora <adapter>] --out <dir>
Writes <out>/states.csv (one row per state and displacement) and <out>/summary.json. Needs <run>/failures.csv.
"""
import argparse
import csv
import json
import os

import numpy as np

STEP_M = 0.05  # metres of commanded end-effector translation per control step at action 1.0 (robosuite OSC_POSE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint, or oft:<checkpoint> for the standard OpenVLA-OFT policy")
    ap.add_argument("--lora", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--episodes_per_task", type=int, default=4, help="per task: this many successful and this many failed episodes")
    ap.add_argument("--states_per_episode", type=int, default=3)
    ap.add_argument("--deltas", default="0.03,0.06", help="displacement magnitudes in metres")
    ap.add_argument("--num_envs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from analyze_failures import task_layout  # sibling script
    from libero.libero import benchmark

    from src.rollout.vec_env import LiberoVecEnv

    rng = np.random.default_rng(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    fails = {(int(r["task_id"]), int(r["trial_id"])): r for r in csv.DictReader(open(os.path.join(args.run, "failures.csv")))}
    eps = [json.loads(l) for l in open(os.path.join(args.run, "episodes.jsonl"))]
    by_task = {}
    for e in eps:
        by_task.setdefault((e["task_id"], int(e["success"])), []).append(e)
    chosen = []
    for (t, s), lst in sorted(by_task.items()):
        chosen += [lst[i] for i in rng.permutation(len(lst))[: args.episodes_per_task]]

    # pre-grasp states: queries before the first chunk that closes the gripper, evenly spread over the approach
    jobs, layouts = [], {}
    for e in chosen:
        if e["task_id"] not in layouts:
            layouts[e["task_id"]] = task_layout(suite, e["task_id"])
        addr, target, _ = layouts[e["task_id"]]
        z = np.load(os.path.join(args.run, "steps", f"t{e['task_id']:02d}_n{e['trial_id']:02d}.npz"))
        close = z["actions"][:, :, -1].min(1) < 0.5
        last = int(np.argmax(close)) if close.any() else len(close) - 1
        if last < 1:
            continue
        for q in sorted({int(round(x)) for x in np.linspace(0, last - 1, args.states_per_episode)}):
            cmd = -1.0 if q == 0 else -float(np.sign(2 * z["actions"][q - 1, -1, -1] - 1))  # env convention: +1 = close
            tgt = z["sim_state"][q, 1 + addr[target] : 1 + addr[target] + 3]
            jobs.append({"task_id": e["task_id"], "trial_id": e["trial_id"], "task": e["task"], "success": int(e["success"]),
                         "mode": fails[(e["task_id"], e["trial_id"])]["mode"], "q": q, "to_first_close": last - q, "t0": int(z["t"][q]),
                         "state": z["sim_state"][q], "gripper_cmd": cmd, "addr": addr[target],
                         "offset_xy": (tgt[:2] - z["eef_pos"][q, :2]).astype(np.float64)})
    deltas = [float(x) for x in args.deltas.split(",")]
    variants = [np.zeros(2)] + [m * np.array(v) for m in deltas for v in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    print(f"{len(chosen)} episodes, {len(jobs)} states, {len(variants)} variants each", flush=True)

    import torch

    from src.policy.rebin import load_policy

    torch.manual_seed(args.seed)
    vec = LiberoVecEnv(args.suite, args.num_envs, max_steps=512, wrist=args.ckpt.startswith(("oft:", "pi05:")))
    policy = load_policy(args.ckpt, args.suite, "cuda:0")
    if args.lora:
        policy.add_lora(adapter_path=args.lora)

    def motion(job, d):
        """Horizontal motion (m) the policy's chunk commands from the restored state with the target moved by d."""
        s = job["state"].copy()
        s[1 + job["addr"] : 1 + job["addr"] + 2] += d
        return s

    rows, pending = [], []
    work = [(j, v) for j in jobs for v in variants]
    i_env = 0
    for k, (job, d) in enumerate(work):
        vec.restore(i_env, job["task_id"], motion(job, d), job["t0"], job["gripper_cmd"])
        pending.append((i_env, job, d))
        i_env += 1
        if i_env == args.num_envs or k == len(work) - 1:
            obs = [vec.recv(i) for i, _, _ in pending]
            out = policy.act([o["rgb"] for o in obs], [j["task"] for _, j, _ in pending], obs=obs)
            for (_, j, d), o, a in zip(pending, obs, out["actions"]):
                rows.append({**{k: v for k, v in j.items() if k not in ("state", "addr", "offset_xy")},
                             "dx": round(float(d[0]), 3), "dy": round(float(d[1]), 3),
                             "offset_x": round(float(j["offset_xy"][0]), 4), "offset_y": round(float(j["offset_xy"][1]), 4),
                             "move_x": round(float(a[:, 0].sum() * STEP_M), 4), "move_y": round(float(a[:, 1].sum() * STEP_M), 4),
                             "gripper_close": int(a[:, -1].min() < 0.5)})
            pending, i_env = [], 0
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(work)} queries", flush=True)
    vec.close()

    base = {(r["task_id"], r["trial_id"], r["q"]): r for r in rows if r["dx"] == 0 and r["dy"] == 0}
    for r in rows:
        b = base[(r["task_id"], r["trial_id"], r["q"])]
        d = np.array([r["dx"], r["dy"]])
        if np.linalg.norm(d) > 0:
            dm = np.array([r["move_x"] - b["move_x"], r["move_y"] - b["move_y"]])
            r["p"] = round(float(dm @ d / (d @ d)), 3)  # 1 = the chunk's motion follows the object fully
            r["dmove_cm"] = round(float(np.linalg.norm(dm)) * 100, 2)
        else:
            r["p"], r["dmove_cm"] = "", ""
    with open(os.path.join(args.out, "states.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def stat(sel):
        v = [r["p"] for r in rows if r["p"] != "" and sel(r)]
        return {"n": len(v), "median_p": round(float(np.median(v)), 3), "mean_p": round(float(np.mean(v)), 3),
                "frac_p_above_0.25": round(float(np.mean(np.array(v) > 0.25)), 3)} if v else None

    summary = {"all": stat(lambda r: True),
               "by_outcome": {g: stat(lambda r, s=s: r["success"] == s) for g, s in (("success", 1), ("failure", 0))},
               "by_delta": {str(m): stat(lambda r, m=m: abs(abs(r["dx"]) + abs(r["dy"]) - m) < 1e-6) for m in deltas},
               "by_phase": {"early (>=3 queries before close)": stat(lambda r: r["to_first_close"] >= 3),
                            "late (<3 queries before close)": stat(lambda r: r["to_first_close"] < 3)},
               "by_mode": {m: stat(lambda r, m=m: r["mode"] == m) for m in sorted({r["mode"] for r in rows})},
               "config": vars(args)}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))
    print("PROBE_COUNTERFACTUAL_DONE", flush=True)


if __name__ == "__main__":
    main()
