"""Where does the policy take its decision from, and when? Sensitivity map along successful episodes.

At logged states of successful episodes (a steps directory written by scripts/eval_libero.py), the simulator is restored
and one input of the policy is changed at a time; the change of its action chunk says what that input drives:
  img_<m>    the target object moved by m metres (before the grasp), or the place target (container) moved by m metres
             while carrying; everything else, the robot included, unchanged. Physics state edit, re-rendered.
  swap       before the grasp, the target and another movable object exchange their places.
  state      only the proprio given to the policy is offset by 5 cm (the images are those of the true state).
  lang       before the grasp, the instruction names another object of the scene instead of the target.
The motion of a chunk is where the hand would go in its first 25 steps (kinematic model of src/rollout/scripted.py).
  follow   = <motion(variant) - motion(base), d> / |d|^2 with d the displacement of the thing that moved (1: the motion
             follows it fully, 0: it ignores it); for "state" d is minus the proprio offset (1: the policy believes the
             proprio over the images and corrects for a hand that is not where it sees it); for "lang" d goes from the
             target to the newly named object.
    python scripts/probe_mechanism.py --steps outputs/week1/pi05_object_steps --suite libero_object \
        --ckpt pi05:<checkpoint> [--lora <adapter>] --out outputs/week1/mech_object_base
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def build_jobs(steps, suite_name, episodes_per_task, states_per_episode, deltas, seed):
    """The probe queries: for logged states of successful episodes, the unchanged state ("base") and its variants (see
    the module docstring). Returns (jobs, chosen episodes); a job carries the simulator state to restore, the proprio
    offset to apply, the instruction, the displacement d of what moved and the state's phase / distances."""
    from analyze_failures import task_layout  # sibling script
    from libero.libero import benchmark

    rng = np.random.default_rng(seed)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    eps = [json.loads(l) for l in open(os.path.join(steps, "episodes.jsonl"))]
    by_task = {}
    for e in eps:
        if e["success"] and os.path.exists(os.path.join(steps, "steps", f"t{e['task_id']:02d}_n{e['trial_id']:02d}.npz")):
            by_task.setdefault(e["task_id"], []).append(e)
    chosen = [lst[i] for t, lst in sorted(by_task.items()) for i in rng.permutation(len(lst))[: episodes_per_task]]
    phrase = lambda n: str(n).rsplit("_", 1)[0].replace("_", " ")  # noqa: E731  "alphabet_soup_1" -> "alphabet soup"
    deltas = [float(x) for x in deltas.split(",")]
    dirs = [np.array(v, dtype=np.float64) for v in ((1, 0), (-1, 0), (0, 1), (0, -1))]

    jobs, layouts = [], {}  # one job = (state, variant): sim state to restore, obs edit, instruction, displacement
    for e in chosen:
        t = e["task_id"]
        if t not in layouts:
            layouts[t] = task_layout(suite, t)
        addr, target, container = layouts[t]
        if target not in addr:
            continue
        z = np.load(os.path.join(steps, "steps", f"t{t:02d}_n{e['trial_id']:02d}.npz"))
        n = len(z["t"])
        close = z["actions"][:, :, -1].min(1) < 0.5  # [0, 1] convention: 0 = close
        first_close = int(np.argmax(close)) if close.any() else n
        for q in sorted({int(round(x)) for x in np.linspace(0, n - 2, states_per_episode)}):
            s0 = np.array(z["sim_state"][q], dtype=np.float64)
            cmd = -1.0 if q == 0 else -float(np.sign(2 * z["actions"][q - 1, -1, -1] - 1))
            xy = {nm: s0[1 + a : 1 + a + 2].copy() for nm, a in addr.items()}
            eef = z["eef_pos"][q].astype(np.float64)
            pre = q < first_close
            base = dict(task_id=t, trial_id=e["trial_id"], task=e["task"], q=q, n_queries=n, phase="pre" if pre else "post",
                        t0=int(z["t"][q]), cmd=cmd, dist_target_cm=round(100 * float(np.linalg.norm(xy[target] - eef[:2])), 1),
                        dist_container_cm=round(100 * float(np.linalg.norm(xy[container] - eef[:2])), 1) if container in xy else "",
                        a_tgt=addr[target], a_con=addr.get(container, -1), eef_xy=eef[:2].copy())
            jobs.append(dict(base, variant="base", state=s0, d=np.zeros(2), lang=e["task"], obs_offset=None))
            moved = target if pre else (container if container in addr else None)
            if moved is not None:  # img_<m>: the thing the current phase is about, moved where nothing else stands
                lo = np.min(list(xy.values()), 0) - 0.1
                hi = np.max(list(xy.values()), 0) + 0.1
                for m in deltas:
                    for v in dirs:
                        new = xy[moved] + m * v
                        clear = all(np.linalg.norm(new - p) > (0.1 if moved == container else 0.07)
                                    for nm, p in xy.items() if nm not in (moved, target if not pre else None))
                        if (new < lo).any() or (new > hi).any() or not clear:
                            continue
                        s = s0.copy()
                        s[1 + addr[moved] : 1 + addr[moved] + 2] = new
                        jobs.append(dict(base, variant=f"img_{m:g}", state=s, d=m * v, lang=e["task"], obs_offset=None))
            if pre:
                others = [nm for nm in addr if nm not in (target, container)]
                if others:  # swap with the nearest other object
                    o = min(others, key=lambda nm: np.linalg.norm(xy[nm] - xy[target]))
                    s = s0.copy()
                    s[1 + addr[target] : 1 + addr[target] + 2], s[1 + addr[o] : 1 + addr[o] + 2] = xy[o], xy[target]
                    jobs.append(dict(base, variant="swap", state=s, d=xy[o] - xy[target], lang=e["task"], obs_offset=None))
                    if phrase(target) in e["task"] and phrase(o) != phrase(target):
                        jobs.append(dict(base, variant="lang", state=s0, d=xy[o] - xy[target],
                                         lang=e["task"].replace(phrase(target), phrase(o)), obs_offset=None))
            for v in dirs[::2]:  # proprio offset by 5 cm along x and along y; the images stay those of the true state
                off = np.array([0.05 * v[0], 0.05 * v[1], 0.0])
                jobs.append(dict(base, variant="state", state=s0, d=-off[:2], lang=e["task"], obs_offset=off))
    return jobs, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--geo", action="store_true", help="the policy has the 3D sub-goal point injection (Pi05Policy.enable_geo)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes_per_task", type=int, default=3)
    ap.add_argument("--states_per_episode", type=int, default=8)
    ap.add_argument("--deltas", default="0.03,0.06,0.12,0.2")
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from src.rollout.scripted import TRACK_GAIN
    from src.rollout.vec_env import LiberoVecEnv

    jobs, chosen = build_jobs(args.steps, args.suite, args.episodes_per_task, args.states_per_episode, args.deltas, args.seed)
    print(f"{len(chosen)} episodes, {len(jobs)} queries", flush=True)

    import torch

    from src.policy.rebin import load_policy

    torch.manual_seed(args.seed)
    vec = LiberoVecEnv(args.suite, args.num_envs, max_steps=600, wrist=True)
    policy = load_policy(args.ckpt, args.suite, "cuda:0")
    if args.geo:
        policy.enable_geo()
    if args.lora:
        policy.add_lora(adapter_path=args.lora)
        if args.geo:
            policy.load_geo(args.lora)

    def motion(chunk):  # where the first 25 steps of a chunk (env units) would bring the hand, horizontally (m)
        return (TRACK_GAIN[:2] * chunk[:25, :2] * 0.05).sum(0)

    rows, pending = [], []
    for k, job in enumerate(jobs):
        vec.restore(len(pending), job["task_id"], job["state"], job["t0"], job["cmd"])
        pending.append(job)
        if len(pending) == args.num_envs or k == len(jobs) - 1:
            obs = [vec.recv(i) for i in range(len(pending))]
            for o, j in zip(obs, pending):
                if j["obs_offset"] is not None:
                    o["eef_pos"] = (o["eef_pos"] + j["obs_offset"]).astype(np.float32)
            chunks = policy.chunk_env([o["rgb"] for o in obs], [j["lang"] for j in pending], obs)
            for j, c in zip(pending, chunks):
                mv = motion(c)
                rows.append({kk: vv for kk, vv in j.items() if kk not in ("state", "obs_offset", "d", "lang", "a_tgt", "a_con", "eef_xy")}
                            | {"dx": round(float(j["d"][0]), 3), "dy": round(float(j["d"][1]), 3),
                               "move_x": round(float(mv[0]), 4), "move_y": round(float(mv[1]), 4),
                               "close_in_chunk": int((c[:25, -1] > 0).any())})
            pending = []
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(jobs)} queries", flush=True)
    vec.close()

    base = {(r["task_id"], r["trial_id"], r["q"]): r for r in rows if r["variant"] == "base"}
    for r in rows:
        b, d = base[(r["task_id"], r["trial_id"], r["q"])], np.array([r["dx"], r["dy"]])
        if r["variant"] == "base" or not np.linalg.norm(d):
            r["follow"] = ""
            continue
        dm = np.array([r["move_x"] - b["move_x"], r["move_y"] - b["move_y"]])
        r["follow"] = round(float(dm @ d / (d @ d)), 3)
    with open(os.path.join(args.out, "states.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def stat(sel):
        v = np.array([r["follow"] for r in rows if r["follow"] != "" and sel(r)], dtype=float)
        return {"n": int(len(v)), "median": round(float(np.median(v)), 2), "mean": round(float(v.mean()), 2)} if len(v) else None

    bins = {"far (>15 cm)": lambda r: r["dist_target_cm"] > 15, "mid (5-15 cm)": lambda r: 5 <= r["dist_target_cm"] <= 15,
            "near (<5 cm)": lambda r: r["dist_target_cm"] < 5}
    variants = sorted({r["variant"] for r in rows} - {"base"})
    summary = {"pre-grasp, by distance hand-target": {b: {v: stat(lambda r, b=b, v=v: r["phase"] == "pre" and bins[b](r) and r["variant"] == v)
                                                          for v in variants} for b in bins},
               "carrying": {v: stat(lambda r, v=v: r["phase"] == "post" and r["variant"] == v) for v in variants},
               "config": vars(args)}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    for part in ("pre-grasp, by distance hand-target", "carrying"):
        print(part)
        if part == "carrying":
            print("  " + "  ".join(f"{v}: {s['median'] if s else '-'} (n {s['n'] if s else 0})" for v, s in summary[part].items()))
        else:
            for b, d in summary[part].items():
                print(f"  {b:14s} " + "  ".join(f"{v}: {s['median'] if s else '-'} (n {s['n'] if s else 0})" for v, s in d.items()))
    print("PROBE_MECHANISM_DONE", flush=True)


if __name__ == "__main__":
    main()
