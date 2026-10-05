"""Takeover test: hand states visited by one policy to another policy and see whether it finishes the task.

For every sampled state of a logged rollout (steps/*.npz hold the simulator state of each query), the env is
restored to that state at its original step count and `--policy` acts greedily until success or the horizon.
This measures, state by state, whether a policy can still solve the task from where the logged policy took it:
with the teacher it tells how good the teacher's corrections are on student-visited states; with two students
it gives the per-state effect of a training change.
    python scripts/takeover_eval.py --run outputs/week1/b0_object_student --policy <ckpt> --name teacher --out <dir>
"""
import argparse
import glob
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="rollout directory with episodes.jsonl and steps/")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--lora", default="", help="optional LoRA adapter directory for --policy")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--fracs", default="0,0.25,0.5,0.75", help="hand-over points as fractions of the logged episode")
    ap.add_argument("--episodes_per_cell", type=int, default=6, help="logged episodes per (task, outcome) cell")
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from src.rollout.vec_env import LiberoVecEnv, mem_available_gb, postprocess_actions

    vec = LiberoVecEnv(args.suite, args.num_envs, args.max_steps)
    from src.policy.token_policy import TokenPolicy

    policy = TokenPolicy(args.policy, args.suite)
    if args.lora:
        policy.add_lora(adapter_path=args.lora)
    tasks = {i: lang for i, lang, _ in vec.task_info()}
    rng = np.random.default_rng(args.seed)
    fracs = [float(x) for x in args.fracs.split(",")]

    # stratified sample: for each task and logged outcome, a few episodes, each handed over at every fraction
    eps = [json.loads(l) for l in open(os.path.join(args.run, "episodes.jsonl"))]
    jobs = []
    for t in sorted(tasks):
        for outcome in (True, False):
            cell = [e for e in eps if e["task_id"] == t and e["success"] == outcome]
            for k in rng.permutation(len(cell))[: args.episodes_per_cell]:
                e = cell[k]
                z = np.load(os.path.join(args.run, "steps", f"t{t:02d}_n{e['trial_id']:02d}.npz"))
                n = len(z["t"])
                for f in fracs:
                    q = min(int(round(f * (n - 1))), n - 1)
                    # last gripper command executed before this state: the wait steps open (-1); afterwards the
                    # policy's gripper output g in [0, 1] (1 = open) was sent to the env as -sign(2g - 1)
                    cmd = -1.0 if q == 0 else float(-np.sign(2 * z["actions"][q - 1][-1, -1] - 1))
                    jobs.append({"task_id": t, "trial_id": e["trial_id"], "query": q, "frac": f, "logged_success": outcome,
                                 "t0": int(z["t"][q]), "state": z["sim_state"][q], "gripper_cmd": cmd})
    out_path = os.path.join(args.out, f"takeover_{args.name}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["task_id"], r["trial_id"], r["query"]) for r in map(json.loads, open(out_path))}
    jobs = [j for j in jobs if (j["task_id"], j["trial_id"], j["query"]) not in done]
    by_task = {}
    for j in jobs:
        by_task.setdefault(j["task_id"], []).append(j)
    print(f"{len(jobs)} takeovers to run ({len(done)} already done)", flush=True)

    slots, last, t_start, n_done = [None] * args.num_envs, [None] * args.num_envs, time.time(), 0

    def start(i):
        if not by_task:
            slots[i] = None
            return
        t = last[i] if last[i] in by_task else max(by_task, key=lambda k: len(by_task[k]))
        job = by_task[t].pop()
        if not by_task[t]:
            del by_task[t]
        last[i] = t
        vec.restore(i, t, job["state"], job["t0"], job["gripper_cmd"])
        slots[i] = {"job": job, "obs": None}

    for i in range(args.num_envs):
        start(i)
    rounds = 0
    with open(out_path, "a") as fout:
        while any(s is not None for s in slots):
            rounds += 1
            if rounds % 20 == 0 and mem_available_gb() < 4:
                raise MemoryError("low RAM on the machine, aborting (finished takeovers are saved)")
            for i, s in enumerate(slots):
                if s is not None and s["obs"] is None:
                    if vec.ready(i) or not any(x is not None and x["obs"] is not None for x in slots):
                        s["obs"] = vec.recv(i)
            act = [i for i, s in enumerate(slots) if s is not None and s["obs"] is not None]
            out = policy.act([slots[i]["obs"]["rgb"] for i in act], [tasks[slots[i]["job"]["task_id"]] for i in act])
            env_actions = postprocess_actions(out["actions"])
            for k, i in enumerate(act):
                vec.step(i, env_actions[k])
            for i in act:
                obs = vec.recv(i)
                slots[i]["obs"] = obs
                if not obs["active"]:
                    j = slots[i]["job"]
                    rec = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in j.items() if k != "state"}
                    rec.update(success=bool(obs["done"]), steps_after=int(obs["t"]) - j["t0"])
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    n_done += 1
                    if n_done % 20 == 0:
                        print(f"[{time.time() - t_start:6.0f}s] {n_done} takeovers done", flush=True)
                    start(i)
    vec.close()

    recs = [json.loads(l) for l in open(out_path)]
    table = {}
    for outcome in (True, False):
        for f in fracs:
            sel = [r["success"] for r in recs if r["logged_success"] == outcome and abs(r["frac"] - f) < 1e-9]
            table[f"logged_{'success' if outcome else 'failure'}@{f}"] = {"n": len(sel), "takeover_success": float(np.mean(sel)) if sel else None}
    json.dump({"policy": args.policy, "lora": args.lora, "run": args.run, "table": table},
              open(os.path.join(args.out, f"takeover_{args.name}.json"), "w"), indent=1)
    print(json.dumps(table, indent=1))
    print("TAKEOVER_DONE", flush=True)


if __name__ == "__main__":
    main()
