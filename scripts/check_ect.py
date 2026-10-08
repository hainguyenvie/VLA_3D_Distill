"""Gate for the ECT baseline (src/rollout/vec_env.py `EnvRunner.ect_replay`, arXiv 2609.39971): on successful episodes of
the base policy (a steps directory written by scripts/eval_libero.py), the share of transformed replays that succeed, per
task and transform. The "identity" transform is the control: the tracking controller following the episode's own path in
the unchanged scene must succeed (nearly) always. The first transformed frame of one episode per task and transform is
saved for inspection.
    LIBERO_VARIANT=pro python scripts/check_ect.py --suite libero_spatial --steps outputs/week1/pi05_spatial_std20_steps \
        --transforms identity,ymirror --per_task 5 --out outputs/week1/check_ect_spatial
"""
import argparse
import contextlib
import glob
import json
import os

import numpy as np

from src.rollout.vec_env import ECT_TRANSFORMS as TRANSFORMS
from src.rollout.vec_env import EnvRunner, postprocess_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--transforms", default="identity,ymirror")
    ap.add_argument("--per_task", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    succ = {(e["task_id"], e["trial_id"]): e["success"] for e in map(json.loads, open(os.path.join(args.steps, "episodes.jsonl")))}
    by_task = {}
    for f in sorted(glob.glob(os.path.join(args.steps, "steps", "t*_n*.npz"))):
        t, n = int(os.path.basename(f)[1:3]), int(os.path.basename(f)[5:7])
        if succ.get((t, n)) and len(by_task.setdefault(t, [])) < args.per_task:
            by_task[t].append((n, f))
    names = args.transforms.split(",")
    res = {}
    for t in sorted(by_task):
        task = suite.get_task(t)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        cfg = dict(suite=args.suite, max_steps=600, num_steps_wait=10, resolution=256, depth=False, wrist=True, perturb=None,
                   counterfactual=None)
        r = EnvRunner(bddl, cfg, contextlib.nullcontext())
        inits = suite.get_task_init_states(t)
        for k, (n, f) in enumerate(by_task[t]):
            z = np.load(f, allow_pickle=True)
            acts = postprocess_actions(z["actions"]).reshape(-1, 7)
            for name in names:
                out = r.ect_replay(inits[n], acts, z["t"], TRANSFORMS[name])
                res.setdefault((t, name), []).append(bool(out["success"]))
                if k == 0 and "rgb" in out:
                    Image.fromarray(out["rgb"][0][::-1, ::-1]).save(os.path.join(args.out, f"t{t:02d}_{name}.png"))
                print(f"  t{t:02d}_n{n:02d} {name:15s} orig {out['orig_success']} replay {out['success']} "
                      f"steps {out.get('n_orig')} -> {out.get('n_replay')}", flush=True)
        r.close()
    print("task " + " ".join(f"{nm:>15s}" for nm in names))
    for t in sorted(by_task):
        print(f"{t:4d} " + " ".join(f"{np.mean(res[(t, nm)]):15.2f}" for nm in names))
    print("all  " + " ".join(f"{np.mean([x for (tt, nm2), v in res.items() if nm2 == nm for x in v]):15.2f}" for nm in names))


if __name__ == "__main__":
    main()
