"""Which objects the `unique` rule of the counterfactuals keeps in place, per task: the target of the retarget / coshift
counterfactuals and the place targets of relocate (BDDL goal on / in), each marked when it has a twin of the same kind in
the scene (the instruction can then only refer to it by where it is).
    python scripts/check_unique.py --suite libero_spatial
"""
import argparse
import contextlib
import os

from src.rollout.vec_env import COUNTERFACTUAL, EnvRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    args = ap.parse_args()
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[args.suite]()
    for t in range(suite.n_tasks):
        task = suite.get_task(t)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        cfg = dict(suite=args.suite, max_steps=600, num_steps_wait=10, resolution=64, depth=False, wrist=False, perturb=None,
                   counterfactual=dict(COUNTERFACTUAL, mode="rr", unique=True))
        r = EnvRunner(bddl, cfg, contextlib.nullcontext())
        r.reset(suite.get_task_init_states(t)[0], counterfactual=True)
        if not getattr(r, "_cs_ok", False):
            print(f"task {t}: no movable target | {task.language}")
            r.close()
            continue
        places = []
        for obj, pl in getattr(r, "_cs_place", {}).items():
            name = pl[3]
            places.append(f"{obj}->{name}{' [twin]' if r._ambiguous(name) else ''}")
        tgt = r._cs_name + (" [twin]" if r._ambiguous(r._cs_name) else "")
        print(f"task {t}: target {tgt}; place {', '.join(places) or '-'}; twins {sorted(r._cs_ambiguous)} | {task.language}")
        r.close()


if __name__ == "__main__":
    main()
