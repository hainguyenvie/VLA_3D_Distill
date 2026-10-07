"""Per task of the LIBERO-PRO Object swap cell: the target's spot in the standard layout and in the cell, the object now
standing at the standard spot (the lure), and the per-task success of an evaluated run.
    cd repo; LIBERO_VARIANT=pro python ../scripts/analyze_target_slots.py   (run from the workspace, see the paths)
"""
import numpy as np, json
from libero.libero import benchmark
from scripts.analyze_failures import task_layout
bm = benchmark.get_benchmark_dict()
base, cell = bm["libero_object"](), bm["libero_object_swap"]()
sr = json.load(open("../outputs/week1/p05_ocd_rotate_s8_iter0008_pro_swap/summary.json"))["per_task"]
for t in range(10):
    addr, target, cont = task_layout(cell, t)
    s0 = np.asarray(base.get_task_init_states(t))[:, 1:]
    s1 = np.asarray(cell.get_task_init_states(t))[:, 1:]
    p0 = s0[:, addr[target]:addr[target] + 2].mean(0); p1 = s1[:, addr[target]:addr[target] + 2].mean(0)
    other = min((np.linalg.norm(s1[:, a:a + 2].mean(0) - p0), n) for n, a in addr.items() if n != target)[1]
    # how many tasks have their target at the new spot in the standard layout? (is the new spot a 'target slot'?)
    print(f"task {t} {target[:-2]:<18} nominal ({p0[0]:+.2f},{p0[1]:+.2f}) -> swap ({p1[0]:+.2f},{p1[1]:+.2f})  lure at nominal: {other[:-2]:<18} rotate_s8 {sr[str(t)]['success_rate']:.2f}")
