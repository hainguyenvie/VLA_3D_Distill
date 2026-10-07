"""Spatial swap: where does a lost_in_transit episode take the bowl? distance of the bowl's lowest point after lifting
(and at the end) to the plate's current spot and to the plate's spot in the standard layout."""
import csv, os, sys, numpy as np
from libero.libero import benchmark
from scripts.analyze_failures import task_layout
bm = benchmark.get_benchmark_dict()
cell, base = bm["libero_spatial_swap"](), bm["libero_spatial"]()
run = sys.argv[1]
rows = [r for r in csv.DictReader(open(os.path.join(run, "failures.csv"))) if r["mode"] == "lost_in_transit"]
out = []
for r in rows:
    t, n = int(r["task_id"]), int(r["trial_id"])
    addr, target, cont = task_layout(cell, t)
    z = np.load(os.path.join(run, "steps", f"t{t:02d}_n{n:02d}.npz"), allow_pickle=True)
    S = np.concatenate([z["sim_state"], z["final_sim_state"][None]])[:, 1:]
    bowl, plate = S[:, addr[target]:addr[target] + 3], S[:, addr[cont]:addr[cont] + 2]
    nominal_plate = np.asarray(base.get_task_init_states(t))[:, 1 + addr[cont]:1 + addr[cont] + 2].mean(0)
    lifted = np.flatnonzero(bowl[:, 2] - bowl[0, 2] > 0.03)
    if not len(lifted):
        continue
    after = bowl[lifted[0]:]
    k = lifted[0] + int(np.argmin(after[:, 2]))  # where it came down after the lift
    out.append((np.linalg.norm(bowl[k, :2] - plate[k]), np.linalg.norm(bowl[k, :2] - nominal_plate), np.linalg.norm(plate[0] - nominal_plate),
                np.linalg.norm(bowl[-1, :2] - plate[-1]), np.linalg.norm(bowl[-1, :2] - nominal_plate)))
a = np.array(out)
print(f"{len(a)} lost_in_transit with a lift; plate moved {np.median(a[:,2])*100:.0f} cm (median)")
print(f"lowest point after lift: to plate now {np.median(a[:,0])*100:.0f} cm, to plate's usual spot {np.median(a[:,1])*100:.0f} cm; "
      f"closer to the usual spot in {np.mean(a[:,1] < a[:,0]):.2f}")
print(f"end of episode: to plate now {np.median(a[:,3])*100:.0f} cm, to usual spot {np.median(a[:,4])*100:.0f} cm; closer to usual spot {np.mean(a[:,4] < a[:,3]):.2f}")
