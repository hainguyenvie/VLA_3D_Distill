"""Contact sheet of evaluated episodes (eval_libero.py without --no_steps): one row per episode, 8 agent-view frames
evenly spaced over its queries; failed episodes first, then one success. Also prints, per episode, the query at which the
gripper first closes and the target's height change (target_pos), to tell a failed grasp from a failed placement.
    python scripts/montage_episodes.py <eval out dir> <out.png> [n failed episodes, default 4]
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def main():
    d, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    eps = [json.loads(l) for l in open(os.path.join(d, "episodes.jsonl")) if l.strip()]
    sel = [e for e in eps if not e["success"]][:n] + [e for e in eps if e["success"]][:1]
    rows = []
    for e in sel:
        z = np.load(os.path.join(d, "steps", f"t{e['task_id']:02d}_n{e['trial_id']:02d}.npz"))
        rgb, nq = z["rgb"], len(z["rgb"])
        qs = np.linspace(0, nq - 1, 8).astype(int)
        rows.append(np.concatenate([rgb[q] for q in qs], axis=1))
        close = np.flatnonzero((z["actions"][:, :, -1] < 0.5).any(1))
        tz = z["target_pos"][:, 2] - z["target_pos"][0, 2] if "target_pos" in z.files else np.zeros(nq)
        print(f"task {e['task_id']} trial {e['trial_id']:2d} success {int(e['success'])} queries {nq:2d} frames {qs.tolist()} "
              f"first close q{close[0] if len(close) else -1} max target rise {tz.max():+.3f} final {tz[-1]:+.3f}")
    img = np.concatenate(rows, axis=0)
    Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2)).save(out)
    print("saved", out, img.shape)


if __name__ == "__main__":
    main()
