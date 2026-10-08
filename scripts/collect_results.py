"""One line per pi0.5 training run of outputs/week1: success rates (%) of its final-adapter evaluations, as written by
scripts/server/pi05_arm.sh (standard suite, LIBERO-PRO cells, LIBERO-Plus Robot init / Layout), and the training's last
logged iteration. Runs are training directories (with train_log.jsonl) whose name matches the prefix.
    python3 scripts/collect_results.py outputs/week1 [--prefix p05] [--csv]
"""
import argparse
import glob
import json
import os

CELLS = ("swap", "temp", "object", "lan", "task")


def rate(path):
    try:
        s = json.load(open(os.path.join(path, "summary.json")))
    except Exception:
        return None
    return round(100 * s["success_rate"], 1) if "success_rate" in s else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--prefix", default="p05")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, args.prefix + "*"))):
        log = os.path.join(d, "train_log.jsonl")
        if not os.path.isfile(log):
            continue
        run = os.path.basename(d)
        its = [json.loads(x) for x in open(log) if x.strip()]
        last = its[-1]["iter"] if its else 0
        std = next((rate(f"{d}_{s}") for s in ("object", "spatial", "goal", "10") if rate(f"{d}_{s}") is not None), None)
        cells = {c: rate(f"{d}_pro_{c}") for c in CELLS}
        plus = rate(os.path.join(args.root, f"plus_{run}_robotlayout"))
        if isinstance(plus, dict):
            plus = {k[:6]: round(100 * v["success_rate"]) for k, v in plus.get("by_category", {}).items()}
        rows.append((run, last, std, cells, plus))
    for run, last, std, cells, plus in rows:
        if args.csv:
            print(",".join([run, str(last), str(std)] + [str(cells[c]) for c in CELLS] + [json.dumps(plus).replace(",", ";")]))
        else:
            cs = " ".join(f"{c} {cells[c]}" for c in CELLS if cells[c] is not None)
            print(f"{run:34s} it {last:2d}  std {std}  {cs}  plus {plus}")


if __name__ == "__main__":
    main()
