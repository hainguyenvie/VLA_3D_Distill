#!/usr/bin/env python3
"""Download public HF model repos into plain folders and verify every LFS file by sha256.

Stdlib only (runs with the system python, no env needed):
    python3 hf_fetch.py [--dataset] <out_root> <repo_id> [<repo_id> ...]
Each repo lands in <out_root>/<repo_id with '/' -> '__'>/ ; resumable; a file `.hf_fetch.json` with the
pinned revision and file list is written only after every file passed its size / sha256 check.
"""
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfetch  # noqa: E402

# Files above BIG bytes are fetched as CONNS parallel 64 MB ranges (env HF_CONNS; each connection is one curl,
# a few % of a core) while FILES files download at once (env HF_FILES): keep FILES x CONNS modest on a shared node.
BIG = 256 << 20
CONNS = int(os.environ.get("HF_CONNS", 8))
FILES = int(os.environ.get("HF_FILES", 4))
# `--dataset` switches the three endpoints to the dataset namespace
KIND = "datasets" if "--dataset" in sys.argv else "models"
API = "https://huggingface.co/api/" + KIND + "/{repo}"
TREE = "https://huggingface.co/api/" + KIND + "/{repo}/tree/{rev}?recursive=1"
RESOLVE = "https://huggingface.co/" + ("datasets/" if KIND == "datasets" else "") + "{repo}/resolve/{rev}/{path}"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(repo, rev, entry, dst_dir):
    dst = os.path.join(dst_dir, entry["path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    size = entry.get("lfs", {}).get("size", entry["size"])
    want = entry.get("lfs", {}).get("oid")
    for attempt in range(5):
        if not (os.path.exists(dst) and os.path.getsize(dst) == size):
            url = RESOLVE.format(repo=repo, rev=rev, path=entry["path"])
            if size > BIG and CONNS > 1:  # one connection tops out well below the link: split into parallel ranges
                try:
                    pfetch.fetch(url, dst, CONNS, chunk=64 << 20)
                except Exception as e:  # noqa: BLE001  (falls back to a single resumable stream below)
                    print(f"[warn] ranged fetch of {entry['path']} failed ({e}); single stream", flush=True)
            if not (os.path.exists(dst) and os.path.getsize(dst) == size):
                subprocess.run(["curl", "-sL", "--retry", "5", "-C", "-", "-o", dst, url], check=False)
        if os.path.exists(dst) and os.path.getsize(dst) == size and (want is None or sha256(dst) == want):
            return entry["path"], size, want
        if os.path.exists(dst) and os.path.getsize(dst) >= size:
            os.remove(dst)  # corrupt or oversized: restart this file
    raise RuntimeError(f"failed to fetch {repo}:{entry['path']}")


def fetch_repo(out_root, repo):
    dst_dir = os.path.join(out_root, repo.replace("/", "__"))
    marker = os.path.join(dst_dir, ".hf_fetch.json")
    if os.path.exists(marker):
        print(f"[skip] {repo} already complete", flush=True)
        return
    rev = get_json(API.format(repo=repo))["sha"]
    files = [e for e in get_json(TREE.format(repo=repo, rev=rev)) if e["type"] == "file"]
    total = sum(e.get("lfs", {}).get("size", e["size"]) for e in files)
    print(f"[start] {repo}@{rev[:8]}: {len(files)} files, {total / 1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(FILES) as ex:
        done = list(ex.map(lambda e: fetch_one(repo, rev, e, dst_dir), files))
    with open(marker + ".tmp", "w") as f:
        json.dump({"repo": repo, "revision": rev, "files": [{"path": p, "size": s, "sha256": h} for p, s, h in done]}, f, indent=1)
    os.replace(marker + ".tmp", marker)
    print(f"[done] {repo}@{rev[:8]} verified ({total / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dataset"]
    root, repos = args[0], args[1:]
    for r in repos:
        fetch_repo(root, r)
    print("FETCH_DONE", flush=True)
