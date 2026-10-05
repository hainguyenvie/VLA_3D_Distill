#!/usr/bin/env python3
"""Download every distribution named in a `pip install --dry-run --report` file with parallel range requests.

    python fetch_report.py <report.json> <wheel_dir> <pinned.txt>
PyPI is throttled per connection from this machine, so each file is fetched through pfetch.py (chunked
ranges) and several files run at once; every file is checked against the sha256 in the report.
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfetch import fetch as fetch_url  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(item, out_dir):
    url = item["download_info"]["url"]
    want = item["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256")
    dst = os.path.join(out_dir, url.split("/")[-1].split("#")[0])
    for _ in range(3):
        fetch_url(url, dst, 12)
        if want is None or sha256(dst) == want:
            return dst
        os.remove(dst)
    raise RuntimeError(f"failed to fetch {url}")


if __name__ == "__main__":
    report, out_dir, pinned = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    items = [i for i in json.load(open(report))["install"] if i["download_info"]["url"].startswith("http")]
    items.sort(key=lambda i: i["metadata"]["name"].lower())
    with ThreadPoolExecutor(4) as ex:
        paths = list(ex.map(lambda i: fetch(i, out_dir), items))
    with open(pinned, "w") as f:
        for i in items:
            f.write(f"{i['metadata']['name']}=={i['metadata']['version']}\n")
    print(f"fetched {len(paths)} files, {sum(os.path.getsize(p) for p in paths) / 1e6:.0f} MB", flush=True)
