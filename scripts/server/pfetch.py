#!/usr/bin/env python3
"""Parallel, resumable range download: pfetch.py <url> <out> [connections=16]

PyPI / GitHub are throttled per connection from this machine (30-100 KB/s) and connections reset often, so
the file is split into small chunks pulled by a pool of connections; a chunk that stalls or breaks is
resumed from the bytes it already has. Stdlib + curl only.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CHUNK = 2 << 20


def content_length(url):
    out = subprocess.run(["curl", "-sIL", "--max-time", "60", url], capture_output=True, text=True).stdout
    sizes = [l.split(":", 1)[1].strip() for l in out.splitlines() if l.lower().startswith("content-length:")]
    return int(sizes[-1]) if sizes else None


def fetch(url, out, conns=16):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    size = None
    for _ in range(5):
        size = content_length(url)
        if size:
            break
    if not size:
        raise RuntimeError(f"no content-length for {url}")
    parts = out + ".parts"
    os.makedirs(parts, exist_ok=True)
    chunks = [(i, s, min(s + CHUNK, size) - 1) for i, s in enumerate(range(0, size, CHUNK))]

    def get(c):
        i, s, e = c
        path, want = os.path.join(parts, f"{i:06d}"), e - s + 1
        for _ in range(200):
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have == want:
                return
            if have > want:
                os.remove(path)
                have = 0
            with open(path, "ab") as f:  # abort a connection that stalls below 2 KB/s for 20 s, then resume
                subprocess.run(["curl", "-sL", "--speed-limit", "2000", "--speed-time", "20", "-r", f"{s + have}-{e}", url], stdout=f)
        raise RuntimeError(f"chunk {i} of {url} did not complete")

    with ThreadPoolExecutor(conns) as ex:
        list(ex.map(get, chunks))
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        for i, _, _ in chunks:
            with open(os.path.join(parts, f"{i:06d}"), "rb") as p:
                f.write(p.read())
    assert os.path.getsize(tmp) == size, f"size mismatch for {out}"
    os.replace(tmp, out)
    for i, _, _ in chunks:
        os.remove(os.path.join(parts, f"{i:06d}"))
    os.rmdir(parts)
    return out


if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    print(f"fetched {sys.argv[2]}")
