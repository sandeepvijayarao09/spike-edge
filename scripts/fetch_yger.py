"""Fetch Yger et al. 2018 paired ground truth (Zenodo 1205233, CC-BY-4.0) into data/yger2018_<rec>/.

Resumable (curl -C -), verifies each archive's byte count and MD5 against the Zenodo listing,
backs off when Zenodo rate-limits (it serves an HTML page instead of the archive), extracts,
deletes the archive and writes a `.complete` marker that scripts/05_real_data.py requires.

Usage: python -I scripts/fetch_yger.py [-j 3] [recording ...]
"""
import argparse, hashlib, json, random, subprocess, tarfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
API = "https://zenodo.org/api/records/1205233"


def listing():
    for attempt in range(30):
        try:
            with urllib.request.urlopen(API, timeout=60) as r:
                files = json.load(r)["files"]
            return {f["key"][:-7]: (f["size"], f["checksum"].split(":")[-1], f["links"]["self"])
                    for f in files if f["key"].endswith(".tar.gz")}
        except Exception as e:              # rate-limited or offline: back off
            print("listing failed:", e, flush=True); time.sleep(30 + random.random() * 60)
    raise SystemExit("could not read the Zenodo listing")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch(rec, size, digest, url):
    d = DATA / f"yger2018_{rec}"
    if (d / ".complete").exists():
        return f"have {rec}"
    d.mkdir(parents=True, exist_ok=True)
    t = d / f"{rec}.tar.gz"
    for attempt in range(40):
        if t.exists() and t.read_bytes()[:15].startswith(b"<!DOCTYPE"):
            t.unlink()                      # rate-limit page, not data
        if t.exists() and t.stat().st_size == size:
            break
        subprocess.run(["curl", "-sL", "-C", "-", "--retry", "3", "-o", str(t), url])
        if t.exists() and t.stat().st_size == size:
            break
        time.sleep(30 + random.random() * 60)
    if not t.exists() or t.stat().st_size != size:
        return f"FAILED {rec}: size"
    if md5(t) != digest:
        t.unlink()
        return f"FAILED {rec}: md5 mismatch (archive deleted, re-run to fetch again)"
    for p in d.iterdir():                   # drop any partial extraction from earlier attempts
        if p != t:
            p.unlink()
    with tarfile.open(t) as tf:
        tf.extractall(d, filter="data")
    t.unlink()
    (d / ".complete").write_text(f"{size} bytes, md5 {digest}\n")
    return f"done {rec}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=3)
    ap.add_argument("recs", nargs="*")
    a = ap.parse_args()
    files = listing()
    recs = a.recs or sorted(files)
    with ThreadPoolExecutor(a.j) as ex:
        for msg in ex.map(lambda r: fetch(r, *files[r]), recs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
