"""Fetch cortical paired recordings (Marques-Smith et al. 2018, bioRxiv 10.1101/370080;
Google Drive "Neuropixel Paired Recordings", licence CC BY-SA 4.0).

Only cells whose juxtacellular-triggered peak-to-peak amplitude is >= 50 uV (Data Summary)
are fetched: below that the cell is not detectable on a single channel. Cells whose raw
file exceeds ~20 GB (c14, c21, c26) are skipped to bound the download.

Each 384-channel int16 30 kHz raw file is streamed through and only the 64 channels around
the summary's peak channel are kept (data/marques_smith/<cell>/<cell>_npx_sub.npy,
shape (n_samples, 64)), so the 3-19 GB raw files never touch the disk.

Usage: python -I scripts/fetch_marques_smith.py [cell ...]
"""
import json, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "marques_smith"
N_CH, KEEP = 384, 64
# Drive file IDs (from the public folder listing) and the summary's chan_highest
CELLS = {
    "c19": dict(chan=183, meta="1D4L959cDVicNzRxEPiFgy6G7VdZd9JS3", spikes="1V-mCs8hf1_1Z6Bndc2Oo-xavrNCAJ13k", raw="1mTaMr2MoL_UITBr09WSYQDo-lqWHkNWA"),
    "c24": dict(chan=218, meta="1A-yOLskumvBkRqA29EG-fzuOOqsVDzx7", spikes="1A4bC2WJTnP85CUOY53sSmaAh2oTKmWMw", raw="1d3m9kv-yFcBXloUrASNDcc9OE7XJu1Ce"),
    "c27": dict(chan=229, meta="1vf2jF_VHYGO8CvmvDmuYGquDOnIhgC-t", spikes="1xpKg560CS1SXHtAC9f2DeTn4g0God49V", raw="1SvMqlgKoGkjBhA5U-p9-OlaNzF-hHez6"),
    "c28": dict(chan=234, meta="1Wmibh7PIOaOzA0kFbLyM0DavjpW-LinW", spikes="1R9inagHodO6xSLteLbIGY6r6czT87bu_", raw="16OjfH5qOcxiBnfgRlME0IfjVWC7huIFN"),
    "c37": dict(chan=194, meta="14n7xukALcOJQkMGvbOZalNn1D0mMxNTH", spikes="1Yg2L6oivbeaS8mG6XRgYSe_owzqpzrML", raw="1RfWbm-yagZT7ckIcVMQMzwTL68ap0IvK"),
    "c45": dict(chan=178, meta="11_SivMzzw7Z5Q-Ec6Dl_m6nJweHH8yAz", spikes="1aBlKG_iSJf2j7eKEBFok4RlEIwNAhWGP", raw="1ZPVV2ypyLq9kf_kM-MlMM6nbCE3Yhq98"),
    "c46": dict(chan=181, meta="1Ph_CG3RwaWwmCg38wopdn1db71mM75GV", spikes="1vUMQYstvArm_lcE0ewNaq5Zm3JOLCc0e", raw="1lWjNaxEWPzOklxEv3JxvBPFYCzf221ad"),
}
SKIPPED_OVER_20GB = {"c14": 24.9, "c21": 24.9, "c26": 28.0}   # GB, 1.38 GB per minute of recording


def url(fid):
    return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"


def fetch_small(fid, path):
    subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(path), url(fid)], check=True)


def fetch_cell(name, spec, layout="interleaved"):
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    sub = d / f"{name}_npx_sub.npy"
    if sub.exists():
        print(name, "already extracted"); return
    fetch_small(spec["meta"], d / f"{name}_expt_meta.csv")
    fetch_small(spec["spikes"], d / f"{name}_extracellular_spikes.npy")
    lo = int(np.clip(spec["chan"] - KEEP // 2, 0, N_CH - KEEP))
    local = d / f"{name}_npx_raw.bin"
    if local.exists():                          # a full raw file fetched earlier
        x = np.memmap(local, np.int16, "r").reshape(-1, N_CH)[:, lo:lo + KEEP]
        np.save(sub, np.ascontiguousarray(x))
        (d / f"{name}_npx_sub.json").write_text(json.dumps(dict(first_channel=lo, n_channels=KEEP, n_samples=len(x), fs=30_000, layout=layout)))
        print(name, x.shape, "from local raw", flush=True)
        return
    proc = subprocess.Popen(["curl", "-sL", "--retry", "3", url(spec["raw"])], stdout=subprocess.PIPE)
    frame = N_CH * 2
    chunk = frame * 30_000                      # 1 s
    parts, buf, total = [], b"", 0
    while True:
        b = proc.stdout.read(chunk)
        if not b:
            break
        buf += b
        n = len(buf) // frame
        if n:
            a = np.frombuffer(buf[: n * frame], np.int16).reshape(n, N_CH)
            parts.append(a[:, lo:lo + KEEP].copy())
            buf = buf[n * frame:]
            total += n
    if proc.wait() != 0 or buf:
        raise RuntimeError(f"{name}: download failed or truncated ({len(buf)} stray bytes)")
    x = np.concatenate(parts)
    np.save(sub, x)
    (d / f"{name}_npx_sub.json").write_text(json.dumps(dict(first_channel=lo, n_channels=KEEP, n_samples=int(total),
                                                            fs=30_000, layout=layout)))
    print(name, x.shape, f"channels {lo}-{lo + KEEP - 1}", flush=True)


if __name__ == "__main__":
    failed = []
    for c in (sys.argv[1:] or list(CELLS)):
        for attempt in range(3):   # a stream cannot resume; retry from scratch
            try:
                fetch_cell(c, CELLS[c]); break
            except RuntimeError as e:
                print(f"{c}: attempt {attempt + 1} failed: {e}", flush=True)
        else:
            failed.append(c)
    print("failed:", failed)
