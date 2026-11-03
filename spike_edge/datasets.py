"""Deterministic detection datasets. Train/val/test differ in seed, so every
template, spike train, noise draw and drift profile is disjoint."""
import numpy as np

from . import FS, PRE, POST
from .frontend import frontend_int, robust_sigma
from .synth import make_channel

CAL_S = 10                      # unsupervised calibration prefix (s)
SNR_LEVELS = [2, 3, 4, 5, 6, 8]


def calibrate(y):
    return robust_sigma(y[FS // 10: FS // 10 + CAL_S * FS])


def channel(seed, dur_s, snrs=None, snr_range=(1.5, 10.0), n_units=(1, 3)):
    rng = np.random.default_rng(seed)
    if snrs is None:
        k = rng.integers(n_units[0], n_units[1] + 1)
        snrs = rng.uniform(*snr_range, k)
    ch, _ = make_channel(rng, dur_s, list(snrs))
    y = frontend_int(ch.raw)
    gt = np.sort(np.concatenate(ch.gt_times)) if ch.gt_times else np.zeros(0, np.int64)
    return dict(raw=ch.raw, y=y, sigma=calibrate(y), gt=gt, snrs=np.asarray(snrs))


def split(name, i):
    """Channel i of split `name`. Test splits are 'test_<level>'; robustness splits
    are 'noise' (no ground-truth unit at all), 'drift' (300 s, SNR 5, so the
    60-300 s drift period actually plays out after the 10 s calibration) and
    'artifact' (test_5 channels with injected stimulation/saturation artifacts)."""
    if name == "train":
        return channel(1_000_000 + i, 60)
    if name == "val":
        return channel(2_000_000 + i, 60)
    if name == "noise":
        return channel(5_000_000 + i, 40, snrs=[])
    if name == "drift":
        rng = np.random.default_rng(6_000_000 + i)
        return channel(6_000_500 + i, 300, snrs=5 * rng.uniform(0.9, 1.1, rng.integers(1, 3)))
    if name == "artifact":
        c = split("test_5", i)
        c["raw"], c["artifacts"] = inject_artifacts(c["raw"], np.random.default_rng(7_000_000 + i))
        c["y"] = frontend_int(c["raw"]); c["sigma"] = calibrate(c["y"])
        return c
    lvl = int(name.split("_")[1])
    rng = np.random.default_rng(3_000_000 + 1000 * lvl + i)
    snrs = lvl * rng.uniform(0.9, 1.1, rng.integers(1, 3))
    return channel(3_000_000 + 1000 * lvl + i + 500, 40, snrs=snrs)


N_CH = {"train": 64, "val": 16, "test": 32, "noise": 32, "drift": 8, "artifact": 32}

ART_RATE_HZ = 0.5


def inject_artifacts(raw, rng):
    """Poisson (0.5/s, after calibration) artifacts: half are a biphasic 0.5+0.5 ms
    +-300 uV stimulation pulse, half a 2 ms rail saturation at -512. Returns raw, onsets."""
    raw = raw.astype(np.int32).copy()
    n = len(raw); lo = CAL_S * FS
    k = rng.poisson(ART_RATE_HZ * (n - lo) / FS)
    on = np.sort(rng.integers(lo, n - 200, k))
    for t, kind in zip(on, rng.integers(0, 2, k)):
        if kind == 0:
            raw[t:t + 10] += 300; raw[t + 10:t + 20] -= 300
        else:
            raw[t:t + 40] = -512
    return np.clip(raw, -512, 511).astype(np.int16), on


def windows(y, idx):
    ok = (idx >= PRE) & (idx + POST <= len(y))
    idx = idx[ok]
    w = y[idx[:, None] + np.arange(-PRE, POST)[None, :]]
    return idx, w
