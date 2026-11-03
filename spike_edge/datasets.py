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
    """Channel i of split `name`. Test splits are 'test_<level>'."""
    if name == "train":
        return channel(1_000_000 + i, 60)
    if name == "val":
        return channel(2_000_000 + i, 60)
    lvl = int(name.split("_")[1])
    rng = np.random.default_rng(3_000_000 + 1000 * lvl + i)
    snrs = lvl * rng.uniform(0.9, 1.1, rng.integers(1, 3))
    return channel(3_000_000 + 1000 * lvl + i + 500, 40, snrs=snrs)


N_CH = {"train": 64, "val": 16, "test": 32}


def windows(y, idx):
    ok = (idx >= PRE) & (idx + POST <= len(y))
    idx = idx[ok]
    w = y[idx[:, None] + np.arange(-PRE, POST)[None, :]]
    return idx, w
