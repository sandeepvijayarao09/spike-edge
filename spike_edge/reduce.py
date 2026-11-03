"""On-implant data-reduction schemes and their bit rates (per channel)."""
import numpy as np

from . import FS, ADC_BITS

BIN = FS // 50   # 20 ms bins -> 50 Hz


def bin_counts(events, n_samples):
    nb = n_samples // BIN
    return np.bincount(np.asarray(events) // BIN, minlength=nb)[:nb]


def sbp_int(y, shift):
    """Spike-band power: sum |y| per bin, >> shift, clamp to uint8 (2 ops/sample)."""
    nb = len(y) // BIN
    s = np.abs(y[:nb * BIN].astype(np.int64)).reshape(nb, BIN).sum(1)
    return np.minimum(s >> shift, 255)


def sbp_shift(y_cal):
    """Smallest shift putting the calibration median of the bin sum below 128."""
    med = int(np.median(np.abs(y_cal[:len(y_cal) // BIN * BIN].astype(np.int64)).reshape(-1, BIN).sum(1)))
    s = 0
    while (med >> s) >= 128:
        s += 1
    return s


def entropy_bits(symbols):
    """Zeroth-order empirical entropy (bits/symbol) = ideal static entropy-coder rate."""
    _, c = np.unique(np.asarray(symbols).ravel(), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def fixed_bits(symbols):
    return int(np.ceil(np.log2(int(np.max(symbols)) + 1))) if np.max(symbols) > 0 else 1


def raw_rates(raw):
    """Uncompressed 10-bit rate and a lossless-compression estimate
    (entropy of the first difference, a cheap predictive coder)."""
    return dict(fixed=ADC_BITS * FS, lossless_est=entropy_bits(np.diff(raw.astype(np.int32))) * FS)
