import numpy as np
from numba import njit

from . import TOL


@njit(cache=True)
def _match(det, gt, tol):
    """Greedy one-to-one matching of two sorted index arrays within +-tol."""
    i = j = tp = 0
    used = np.zeros(len(gt), np.bool_)
    hit = np.zeros(len(det), np.bool_)
    for i in range(len(det)):
        while j < len(gt) and gt[j] < det[i] - tol:
            j += 1
        k = j
        while k < len(gt) and gt[k] <= det[i] + tol:
            if not used[k]:
                used[k] = True; hit[i] = True; tp += 1
                break
            k += 1
    return tp, hit, used


def match(det, gt, tol=TOL):
    det = np.sort(np.asarray(det, np.int64)); gt = np.sort(np.asarray(gt, np.int64))
    return _match(det, gt, tol)


def prf(tp, n_det, n_gt):
    p = tp / n_det if n_det else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def bootstrap_f1(counts, n_boot=2000, seed=0):
    """counts: array (n_channels, 3) of (tp, n_det, n_gt). Resample channels."""
    counts = np.asarray(counts, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), (n_boot, len(counts)))
    s = counts[idx].sum(1)
    p = s[:, 0] / np.maximum(s[:, 1], 1); r = s[:, 0] / np.maximum(s[:, 2], 1)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return np.percentile(f, [2.5, 97.5])
