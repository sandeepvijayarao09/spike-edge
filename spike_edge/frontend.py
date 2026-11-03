"""Integer front-end and event logic shared by every detector.

Front-end (all int32, Q8 state, ~8 add/shift ops per sample, no multiplies):
    xs  = x << 8
    s1 += (xs - s1) >> KS ; h1 = xs - s1          # 1st-order high-pass (~200 Hz)
    s2 += (h1 - s2) >> KS ; h2 = h1 - s2          # 2nd high-pass stage
    s3 += (h2 - s3) >> KF                         # 1st-order low-pass (~2.3 kHz)
    y   = s3 >> 6                                 # output in 0.25 uV units
`>>` is an arithmetic (floor) shift, identical in numpy/numba and in the C port.
"""
import numpy as np
from numba import njit

from . import SEARCH, DEAD

KS, KF = 4, 1


@njit(cache=True)
def frontend_int(x):
    n = len(x)
    y = np.empty(n, np.int32)
    s1 = np.int32(0); s2 = np.int32(0); s3 = np.int32(0)
    if n:
        s1 = np.int32(x[0]) << 8   # start settled on the first sample
    for i in range(n):
        xs = np.int32(x[i]) << 8
        s1 += (xs - s1) >> KS
        h1 = xs - s1
        s2 += (h1 - s2) >> KS
        h2 = h1 - s2
        s3 += (h2 - s3) >> KF
        y[i] = s3 >> 6
    return y


@njit(cache=True)
def frontend_float(x):
    """Same recurrences in float64 (no rounding), same output units."""
    n = len(x)
    y = np.empty(n)
    a, b = 1.0 / (1 << KS), 1.0 / (1 << KF)
    s1 = x[0] * 256.0 if n else 0.0
    s2 = 0.0; s3 = 0.0
    for i in range(n):
        xs = x[i] * 256.0
        s1 += (xs - s1) * a
        h1 = xs - s1
        s2 += (h1 - s2) * a
        h2 = h1 - s2
        s3 += (h2 - s3) * b
        y[i] = s3 / 64.0
    return y


def robust_sigma(y):
    return float(np.median(np.abs(y)) / 0.6745)


@njit(cache=True)
def threshold_events(y, thr, dead=DEAD):
    """Negative threshold crossing -> trough within SEARCH samples -> dead time.

    `thr` is a positive integer; a crossing is y[n] < -thr. Returns trough
    indices. With dead=DEAD this mirrors c/spike_edge.c exactly.
    """
    n = len(y)
    out = np.empty(n // dead + 1, np.int64)
    k = 0
    in_ev = False
    best = 0; best_i = 0; end = 0; dead_until = 0
    for i in range(n):
        if in_ev:
            if y[i] < best:
                best = y[i]; best_i = i
            if i >= end:
                out[k] = best_i; k += 1
                in_ev = False
                dead_until = best_i + dead
        elif i >= dead_until and y[i] < -thr:
            in_ev = True
            best = y[i]; best_i = i; end = i + SEARCH
    return out[:k]


@njit(cache=True)
def peak_events(e, thr):
    """Positive-going version on an energy signal (NEO): crossing -> argmax."""
    n = len(e)
    out = np.empty(n // DEAD + 1, np.int64)
    k = 0
    in_ev = False
    best = 0.0; best_i = 0; end = 0; dead_until = 0
    for i in range(n):
        if in_ev:
            if e[i] > best:
                best = e[i]; best_i = i
            if i >= end:
                out[k] = best_i; k += 1
                in_ev = False
                dead_until = best_i + DEAD
        elif i >= dead_until and e[i] > thr:
            in_ev = True
            best = e[i]; best_i = i; end = i + SEARCH
    return out[:k]


def neo(y):
    """Nonlinear energy operator psi[n] = y[n]^2 - y[n-1] y[n+1], Bartlett-smoothed."""
    y = y.astype(np.float64)
    psi = np.zeros_like(y)
    psi[1:-1] = y[1:-1] ** 2 - y[:-2] * y[2:]
    w = np.bartlett(7)
    return np.convolve(psi, w / w.sum(), mode="same")
