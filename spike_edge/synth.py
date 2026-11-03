"""Synthetic single-channel extracellular recordings with ground-truth spikes.

Each channel is an independent electrode (implant-style, no spatial sorting):
  * 0-3 "near" units with known spike times (ground truth), velocity-tuned
  * ~20 "far" background units below detectability (the hash), also weakly tuned
  * white + pink thermal/biological noise, 1/f^2 LFP below ~100 Hz
  * slow amplitude drift per unit (proxy for electrode micro-motion)
  * 10-bit ADC quantisation at 1 uV/LSB
SNR of a unit = |filtered trough| / robust sigma of the filtered background,
both measured through the same integer front-end used by every detector.
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import FS, LSB_UV
from .frontend import frontend_float, frontend_int, robust_sigma

TLEN, TPEAK = 64, 20  # template length and trough index (samples)


def make_template(rng):
    t = np.arange(TLEN, dtype=np.float64)
    w1 = rng.uniform(1.6, 4.0)          # trough width   (0.08-0.2 ms)
    w2 = rng.uniform(4.0, 10.0)         # repolarisation width
    d = rng.uniform(5.0, 12.0)          # trough-to-peak delay
    r = rng.uniform(0.2, 0.6)           # peak/trough ratio
    pre = rng.uniform(0.0, 0.15)        # small pre-trough bump
    w = (-np.exp(-0.5 * ((t - TPEAK) / w1) ** 2)
         + r * np.exp(-0.5 * ((t - TPEAK - d) / w2) ** 2)
         + pre * np.exp(-0.5 * ((t - TPEAK + 2.5 * w1) / w1) ** 2))
    w -= np.linspace(w[0], w[-1], TLEN)  # zero the ends
    return w / -w.min()


def velocity(rng, n_ms, tau_s=0.3):
    """Smooth 2D velocity at 1 kHz (unit variance per axis)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, 1.0 / (2 * np.pi * tau_s), fs=1000, output="sos")
    v = sosfiltfilt(sos, rng.standard_normal((n_ms, 2)), axis=0)
    return v / v.std(axis=0)


def tuned_spikes(rng, vel, base, depth):
    """Cosine-tuned inhomogeneous Poisson train with 1.5 ms refractory."""
    pd = rng.uniform(0, 2 * np.pi)
    drive = vel[:, 0] * np.cos(pd) + vel[:, 1] * np.sin(pd)
    rate = np.clip(base * (1 + depth * drive), 0.5, None)  # Hz, per ms
    ms = np.flatnonzero(rng.random(len(rate)) < rate / 1000.0)
    t = ms * (FS // 1000) + rng.integers(0, FS // 1000, len(ms))
    keep = np.ones(len(t), bool)
    last = -10**9
    for i, ti in enumerate(t):
        if ti - last < 30:
            keep[i] = False
        else:
            last = ti
    return t[keep]


@njit(cache=True)
def _add_spikes(x, times, amps, tmpl):
    n = len(x)
    for i in range(len(times)):
        t0 = times[i] - TPEAK
        for j in range(TLEN):
            k = t0 + j
            if 0 <= k < n:
                x[k] += amps[i] * tmpl[j]


def colored_noise(rng, n, exponent):
    f = np.fft.rfftfreq(n, 1 / FS)
    spec = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    f[0] = f[1]
    spec *= f ** (-exponent / 2)
    x = np.fft.irfft(spec, n)
    return x / x.std()


@dataclass
class Channel:
    raw: np.ndarray                 # int16 ADC counts
    gt_times: list = field(default_factory=list)   # per near unit, trough sample idx
    gt_snr: list = field(default_factory=list)
    sigma_y: float = 0.0            # robust sigma of filtered background (front-end units)


def make_channel(rng, dur_s, unit_snrs, vel=None, n_bg=20, drift=0.25,
                 white_uv=4.0, pink_uv=3.0, lfp_uv=40.0, bg_tuned=True):
    n = int(dur_s * FS)
    if vel is None:
        vel = velocity(rng, int(dur_s * 1000))
    t_s = np.arange(n) / FS
    bg = (white_uv * rng.standard_normal(n) + pink_uv * colored_noise(rng, n, 1.0)
          + lfp_uv * colored_noise(rng, n, 2.0))
    # far units -> hash. Own RNG stream, so toggling bg_tuned leaves the noise and
    # the near (ground-truth) units identical: a paired control.
    brng = np.random.default_rng(rng.integers(2**62))
    for _ in range(n_bg):
        st = tuned_spikes(brng, vel, brng.uniform(5, 25), 0.5 if bg_tuned else 0.0)
        amp = brng.uniform(0.3, 1.5) * white_uv * np.ones(len(st))
        _add_spikes(bg, st, amp, make_template(brng))
    sig_y = robust_sigma(frontend_float(bg)[FS // 10:])
    x = bg.copy()
    ch = Channel(raw=None, sigma_y=sig_y)
    for snr in unit_snrs:
        tm = make_template(rng)
        ft = frontend_float(np.r_[np.zeros(200), tm, np.zeros(200)])
        gain, off = -ft.min(), int(np.argmin(ft)) - (200 + TPEAK)
        st = tuned_spikes(rng, vel, rng.uniform(5, 40), rng.uniform(0.4, 0.9))
        ph, per = rng.uniform(0, 2 * np.pi), rng.uniform(60, 300)
        dr = 1 + drift * np.sin(2 * np.pi * t_s[st] / per + ph)
        amp = snr * sig_y / gain * dr
        _add_spikes(x, st, amp, tm)
        ch.gt_times.append(st + off)
        ch.gt_snr.append(snr)
    ch.raw = np.clip(np.round(x / LSB_UV), -512, 511).astype(np.int16)
    return ch, vel
