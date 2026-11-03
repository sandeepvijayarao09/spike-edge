"""Data reduction vs. decoding: bits/s per channel and held-out velocity R^2.

Usage: python scripts/02_reduction.py [--untuned-hash]
Three simulated 32-channel arrays (seeds 0-2). With --untuned-hash the very same
recordings are regenerated with the background units' velocity tuning switched off
(paired control: does spike-band power win only because the hash carries signal?).
Writes results/reduction[_untuned].json
"""
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spike_edge import FS
from spike_edge import qnn
from spike_edge.datasets import calibrate, windows, CAL_S
from spike_edge.decode import fit_eval, paired_diff_ci
from spike_edge.frontend import frontend_int, threshold_events
from spike_edge.reduce import (BIN, COUNT_BITS, bin_counts, sbp_int, sbp_shift, entropy_bits, raw_rates,
                               saturate, huffman_bits, adaptive_ac_bits, framing_bits_per_s_per_channel)
from spike_edge.synth import make_channel, velocity

N_CH, DUR = 32, 300
UNTUNED = "--untuned-hash" in sys.argv
MODEL = "mlp8"
N_SEEDS = 3
BASE = "threshold counts, tuned"   # strongest threshold baseline is the comparator
DET = json.loads((ROOT / "results" / "detection.json").read_text())["selected"]


def rate_matched_k(y, s, n_target, dead):
    """Label-free: the threshold whose event count on the calibration segment
    is closest to `n_target` (the learned detector's count there)."""
    cal = y[: CAL_S * FS]
    ks = np.arange(2.5, 8.01, 0.0625)
    n = np.array([len(threshold_events(cal, int(round(k * s)), dead)) for k in ks])
    return float(ks[int(np.argmin(np.abs(n - n_target)))])


def one_array(seed, q, meta):
    rng = np.random.default_rng(4_000_000 + seed)
    vel = velocity(rng, DUR * 1000)
    nb = DUR * FS // BIN
    V = vel[: nb * 20].reshape(nb, 20, 2).mean(1)
    feats = {}
    add = lambda k, v: feats.setdefault(k, []).append(v)
    raw_r, n_units, sat = [], 0, []
    k_dt, dead_dt = DET["tc_dead_k"], DET["tc_dead_samples"]
    t = time.time()
    for c in range(N_CH):
        k = rng.poisson(1.5)
        snrs = np.clip(rng.lognormal(np.log(4), 0.5, k), 1.5, 12)
        ch, _ = make_channel(rng, DUR, list(snrs), vel=vel, bg_tuned=not UNTUNED)
        n_units += k
        y = frontend_int(ch.raw); s = calibrate(y)
        n = len(y)
        add("tc4.5", bin_counts(threshold_events(y, int(round(4.5 * s))), n))
        add("tc3.5", bin_counts(threshold_events(y, int(round(3.5 * s))), n))
        add("tc_tuned", bin_counts(threshold_events(y, int(round(k_dt * s)), dead_dt), n))
        add("sbp8", sbp_int(y, sbp_shift(y[:CAL_S * FS])))
        add("sbp_float", np.abs(y[: nb * BIN].astype(float)).reshape(nb, BIN).mean(1))
        idx = threshold_events(y, int(round(meta["gate_k"] * s)))
        idx, w = windows(y, idx)
        ev = idx[qnn.run_int(q, qnn.quantize_input(w, qnn.input_gain(s))) > meta["logit_thr"]]
        add("learned", bin_counts(ev, n))
        k_rm = rate_matched_k(y, s, int(np.sum(ev < CAL_S * FS)), dead_dt)
        add("tc_rate_matched", bin_counts(threshold_events(y, int(round(k_rm * s)), dead_dt), n))
        gt_all = np.concatenate(ch.gt_times) if ch.gt_times else np.zeros(0, int)
        add("oracle_unsorted", bin_counts(gt_all, n))
        for g in ch.gt_times:
            add("oracle_sorted", bin_counts(g, n))
        raw_r.append(raw_rates(ch.raw))
    print(f"generated {N_CH} ch x {DUR}s, {n_units} near units, {time.time()-t:.0f}s", flush=True)
    F = {k: np.stack(v, 1) for k, v in feats.items()}

    reps = {
        "raw broadband (off-chip TC3.5+TC4.5+SBP)": [F["tc3.5"], F["tc4.5"], F["sbp_float"]],
        "threshold counts −4.5σ": [F["tc4.5"]],
        "threshold counts −3.5σ": [F["tc3.5"]],
        BASE: [F["tc_tuned"]],
        "threshold counts, rate-matched to learned (label-free)": [F["tc_rate_matched"]],
        "spike-band power (8-bit)": [F["sbp8"]],
        f"learned gated {MODEL} counts": [F["learned"]],
        "learned counts + SBP": [F["learned"], F["sbp8"]],
        "oracle unsorted GT counts": [F["oracle_unsorted"]],
        "oracle sorted GT counts (per unit)": [F["oracle_sorted"]],
    }
    n_tr = int(0.7 * nb)   # coders are fitted on the decoder's train portion only
    res = {}
    for name, parts in reps.items():
        if name.startswith("raw"):
            rates = dict(fixed=float(np.mean([r["fixed"] for r in raw_r])),
                         entropy=float(np.mean([r["lossless_est"] for r in raw_r])), huffman=None, arithmetic=None)
            X = np.concatenate(parts, 1)
        else:
            rates = dict(fixed=0.0, entropy=0.0, huffman=0.0, arithmetic=0.0)
            Xs = []
            for p in parts:
                is_count = p.dtype.kind in "iu" and p.max() < 256 and p is not F["sbp8"]
                sym = saturate(p) if is_count else p          # what would actually be sent
                Xs.append(sym)
                w = COUNT_BITS if is_count else 8
                alpha = (1 << w)
                cols = range(p.shape[1])
                rates["fixed"] += w * 50
                rates["entropy"] += np.mean([entropy_bits(sym[:, j]) for j in cols]) * 50
                rates["huffman"] += np.mean([huffman_bits(sym[:n_tr, j], sym[n_tr:, j]) for j in cols]) * 50
                rates["arithmetic"] += np.mean([adaptive_ac_bits(sym[:, j], alpha) for j in cols]) * 50
                if is_count:
                    sat.append((name, float(np.mean(p > (1 << COUNT_BITS) - 1))))
            X = np.concatenate(Xs, 1)
            if "sorted" in name:   # per-unit streams -> per-channel rate
                for kk in rates:
                    rates[kk] *= X.shape[1] / N_CH
        d = fit_eval(X, V)
        res[name] = dict(bits_per_s=rates, **d)
    for name, d in res.items():
        d["diff_vs_base_ci"] = paired_diff_ci(d["y"], d["pred"], res[BASE]["pred"])
        b = d["bits_per_s"]
        print(f"  seed {seed} {name:52s} fixed {b['fixed']:8.0f} H {b['entropy']:7.1f} huff {b['huffman'] or 0:7.1f} "
              f"AC {b['arithmetic'] or 0:7.1f}  R2 {d['r2']:.3f}  vs base [{d['diff_vs_base_ci'][0]:+.3f},{d['diff_vs_base_ci'][1]:+.3f}]", flush=True)
    return res, n_units, sat


def main():
    _, q, meta = qnn.load(ROOT / "models" / f"{MODEL}.npz")
    runs = [one_array(s, q, meta) for s in range(N_SEEDS)]
    out = dict(config=dict(n_channels=N_CH, dur_s=DUR, bin_ms=20, n_seeds=N_SEEDS, n_near_units=[n for _, n, _ in runs],
                           hash_tuned=not UNTUNED, model=MODEL, base=BASE, count_bits=COUNT_BITS,
                           framing_bits_per_s_per_channel_1024ch=framing_bits_per_s_per_channel(),
                           saturated_bin_fraction_max=max(f for _, _, s in runs for _, f in s)), reps={})
    for name in runs[0][0]:
        per = [r[name] for r, _, _ in runs]
        r2s = [p["r2"] for p in per]
        rates = {k: (None if per[0]["bits_per_s"][k] is None else float(np.mean([p["bits_per_s"][k] for p in per])))
                 for k in per[0]["bits_per_s"]}
        out["reps"][name] = dict(bits_per_s=rates, r2_mean=float(np.mean(r2s)), r2_per_seed=r2s,
                                 r2_ci_per_seed=[p["ci"] for p in per],
                                 diff_vs_base_ci_per_seed=[p["diff_vs_base_ci"] for p in per])
        print(f"{name:52s} AC {rates['arithmetic'] or rates['entropy']:9.1f} b/s  R2 mean {np.mean(r2s):.3f} seeds {np.round(r2s, 3)}")
    sfx = "_untuned" if UNTUNED else ""
    (ROOT / "results" / f"reduction{sfx}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
