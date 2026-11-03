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
from spike_edge.reduce import BIN, bin_counts, sbp_int, sbp_shift, entropy_bits, fixed_bits, raw_rates
from spike_edge.synth import make_channel, velocity

N_CH, DUR = 32, 300
UNTUNED = "--untuned-hash" in sys.argv
MODEL = "cnn8"


N_SEEDS = 3
BASE = "threshold counts -4.5σ"


def one_array(seed, q, meta):
    rng = np.random.default_rng(4_000_000 + seed)
    vel = velocity(rng, DUR * 1000)
    nb = DUR * FS // BIN
    V = vel[: nb * 20].reshape(nb, 20, 2).mean(1)
    feats = {k: [] for k in ["tc4.5", "tc3.5", "sbp8", "learned", "sbp_float", "oracle_unsorted", "oracle_sorted"]}
    raw_r, n_units = [], 0
    t = time.time()
    for c in range(N_CH):
        k = rng.poisson(1.5)
        snrs = np.clip(rng.lognormal(np.log(4), 0.5, k), 1.5, 12)
        ch, _ = make_channel(rng, DUR, list(snrs), vel=vel, bg_tuned=not UNTUNED)
        n_units += k
        y = frontend_int(ch.raw); s = calibrate(y)
        n = len(y)
        feats["tc4.5"].append(bin_counts(threshold_events(y, int(round(4.5 * s))), n))
        feats["tc3.5"].append(bin_counts(threshold_events(y, int(round(3.5 * s))), n))
        feats["sbp8"].append(sbp_int(y, sbp_shift(y[:CAL_S * FS])))
        feats["sbp_float"].append(np.abs(y[: nb * BIN].astype(float)).reshape(nb, BIN).mean(1))
        idx = threshold_events(y, int(round(meta["gate_k"] * s)))
        idx, w = windows(y, idx)
        keep = qnn.run_int(q, qnn.quantize_input(w, qnn.input_gain(s))) > meta["logit_thr"]
        feats["learned"].append(bin_counts(idx[keep], n))
        gt_all = np.concatenate(ch.gt_times) if ch.gt_times else np.zeros(0, int)
        feats["oracle_unsorted"].append(bin_counts(gt_all, n))
        for g in ch.gt_times:
            feats["oracle_sorted"].append(bin_counts(g, n))
        raw_r.append(raw_rates(ch.raw))
    print(f"generated {N_CH} ch x {DUR}s, {n_units} near units, {time.time()-t:.0f}s", flush=True)
    F = {k: np.stack(v, 1) for k, v in feats.items()}

    reps = {
        "raw broadband (off-chip TC3.5+TC4.5+SBP)": (np.c_[F["tc3.5"], F["tc4.5"], F["sbp_float"]],
            float(np.mean([r["fixed"] for r in raw_r])), float(np.mean([r["lossless_est"] for r in raw_r]))),
        "threshold counts -4.5σ": (F["tc4.5"], None, None),
        "spike-band power (8-bit)": (F["sbp8"], None, None),
        f"learned gated {MODEL} counts": (F["learned"], None, None),
        "learned counts + SBP": (np.c_[F["learned"], F["sbp8"]], None, None),
        "oracle unsorted GT counts": (F["oracle_unsorted"], None, None),
        "oracle sorted GT counts (per unit)": (F["oracle_sorted"], None, None),
    }
    res = {}
    for name, (X, fixed, ent) in reps.items():
        if fixed is None:
            parts = [F["learned"], F["sbp8"]] if name.startswith("learned counts +") else [X]
            fixed = sum(fixed_bits(p) * 50 for p in parts)
            ent = sum(np.mean([entropy_bits(p[:, j]) for j in range(p.shape[1])]) * 50 for p in parts)
            if "sorted" in name:   # per-unit streams -> per-channel rate
                fixed *= X.shape[1] / N_CH; ent *= X.shape[1] / N_CH
        d = fit_eval(X, V)
        res[name] = dict(bits_per_s_fixed=float(fixed), bits_per_s_entropy=float(ent), **d)
    for name, d in res.items():
        d["diff_vs_tc_ci"] = paired_diff_ci(d["y"], d["pred"], res[BASE]["pred"])
        print(f"  seed {seed} {name:42s} {d['bits_per_s_fixed']:9.0f} b/s (H {d['bits_per_s_entropy']:8.1f})  R2 {d['r2']:.3f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}]  vs TC [{d['diff_vs_tc_ci'][0]:+.3f},{d['diff_vs_tc_ci'][1]:+.3f}]", flush=True)
    return res, n_units


def main():
    _, q, meta = qnn.load(ROOT / "models" / f"{MODEL}.npz")
    runs = [one_array(s, q, meta) for s in range(N_SEEDS)]
    out = dict(config=dict(n_channels=N_CH, dur_s=DUR, bin_ms=20, n_seeds=N_SEEDS,
                           n_near_units=[n for _, n in runs], hash_tuned=not UNTUNED, model=MODEL), reps={})
    for name in runs[0][0]:
        per = [r[name] for r, _ in runs]
        r2s = [p["r2"] for p in per]
        out["reps"][name] = dict(
            bits_per_s_fixed=float(np.mean([p["bits_per_s_fixed"] for p in per])),
            bits_per_s_entropy=float(np.mean([p["bits_per_s_entropy"] for p in per])),
            r2_mean=float(np.mean(r2s)), r2_per_seed=r2s,
            r2_ci_per_seed=[p["ci"] for p in per], diff_vs_tc_ci_per_seed=[p["diff_vs_tc_ci"] for p in per])
        o = out["reps"][name]
        print(f"{name:42s} {o['bits_per_s_fixed']:9.0f} b/s (H {o['bits_per_s_entropy']:8.1f})  R2 mean {o['r2_mean']:.3f} seeds {np.round(r2s, 3)}")
    sfx = "_untuned" if UNTUNED else ""
    (ROOT / "results" / f"reduction{sfx}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
