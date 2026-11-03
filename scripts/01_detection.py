"""Detection benchmark: threshold crossing, NEO, and gated int8 classifiers
(logistic regression, matched filter, MLP, CNNs) on the same -3 sigma candidates.

Writes results/detection.json and models/<arch>.npz.
"""
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spike_edge import FS, TOL, WIN, DEAD, PRE
from spike_edge.datasets import split, windows, N_CH, SNR_LEVELS, CAL_S
from spike_edge.frontend import threshold_events, peak_events, neo
from spike_edge.metrics import match, prf, bootstrap_f1
from spike_edge import qnn

GATE_K = 3.0
LEARNED = sys.argv[1:] or ["cnn8", "cnn4", "mlp8", "lin32"]


def load_split(name, n):
    t = time.time()
    chs = [split(name, i) for i in range(n)]
    print(f"{name}: {n} ch in {time.time()-t:.0f}s", flush=True)
    return chs


def candidates(ch, k=GATE_K):
    idx = threshold_events(ch["y"], int(round(k * ch["sigma"])))
    idx, w = windows(ch["y"], idx)
    return idx, qnn.quantize_input(w, qnn.input_gain(ch["sigma"]))


def labels(idx, gt):
    _, hit, _ = match(idx, gt)
    return hit.astype(np.int8)


def neo_energy(ch):
    e = neo(ch["y"])
    return e, e[FS // 10: FS // 10 + CAL_S * FS].mean()


def counts(det, gt, lo, hi):
    det = det[(det >= lo) & (det < hi)]
    gt = gt[(gt >= lo) & (gt < hi)]
    tp, _, _ = match(det, gt)
    return (tp, len(det), len(gt))


def pooled_f1(cnts):
    return prf(*np.sum(cnts, 0))[2]


def tc(ch, k, dead=DEAD):
    return threshold_events(ch["y"], int(round(k * ch["sigma"])), dead)


def select_baselines(val, res):
    """All baseline operating points are chosen on val (mixed SNR 1.5-10)."""
    full = lambda c: (0, len(c["y"]))
    ks = np.arange(3.0, 6.01, 0.25)
    f_tc = [pooled_f1([counts(tc(c, k), c["gt"], *full(c)) for c in val]) for k in ks]
    k_best = float(ks[int(np.argmax(f_tc))])
    # filter ringing makes large spikes re-cross ~1-1.5 ms later; give the threshold
    # baseline a tuned dead time too, so the learned model gets no free win from that
    grid = [(k, dd) for k in ks for dd in (20, 30, 40, 60)]
    f_td = [pooled_f1([counts(tc(c, k, dd), c["gt"], *full(c)) for c in val]) for k, dd in grid]
    k_dt, dead_dt = grid[int(np.argmax(f_td))]
    cs = np.arange(2, 40.1, 1.0)
    neos = [neo_energy(c) for c in val]
    f_neo = []
    for cc in cs:
        cn = [counts(peak_events(e, cc * m), c["gt"], *full(c)) for (e, m), c in zip(neos, val)]
        f_neo.append(pooled_f1(cn))
    c_best = float(cs[int(np.argmax(f_neo))])
    res["selected"].update(tc_best_k=k_best, tc_dead_k=float(k_dt), tc_dead_samples=int(dead_dt), neo_c=c_best)
    print("TC best k", k_best, "| TC tuned k/dead", k_dt, dead_dt, "| NEO C", c_best, flush=True)
    return k_best, k_dt, dead_dt, c_best


def best_threshold(logits, idx_list, chs):
    """Logit threshold maximising pooled val F1; grid = logit quantiles (scale-free)."""
    grid = np.unique(np.quantile(logits, np.linspace(0, 1, 201)))
    fs = []
    for th in grid:
        cn, o = [], 0
        for i, c in zip(idx_list, chs):
            cn.append(counts(i[logits[o:o + len(i)] > th], c["gt"], 0, len(c["y"])))
            o += len(i)
        fs.append(pooled_f1(cn))
    j = int(np.argmax(fs))
    return grid[j], fs[j]


def fit_models(train, val, res):
    def build(chs):
        X, Y, I = [], [], []
        for c in chs:
            idx, x = candidates(c)
            X.append(x); Y.append(labels(idx, c["gt"])); I.append(idx)
        return np.concatenate(X), np.concatenate(Y), I
    Xtr, Ytr, _ = build(train)
    Xva, Yva, idx_va = build(val)
    hits = sum(counts(i, c["gt"], 0, len(c["y"]))[0] for i, c in zip(idx_va, val))
    print(f"train windows {len(Xtr)} pos {Ytr.mean():.3f}; val gate recall {hits / sum(len(c['gt']) for c in val):.3f}")
    from spike_edge.train import train as fit, float_logits

    models = {}
    for name in LEARNED + ["tmpl"]:
        t = time.time()
        if name == "tmpl":   # matched filter: mean positive training window, no training
            w = (Xtr[Ytr == 1].astype(np.float64) * qnn.IN_SCALE).mean(0)
            fl, act_max = [(w[None, :], np.zeros(1))], []
            ffn = lambda X, w=w: (X.astype(np.float64) * qnn.IN_SCALE) @ w
        else:
            net, fl, act_max, _ = fit(name, Xtr, Ytr, Xva, Yva, WIN, epochs=15)
            ffn = lambda X, net=net: float_logits(net, X)
        q = qnn.quantize_model(fl, qnn.ARCHS[name], act_max)
        thf, ff = best_threshold(ffn(Xva), idx_va, val)
        thi, fi = best_threshold(qnn.run_int(q, Xva).astype(np.int64), idx_va, val)
        thi = int(np.floor(thi))
        mac = qnn.macs(qnn.ARCHS[name], WIN)
        print(f"{name}: {time.time()-t:.0f}s  val F1 float {ff:.4f} int8 {fi:.4f}  MACs {mac}", flush=True)
        qnn.save(ROOT / "models" / f"{name}.npz", q, name,
                 dict(gate_k=GATE_K, logit_thr=thi, act_max=np.array(act_max)))
        models[name] = (ffn, q, thf, thi)
        res["selected"][name] = dict(float_logit_thr=float(thf), int_logit_thr=thi, val_f1_float=ff,
                                     val_f1_int8=fi, macs=mac, weights_bytes=qnn.flash_bytes(q))
    return models


def detectors(c, sel, models):
    """All detector outputs (event index arrays) for one channel."""
    k_best, k_dt, dead_dt, c_best = sel
    out = {"tc4.5": tc(c, 4.5), "tc_best": tc(c, k_best), "tc_dead": tc(c, k_dt, dead_dt)}
    e, m = neo_energy(c)
    out["neo"] = peak_events(e, c_best * m)
    idx, x = candidates(c)
    out["gate"] = idx
    for a, (ffn, q, thf, thi) in models.items():
        out[f"{a}_float"] = idx[ffn(x) > thf]
        out[f"{a}_int8"] = idx[qnn.run_int(q, x) > thi]
    return out, int(np.sum(idx >= CAL_S * FS))   # candidates in the scored segment


def summarize(per):
    out = {}
    for m, cn in per.items():
        p, r, f = prf(*np.sum(cn, 0))
        lo_, hi_ = bootstrap_f1(cn)
        out[m] = dict(precision=p, recall=r, f1=f, f1_ci=[float(lo_), float(hi_)], n_gt=int(np.sum(cn, 0)[2]))
    return out


def snr_conventions(chs):
    """Trough/sigma (our convention) and peak-to-peak/sigma from the GT spike-triggered average."""
    tr, pp = [], []
    for c in chs:
        g = c["gt"][(c["gt"] > 40) & (c["gt"] < len(c["y"]) - 40)]
        sta = c["y"][g[:, None] + np.arange(-20, 30)].mean(0)
        tr.append(-sta.min() / c["sigma"]); pp.append((sta.max() - sta.min()) / c["sigma"])
    return dict(trough_over_sigma=float(np.mean(tr)), ptp_over_sigma=float(np.mean(pp)))


def main():
    train, val = load_split("train", N_CH["train"]), load_split("val", N_CH["val"])
    res = {"config": dict(fs=FS, tol_samples=TOL, dead_samples=DEAD, gate_k=GATE_K, win=WIN,
                          eval_start_s=CAL_S, crossing="negative-going only"), "selected": {}, "levels": {}}
    sel = select_baselines(val, res)
    models = fit_models(train, val, res)

    # --- held-out test, per SNR level ----------------------------------------
    for lvl in SNR_LEVELS:
        chs = load_split(f"test_{lvl}", N_CH["test"])
        per, ncand, nsamp = {}, 0, 0
        for c in chs:
            lo, hi = CAL_S * FS, len(c["y"])
            dets, nc = detectors(c, sel, models)
            for m, d in dets.items():
                per.setdefault(m, []).append(counts(d, c["gt"], lo, hi))
            ncand += nc; nsamp += hi - lo
        out = summarize(per)
        out["candidates_per_s"] = float(ncand / (nsamp / FS))
        out["snr_conventions"] = snr_conventions(chs)
        res["levels"][str(lvl)] = out
        f1s = {m: round(v["f1"], 3) for m, v in out.items() if isinstance(v, dict) and "f1" in v}
        print(lvl, f1s, f"cand/s {out['candidates_per_s']:.0f}", flush=True)

    # --- robustness ------------------------------------------------------------
    # 1) no ground-truth unit at all: every event is a false positive
    chs = load_split("noise", N_CH["noise"])
    fp = {}
    for c in chs:
        lo = CAL_S * FS
        for m, d in detectors(c, sel, models)[0].items():
            fp.setdefault(m, []).append(np.sum(d >= lo) / ((len(c["y"]) - lo) / FS))
    res["noise_only_fp_per_s"] = {m: dict(mean=float(np.mean(v)), max=float(np.max(v))) for m, v in fp.items()}
    print("noise-only FP/s", {m: round(v["mean"], 2) for m, v in res["noise_only_fp_per_s"].items()}, flush=True)

    # 2) drift over 300 s with thresholds calibrated on the first 10 s only: F1 per 60 s block
    chs = load_split("drift", N_CH["drift"])
    blocks = [(CAL_S * FS, 60 * FS)] + [(a * FS, (a + 60) * FS) for a in range(60, 300, 60)]
    per = {}
    for c in chs:
        dets = detectors(c, sel, models)[0]
        for m, d in dets.items():
            for b, (lo, hi) in enumerate(blocks):
                per.setdefault(m, [[] for _ in blocks])[b].append(counts(d, c["gt"], lo, hi))
    res["drift_f1_per_60s_block"] = {m: [pooled_f1(b) for b in v] for m, v in per.items()}
    print("drift F1 per block", {m: np.round(v, 3).tolist() for m, v in res["drift_f1_per_60s_block"].items() if m in ("tc_dead", "mlp8_int8", "lin32_int8")}, flush=True)

    # 3) injected artifacts on the SNR-5 test channels
    chs = load_split("artifact", N_CH["artifact"])
    per, near = {}, {}
    for c in chs:
        lo, hi = CAL_S * FS, len(c["y"])
        for m, d in detectors(c, sel, models)[0].items():
            per.setdefault(m, []).append(counts(d, c["gt"], lo, hi))
            # events within 3 ms of an artifact onset that are not matched to GT
            _, hit, _ = match(d, c["gt"])
            fpd = d[~hit]
            j = np.searchsorted(c["artifacts"], fpd)
            dist = np.minimum(np.abs(fpd - c["artifacts"][np.clip(j - 1, 0, None)]), np.abs(c["artifacts"][np.clip(j, None, len(c["artifacts"]) - 1)] - fpd))
            near.setdefault(m, [0, 0])
            near[m][0] += int(np.sum(dist < 60)); near[m][1] += len(c["artifacts"])
    res["artifacts"] = {"rate_hz": 0.5, "f1": {m: v["f1"] for m, v in summarize(per).items()},
                        "fp_per_artifact": {m: v[0] / max(v[1], 1) for m, v in near.items()}}
    print("artifact F1", {m: round(v, 3) for m, v in res["artifacts"]["f1"].items()}, flush=True)

    (ROOT / "results" / "detection.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
