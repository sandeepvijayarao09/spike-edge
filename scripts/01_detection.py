"""Detection benchmark: threshold crossing, NEO, gated int8 CNN/MLP.

Writes results/detection.json, results/detection_table.md, models/<arch>.npz.
"""
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spike_edge import FS, TOL, WIN, DEAD
from spike_edge.datasets import split, windows, N_CH, SNR_LEVELS, CAL_S
from spike_edge.frontend import threshold_events, peak_events, neo
from spike_edge.metrics import match, prf, bootstrap_f1
from spike_edge import qnn

GATE_K = 3.0
ARCH_NAMES = sys.argv[1:] or ["cnn8", "cnn4", "mlp8"]


def load_split(name, n):
    t = time.time()
    chs = [split(name, i) for i in range(n)]
    print(f"{name}: {n} ch in {time.time()-t:.0f}s", flush=True)
    return chs


def candidates(ch, k=GATE_K):
    idx = threshold_events(ch["y"], int(round(k * ch["sigma"])))
    idx, w = windows(ch["y"], idx)
    g = qnn.input_gain(ch["sigma"])
    return idx, qnn.quantize_input(w, g)


def labels(idx, gt):
    _, hit, _ = match(idx, gt)
    return hit.astype(np.int8)


def neo_thr(ch, c):
    e = neo(ch["y"])
    return e, c * e[FS // 10: FS // 10 + CAL_S * FS].mean()


def counts(det, gt, lo, hi):
    det = det[(det >= lo) & (det < hi)]; gt = gt[(gt >= lo) & (gt < hi)]
    tp, _, _ = match(det, gt)
    return (tp, len(det), len(gt))


def pooled_f1(cnts):
    s = np.sum(cnts, 0); return prf(*s)[2]


def main():
    train, val = load_split("train", N_CH["train"]), load_split("val", N_CH["val"])
    res = {"config": dict(fs=FS, tol_samples=TOL, dead_samples=DEAD, gate_k=GATE_K,
                          win=WIN, eval_start_s=CAL_S), "selected": {}, "levels": {}}

    # --- choose operating points on val (mixed SNR 1.5-10) -------------------
    ks = np.arange(3.0, 6.01, 0.25)
    f_tc = [pooled_f1([counts(threshold_events(c["y"], int(round(k * c["sigma"]))), c["gt"], 0, len(c["y"])) for c in val]) for k in ks]
    k_best = float(ks[int(np.argmax(f_tc))])
    # filter ringing makes large spikes re-cross ~1-1.5 ms later; give the threshold
    # baseline a tuned dead time too, so the learned model gets no free win from that
    grid = [(k, dd) for k in ks for dd in (20, 30, 40, 60)]
    f_td = [pooled_f1([counts(threshold_events(c["y"], int(round(k * c["sigma"])), dd), c["gt"], 0, len(c["y"])) for c in val]) for k, dd in grid]
    k_dt, dead_dt = grid[int(np.argmax(f_td))]
    res["selected"].update(tc_dead_k=float(k_dt), tc_dead_samples=int(dead_dt))
    print("TC tuned k/dead", k_dt, dead_dt)
    cs = np.arange(2, 40.1, 1.0)
    neos = [neo_thr(c, 1.0) for c in val]
    f_neo = [pooled_f1([counts(peak_events(e, cc * t), c["gt"], 0, len(c["y"])) for (e, t), c in zip(neos, val)]) for cc in cs]
    c_best = float(cs[int(np.argmax(f_neo))])
    res["selected"].update(tc_best_k=k_best, neo_c=c_best)
    print("TC best k", k_best, "NEO C", c_best)

    # --- learned candidate classifier ---------------------------------------
    def build(chs):
        X, Y, meta = [], [], []
        for c in chs:
            idx, x = candidates(c); X.append(x); Y.append(labels(idx, c["gt"])); meta.append(idx)
        return np.concatenate(X), np.concatenate(Y), meta
    Xtr, Ytr, _ = build(train); Xva, Yva, idx_va = build(val)
    gate_rec = sum(counts(i, c["gt"], 0, len(c["y"]))[0] for i, c in zip(idx_va, val)) / sum(len(c["gt"]) for c in val)
    print(f"train windows {len(Xtr)} pos {Ytr.mean():.3f}; val gate recall {gate_rec:.3f}")
    from spike_edge.train import train as fit, float_logits

    models = {}
    for name in ARCH_NAMES:
        t = time.time()
        net, fl, act_max, _ = fit(name, Xtr, Ytr, Xva, Yva, WIN, epochs=15)
        q = qnn.quantize_model(fl, qnn.ARCHS[name], act_max)
        li = qnn.run_int(q, Xva)
        lf = float_logits(net, Xva)

        def best_thr(logits, grid):
            fs = []
            for th in grid:
                cn, o = [], 0
                for i, c in zip(idx_va, val):
                    d = i[logits[o:o + len(i)] > th]; o += len(i)
                    cn.append(counts(d, c["gt"], 0, len(c["y"])))
                fs.append(pooled_f1(cn))
            return grid[int(np.argmax(fs))], max(fs)
        thf, ff = best_thr(lf, np.linspace(-6, 6, 49))
        sc = q[-1]["logit_scale"]
        thi, fi = best_thr(li, np.round(np.linspace(-6, 6, 49) / sc).astype(np.int64))
        print(f"{name}: {time.time()-t:.0f}s  val F1 float {ff:.4f} int8 {fi:.4f}  MACs {qnn.macs(qnn.ARCHS[name], WIN)}", flush=True)
        qnn.save(ROOT / "models" / f"{name}.npz", q, name,
                 dict(gate_k=GATE_K, logit_thr=int(thi), act_max=np.array(act_max)))
        models[name] = (net, q, thf, int(thi))
        res["selected"][name] = dict(float_logit_thr=float(thf), int_logit_thr=int(thi),
                                     val_f1_float=ff, val_f1_int8=fi,
                                     macs=qnn.macs(qnn.ARCHS[name], WIN), flash_bytes=qnn.flash_bytes(q))

    # --- test per SNR level -------------------------------------------------
    for lvl in SNR_LEVELS:
        chs = load_split(f"test_{lvl}", N_CH["test"])
        per = {m: [] for m in ["tc4.5", "tc_best", "tc_dead", "neo", "gate"] + [f"{a}_{p}" for a in models for p in ("float", "int8")]}
        ncand, nsamp = 0, 0
        for c in chs:
            y, gt, lo, hi = c["y"], c["gt"], CAL_S * FS, len(c["y"])
            per["tc4.5"].append(counts(threshold_events(y, int(round(4.5 * c["sigma"]))), gt, lo, hi))
            per["tc_best"].append(counts(threshold_events(y, int(round(k_best * c["sigma"]))), gt, lo, hi))
            per["tc_dead"].append(counts(threshold_events(y, int(round(k_dt * c["sigma"])), dead_dt), gt, lo, hi))
            e, t = neo_thr(c, c_best); per["neo"].append(counts(peak_events(e, t), gt, lo, hi))
            idx, x = candidates(c)
            per["gate"].append(counts(idx, gt, lo, hi))
            ncand += np.sum(idx >= lo); nsamp += hi - lo
            for a, (net, q, thf, thi) in models.items():
                per[f"{a}_float"].append(counts(idx[float_logits(net, x) > thf], gt, lo, hi))
                per[f"{a}_int8"].append(counts(idx[qnn.run_int(q, x) > thi], gt, lo, hi))
        out = {}
        for m, cn in per.items():
            p, r, f = prf(*np.sum(cn, 0)); lo_, hi_ = bootstrap_f1(cn)
            out[m] = dict(precision=p, recall=r, f1=f, f1_ci=[float(lo_), float(hi_)], n_gt=int(np.sum(cn, 0)[2]))
        out["candidates_per_s"] = float(ncand / (nsamp / FS))
        res["levels"][str(lvl)] = out
        print(lvl, {m: round(v["f1"], 3) for m, v in out.items() if isinstance(v, dict)}, f"cand/s {out['candidates_per_s']:.0f}", flush=True)

    (ROOT / "results" / "detection.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
