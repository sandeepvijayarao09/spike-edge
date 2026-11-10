"""Frozen detectors on real paired ground truth. Nothing is re-trained or re-tuned on
real data: thresholds, int8 models and logit thresholds come from synthetic validation.

Datasets (each recording = one patched cell = one independent sample):
  * Yger et al. 2018, eLife 7:e34518 (Zenodo 1205233, CC-BY-4.0): mouse retina in vitro,
    252-ch MEA at 20 kHz, already high-passed at 100 Hz; loose-patch spike times.
  * Marques-Smith et al. 2018, bioRxiv 10.1101/370080 (CC BY-SA 4.0): rat cortex in vivo,
    Neuropixels at 30 kHz; juxtacellular spike times already converted to probe samples.
    Here high-passed at 100 Hz (3rd-order Butterworth, causal, to match Yger's preprocessing)
    and resampled to 20 kHz (polyphase 2/3). Fetched by scripts/fetch_marques_smith.py.

Every electrode on which the cell's front-end STA trough is >= 4 sigma is an independent
implant-style channel: rescaled to robust sigma = 5 counts, clipped to 10 bits.

Only one neuron per recording is labelled, so precision is NOT measurable. Comparisons:
  1. frozen operating points (label-free): recall and event rate of every detector;
  2. label-free rate matching (PRIMARY): each gated model's logit threshold is set so that
     its event count on the unlabelled 10 s calibration segment equals the tuned threshold
     detector's count there; recall is then measured on the rest;
  3. oracle matched rate: recall read off each detector's swept recall-vs-rate envelope at
     the tuned threshold's event rate (threshold envelope over k x {1, 2} ms dead time).
Per cell, the native-electrode Delta recall vs the tuned threshold is averaged over that
cell's electrodes; cells are then counted as wins (> +0.01), ties or losses, with a
bootstrap CI over cells and a two-sided sign test.

Real-signal SNR sweep (secondary): mix = a * electrode + sqrt(1 - a^2) * donor, donor from a
*different* recording of the same dataset (same-recording channels share reference noise,
rho ~ -0.65 on Yger 20160426_patch3), scaled to the electrode's front-end sigma; mixes whose
measured SNR is not below native are dropped.

Usage: python -I scripts/05_real_data.py   (uses every recording found under data/)
"""
import json, re, sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spike_edge import FS, qnn
from spike_edge.datasets import calibrate, windows, CAL_S
from spike_edge.frontend import frontend_int, frontend_float, robust_sigma, threshold_events, peak_events, neo
from spike_edge.metrics import match

MIN_SNR = 4.0        # front-end STA trough / sigma needed to use an electrode
NOISE_COUNTS = 5.0   # rescale so robust sigma ~ 5 counts, as in the synthetic 10-bit ADC model
TARGETS = [None, 6.0, 5.0, 4.0, 3.0]
GATED = ("mlp8", "cnn8", "lin32", "tmpl")
STAT_S = 20          # seconds used to rank electrodes and pick donors
TIE = 0.01


# --------------------------------------------------------------------------- datasets
class Yger:
    dataset = "Yger 2018 (mouse retina, MEA)"

    def __init__(self, d):
        d = Path(d)
        meta = dict(re.findall(r"(\w+)\s*=\s*(\S+)", next(d.glob("*.txt")).read_text()))
        raw = next(p for p in d.glob("*.raw") if not p.name.endswith("juxta.raw"))
        self.x = np.memmap(raw, np.uint16, "r", offset=int(float(meta["padding"]))).reshape(-1, int(float(meta.get("total_nb_channels", 256))))
        trig = list(d.glob("*.triggers.npy"))
        self.reconstructed = not trig
        if trig:
            self.tr = np.load(trig[0]).astype(np.int64)
        else:  # some recordings ship without triggers: rebuild them as the authors describe
            self.tr = juxta_triggers(np.fromfile(next(d.glob("*.juxta.raw")), np.float32), float(meta["threshold"]))
        self.name = meta["name"]
        self.n = self.x.shape[0]
        self.channels = [k for k in range(self.x.shape[1]) if k != int(float(meta["juxta_channel"]))]

    def chan(self, k, n=None):
        v = self.x[: n or self.n, k].astype(np.float64)
        return v - np.median(v[: STAT_S * FS])


class MarquesSmith:
    dataset = "Marques-Smith 2018 (rat cortex, Neuropixels)"

    def __init__(self, d):
        from scipy.signal import butter
        d = Path(d); c = d.name
        self.sub = np.load(d / f"{c}_npx_sub.npy", mmap_mode="r")
        self.meta = json.loads((d / f"{c}_npx_sub.json").read_text())
        t30 = np.load(d / f"{c}_extracellular_spikes.npy", allow_pickle=False).astype(np.int64)
        self.tr = np.round(t30 * 2 / 3).astype(np.int64)
        self.name, self.reconstructed = f"cortex/{c}", False
        self.n = self.sub.shape[0] * 2 // 3
        self.channels = list(range(self.sub.shape[1]))
        self.sos = butter(3, 100, "highpass", fs=30_000, output="sos")

    def chan(self, k, n=None):
        from scipy.signal import resample_poly, sosfilt
        m = None if n is None else (n * 3) // 2 + 3
        v = sosfilt(self.sos, self.sub[:m, k].astype(np.float64))
        v = resample_poly(v, 2, 3)
        return v[: n or self.n]


def juxta_triggers(j, k, dead=40):
    """Negative peaks below -k*MAD, merged within 2 ms. Reproduces the official
    .triggers.npy exactly on 20160426_patch3 (1691/1691) and 20170622_patch1 (4541/4541)."""
    med = np.median(j); z = med - j
    idx = np.flatnonzero((z[1:-1] > k * np.median(np.abs(j - med))) & (z[1:-1] >= z[:-2]) & (z[1:-1] > z[2:])) + 1
    keep = []
    for i in idx:
        if keep and i - keep[-1] <= dead:
            if z[i] > z[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    return np.array(keep, np.int64)


def discover():
    data = ROOT / "data"
    # .complete is written only after the archive size matched Zenodo and extracted cleanly
    recs = [Yger(d) for d in sorted(data.glob("yger2018_*")) if (d / ".complete").exists()]
    recs += [MarquesSmith(d) for d in sorted((data / "marques_smith").glob("c*")) if list(d.glob("*_npx_sub.json"))]
    return recs


# --------------------------------------------------------------------------- helpers
def fe_sta_snr(v, tr):
    """Front-end trough SNR of the labelled cell on a (scaled) channel snippet."""
    y = frontend_float(v)
    s = robust_sigma(y[FS // 10:])
    t = tr[(tr > 60) & (tr < len(y) - 60)]
    if len(t) < 10 or s <= 0:
        return 0.0
    sta = y[t[:, None] + np.arange(-40, 41)].mean(0)
    return float(-sta.min() / s)


def scaled(v):
    s = robust_sigma(v[: STAT_S * FS])
    return v * NOISE_COUNTS / max(s, 1e-9)


def channel_stats(rec):
    """Rank electrodes on a window spanning the first 500 GT spikes (at least STAT_S s):
    some cells are patched only after the recording starts (cortex/c24: first spike at 39 s)."""
    out = []
    first = rec.tr[(rec.tr > 60)]
    n = int(min(rec.n, max(STAT_S * FS, first[min(499, len(first) - 1)] + FS))) if len(first) else STAT_S * FS
    for k in rec.channels:
        v = rec.chan(k, n)
        if robust_sigma(v) < 1e-6:
            continue
        out.append((fe_sta_snr(scaled(v), rec.tr[rec.tr < n]), k, robust_sigma(frontend_float(scaled(v))[FS // 10:])))
    return sorted(out, reverse=True)


def envelope_recall(points, rate):
    """Best recall achievable at an event rate <= `rate` (upper-left envelope)."""
    ok = [r for e, r in points if e <= rate + 1e-3]   # curve rates are rounded to 1e-3
    return max(ok) if ok else None


def bitexact(ch16, model, wd):
    from spike_edge.export_c import build, run_harness
    name, q, mm = model
    y = frontend_int(ch16); sig = calibrate(y)
    gate, gain, thr = int(round(3.0 * sig)), qnn.input_gain(sig), int(mm["logit_thr"])
    idx, w = windows(y, threshold_events(y, gate))
    lg = qnn.run_int(q, qnn.quantize_input(w, gain))
    yc, ev = run_harness(build(name), ch16, gate, gain, thr, wd)
    return bool(np.array_equal(yc.astype(np.int32), y) and np.array_equal(ev[:, 0], idx) and np.array_equal(ev[:, 1], lg))


def evaluate(y, tr, det, models):
    sig = calibrate(y)
    t = tr[(tr > 60) & (tr < len(y) - 60)]
    sta = y[t[:, None] + np.arange(-40, 41)].mean(0)
    lag = int(np.argmin(sta)) - 40          # patch spike peak -> extracellular trough
    gt = t + lag
    lo = CAL_S * FS; cal = (FS // 10, lo)
    gt = gt[gt >= lo]; dur = (len(y) - lo) / FS

    def score(ev):
        ev = ev[ev >= lo]; tp, _, _ = match(ev, gt)
        return dict(recall=tp / len(gt), events_per_s=len(ev) / dur)

    r = dict(frontend_snr=float(-sta.min() / sig), frontend_ptp_snr=float((sta.max() - sta.min()) / sig),
             gt_lag_samples=lag, n_gt=int(len(gt)), gt_rate_hz=len(gt) / dur, frozen={}, curves={})
    f = r["frozen"]
    f["tc4.5"] = score(threshold_events(y, int(round(4.5 * sig))))
    tc_dead = threshold_events(y, int(round(det["tc_dead_k"] * sig)), det["tc_dead_samples"])
    f["tc_dead"] = score(tc_dead)
    e = neo(y)
    f["neo"] = score(peak_events(e, det["neo_c"] * e[FS // 10: FS // 10 + CAL_S * FS].mean()))
    idx, w = windows(y, threshold_events(y, int(round(3.0 * sig))))
    xq = qnn.quantize_input(w, qnn.input_gain(sig))
    f["gate"] = score(idx)
    logits = {}
    for m in GATED:
        _, q, mm = models[m]
        logits[m] = qnn.run_int(q, xq).astype(np.int64)
        f[f"{m}_int8"] = score(idx[logits[m] > mm["logit_thr"]])

    ks = np.arange(2.5, 8.01, 0.125)
    r["curves"]["threshold"] = [(round(s["events_per_s"], 3), round(s["recall"], 4)) for dd in (20, 40) for k in ks
                                for s in [score(threshold_events(y, int(round(k * sig)), dd))]]
    for m in GATED:
        ths = np.unique(np.quantile(logits[m], np.linspace(0, 1, 121)))
        r["curves"][m] = [(round(s["events_per_s"], 3), round(s["recall"], 4)) for th in ths for s in [score(idx[logits[m] > th])]]
    rate = f["tc_dead"]["events_per_s"]
    r["matched_rate_events_per_s"] = rate
    r["oracle_recall_at_matched_rate"] = {k: envelope_recall(v, rate) for k, v in r["curves"].items()}
    del r["curves"]   # ~3 MB of raw sweeps; only the matched-rate read-outs are reported

    in_cal = (idx >= cal[0]) & (idx < cal[1])
    target = int(np.sum((tc_dead >= cal[0]) & (tc_dead < cal[1])))
    r["label_free_rate_matched"] = {"tc_dead": f["tc_dead"]}
    for m in GATED:
        lg = np.sort(logits[m][in_cal])[::-1]
        th = lg[target] if target < len(lg) else (lg[-1] - 1 if len(lg) else 0)   # keep exactly `target` cal events
        r["label_free_rate_matched"][m] = score(idx[logits[m] > th])
    return r


# --------------------------------------------------------------------------- cell stats
def sign_test(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def cell_summary(results):
    """Per cell: mean over native electrodes of Delta recall vs the tuned threshold."""
    keys = {"label_free": lambda r, m: r["label_free_rate_matched"][m]["recall"] - r["label_free_rate_matched"]["tc_dead"]["recall"],
            "oracle_matched": lambda r, m: (None if r["oracle_recall_at_matched_rate"][m] is None else
                                            r["oracle_recall_at_matched_rate"][m] - r["oracle_recall_at_matched_rate"]["threshold"]),
            "frozen": lambda r, m: r["frozen"][f"{m}_int8"]["recall"] - r["frozen"]["tc_dead"]["recall"]}
    cells = {}
    for r in results:
        if r["target_snr"] is None:
            cells.setdefault(r["recording"], dict(dataset=r["dataset"], rows=[]))["rows"].append(r)
    out = {"per_cell": {}, "aggregate": {}}
    for c, d in cells.items():
        rows = d["rows"]
        out["per_cell"][c] = dict(dataset=d["dataset"], n_electrodes=len(rows), snr=[r["frontend_snr"] for r in rows],
                                  **{f"{k}_{m}": float(np.mean([v for v in (f(r, m) for r in rows) if v is not None]))
                                     for k, f in keys.items() for m in GATED},
                                  rate_ratio_label_free_mlp8=float(np.mean([r["label_free_rate_matched"]["mlp8"]["events_per_s"] /
                                                                            max(r["frozen"]["tc_dead"]["events_per_s"], 1e-9) for r in rows])))
    rng = np.random.default_rng(0)
    for scope in ["all"] + sorted({d["dataset"] for d in out["per_cell"].values()}):
        cs = [v for v in out["per_cell"].values() if scope == "all" or v["dataset"] == scope]
        agg = {"n_cells": len(cs)}
        for k in keys:
            for m in GATED:
                x = np.array([v[f"{k}_{m}"] for v in cs])
                if not len(x):
                    continue
                w, l = int(np.sum(x > TIE)), int(np.sum(x < -TIE))
                bs = rng.choice(x, (10000, len(x))).mean(1)
                agg[f"{k}_{m}"] = dict(mean=float(x.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                       wins=w, ties=len(x) - w - l, losses=l, sign_test_p=sign_test(w, l))
        out["aggregate"][scope] = agg
    return out


# --------------------------------------------------------------------------- main
def main():
    det = json.loads((ROOT / "results" / "detection.json").read_text())["selected"]
    models = {m: qnn.load(ROOT / "models" / f"{m}.npz") for m in GATED}
    recs = discover()
    out = {"sources": ["Yger et al. 2018 eLife 7:e34518; zenodo.org/records/1205233 (CC-BY-4.0)",
                       "Marques-Smith et al. 2018 bioRxiv 10.1101/370080; Neuropixel Paired Recordings (CC BY-SA 4.0)"],
           "frozen_from_synthetic": det, "min_frontend_snr": MIN_SNR, "results": [], "excluded_mixes": [], "excluded_electrodes": [], "cells_without_electrode": []}
    stats = {}
    for rec in recs:
        stats[rec.name] = channel_stats(rec)
        print(f"{rec.name}: {rec.n / FS:.0f}s, {len(rec.tr)} GT spikes, best fe-SNR "
              f"{[(k, round(v, 1)) for v, k, _ in stats[rec.name][:4]]}", flush=True)
    for ri, rec in enumerate(recs):
        ranked = [e for e in stats[rec.name] if e[0] >= MIN_SNR]
        if not ranked:
            out["cells_without_electrode"].append(dict(recording=rec.name, dataset=rec.dataset,
                                                       best_frontend_snr=stats[rec.name][0][0] if stats[rec.name] else None))
            continue
        same = [o for o in recs if o.dataset == rec.dataset and o is not rec]
        donor_rec = same[0] if same else None
        donors = []
        if donor_rec is not None:
            ds = stats[donor_rec.name]
            q25, q75 = np.percentile([s for _, _, s in ds], [25, 75])
            donors = [k for v, k, s in ds if v < 0.5 and q25 <= s <= q75][:4]
        for snr0, k, _ in ranked:
            base = scaled(rec.chan(k))
            native = None
            for i, target in enumerate(TARGETS):
                if target is None:
                    sig_in, a, donor, rho = base, 1.0, None, None
                else:
                    if not donors or target >= native - 0.25:
                        continue
                    a = target / native
                    dk = donors[i % len(donors)]
                    dn = np.resize(scaled(donor_rec.chan(dk)), len(base))
                    dn *= robust_sigma(frontend_float(base[: STAT_S * FS])[FS // 10:]) / robust_sigma(frontend_float(dn[: STAT_S * FS])[FS // 10:])
                    rho = float(np.corrcoef(frontend_float(base[: STAT_S * FS]), frontend_float(dn[: STAT_S * FS]))[0, 1])
                    sig_in, donor = a * base + np.sqrt(1 - a * a) * dn, f"{donor_rec.name}/e{dk}"
                ch16 = np.clip(np.round(sig_in), -512, 511).astype(np.int16)
                r = evaluate(frontend_int(ch16), rec.tr, det, models)
                r.update(recording=rec.name, dataset=rec.dataset, triggers_reconstructed=rec.reconstructed, electrode=int(k),
                         target_snr=target, attenuation=a, donor=donor, donor_noise_corr=rho)
                if target is None:
                    native = r["frontend_snr"]
                    if native < MIN_SNR:   # 20 s shortlist said >= 4 sigma, full recording disagrees
                        out["excluded_electrodes"].append(dict(recording=rec.name, electrode=int(k), shortlist_snr=snr0, full_snr=native))
                        break
                    r["c_bitexact"] = bitexact(ch16, models["mlp8"], ROOT / "build" / "real_bx")
                elif r["frontend_snr"] >= native - 0.25:
                    out["excluded_mixes"].append({k2: r[k2] for k2 in ("recording", "electrode", "target_snr", "frontend_snr", "donor")})
                    continue
                out["results"].append(r)
                lf = r["label_free_rate_matched"]
                print(f"  {rec.name} e{k} {'native' if target is None else f'mix {target}'}: SNR {r['frontend_snr']:.1f} n_gt {r['n_gt']} | "
                      f"label-free @~{lf['tc_dead']['events_per_s']:.0f}/s: TC {lf['tc_dead']['recall']:.3f} MLP {lf['mlp8']['recall']:.3f} "
                      f"LIN {lf['lin32']['recall']:.3f} MF {lf['tmpl']['recall']:.3f}", flush=True)
    have = {r["recording"] for r in out["results"]}
    for rec in recs:
        if rec.name not in have and rec.name not in {c["recording"] for c in out["cells_without_electrode"]}:
            out["cells_without_electrode"].append(dict(recording=rec.name, dataset=rec.dataset,
                                                       best_frontend_snr=stats[rec.name][0][0] if stats[rec.name] else None))
    out["cells"] = cell_summary(out["results"])
    for scope, agg in out["cells"]["aggregate"].items():
        print(scope, {k: (round(v["mean"], 3), v["wins"], v["ties"], v["losses"]) for k, v in agg.items() if isinstance(v, dict) and k.startswith("label_free")})
    print("cells without a >=4 sigma electrode:", [c["recording"] for c in out["cells_without_electrode"]])
    (ROOT / "results" / "real_paired_gt.json").write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
