"""Frozen detectors on real paired ground truth (Yger et al. 2018, eLife 7:e34518;
Zenodo 10.5281/zenodo.1205233, CC-BY-4.0): mouse retina, 252-ch MEA at 20 kHz,
loose-patch spike times of one ganglion cell per recording.

Nothing is re-trained or re-tuned: every threshold, the int8 models and their
logit thresholds are the ones chosen on synthetic validation data.

Only one neuron per recording is labelled while the electrode also sees others,
so precision is NOT measurable. Reported: GT recall at the frozen operating point,
the event rate, and recall at a matched event rate (each detector's threshold swept).

Real-signal SNR sweep: mix = a * electrode + sqrt(1 - a^2) * donor, where the donor is
a distant electrode on which the GT unit is invisible (STA SNR < 0.5), scaled so its
robust sigma *after the front-end* equals the electrode's. Targets are front-end SNRs. Waveforms, noise and the other (unlabelled) neurons stay real; only
the GT unit's SNR drops to a * native.

Usage: python -I scripts/05_real_data.py data/yger2018 [data/yger2018_...] ...
"""
import json, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spike_edge import FS, qnn
from spike_edge.datasets import calibrate, windows, CAL_S
from spike_edge.frontend import frontend_int, frontend_float, robust_sigma, threshold_events, peak_events, neo
from spike_edge.metrics import match

MIN_SNR = 4.0        # use electrodes where the GT unit's raw STA trough / sigma >= this
NOISE_COUNTS = 5.0   # rescale so raw robust sigma ~ 5 counts, as in the synthetic 10-bit ADC model
TARGETS = [None, 6.0, 5.0, 4.0, 3.0]   # None = native SNR


def load(d):
    d = Path(d)
    meta = dict(re.findall(r"(\w+)\s*=\s*(\S+)", next(d.glob("*.txt")).read_text()))
    raw = next(p for p in d.glob("*.raw") if not p.name.endswith("juxta.raw"))
    n_ch = int(meta.get("total_nb_channels", 256))  # 16x16 MEA
    x = np.memmap(raw, np.uint16, "r", offset=int(meta["padding"])).reshape(-1, n_ch)
    trig = list(d.glob("*.triggers.npy"))
    if trig:
        tr = np.load(trig[0]).astype(np.int64)
    else:  # 20160426_patch2 ships without triggers: rebuild them as the authors describe
        j = np.fromfile(next(d.glob("*.juxta.raw")), np.float32)
        tr = juxta_triggers(j, float(meta["threshold"]))
        meta["triggers_reconstructed"] = True
    return x, tr, meta


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


def electrodes(x, tr, juxta):
    """All extracellular channels as (raw STA trough / robust sigma, channel, sigma), best first."""
    seg = slice(0, 20 * FS)
    t = tr[(tr > 40) & (tr < x.shape[0] - 40)][:500]
    out = []
    for k in range(x.shape[1]):
        if k == juxta:
            continue
        c = x[seg, k].astype(np.float64); med = np.median(c)
        s = np.median(np.abs(c - med)) / 0.6745
        if s < 1:
            continue
        sta = x[t[:, None] + np.arange(-40, 40), k].astype(np.float64).mean(0) - med
        out.append((float(-sta.min() / s), k, s))
    return sorted(out, reverse=True)


def evaluate(y, tr, det, models):
    sig = calibrate(y)
    t = tr[(tr > 60) & (tr < len(y) - 60)]
    sta = y[t[:, None] + np.arange(-40, 41)].mean(0)
    lag = int(np.argmin(sta)) - 40          # patch spike peak -> extracellular trough
    gt = t + lag
    lo = CAL_S * FS; gt = gt[gt >= lo]; dur = (len(y) - lo) / FS

    def score(ev):
        ev = ev[ev >= lo]; tp, _, _ = match(ev, gt)
        return dict(recall=tp / len(gt), events_per_s=len(ev) / dur)

    r = dict(frontend_snr=float(-sta.min() / sig), gt_lag_samples=lag, n_gt=int(len(gt)),
             gt_rate_hz=len(gt) / dur, frozen={}, curves={})
    f = r["frozen"]
    f["tc4.5"] = score(threshold_events(y, int(round(4.5 * sig))))
    f["tc_dead"] = score(threshold_events(y, int(round(det["tc_dead_k"] * sig)), det["tc_dead_samples"]))
    e = neo(y)
    f["neo"] = score(peak_events(e, det["neo_c"] * e[FS // 10: FS // 10 + CAL_S * FS].mean()))
    idx, w = windows(y, threshold_events(y, int(round(3.0 * sig))))
    xq = qnn.quantize_input(w, qnn.input_gain(sig))
    f["gate"] = score(idx)
    logits = {}
    for m, (_, q, mm) in models.items():
        logits[m] = qnn.run_int(q, xq)
        f[f"{m}_int8"] = score(idx[logits[m] > mm["logit_thr"]])
    r["curves"]["threshold_tuned_dead"] = [dict(k=float(k), **score(threshold_events(y, int(round(k * sig)), det["tc_dead_samples"])))
                                           for k in np.arange(2.5, 8.01, 0.25)]
    sc = models["mlp8"][1][-1]["logit_scale"]
    r["curves"]["gated_mlp8"] = [dict(logit=float(th), **score(idx[logits["mlp8"] > th / sc])) for th in np.arange(-8, 10.01, 0.5)]
    rate = f["tc_dead"]["events_per_s"]
    r["matched_rate_events_per_s"] = rate
    r["recall_at_matched_rate"] = {}
    for name, cur in r["curves"].items():
        pts = sorted((p["events_per_s"], p["recall"]) for p in cur)
        xs, ys = zip(*pts)
        # only interpolate inside the swept range; outside it the value is not measured
        r["recall_at_matched_rate"][name] = float(np.interp(rate, xs, ys)) if xs[0] <= rate <= xs[-1] else None
    return r


def bitexact(ch16, model, wd):
    from spike_edge.export_c import write_header, build, run_harness
    name, q, mm = model
    write_header(name, q)
    y = frontend_int(ch16); sig = calibrate(y)
    gate, gain, thr = int(round(3.0 * sig)), qnn.input_gain(sig), int(mm["logit_thr"])
    idx, w = windows(y, threshold_events(y, gate))
    lg = qnn.run_int(q, qnn.quantize_input(w, gain))
    yc, ev = run_harness(build(name), ch16, gate, gain, thr, wd)
    return bool(np.array_equal(yc.astype(np.int32), y) and np.array_equal(ev[:, 0], idx) and np.array_equal(ev[:, 1], lg))


def main():
    det = json.loads((ROOT / "results" / "detection.json").read_text())["selected"]
    models = {m: qnn.load(ROOT / "models" / f"{m}.npz") for m in ("cnn8", "cnn4", "mlp8")}
    out = {"source": "Yger et al. 2018 eLife 7:e34518; zenodo.org/records/1205233 (CC-BY-4.0)",
           "frozen_from_synthetic": det, "results": []}
    for d in sys.argv[1:]:
        x, tr, meta = load(d)
        allr = electrodes(x, tr, int(meta["juxta_channel"]))
        ranked = [e for e in allr if e[0] >= MIN_SNR]
        donors = [(k, s) for v, k, s in allr[::-1] if v < 0.5][:4]
        print(d, f"{x.shape[0]/FS:.0f}s, {len(tr)} GT spikes; electrodes:",
              [(k, round(v, 1)) for v, k, _ in ranked], "donors:", [k for k, _ in donors], flush=True)

        def norm(k, s):
            return (x[:, k].astype(np.float64) - np.median(x[: 20 * FS, k])) * NOISE_COUNTS / s

        def fe_sigma(v):
            return robust_sigma(frontend_float(v[: 20 * FS])[FS // 10:])

        for raw_snr, k, s_raw in ranked:
            base = norm(k, s_raw)
            native = evaluate(frontend_int(np.clip(np.round(base), -512, 511).astype(np.int16)), tr, det, models)["frontend_snr"]
            for i, target in enumerate(TARGETS):
                if target is not None and target >= native:
                    continue
                a = 1.0 if target is None else target / native
                donor = None if target is None else donors[i % len(donors)][0]
                if target is None:
                    sig_in = base
                else:
                    dn = norm(*donors[i % len(donors)])
                    sig_in = a * base + np.sqrt(1 - a * a) * dn * (fe_sigma(base) / fe_sigma(dn))
                y = frontend_int(np.clip(np.round(sig_in), -512, 511).astype(np.int16))
                ch16 = np.clip(np.round(sig_in), -512, 511).astype(np.int16)
                if target is None:          # C pipeline on real data, bit for bit
                    r_bx = bitexact(ch16, models["mlp8"], ROOT / "build" / "real_bx")
                r = dict(recording=meta["name"], triggers_reconstructed=bool(meta.get("triggers_reconstructed")), electrode=k, raw_snr=raw_snr, target_snr=target,
                         c_bitexact_native=r_bx,
                         attenuation=a, donor=donor, **evaluate(y, tr, det, models))
                out["results"].append(r)
                mr = r["recall_at_matched_rate"]
                print(f"  e{k} target {target}: fe-SNR {r['frontend_snr']:.1f} n_gt {r['n_gt']} | " +
                      " ".join(f"{m}={v['recall']:.3f}@{v['events_per_s']:.0f}/s" for m, v in r["frozen"].items()) +
                      f" | matched {r['matched_rate_events_per_s']:.0f}/s: TC {mr['threshold_tuned_dead']} MLP {mr['gated_mlp8']}", flush=True)
    (ROOT / "results" / "real_yger2018.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
