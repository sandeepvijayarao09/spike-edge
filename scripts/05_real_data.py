"""Frozen detectors on real paired ground truth (Yger et al. 2018, eLife 7:e34518;
Zenodo 10.5281/zenodo.1205233, CC-BY-4.0): mouse retina, 252-ch MEA at 20 kHz,
loose-patch spike times of one ganglion cell per recording.

Nothing is re-trained or re-tuned on real data: thresholds, int8 models and their
logit thresholds are the ones chosen on synthetic validation data.

Only one neuron per recording is labelled while each electrode also sees others,
so precision is NOT measurable. Three comparisons, from most to least deployable:
  1. frozen operating points (label-free): GT recall and event rate of each detector;
  2. label-free rate matching: each gated model's logit threshold is set so that its
     event count on the unlabelled 10 s calibration segment equals the tuned
     threshold detector's count there; recall is then measured on the rest;
  3. oracle matched rate (needs labels/known rate): GT recall read off each detector's
     swept recall-vs-rate envelope at the tuned threshold's event rate. The threshold
     envelope takes the best of 1 ms and 2 ms dead time at every rate.

Real-signal SNR sweep: mix = a * electrode + sqrt(1 - a^2) * donor. Donors come from a
*different* recording: same-recording channels share reference noise (rho ~ -0.65 on
20160426_patch3), which made mixing cancel noise and raise SNR. Donors are scaled to the
electrode's robust sigma after the front-end; near-dead channels are excluded; any mix
whose measured SNR is not below native is dropped.

Usage: python -I scripts/05_real_data.py data/yger2018_<rec> [data/yger2018_<rec> ...]
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

MIN_SNR = 4.0        # electrodes where the GT unit's raw STA trough / sigma >= this
NOISE_COUNTS = 5.0   # rescale so raw robust sigma ~ 5 counts, as in the synthetic 10-bit ADC model
TARGETS = [None, 6.0, 5.0, 4.0, 3.0]   # front-end SNR targets; None = native
GATED = ("mlp8", "cnn8", "lin32", "tmpl")


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


def channel_stats(x, tr, skip):
    """Per channel: (raw STA trough / robust sigma, channel, robust sigma), best first."""
    seg = slice(0, 20 * FS)
    t = tr[(tr > 40) & (tr < x.shape[0] - 40)][:500]
    out = []
    for k in range(x.shape[1]):
        if k in skip:
            continue
        c = x[seg, k].astype(np.float64); med = np.median(c)
        s = np.median(np.abs(c - med)) / 0.6745
        if s < 1:
            continue
        sta = x[t[:, None] + np.arange(-40, 40), k].astype(np.float64).mean(0) - med
        out.append((float(-sta.min() / s), k, s))
    return sorted(out, reverse=True)


def normed(x, k, s, n=None):
    v = (x[:, k].astype(np.float64) - np.median(x[: 20 * FS, k])) * NOISE_COUNTS / s
    return v if n is None else np.resize(v, n)   # tile a shorter donor recording


def fe_sigma(v):
    return robust_sigma(frontend_float(v[: 20 * FS])[FS // 10:])


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

    # swept curves: (events/s, recall)
    ks = np.arange(2.5, 8.01, 0.125)
    r["curves"]["threshold"] = [(round(s["events_per_s"], 3), round(s["recall"], 4)) for dd in (20, 40) for k in ks
                                for s in [score(threshold_events(y, int(round(k * sig)), dd))]]
    for m in GATED:
        ths = np.unique(np.quantile(logits[m], np.linspace(0, 1, 121)))
        r["curves"][m] = [(round(s["events_per_s"], 3), round(s["recall"], 4)) for th in ths for s in [score(idx[logits[m] > th])]]

    rate = f["tc_dead"]["events_per_s"]
    r["matched_rate_events_per_s"] = rate
    r["oracle_recall_at_matched_rate"] = {k: envelope_recall(v, rate) for k, v in r["curves"].items()}

    # label-free: match the tuned threshold's event count on the unlabelled calibration segment
    in_cal = (idx >= cal[0]) & (idx < cal[1])
    target = int(np.sum((tc_dead >= cal[0]) & (tc_dead < cal[1])))
    r["label_free_rate_matched"] = {"tc_dead": f["tc_dead"]}
    for m in GATED:
        lg = np.sort(logits[m][in_cal])[::-1]
        th = lg[target] if target < len(lg) else lg[-1] - 1   # keep exactly `target` cal events
        r["label_free_rate_matched"][m] = score(idx[logits[m] > th])
    return r


def main():
    det = json.loads((ROOT / "results" / "detection.json").read_text())["selected"]
    models = {m: qnn.load(ROOT / "models" / f"{m}.npz") for m in GATED}
    recs = [(d, *load(d)) for d in sys.argv[1:]]
    out = {"source": "Yger et al. 2018 eLife 7:e34518; zenodo.org/records/1205233 (CC-BY-4.0)",
           "frozen_from_synthetic": det, "results": [], "excluded_mixes": []}
    for ri, (d, x, tr, meta) in enumerate(recs):
        stats = channel_stats(x, tr, {int(meta["juxta_channel"])})
        ranked = [e for e in stats if e[0] >= MIN_SNR]
        # donors: another recording, channels with typical (not near-dead) noise
        _, dx, dtr, dmeta = recs[(ri + 1) % len(recs)]
        dst = channel_stats(dx, dtr, {int(dmeta["juxta_channel"])})
        q25, q75 = np.percentile([s for _, _, s in dst], [25, 75])
        donors = [(k, s) for v, k, s in dst if v < 0.5 and q25 <= s <= q75][:4]
        print(d, f"{x.shape[0]/FS:.0f}s, {len(tr)} GT spikes; electrodes:",
              [(k, round(v, 1)) for v, k, _ in ranked], f"donors from {dmeta['name']}:", [k for k, _ in donors], flush=True)
        for raw_snr, k, s_raw in ranked:
            base = normed(x, k, s_raw)
            native = None
            for i, target in enumerate(TARGETS):
                if target is None:
                    sig_in, a, donor, rho = base, 1.0, None, None
                else:
                    if target >= native - 0.25:
                        continue
                    a = target / native
                    dk, ds = donors[i % len(donors)]
                    dn = normed(dx, dk, ds, len(base))
                    dn *= fe_sigma(base) / fe_sigma(dn)
                    rho = float(np.corrcoef(frontend_float(base[: 20 * FS]), frontend_float(dn[: 20 * FS]))[0, 1])
                    sig_in, donor = a * base + np.sqrt(1 - a * a) * dn, f"{dmeta['name']}/e{dk}"
                ch16 = np.clip(np.round(sig_in), -512, 511).astype(np.int16)
                r = evaluate(frontend_int(ch16), tr, det, models)
                r.update(recording=meta["name"], triggers_reconstructed=bool(meta.get("triggers_reconstructed")),
                         electrode=k, raw_snr=raw_snr, target_snr=target, attenuation=a, donor=donor, donor_noise_corr=rho)
                if target is None:
                    native = r["frontend_snr"]
                    r["c_bitexact"] = bitexact(ch16, models["mlp8"], ROOT / "build" / "real_bx")
                elif r["frontend_snr"] >= native - 0.25:
                    out["excluded_mixes"].append({k2: r[k2] for k2 in ("recording", "electrode", "target_snr", "frontend_snr", "donor")})
                    continue
                out["results"].append(r)
                o = r["oracle_recall_at_matched_rate"]; lf = r["label_free_rate_matched"]
                fz = " ".join(f"{m}={v['recall']:.3f}@{v['events_per_s']:.0f}/s" for m, v in r["frozen"].items()
                              if m in ("tc4.5", "tc_dead", "mlp8_int8", "lin32_int8"))
                print(f"  e{k} {'native' if target is None else f'target {target}'}: SNR {r['frontend_snr']:.1f} | {fz} | "
                      f"oracle@{r['matched_rate_events_per_s']:.0f}/s TC {o['threshold']} MLP {o['mlp8']} LIN {o['lin32']} | "
                      f"label-free MLP {lf['mlp8']['recall']:.3f}@{lf['mlp8']['events_per_s']:.0f}/s", flush=True)

    cells = {}
    for r in out["results"]:
        o = r["oracle_recall_at_matched_rate"]
        for m in GATED:
            if o[m] is not None and o["threshold"] is not None:
                cells.setdefault(r["recording"], {}).setdefault(m, []).append(o[m] - o["threshold"])
    out["per_cell_delta_recall_vs_threshold"] = {
        c: {m: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), n_conditions=len(v)) for m, v in dd.items()}
        for c, dd in cells.items()}
    for c, dd in out["per_cell_delta_recall_vs_threshold"].items():
        print(c, {m: round(v["mean"], 3) for m, v in dd.items()})
    print("excluded mixes:", out["excluded_mixes"])
    (ROOT / "results" / "real_yger2018.json").write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
