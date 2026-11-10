"""Figures + markdown tables from results/*.json; fills the marked blocks in README.md."""
import json, re, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "results"; FIG = ROOT / "figures"
det = json.loads((R / "detection.json").read_text())
emb = json.loads((R / "embedded.json").read_text())
red = {k: json.loads((R / f"reduction{k}.json").read_text()) for k in ("", "_untuned")}
REAL = R / "real_paired_gt.json"
real = json.loads(REAL.read_text()) if REAL.exists() else None
LV = list(det["levels"])
SEL = det["selected"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})

NAMES = {"tc4.5": "Threshold −4.5σ", "tc_best": f"Threshold −{SEL['tc_best_k']}σ (tuned)",
         "tc_dead": f"Threshold −{SEL['tc_dead_k']}σ, {SEL['tc_dead_samples']/20:g} ms dead (tuned)",
         "neo": "NEO (tuned)", "gate": "Gate only (−3σ)",
         "tmpl_int8": "Gated matched filter int8 (32 MACs)", "lin32_int8": "Gated logistic regression int8 (32 MACs)",
         "mlp8_int8": "Gated MLP-8 int8 (264 MACs)", "cnn4_int8": "Gated CNN-4 int8 (592 MACs)",
         "cnn8_int8": "Gated CNN-8 int8 (1760 MACs)"}
TH_OPS = emb["baselines"]["threshold_ops_per_sample"]
OPS = {"tc4.5": TH_OPS, "tc_best": TH_OPS, "tc_dead": TH_OPS, "gate": TH_OPS, "neo": emb["baselines"]["neo_ops_per_sample"]}
for m in emb["models"]:
    OPS[f"{m}_int8"] = emb["models"][m]["ops_per_sample"]


def rodata(m):
    mm = m.rsplit("_", 1)[0]
    return str(emb["models"][mm]["m4_sizes_bytes"]["rodata"]) if mm in emb["models"] else "0"


def fmt_f1(v):
    return f"{v['f1']:.3f} [{v['f1_ci'][0]:.2f}–{v['f1_ci'][1]:.2f}]"


def table_detection():
    conv = [det["levels"][l]["snr_conventions"] for l in LV]
    hdr = ("| Detector | ops/sample | M4 .rodata (B) | " + " | ".join(f"F1 @ SNR {l} (ptp {c['ptp_over_sigma']:.1f})" for l, c in zip(LV, conv))
           + " |\n|---|---:|---:|" + "---|" * len(LV) + "\n")
    rows = [f"| {nm} | {OPS[m]:.1f} | {rodata(m)} | " + " | ".join(fmt_f1(det["levels"][l][m]) for l in LV) + " |" for m, nm in NAMES.items()]
    pr = "\n\n| Detector | " + " | ".join(f"P / R @ {l}" for l in LV) + " |\n|---|" + "---|" * len(LV) + "\n"
    pr += "\n".join(f"| {NAMES[m]} | " + " | ".join(f"{det['levels'][l][m]['precision']:.2f} / {det['levels'][l][m]['recall']:.2f}" for l in LV) + " |" for m in NAMES)
    fq = "\n\n| Gated classifier | MACs/inference | val F1 float | val F1 int8 |\n|---|---:|---:|---:|\n"
    fq += "\n".join(f"| {m} | {SEL[m]['macs']} | {SEL[m]['val_f1_float']:.4f} | {SEL[m]['val_f1_int8']:.4f} |" for m in ("tmpl", "lin32", "mlp8", "cnn4", "cnn8"))
    return hdr + "\n".join(rows) + pr + fq


def table_robustness():
    fp = det["noise_only_fp_per_s"]; dr = det["drift_f1_per_60s_block"]; ar = det["artifacts"]
    out = ("| Detector | false events/s, no unit on channel (mean / worst ch) | F1 per 60 s block over 300 s of drift (first block after calibration) | "
           f"F1 with artifacts ({ar['rate_hz']}/s, SNR 5) | unmatched events within 3 ms of an artifact, per artifact |\n|---|---|---|---:|---:|\n")
    for m, nm in NAMES.items():
        out += (f"| {nm} | {fp[m]['mean']:.2f} / {fp[m]['max']:.2f} | {' · '.join(f'{v:.3f}' for v in dr[m])} | "
                f"{ar['f1'][m]:.3f} | {ar['fp_per_artifact'][m]:.2f} |\n")
    out += (f"\nSNR-5 clean-test F1 for reference: " + ", ".join(f"{NAMES[m].split(' (')[0]} {det['levels']['5'][m]['f1']:.3f}" for m in ("tc_dead", "lin32_int8", "mlp8_int8")) + ".\n")
    return out


def table_reduction():
    out = ""
    for key, title in (("", "Background hash velocity-tuned (default)"), ("_untuned", "Control: same recordings, hash untuned")):
        reps = red[key]["reps"]; cfg = red[key]["config"]
        out += (f"\n**{title}**: mean over {cfg['n_seeds']} simulated 32-ch arrays. Counts are sent as {cfg['count_bits']}-bit saturating "
                f"symbols (max saturated-bin fraction {cfg['saturated_bin_fraction_max']:.4f}). Add ≈{cfg['framing_bits_per_s_per_channel_1024ch']:.1f} b/s/ch "
                f"framing for a 1024-ch frame header (assumed 64 bit / 20 ms).\n\n")
        out += ("| Representation | fixed-width b/s/ch | Huffman b/s/ch (≥1 bit/bin) | adaptive arithmetic b/s/ch | decode R² mean (per seed) | ΔR² vs tuned threshold, 95% CI per seed |\n"
                "|---|---:|---:|---:|---|---|\n")
        for n, v in reps.items():
            b = v["bits_per_s"]
            f = lambda x: "—" if x is None else f"{x:,.0f}"
            ci = "—" if n == cfg["base"] else " ".join(f"[{d[0]:+.3f}, {d[1]:+.3f}]" for d in v["diff_vs_base_ci_per_seed"])
            ac = f(b["arithmetic"]) if b["arithmetic"] is not None else f"{f(b['entropy'])} (lossless est.)"
            out += (f"| {n} | {f(b['fixed'])} | {f(b['huffman'])} | {ac} | {v['r2_mean']:.3f} "
                    f"({', '.join(f'{x:.3f}' for x in v['r2_per_seed'])}) | {ci} |\n")
    return out


def table_embedded():
    out = ("| Model | MACs/inf | ops/sample (avg) | M4 .text (B) | M4 .rodata (B) | state/ch (B) | stack scratch (B) | host ns/sample | host ns/inference | bit-exact (events checked) |\n"
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for m, v in emb["models"].items():
        out += (f"| {m} | {v['macs_per_inference']} | {v['ops_per_sample']:.1f} | {v['m4_sizes_bytes']['text']} | {v['m4_sizes_bytes']['rodata']} | "
                f"{v['ram_bytes']['state']} | {v['ram_bytes']['activations_stack'] + v['ram_bytes']['window_stack']} | {v['host']['ns_per_sample_gated_model']:.2f} | "
                f"{v['host']['ns_per_inference']:.0f} | {'yes' if v['bit_exact'] else 'NO'} ({v['bitexact_events']:,}) |\n")
    a = emb["assumptions"]; th = emb["threshold_only_array_budget"]
    out += (f"\n**{a['n_array']}-channel budget** (mean candidate rate {emb['candidates_per_s_mean_over_test_levels']:.0f}/s per channel). "
            f"The energy column uses an **assumed** {a['pj_per_op'][0]}–{a['pj_per_op'][1]} pJ per integer op (not sourced). It scales linearly, so plug in your process.\n\n"
            "| Pipeline | classifier MAC/s per ch | total Mops/s (1024 ch) | of which classifier MMAC/s | state SRAM (KiB) | shared weights (B) | mW at assumed pJ/op |\n"
            "|---|---:|---:|---:|---:|---:|---|\n"
            f"| threshold only | 0 | {th['total_mops_per_s']:.0f} | 0 | — | 0 | {th['mw_at_assumed_pj_per_op'][0]:.2f}–{th['mw_at_assumed_pj_per_op'][1]:.2f} |\n")
    for m, v in emb["models"].items():
        b = v["array_budget"]
        out += (f"| gated {m} | {b['classifier_mac_per_s_per_channel']:,.0f} | {b['total_mops_per_s']:.0f} | {b['classifier_mmac_per_s']:.1f} | "
                f"{b['state_sram_kib']:.0f} | {b['weights_bytes_shared']} | {b['mw_at_assumed_pj_per_op'][0]:.2f}–{b['mw_at_assumed_pj_per_op'][1]:.2f} |\n")
    out += "\nSecondary, Cortex-M4 **estimate** (cycles only; MCU power depends on the part, so it is not quoted):\n\n| Model | kernel | MHz per channel |\n|---|---|---|\n"
    for m, v in emb["models"].items():
        for k, e in v["m4_estimate"].items():
            out += f"| {m} | {k} | {e['mhz_per_channel'][0]:.2f}–{e['mhz_per_channel'][1]:.2f} |\n"
    return out


def win(a, b):
    """'a vs b' with the clearly higher one (> 0.01) in bold."""
    if a is None or b is None:
        return f"{a} vs {b}"
    fa, fb = f"{a:.3f}", f"{b:.3f}"
    if b - a > 0.01: fb = f"**{fb}**"
    elif a - b > 0.01: fa = f"**{fa}**"
    return f"{fa} vs {fb}"


def table_real():
    if real is None:
        return "_not run_"
    pc, ag = real["cells"]["per_cell"], real["cells"]["aggregate"]
    M = [("mlp8", "MLP-8"), ("cnn8", "CNN-8"), ("lin32", "logistic"), ("tmpl", "matched filter")]
    fmt = lambda v: f"{v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] · {v['wins']}/{v['ties']}/{v['losses']} · p={v['sign_test_p']:.3g}"
    out = ("**Across cells** (one sample per cell = mean over its native ≥4σ electrodes). Entries are mean Δrecall vs the tuned "
           "threshold [95% bootstrap CI over cells] · wins/ties/losses (±0.01) · two-sided sign test p.\n\n"
           "| Scope | cells | comparison | " + " | ".join(n for _, n in M) + " |\n|---|---:|---|" + "---|" * len(M) + "\n")
    for scope, d in ag.items():
        for key, lab in (("label_free", "label-free rate-matched (primary)"), ("oracle_matched", "oracle matched rate")):
            out += f"| {scope} | {d['n_cells']} | {lab} | " + " | ".join(fmt(d[f'{key}_{m}']) for m, _ in M) + " |\n"
    out += ("\n**Per cell** (label-free Δrecall vs tuned threshold, native electrodes):\n\n"
            "| Cell | dataset | electrodes | SNR range | threshold recall | " + " | ".join(f"Δ {n}" for _, n in M) + " | MLP/threshold event-rate ratio |\n"
            "|---|---|---:|---|---:|" + "---:|" * len(M) + "---:|\n")
    nat = {}
    for r in real["results"]:
        if r["target_snr"] is None:
            nat.setdefault(r["recording"], []).append(r["label_free_rate_matched"]["tc_dead"]["recall"])
    for c, v in sorted(pc.items(), key=lambda kv: (kv[1]["dataset"], kv[0])):
        out += (f"| {c} | {v['dataset'].split(' (')[0]} | {v['n_electrodes']} | {min(v['snr']):.1f}–{max(v['snr']):.1f} | {np.mean(nat[c]):.3f} | "
                + " | ".join(f"{v[f'label_free_{m}']:+.3f}" for m, _ in M) + f" | {v['rate_ratio_label_free_mlp8']:.2f} |\n")
    mixes = [r for r in real["results"] if r["target_snr"] is not None]
    if mixes:
        out += "\n**Real-signal SNR sweep** (attenuated electrode + cross-recording noise; rows reuse the same cells, so not independent):\n\n"
        out += "| measured SNR | conditions | mean label-free Δ MLP-8 | mean Δ logistic | mean Δ matched filter |\n|---|---:|---:|---:|---:|\n"
        for lo, hi in ((0, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 99)):
            rr = [r for r in mixes if lo <= r["frontend_snr"] < hi]
            if rr:
                d = lambda m: np.mean([r["label_free_rate_matched"][m]["recall"] - r["label_free_rate_matched"]["tc_dead"]["recall"] for r in rr])
                out += f"| {lo:g}–{hi:g} | {len(rr)} | {d('mlp8'):+.3f} | {d('lin32'):+.3f} | {d('tmpl'):+.3f} |\n"
    bx = [r["c_bitexact"] for r in real["results"] if r["target_snr"] is None]
    out += (f"\nC pipeline bit-exact with Python on {sum(bx)}/{len(bx)} native real electrodes. "
            f"Cells with no electrode ≥ {real['min_frontend_snr']:g}σ (excluded): {len(real['cells_without_electrode'])} "
            f"({', '.join(c['recording'] for c in real['cells_without_electrode']) or 'none'}). "
            f"Shortlisted electrodes dropped because the full recording was < {real['min_frontend_snr']:g}σ: {len(real['excluded_electrodes'])}. "
            f"Mixes dropped (SNR not below native): {len(real['excluded_mixes'])}.\n")
    return out


def fig_real():
    if real is None:
        return
    pc = real["cells"]["per_cell"]
    cells = sorted(pc, key=lambda c: (pc[c]["dataset"], pc[c]["label_free_mlp8"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    for j, (m, c, mk, lab) in enumerate([("mlp8", "#59a14f", "^", "MLP-8 int8"), ("lin32", "#f28e2b", "s", "logistic int8"),
                                         ("tmpl", "#4e79a7", "o", "matched filter int8")]):
        ax.scatter(np.arange(len(cells)) + (j - 1) * 0.22, [pc[c][f"label_free_{m}"] for c in cells], color=c, marker=mk, s=22, label=lab)
    ax.axhline(0, color="#999", lw=0.8)
    for i, c in enumerate(cells):
        if i and pc[c]["dataset"] != pc[cells[i - 1]]["dataset"]:
            ax.axvline(i - 0.5, color="#ccc", lw=0.8, ls="--")
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([c.replace("cortex/", "") for c in cells], rotation=75, fontsize=5.5)
    ax.set_ylabel("Δ GT recall vs tuned threshold\n(label-free, same event budget)")
    ax.set_title("Per cell, native electrodes (dashed line separates cortex | retina)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axs[1]
    rr = real["results"]
    for key, c, mk, lab in [("tc_dead", "#444", "o", "tuned threshold"), ("lin32", "#f28e2b", "s", "logistic int8"), ("mlp8", "#59a14f", "^", "MLP-8 int8")]:
        ax.scatter([r["frontend_snr"] for r in rr], [r["label_free_rate_matched"][key]["recall"] for r in rr], color=c, marker=mk, s=12, alpha=0.7, label=lab)
    ax.set_xlabel("measured front-end SNR (native + real-signal mixes)"); ax.set_ylabel("GT recall (label-free)")
    ax.set_title("All electrodes and mixes", fontsize=9); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Real paired ground truth, frozen int8 models (Yger 2018 retina + Marques-Smith 2018 cortex)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "real_recall_matched_rate.png"); plt.close(fig)


def fig_f1():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.array([float(l) for l in LV])
    for m, c, ls in [("tc4.5", "#888", "--"), ("tc_dead", "#444", "-"), ("neo", "#b07aa1", "-"), ("gate", "#bbb", ":"),
                     ("lin32_int8", "#f28e2b", "-"), ("mlp8_int8", "#59a14f", "-"), ("cnn8_int8", "#4e79a7", "--")]:
        f = np.array([det["levels"][l][m]["f1"] for l in LV])
        ci = np.array([det["levels"][l][m]["f1_ci"] for l in LV]).T
        ax.errorbar(x, f, yerr=[f - ci[0], ci[1] - f], color=c, ls=ls, marker="o", ms=3.5, capsize=2, label=NAMES[m])
    ax.set_xlabel("unit SNR (|filtered trough| / robust σ)"); ax.set_ylabel("F1 (±0.5 ms match)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("Synthetic detection F1 vs SNR (held-out channels, 95% bootstrap CI)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "f1_vs_snr.png"); plt.close(fig)


def fig_rate():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, key, title in zip(axs, ("", "_untuned"), ("hash tuned", "control: hash untuned")):
        for i, (n, v) in enumerate(red[key]["reps"].items()):
            r = np.array(v["r2_per_seed"]); b = v["bits_per_s"]
            rate = b["arithmetic"] if b["arithmetic"] is not None else b["entropy"]
            ax.errorbar(rate, r.mean(), yerr=[[r.mean() - r.min()], [r.max() - r.mean()]],
                        marker="osDv^<>ph*"[i % 10], ms=6, capsize=2, label=n, ls="none")
        ax.set_xscale("log"); ax.set_xlabel("bits/s per channel (adaptive arithmetic; raw: lossless est.)")
        ax.set_title(title, fontsize=9.5)
    axs[0].set_ylabel("held-out velocity R² (mean, seed range)")
    axs[1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "bandwidth_vs_r2.png"); plt.close(fig)


def fig_pareto(level="4"):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for m, nm in NAMES.items():
        ax.scatter(OPS[m], det["levels"][level][m]["f1"], s=30)
        ax.annotate(nm.replace(" int8", ""), (OPS[m], det["levels"][level][m]["f1"]), fontsize=6, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("integer ops per sample (incl. 13-op front-end)"); ax.set_ylabel(f"F1 @ SNR {level}")
    ax.set_title("Compute vs accuracy (synthetic)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "ops_vs_f1.png"); plt.close(fig)


def fig_trace():
    from spike_edge import FS, qnn
    from spike_edge.datasets import split, windows
    from spike_edge.frontend import threshold_events
    c = split("test_6", 3)
    _, q, meta = qnn.load(ROOT / "models" / "mlp8.npz")
    g = c["gt"][c["gt"] > 12 * FS]
    dens = [np.sum((g >= a) & (g < a + 0.15 * FS)) for a in g[:200]]   # busiest 150 ms window
    t0 = int(g[int(np.argmax(dens))]) - FS // 100; t1 = t0 + int(0.15 * FS)
    y = c["y"]
    tc = threshold_events(y, int(round(4.5 * c["sigma"])))
    idx, w = windows(y, threshold_events(y, int(round(3 * c["sigma"]))))
    lrn = idx[qnn.run_int(q, qnn.quantize_input(w, qnn.input_gain(c["sigma"]))) > meta["logit_thr"]]
    fig, axs = plt.subplots(2, 1, figsize=(8, 3.8), sharex=True)
    tt = np.arange(t0, t1) / FS * 1000
    axs[0].plot(tt, c["raw"][t0:t1], lw=0.5, color="#555"); axs[0].set_ylabel("raw (µV)")
    axs[1].plot(tt, y[t0:t1] / 4, lw=0.5, color="#333"); axs[1].set_ylabel("front-end (µV)")
    for arr, col, lab, off in [(c["gt"], "k", "ground truth", 1.0), (tc, "#e15759", "TC −4.5σ", 1.25), (lrn, "#59a14f", "gated MLP-8", 1.5)]:
        s = arr[(arr >= t0) & (arr < t1)]
        axs[1].scatter(s / FS * 1000, np.full(len(s), -off * 5 * c["sigma"] / 4 - 10), marker="|", s=60, color=col, label=lab)
    axs[1].axhline(-4.5 * c["sigma"] / 4, color="#e15759", lw=0.6, ls="--")
    axs[1].legend(fontsize=7, frameon=False, ncol=3, loc="upper right"); axs[1].set_xlabel("time (ms)")
    fig.suptitle("Synthetic channel, nominal SNR 6 (test split): ticks = ground truth / detections", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "example_trace.png"); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_f1(); fig_rate(); fig_pareto(); fig_trace(); fig_real()
    blocks = {"real": table_real(), "detection": table_detection(), "robustness": table_robustness(),
              "reduction": table_reduction(), "embedded": table_embedded()}
    (R / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in blocks.items()))
    readme = ROOT / "README.md"
    if readme.exists():
        s = readme.read_text()
        for k, v in blocks.items():
            s = re.sub(rf"(<!-- BEGIN {k} -->\n).*?(<!-- END {k} -->)", lambda mt: mt.group(1) + v.strip() + "\n" + mt.group(2), s, flags=re.S)
        readme.write_text(s)


if __name__ == "__main__":
    main()
