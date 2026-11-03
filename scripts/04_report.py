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
LV = list(det["levels"])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})

NAMES = {"tc4.5": "Threshold −4.5σ", "tc_best": f"Threshold −{det['selected']['tc_best_k']}σ (tuned)",
         "tc_dead": f"Threshold −{det['selected']['tc_dead_k']}σ, {det['selected']['tc_dead_samples']/20:g} ms dead (tuned)",
         "neo": "NEO (tuned)", "gate": "Gate only (−3σ)", "cnn8_int8": "Gated CNN-8 int8",
         "cnn4_int8": "Gated CNN-4 int8", "mlp8_int8": "Gated MLP-8 int8"}
OPS = {"tc4.5": emb["baselines"]["threshold_ops_per_sample"], "tc_best": emb["baselines"]["threshold_ops_per_sample"],
       "tc_dead": emb["baselines"]["threshold_ops_per_sample"], "neo": emb["baselines"]["neo_ops_per_sample"],
       "gate": emb["baselines"]["threshold_ops_per_sample"]}
for m in ("cnn8", "cnn4", "mlp8"):
    OPS[f"{m}_int8"] = emb["models"][m]["ops_per_sample"]


def fmt_f1(v):
    return f"{v['f1']:.3f} [{v['f1_ci'][0]:.2f}–{v['f1_ci'][1]:.2f}]"


def table_detection():
    hdr = "| Detector | ops/sample | flash (B) | " + " | ".join(f"F1 @ SNR {l}" for l in LV) + " |\n"
    hdr += "|---|---:|---:|" + "---|" * len(LV) + "\n"
    rows = []
    for m, nm in NAMES.items():
        mm = m.split("_")[0]
        flash = f"{emb['models'][mm]['m4_sizes_bytes']['rodata']}" if mm in emb["models"] else "0"
        rows.append(f"| {nm} | {OPS[m]:.1f} | {flash} | " + " | ".join(fmt_f1(det["levels"][l][m]) for l in LV) + " |")
    pr = "\n\n| Detector | " + " | ".join(f"P / R @ {l}" for l in LV) + " |\n|---|" + "---|" * len(LV) + "\n"
    pr += "\n".join(f"| {NAMES[m]} | " + " | ".join(f"{det['levels'][l][m]['precision']:.2f} / {det['levels'][l][m]['recall']:.2f}" for l in LV) + " |" for m in NAMES)
    fq = "\n\n| Model | MACs/inference | val F1 float | val F1 int8 |\n|---|---:|---:|---:|\n"
    fq += "\n".join(f"| {m} | {det['selected'][m]['macs']} | {det['selected'][m]['val_f1_float']:.4f} | {det['selected'][m]['val_f1_int8']:.4f} |" for m in ("cnn8", "cnn4", "mlp8"))
    return hdr + "\n".join(rows) + pr + fq


def table_reduction():
    out = ""
    for key, title in (("", "Background hash velocity-tuned (default)"), ("_untuned", "Control: same recordings, hash untuned")):
        reps = red[key]["reps"]
        out += f"\n**{title}** — mean over {red[key]['config']['n_seeds']} simulated 32-ch arrays; per-seed R² in brackets\n\n"
        out += "| Representation | bits/s/ch (fixed-width) | bits/s/ch (entropy-coded) | decode R² | ΔR² vs TC −4.5σ (95% CI, seed 0) |\n|---|---:|---:|---:|---|\n"
        for n, v in reps.items():
            d = v["diff_vs_tc_ci_per_seed"][0]
            out += (f"| {n} | {v['bits_per_s_fixed']:,.0f} | {v['bits_per_s_entropy']:,.0f} | {v['r2_mean']:.3f} "
                    f"({', '.join(f'{x:.3f}' for x in v['r2_per_seed'])}) | {'—' if n.startswith('threshold') else f'[{d[0]:+.3f}, {d[1]:+.3f}]'} |\n")
    return out


def table_embedded():
    out = "| Model | MACs/inf | ops/sample (avg) | M4 .text (B) | M4 .rodata (B) | RAM/ch state (B) | shared scratch (B) | host ns/sample | host ns/inference | bit-exact (events checked) |\n"
    out += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    for m, v in emb["models"].items():
        out += (f"| {m} | {v['macs_per_inference']} | {v['ops_per_sample']:.1f} | {v['m4_sizes_bytes']['text']} | {v['m4_sizes_bytes']['rodata']} | "
                f"{v['ram_bytes']['state']} | {v['ram_bytes']['activations'] + v['ram_bytes']['window']} | {v['host']['ns_per_sample_gated_model']:.2f} | "
                f"{v['host']['ns_per_inference']:.0f} | {'yes' if v['bit_exact'] else 'NO'} ({v['bitexact_events']:,}) |\n")
    out += f"\nThreshold-only path on host: {emb['models']['cnn8']['host']['ns_per_sample_threshold']:.2f} ns/sample; front-end alone {emb['models']['cnn8']['host']['ns_per_sample_frontend']:.2f} ns/sample.\n"
    out += "\n**Cortex-M4 ESTIMATE (not measured on hardware)** per channel at 20 kHz, mean candidate rate "
    out += f"{emb['candidates_per_s_mean_over_test_levels']:.0f}/s:\n\n| Model | kernel | MHz/channel | µW/channel |\n|---|---|---|---|\n"
    for m, v in emb["models"].items():
        for k, e in v["m4_estimate"].items():
            out += f"| {m} | {k} | {e['mhz_per_channel'][0]:.2f}–{e['mhz_per_channel'][1]:.2f} | {e['uw_per_channel'][0]:.0f}–{e['uw_per_channel'][1]:.0f} |\n"
    return out


REAL = R / "real_yger2018.json"


def win(a, b):
    """'a vs b' with the clearly higher one (> 0.01) in bold."""
    if a is None or b is None:
        return f"{a} vs {b}"
    fa, fb = f"{a:.3f}", f"{b:.3f}"
    if b - a > 0.01: fb = f"**{fb}**"
    elif a - b > 0.01: fa = f"**{fa}**"
    return f"{fa} vs {fb}"


def table_real():
    if not REAL.exists():
        return "_not run_"
    rr = json.loads(REAL.read_text())["results"]
    out = ("| Recording / electrode | condition | front-end SNR | GT rate (Hz) | Threshold −4.5σ | Threshold tuned | NEO | Gated MLP-8 int8 | Gated CNN-8 int8 | "
           "matched rate (ev/s) | **recall @ matched rate: threshold vs MLP-8** |\n|---|---|---:|---:|---|---|---|---|---|---:|---|\n")
    for r in sorted(rr, key=lambda r: (r["recording"], r["electrode"], -r["frontend_snr"])):
        f = r["frozen"]; m = r["recall_at_matched_rate"]
        cell = lambda k: f"{f[k]['recall']:.3f} @ {f[k]['events_per_s']:.0f}/s"
        cond = "native" if r["target_snr"] is None else f"mix (×{r['attenuation']:.2f} + donor e{r['donor']})"
        mm = lambda v: "n/a" if v is None else f"{v:.3f}"
        out += (f"| {r['recording']} / e{r['electrode']} | {cond} | {r['frontend_snr']:.1f} | {r['gt_rate_hz']:.1f} | {cell('tc4.5')} | {cell('tc_dead')} | "
                f"{cell('neo')} | {cell('mlp8_int8')} | {cell('cnn8_int8')} | {r['matched_rate_events_per_s']:.0f} | "
                f"{win(m['threshold_tuned_dead'], m['gated_mlp8'])} |\n")
    bx = [r["c_bitexact_native"] for r in rr if r["target_snr"] is None]
    out += f"\nC pipeline bit-exact with Python on the native real electrodes: {sum(bx)}/{len(bx)}.\n"
    return out


def fig_real():
    if not REAL.exists():
        return
    rr = json.loads(REAL.read_text())["results"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for key, c, lab in [("threshold_tuned_dead", "#444", "threshold (tuned k sweep)"), ("gated_mlp8", "#59a14f", "gated MLP-8 int8 (logit sweep)")]:
        pts = sorted((r["frontend_snr"], r["recall_at_matched_rate"][key]) for r in rr if r["recall_at_matched_rate"][key] is not None)
        ax.scatter(*zip(*pts), color=c, s=22, label=lab)
    ax.set_xlabel("measured front-end SNR of the GT unit"); ax.set_ylabel("GT recall at matched event rate")
    ax.set_title("Real paired GT (Yger 2018 retina), frozen models", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right"); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(FIG / "real_recall_matched_rate.png"); plt.close(fig)


def fig_f1():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.array([float(l) for l in LV])
    for m, c, ls in [("tc4.5", "#888", "--"), ("tc_dead", "#444", "-"), ("neo", "#b07aa1", "-"), ("gate", "#bbb", ":"),
                     ("mlp8_int8", "#59a14f", "-"), ("cnn8_int8", "#4e79a7", "-")]:
        f = [det["levels"][l][m]["f1"] for l in LV]
        ci = np.array([det["levels"][l][m]["f1_ci"] for l in LV]).T
        ax.errorbar(x, f, yerr=[f - ci[0], ci[1] - f], color=c, ls=ls, marker="o", ms=3.5, capsize=2, label=NAMES[m])
    ax.set_xlabel("unit SNR (|filtered trough| / robust σ)"); ax.set_ylabel("F1 (±0.5 ms match)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("Detection F1 vs SNR (held-out synthetic channels, 95% bootstrap CI)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "f1_vs_snr.png"); plt.close(fig)


def fig_rate():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, key, title in zip(axs, ("", "_untuned"), ("hash tuned", "control: hash untuned")):
        for i, (n, v) in enumerate(red[key]["reps"].items()):
            r = np.array(v["r2_per_seed"])
            ax.errorbar(v["bits_per_s_entropy"], r.mean(), yerr=[[r.mean() - r.min()], [r.max() - r.mean()]],
                        marker="os^Dv<>"[i], ms=6, capsize=2, label=n, ls="none")
        ax.set_xscale("log"); ax.set_xlabel("bits/s per channel (entropy-coded)"); ax.set_title(title, fontsize=9.5)
    axs[0].set_ylabel("held-out velocity R²")
    axs[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "bandwidth_vs_r2.png"); plt.close(fig)


def fig_pareto(level="4"):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m, nm in NAMES.items():
        ax.scatter(OPS[m], det["levels"][level][m]["f1"], s=30)
        ax.annotate(nm, (OPS[m], det["levels"][level][m]["f1"]), fontsize=6.5, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("integer ops per sample (incl. 13-op front-end)"); ax.set_ylabel(f"F1 @ SNR {level}")
    ax.set_title("Compute vs accuracy", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "ops_vs_f1.png"); plt.close(fig)


def fig_trace():
    from spike_edge import FS, qnn
    from spike_edge.datasets import split, windows
    from spike_edge.frontend import threshold_events
    c = split("test_6", 3)
    _, q, meta = qnn.load(ROOT / "models" / "cnn8.npz")
    g = c["gt"][c["gt"] > 12 * FS]
    dens = [np.sum((g >= a) & (g < a + 0.15 * FS)) for a in g[:200]]   # busiest 150 ms window
    t0 = int(g[int(np.argmax(dens))]) - FS // 100; t1 = t0 + int(0.15 * FS)
    y = c["y"]
    tc = threshold_events(y, int(round(4.5 * c["sigma"])))
    idx, w = windows(y, threshold_events(y, int(round(3 * c["sigma"]))))
    cnn = idx[qnn.run_int(q, qnn.quantize_input(w, qnn.input_gain(c["sigma"]))) > meta["logit_thr"]]
    fig, axs = plt.subplots(2, 1, figsize=(8, 3.8), sharex=True)
    tt = np.arange(t0, t1) / FS * 1000
    axs[0].plot(tt, c["raw"][t0:t1], lw=0.5, color="#555"); axs[0].set_ylabel("raw (µV)")
    axs[1].plot(tt, y[t0:t1] / 4, lw=0.5, color="#333"); axs[1].set_ylabel("front-end (µV)")
    for arr, col, lab, off in [(c["gt"], "k", "ground truth", 1.0), (tc, "#e15759", "TC −4.5σ", 1.25), (cnn, "#4e79a7", "gated CNN-8", 1.5)]:
        s = arr[(arr >= t0) & (arr < t1)]
        axs[1].scatter(s / FS * 1000, np.full(len(s), -off * 5 * c["sigma"] / 4 - 10), marker="|", s=60, color=col, label=lab)
    axs[1].axhline(-4.5 * c["sigma"] / 4, color="#e15759", lw=0.6, ls="--")
    axs[1].legend(fontsize=7, frameon=False, ncol=3, loc="upper right"); axs[1].set_xlabel("time (ms)")
    fig.suptitle("Synthetic channel, nominal SNR 6 (test split): ticks = ground truth / detections", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "example_trace.png"); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_f1(); fig_rate(); fig_pareto(); fig_trace(); fig_real()
    blocks = {"real": table_real(), "detection": table_detection(), "reduction": table_reduction(), "embedded": table_embedded()}
    (R / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in blocks.items()))
    readme = ROOT / "README.md"
    if readme.exists():
        s = readme.read_text()
        for k, v in blocks.items():
            s = re.sub(rf"(<!-- BEGIN {k} -->\n).*?(<!-- END {k} -->)", lambda mt: mt.group(1) + v.strip() + "\n" + mt.group(2), s, flags=re.S)
        readme.write_text(s)


if __name__ == "__main__":
    main()
