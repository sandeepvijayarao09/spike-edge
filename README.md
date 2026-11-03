# spike-edge

**Spike detection and data reduction under implant-like compute and bandwidth limits.**
An integer-only front-end, threshold and NEO baselines, and a tiny int8 classifier that
runs only on threshold candidates; C that matches the Python bit for bit; and a
bandwidth-vs-decoding comparison of what an implant could send over the radio.

> **Models are trained and tuned only on my own simulator.** They are then tested, frozen, on real paired
> ground truth (3 retinal ganglion cells, [§0](#0-real-paired-ground-truth-frozen-models-no-re-tuning)).
> The detection, bandwidth and decoding tables in §1–2 are synthetic
> (see [Limitations](#limitations-read-this-first)). The tables are produced by `scripts/04_report.py`
> straight from `results/*.json`.

![F1 vs SNR](figures/f1_vs_snr.png)

## TL;DR

* **Real data (the result that matters most).** On Yger et al. 2018 paired recordings (3 cells,
  5 electrodes plus a real-signal SNR sweep, 20 conditions), the frozen int8 MLP is compared with the
  tuned threshold at the same event rate. It has higher ground-truth recall in **13/20** conditions,
  ties (|Δ| ≤ 0.01) in 6, and loses in 1. The median Δrecall is +0.016. The gains concentrate at SNR ≈ 3–4.5:
  +0.14 on the one natively low-SNR electrode (0.58 vs 0.44 at SNR 3.2). On one cell (20170622), the
  attenuated mixes show no gain (Δ from −0.018 to +0.004). The synthetic F1 gaps below are larger
  than this, as expected for a simulator. Conditions share neurons, so these are not 20 independent
  samples. Precision is not measurable here.
* **Detection.** A −3σ gate followed by a 264-MAC int8 MLP beats the strongest threshold baseline
  (k and dead time both tuned on val) at every SNR from 3 to 8. F1 at SNR 4 is **0.875 vs 0.758**; at
  SNR 3 it is **0.673 vs 0.486**. The 95% channel-bootstrap CIs do not overlap from SNR 4 up. At SNR 2
  every method fails (F1 ≤ 0.31). There, the raw −3σ gate has the best F1 because the classifier trades
  recall for precision.
* **Model size barely matters here.** MLP-8 (264 MACs), CNN-4 (592) and CNN-8 (1760) are within CI of
  one another, and int8 post-training quantisation changes val F1 by less than 0.001. The smallest model
  is the right choice.
* **Compute.** The classifier runs only on about 49 candidates/s, so it adds about 1–6 ops/sample to the
  15-op front-end plus threshold. The C code is **982 B of Cortex-M4 code, 284–568 B of weights, and
  192 B of state per channel**. It is **bit-exact** with the Python integer reference across
  6.4 M samples / 21 k events for all three models.
* **Bandwidth.** Learned spike counts cost **49 b/s/ch** entropy-coded, about 1/1900 of the 92 kb/s
  lossless-raw estimate (200 kb/s uncompressed). They decode velocity with R² **0.782** vs **0.709** for
  −4.5σ threshold counts, The paired block-bootstrap ΔR² CI excludes 0 on all 3 simulated arrays (seed 0: [+0.07, +0.13]). 8-bit spike-band power reaches about
  the same R² (0.797) at 5× the bits. Combining the two gives the best on-chip result (0.810, 315 b/s).
  The control makes the background hash untuned on otherwise identical recordings; SBP drops to 0.780
  and learned counts are unchanged (0.783).
* **Still on the table:** perfect detection (oracle unsorted, 0.871) and sorting (oracle sorted, 0.910)
  would add another 0.09–0.13 R² beyond the learned detector.

## Pipeline

```
raw int16 @ 20 kHz ──► integer band-pass (~200 Hz–2.3 kHz, 13 add/shift ops, no multiplies)
                        │
                        ├─► spike-band power: Σ|y| per 20 ms bin ──► 8-bit value
                        │
                        └─► gate: y < −3σ ──► trough search (0.75 ms) ──► 1 ms dead time
                                               │
                                               └─► 32-sample window ──► int8 CNN/MLP ──► spike / not spike
                                                                                        └─► count per 20 ms bin
```

* **Front-end** ([spike_edge/frontend.py](spike_edge/frontend.py), [c/spike_edge.c](c/spike_edge.c)): two first-order
  EMA high-passes and one EMA low-pass on Q8 integer state. These are shift-only filters, the kind a
  mixed-signal ASIC can afford. The same front-end feeds every detector, so the detector comparison is fair.
* **Calibration**: σ is a MAD estimate over the first 10 s, with no labels. All thresholds, the int8
  input gain and the SBP scale are set from it. Scoring starts after the calibration window.
* **Baselines**: threshold at −4.5σ, threshold with k tuned on val, threshold with k and dead time both
  tuned on val, and NEO (ψ = y² − y₋y₊, Bartlett-smoothed, C·mean(ψ) with C tuned on val).
* **Learned**: a −3σ gate proposes candidates, and a classifier sees 0.5 ms before and 1.1 ms after the trough.
  There are three sizes (`cnn8`, `cnn4`, `mlp8`). Each is trained in float (PyTorch), then post-training
  quantised to int8 (per-channel weight scales, int32 accumulation, TFLite-style fixed-point requantisation).
  The classifier runs only on candidates (≈28–65/s on the test sets), never per sample.
* **Scoring**: one-to-one greedy matching within ±0.5 ms. Operating points are chosen on a validation
  split (mixed SNR 1.5–10). Test channels use disjoint seeds, so templates, noise, spike trains and drift
  are all disjoint. CIs come from a channel bootstrap.

## Results

### 0. Real paired ground truth (frozen models, no re-tuning)

The data is [Yger et al. 2018](https://elifesciences.org/articles/34518) from
[Zenodo 1205233](https://zenodo.org/records/1205233) (CC-BY-4.0): mouse retina on a 252-electrode MEA at
20 kHz, plus loose-patch spike times for one ganglion cell per recording. I evaluate each electrode where
that cell's spike-triggered trough is ≥ 4σ, as an independent implant-style channel. The signal is
rescaled to a robust σ of 5 counts and clipped to 10 bits, as in the simulator. Every threshold and model
is frozen from the synthetic validation set.

Only one neuron per recording is labelled, so **precision cannot be measured**. Events not matched to it
may be real spikes from other cells. I therefore report ground-truth recall at each frozen operating
point and the event rate. The fair comparison is **recall at a matched event rate**: each detector's
threshold is swept, and recall is read off at the event rate of the tuned threshold baseline. Rows marked
"mix" are a real-signal SNR sweep. The electrode is attenuated by a factor a and real noise from a distant
electrode, where the cell is invisible, is added, matched in front-end σ. Waveforms, noise and other
neurons all stay real. The x-axis is the *measured* SNR of the mixture.

<!-- BEGIN real -->
| Recording / electrode | condition | front-end SNR | GT rate (Hz) | Threshold −4.5σ | Threshold tuned | NEO | Gated MLP-8 int8 | Gated CNN-8 int8 | matched rate (ev/s) | **recall @ matched rate: threshold vs MLP-8** |
|---|---|---:|---:|---|---|---|---|---|---:|---|
| 20160426/patch_2 / e232 | mix (×0.84 + donor e79) | 7.9 | 4.4 | 0.999 @ 5/s | 0.982 @ 11/s | 0.998 @ 11/s | 0.996 @ 5/s | 0.996 @ 5/s | 11 | 0.982 vs **0.996** |
| 20160426/patch_2 / e232 | mix (×0.63 + donor e95) | 5.9 | 4.4 | 0.953 @ 4/s | 0.994 @ 6/s | 0.870 @ 7/s | 0.999 @ 4/s | 0.999 @ 4/s | 6 | 0.994 vs 0.999 |
| 20160426/patch_2 / e232 | native | 4.7 | 4.4 | 0.754 @ 3/s | 0.947 @ 5/s | 0.672 @ 5/s | 0.994 @ 4/s | 0.994 @ 4/s | 5 | 0.947 vs **0.994** |
| 20160426/patch_3 / e130 | mix (×0.77 + donor e110) | 12.0 | 9.4 | 1.000 @ 14/s | 0.983 @ 19/s | 1.000 @ 22/s | 0.999 @ 17/s | 0.999 @ 17/s | 19 | 0.983 vs **0.999** |
| 20160426/patch_3 / e130 | mix (×0.64 + donor e71) | 10.1 | 9.4 | 1.000 @ 14/s | 0.979 @ 24/s | 0.997 @ 32/s | 1.000 @ 15/s | 1.000 @ 16/s | 24 | 0.979 vs **1.000** |
| 20160426/patch_3 / e130 | native | 7.8 | 9.4 | 0.999 @ 12/s | 0.994 @ 14/s | 0.976 @ 17/s | 1.000 @ 13/s | 1.000 @ 13/s | 14 | 0.994 vs 1.000 |
| 20160426/patch_3 / e130 | mix (×0.51 + donor e79) | 6.5 | 9.4 | 0.992 @ 11/s | 0.996 @ 13/s | 0.870 @ 23/s | 0.999 @ 12/s | 0.999 @ 12/s | 13 | 0.996 vs 0.999 |
| 20160426/patch_3 / e130 | mix (×0.39 + donor e126) | 3.1 | 9.4 | 0.181 @ 2/s | 0.467 @ 8/s | 0.133 @ 13/s | 0.409 @ 4/s | 0.431 @ 4/s | 8 | 0.467 vs **0.655** |
| 20160426/patch_3 / e228 | native | 8.5 | 9.4 | 0.996 @ 16/s | 0.979 @ 22/s | 0.996 @ 25/s | 0.996 @ 20/s | 0.996 @ 20/s | 22 | 0.979 vs **0.996** |
| 20160426/patch_3 / e228 | mix (×0.71 + donor e110) | 4.8 | 9.4 | 0.815 @ 12/s | 0.953 @ 14/s | 0.611 @ 16/s | 0.988 @ 15/s | 0.989 @ 15/s | 14 | 0.953 vs **0.986** |
| 20160426/patch_3 / e228 | mix (×0.59 + donor e71) | 4.4 | 9.4 | 0.700 @ 11/s | 0.909 @ 16/s | 0.417 @ 14/s | 0.979 @ 14/s | 0.980 @ 15/s | 16 | 0.909 vs **0.984** |
| 20160426/patch_3 / e228 | mix (×0.47 + donor e79) | 3.5 | 9.4 | 0.362 @ 6/s | 0.670 @ 11/s | 0.230 @ 8/s | 0.912 @ 13/s | 0.907 @ 13/s | 11 | 0.670 vs **0.810** |
| 20160426/patch_3 / e228 | mix (×0.35 + donor e126) | 2.7 | 9.4 | 0.084 @ 1/s | 0.285 @ 5/s | 0.122 @ 13/s | 0.429 @ 7/s | 0.426 @ 6/s | 5 | 0.285 vs **0.338** |
| 20170622/patch1 / e146 | native | 3.2 | 15.1 | 0.177 @ 4/s | 0.443 @ 14/s | 0.414 @ 39/s | 0.664 @ 18/s | 0.625 @ 16/s | 14 | 0.443 vs **0.582** |
| 20170622/patch1 / e146 | mix (×0.93 + donor e126) | 3.0 | 15.1 | 0.143 @ 3/s | 0.397 @ 12/s | 0.265 @ 28/s | 0.596 @ 15/s | 0.548 @ 14/s | 12 | 0.397 vs **0.529** |
| 20170622/patch1 / e178 | native | 8.6 | 15.1 | 0.998 @ 21/s | 0.982 @ 33/s | 0.999 @ 43/s | 0.998 @ 38/s | 0.998 @ 35/s | 33 | 0.982 vs **0.997** |
| 20170622/patch1 / e178 | mix (×0.70 + donor e128) | 6.0 | 15.1 | 0.957 @ 16/s | 0.986 @ 20/s | 0.972 @ 36/s | 0.980 @ 21/s | 0.971 @ 20/s | 20 | 0.986 vs 0.978 |
| 20170622/patch1 / e178 | mix (×0.58 + donor e43) | 5.0 | 15.1 | 0.793 @ 12/s | 0.936 @ 17/s | 0.899 @ 34/s | 0.919 @ 17/s | 0.888 @ 16/s | 17 | **0.936** vs 0.918 |
| 20170622/patch1 / e178 | mix (×0.46 + donor e222) | 3.9 | 15.1 | 0.418 @ 6/s | 0.719 @ 12/s | 0.674 @ 29/s | 0.697 @ 12/s | 0.645 @ 11/s | 12 | 0.719 vs 0.723 |
| 20170622/patch1 / e178 | mix (×0.35 + donor e126) | 3.1 | 15.1 | 0.186 @ 3/s | 0.507 @ 10/s | 0.134 @ 15/s | 0.314 @ 6/s | 0.287 @ 5/s | 10 | 0.507 vs 0.507 |

C pipeline bit-exact with Python on the native real electrodes: 5/5.
<!-- END real -->

![real](figures/real_recall_matched_rate.png)

### 1. Detection: detector × F1 × ops × memory

32 held-out channels × 30 s scored per SNR level, with 1–2 ground-truth units per channel at the nominal SNR ±10%.

<!-- BEGIN detection -->
| Detector | ops/sample | flash (B) | F1 @ SNR 2 | F1 @ SNR 3 | F1 @ SNR 4 | F1 @ SNR 5 | F1 @ SNR 6 | F1 @ SNR 8 |
|---|---:|---:|---|---|---|---|---|---|
| Threshold −4.5σ | 15.0 | 0 | 0.025 [0.02–0.03] | 0.209 [0.16–0.25] | 0.540 [0.47–0.61] | 0.757 [0.70–0.81] | 0.890 [0.86–0.92] | 0.938 [0.92–0.96] |
| Threshold −4.0σ (tuned) | 15.0 | 0 | 0.079 [0.06–0.10] | 0.381 [0.32–0.44] | 0.701 [0.64–0.76] | 0.840 [0.80–0.88] | 0.920 [0.90–0.94] | 0.904 [0.87–0.94] |
| Threshold −3.75σ, 2 ms dead (tuned) | 15.0 | 0 | 0.139 [0.11–0.17] | 0.486 [0.42–0.54] | 0.758 [0.71–0.80] | 0.867 [0.84–0.90] | 0.935 [0.92–0.95] | 0.950 [0.94–0.96] |
| NEO (tuned) | 24.0 | 0 | 0.138 [0.12–0.15] | 0.370 [0.32–0.42] | 0.662 [0.61–0.71] | 0.766 [0.73–0.80] | 0.848 [0.82–0.88] | 0.928 [0.91–0.95] |
| Gate only (−3σ) | 15.0 | 0 | 0.308 [0.27–0.34] | 0.561 [0.52–0.60] | 0.711 [0.68–0.74] | 0.710 [0.68–0.73] | 0.716 [0.68–0.75] | 0.690 [0.65–0.73] |
| Gated CNN-8 int8 | 20.6 | 568 | 0.219 [0.17–0.28] | 0.676 [0.60–0.74] | 0.877 [0.83–0.91] | 0.945 [0.93–0.96] | 0.972 [0.97–0.98] | 0.977 [0.97–0.98] |
| Gated CNN-4 int8 | 17.2 | 284 | 0.234 [0.18–0.29] | 0.686 [0.61–0.75] | 0.879 [0.84–0.92] | 0.943 [0.93–0.95] | 0.969 [0.96–0.97] | 0.975 [0.97–0.98] |
| Gated MLP-8 int8 | 15.9 | 428 | 0.216 [0.16–0.27] | 0.673 [0.59–0.74] | 0.875 [0.83–0.91] | 0.944 [0.93–0.96] | 0.972 [0.97–0.98] | 0.978 [0.97–0.98] |

| Detector | P / R @ 2 | P / R @ 3 | P / R @ 4 | P / R @ 5 | P / R @ 6 | P / R @ 8 |
|---|---|---|---|---|---|---|
| Threshold −4.5σ | 0.78 / 0.01 | 0.97 / 0.12 | 0.99 / 0.37 | 0.98 / 0.62 | 0.98 / 0.82 | 0.91 / 0.97 |
| Threshold −4.0σ (tuned) | 0.69 / 0.04 | 0.91 / 0.24 | 0.96 / 0.55 | 0.95 / 0.75 | 0.94 / 0.91 | 0.84 / 0.98 |
| Threshold −3.75σ, 2 ms dead (tuned) | 0.60 / 0.08 | 0.85 / 0.34 | 0.94 / 0.63 | 0.94 / 0.80 | 0.95 / 0.92 | 0.95 / 0.95 |
| NEO (tuned) | 0.21 / 0.10 | 0.48 / 0.30 | 0.75 / 0.59 | 0.79 / 0.74 | 0.83 / 0.86 | 0.90 / 0.95 |
| Gate only (−3σ) | 0.32 / 0.29 | 0.49 / 0.66 | 0.61 / 0.86 | 0.57 / 0.94 | 0.56 / 0.98 | 0.53 / 0.98 |
| Gated CNN-8 int8 | 0.79 / 0.13 | 0.93 / 0.53 | 0.97 / 0.80 | 0.96 / 0.93 | 0.97 / 0.98 | 0.97 / 0.98 |
| Gated CNN-4 int8 | 0.78 / 0.14 | 0.92 / 0.55 | 0.97 / 0.81 | 0.96 / 0.93 | 0.96 / 0.98 | 0.97 / 0.98 |
| Gated MLP-8 int8 | 0.79 / 0.12 | 0.93 / 0.53 | 0.97 / 0.80 | 0.96 / 0.93 | 0.97 / 0.98 | 0.97 / 0.98 |

| Model | MACs/inference | val F1 float | val F1 int8 |
|---|---:|---:|---:|
| cnn8 | 1760 | 0.9164 | 0.9165 |
| cnn4 | 592 | 0.9163 | 0.9164 |
| mlp8 | 264 | 0.9165 | 0.9167 |
<!-- END detection -->

### 2. Data reduction: representation × bits/s × decode R²

32-channel arrays, 300 s each, sharing one smooth 2D velocity trace. Units are cosine-tuned, with 0–4 per
channel and log-normal SNR (median 4). The decoder is a ridge Wiener filter (100 ms causal history). It is
trained on the first 70% of the recording and tested on the last 30%, with alpha chosen by contiguous
K-fold inside train. Nothing is shuffled. CIs come from a 5 s block bootstrap. "Fixed-width" means
ceil(log2(max+1)) bits per 20 ms bin. "Entropy-coded" is the zeroth-order empirical entropy, i.e. what an
ideal static entropy coder would reach. The raw row's entropy is a lossless first-difference estimate.

<!-- BEGIN reduction -->
**Background hash velocity-tuned (default)** — mean over 3 simulated 32-ch arrays; per-seed R² in brackets

| Representation | bits/s/ch (fixed-width) | bits/s/ch (entropy-coded) | decode R² | ΔR² vs TC −4.5σ (95% CI, seed 0) |
|---|---:|---:|---:|---|
| raw broadband (off-chip TC3.5+TC4.5+SBP) | 200,000 | 92,080 | 0.805 (0.811, 0.750, 0.852) | [+0.089, +0.144] |
| threshold counts -4.5σ | 200 | 33 | 0.709 (0.697, 0.657, 0.775) | — |
| spike-band power (8-bit) | 400 | 266 | 0.797 (0.795, 0.743, 0.854) | [+0.078, +0.121] |
| learned gated cnn8 counts | 183 | 49 | 0.782 (0.792, 0.736, 0.817) | [+0.073, +0.127] |
| learned counts + SBP | 583 | 315 | 0.810 (0.818, 0.763, 0.850) | [+0.094, +0.154] |
| oracle unsorted GT counts | 200 | 59 | 0.871 (0.895, 0.834, 0.885) | [+0.165, +0.249] |
| oracle sorted GT counts (per unit) | 267 | 81 | 0.910 (0.934, 0.879, 0.917) | [+0.203, +0.291] |

**Control: same recordings, hash untuned** — mean over 3 simulated 32-ch arrays; per-seed R² in brackets

| Representation | bits/s/ch (fixed-width) | bits/s/ch (entropy-coded) | decode R² | ΔR² vs TC −4.5σ (95% CI, seed 0) |
|---|---:|---:|---:|---|
| raw broadband (off-chip TC3.5+TC4.5+SBP) | 200,000 | 92,081 | 0.789 (0.798, 0.739, 0.831) | [+0.082, +0.127] |
| threshold counts -4.5σ | 200 | 34 | 0.712 (0.697, 0.661, 0.779) | — |
| spike-band power (8-bit) | 400 | 266 | 0.780 (0.787, 0.728, 0.825) | [+0.072, +0.111] |
| learned gated cnn8 counts | 200 | 49 | 0.783 (0.788, 0.749, 0.812) | [+0.073, +0.121] |
| learned counts + SBP | 600 | 316 | 0.796 (0.807, 0.758, 0.824) | [+0.088, +0.140] |
| oracle unsorted GT counts | 200 | 59 | 0.871 (0.895, 0.834, 0.885) | [+0.166, +0.250] |
| oracle sorted GT counts (per unit) | 267 | 81 | 0.910 (0.934, 0.879, 0.917) | [+0.204, +0.293] |
<!-- END reduction -->

![bandwidth vs R2](figures/bandwidth_vs_r2.png)

### 3. Embedded cost

<!-- BEGIN embedded -->
| Model | MACs/inf | ops/sample (avg) | M4 .text (B) | M4 .rodata (B) | RAM/ch state (B) | shared scratch (B) | host ns/sample | host ns/inference | bit-exact (events checked) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cnn4 | 592 | 17.2 | 982 | 284 | 192 | 288 | 4.04 | 297 | yes (21,139) |
| cnn8 | 1760 | 20.6 | 982 | 568 | 192 | 288 | 5.19 | 770 | yes (21,139) |
| mlp8 | 264 | 15.9 | 982 | 428 | 192 | 288 | 3.37 | 20 | yes (21,139) |

Threshold-only path on host: 3.36 ns/sample; front-end alone 3.24 ns/sample.

**Cortex-M4 ESTIMATE (not measured on hardware)** per channel at 20 kHz, mean candidate rate 49/s:

| Model | kernel | MHz/channel | µW/channel |
|---|---|---|---|
| cnn4 | plain C (this code) | 0.46–0.92 | 5–92 |
| cnn4 | CMSIS-NN style SMLAD | 0.41–0.83 | 4–83 |
| cnn8 | plain C (this code) | 0.57–1.14 | 6–114 |
| cnn8 | CMSIS-NN style SMLAD | 0.44–0.89 | 4–89 |
| mlp8 | plain C (this code) | 0.43–0.85 | 4–85 |
| mlp8 | CMSIS-NN style SMLAD | 0.41–0.81 | 4–81 |
<!-- END embedded -->

How to read these numbers:

* **Measured** on this machine: bit-exactness, host latency (Apple M5 Pro, `cc -O2`, median of 15 runs),
  and Cortex-M4 code/data sizes from `clang --target=thumbv7em-none-eabihf -mcpu=cortex-m4 -Os`.
  The sizes are object-file section sizes; there is no linker script or startup code.
* **Estimated** (never run on a chip): M4 cycles = 20–40 cycles/sample for the front-end and state
  machine, plus candidate rate × MACs × cycles/MAC. Plain C costs 2–4 cycles/MAC; a CMSIS-NN `SMLAD`
  kernel costs 0.5–1. Power = MHz × 10–100 µW/MHz, a deliberately wide range for low-power M4-class
  MCUs. These are assumptions, not datasheet values for any specific part.
* **What this shows**: the per-sample front-end dominates. Because the classifier is gated, it adds only
  a few ops per sample on average. That is why the 8× larger CNN costs almost the same as the MLP. The
  remaining lever is the front-end, which on a real implant would sit in analog or ASIC logic.

![ops vs F1](figures/ops_vs_f1.png)

![example](figures/example_trace.png)

## Limitations (read this first)

* **The real-data test is small and not cortex.** It covers 3 mouse retinal ganglion cells in vitro on a
  planar MEA, not chronic cortical implants. The data was already high-passed at 100 Hz and is rescaled
  to my 10-bit model. There is one labelled neuron per recording, so precision is unmeasured. The
  SNR-sweep mixes reuse the same neurons, and on some electrodes mixing *raised* the measured SNR,
  probably because noise is shared across channels. For 20160426_patch2 the authors' `triggers.npy` is
  missing. I rebuilt it with their stated rule (−k·MAD peaks on the patch trace), and that rule
  reproduces the official triggers exactly on both other recordings.
* **Training and tuning are synthetic only.** The waveforms are parametric (Gaussian trough, repolarisation, small pre-bump).
  The noise is white + 1/f + 1/f² LFP plus 20 sub-threshold "hash" units. Drift is a ±25% sinusoidal
  amplitude modulation. There is no shape drift, no bursting with amplitude attenuation, no
  overlapping-spike interactions beyond linear summation, and no motion or EMI artifacts. A learned
  detector can latch onto quirks of a simulator. §0 confirms that the synthetic F1 gains are much
  larger than the real ones.
* **SNR is nominal.** It is defined as |filtered trough| / robust σ of the background. Other papers use
  peak-to-peak / σ or RMS-based definitions, so absolute SNR values are not comparable across papers.
  Measured trough/σ for a nominal-3 unit is about 2.7 because the drift averages down.
* **The threshold baseline is partly penalised by filter ringing.** The 2nd-order high-pass makes large
  spikes cross threshold again about 1–1.5 ms later. The `tc_dead` row tunes the dead time to remove that
  free win for the learned model; compare against it, not against −4.5σ.
* **Gate ceiling.** The learned detector cannot recover spikes that never cross −3σ, which is why it
  fails at SNR 2.
* **Decoding is a toy.** The velocity tuning is cosine and linear, and so is the decoder. The tuned
  "hash" deliberately makes spike-band power informative; the untuned control shows how much of SBP's
  advantage depends on that assumption. Real implants show SBP ≈ TC or better, but that is literature,
  not something tested here.
* **No real chip.** The M4 numbers are estimates. Host latency on an M5 says nothing about a 64 MHz MCU
  except that the code is not pathological.
* **No spike sorting.** Everything is per channel and unsorted. The "oracle sorted" row only shows
  what sorting could add.

## Reproduce

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -e ".[train,test]"
make all     # detection (~8 min) → reduction (~3 min) → embedded → report; all seeds fixed
make test    # unit tests + C-vs-Python bit-exactness for every committed model (no torch needed)
```

The int8 models are committed in `models/` and their C headers in `c/generated/`, so CI and the C path
do not need PyTorch.

Real-data evaluation (§0) downloads about 4.7 GB of the CC-BY-4.0 Yger et al. 2018 data into the
git-ignored `data/` folder:

```bash
make real
```

## Layout

| Path | What |
|---|---|
| `spike_edge/synth.py` | ground-truth generator (templates, tuned Poisson trains, noise, drift, 10-bit ADC) |
| `spike_edge/frontend.py` | integer front-end, threshold / NEO event logic |
| `spike_edge/qnn.py` | int8 quantisation and the exact integer reference inference |
| `spike_edge/train.py` | float training (PyTorch) |
| `spike_edge/reduce.py`, `decode.py` | representations, bit rates, Wiener-filter decoding |
| `c/spike_edge.[ch]` | C99 streaming implementation: no floats, no malloc, 192 B state per channel |
| `c/harness.c`, `c/bench.c` | bit-exact I/O harness, host latency bench |
| `scripts/01–04` | experiments → `results/*.json` → figures + tables |
| `scripts/05_real_data.py` | frozen models on Yger 2018 paired ground truth |

## Next steps

1. **More real ground truth, and cortex.** Run all 19 Yger recordings: the rest are about 2 GB each,
   so ask before downloading them. Then cortical paired data (Marques-Smith et al. 2018 Neuropixels +
   patch) to test the cortex/implant setting. Then fine-tune on real noise with injected templates, and
   check whether the 20170622 null result moves.
2. **On hardware.** Flash an nRF52 or STM32 board, count cycles with DWT, and measure current. That replaces
   the estimate in §3. Then add a CMSIS-NN `arm_convolve_s8` path and check it stays bit-exact.
3. **Harder synthetic data.** Waveform shape drift, bursts with amplitude attenuation, MEArec biophysical
   templates, and motion/EMI artifacts. Also add a calibration-drift test where σ is re-estimated online.
4. **Shared hardware.** Share the front-end across channels in a 1024-channel budget, and lower the gate
   (−2.5σ) to see whether recall improves at SNR 2–3 at an acceptable candidate rate.
5. **Better learned scheme.** Send a per-event 1–2 bit shape code (on-chip clustering) next to the counts,
   aiming to close part of the 0.13 R² gap to the sorted oracle at under 100 b/s.
