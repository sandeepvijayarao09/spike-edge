## real

**Across cells** (one sample per cell = mean over its native ≥4σ electrodes). Entries are mean Δrecall vs the tuned threshold [95% bootstrap CI over cells] · wins/ties/losses (±0.01) · two-sided sign test p.

| Scope | cells | comparison | MLP-8 | CNN-8 | logistic | matched filter |
|---|---:|---|---|---|---|---|
| all | 20 | label-free rate-matched (primary) | +0.025 [+0.013, +0.038] · 16/2/2 · p=0.00131 | +0.025 [+0.013, +0.039] · 16/3/1 · p=0.000275 | -0.017 [-0.047, +0.011] · 10/2/8 · p=0.815 | +0.014 [-0.003, +0.029] · 14/3/3 · p=0.0127 |
| all | 20 | oracle matched rate | +0.008 [-0.001, +0.017] · 7/11/2 · p=0.18 | +0.008 [-0.002, +0.018] · 7/11/2 · p=0.18 | -0.033 [-0.061, -0.008] · 4/6/10 · p=0.18 | -0.004 [-0.019, +0.009] · 6/10/4 · p=0.754 |
| Marques-Smith 2018 (rat cortex, Neuropixels) | 1 | label-free rate-matched (primary) | +0.076 [+0.076, +0.076] · 1/0/0 · p=1 | +0.082 [+0.082, +0.082] · 1/0/0 · p=1 | +0.029 [+0.029, +0.029] · 1/0/0 · p=1 | +0.051 [+0.051, +0.051] · 1/0/0 · p=1 |
| Marques-Smith 2018 (rat cortex, Neuropixels) | 1 | oracle matched rate | +0.055 [+0.055, +0.055] · 1/0/0 · p=1 | +0.064 [+0.064, +0.064] · 1/0/0 · p=1 | +0.015 [+0.015, +0.015] · 1/0/0 · p=1 | +0.037 [+0.037, +0.037] · 1/0/0 · p=1 |
| Yger 2018 (mouse retina, MEA) | 19 | label-free rate-matched (primary) | +0.023 [+0.011, +0.036] · 15/2/2 · p=0.00235 | +0.022 [+0.011, +0.035] · 15/3/1 · p=0.000519 | -0.019 [-0.051, +0.010] · 9/2/8 · p=1 | +0.012 [-0.006, +0.028] · 13/3/3 · p=0.0213 |
| Yger 2018 (mouse retina, MEA) | 19 | oracle matched rate | +0.006 [-0.002, +0.014] · 6/11/2 · p=0.289 | +0.005 [-0.004, +0.013] · 6/11/2 · p=0.289 | -0.036 [-0.065, -0.010] · 3/6/10 · p=0.0923 | -0.006 [-0.021, +0.007] · 5/10/4 · p=1 |

**Per cell** (label-free Δrecall vs tuned threshold, native electrodes):

| Cell | dataset | electrodes | SNR range | threshold recall | Δ MLP-8 | Δ CNN-8 | Δ logistic | Δ matched filter | MLP/threshold event-rate ratio |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| cortex/c37 | Marques-Smith 2018 | 4 | 4.5–7.4 | 0.852 | +0.076 | +0.082 | +0.029 | +0.051 | 1.08 |
| 20160415/patch_2_MEA | Yger 2018 | 3 | 4.5–7.7 | 0.903 | +0.037 | +0.029 | -0.088 | -0.008 | 1.02 |
| 20160426/patch_2 | Yger 2018 | 1 | 4.7–4.7 | 0.947 | +0.047 | +0.047 | +0.047 | +0.047 | 1.01 |
| 20160426/patch_3 | Yger 2018 | 2 | 7.8–8.5 | 0.987 | +0.011 | +0.011 | +0.006 | +0.011 | 1.00 |
| 20170621/patch1_converted | Yger 2018 | 4 | 4.9–15.2 | 0.941 | +0.012 | +0.013 | -0.038 | -0.013 | 0.98 |
| 20170622/patch1 | Yger 2018 | 1 | 8.6–8.6 | 0.982 | +0.015 | +0.015 | -0.101 | +0.011 | 0.97 |
| 20170622/patch2 | Yger 2018 | 7 | 5.9–19.6 | 0.947 | +0.040 | +0.040 | +0.033 | +0.037 | 1.01 |
| 20170623/patch1 | Yger 2018 | 10 | 4.2–35.7 | 0.970 | +0.021 | +0.021 | +0.019 | +0.019 | 1.03 |
| 20170627/patch1 | Yger 2018 | 2 | 4.3–6.9 | 0.962 | +0.018 | +0.018 | -0.036 | +0.013 | 0.99 |
| 20170629/patch2 | Yger 2018 | 10 | 4.1–24.3 | 0.972 | +0.003 | +0.005 | -0.018 | -0.005 | 1.01 |
| 20170629/patch3 | Yger 2018 | 4 | 4.1–11.6 | 0.954 | -0.044 | -0.045 | -0.185 | -0.091 | 0.99 |
| 20170630/patch1 | Yger 2018 | 4 | 5.5–14.7 | 0.991 | +0.006 | +0.006 | -0.043 | +0.004 | 0.96 |
| 20170706/patch1 | Yger 2018 | 1 | 17.9–17.9 | 0.972 | +0.027 | +0.027 | +0.007 | +0.025 | 1.02 |
| 20170706/patch2 | Yger 2018 | 1 | 10.2–10.2 | 0.926 | +0.018 | +0.018 | +0.015 | +0.016 | 0.94 |
| 20170706/patch3 | Yger 2018 | 3 | 4.4–11.5 | 0.835 | +0.100 | +0.101 | +0.090 | +0.094 | 0.98 |
| 20170713/patch1 | Yger 2018 | 3 | 4.0–14.0 | 0.895 | +0.034 | +0.033 | +0.019 | +0.033 | 1.02 |
| 20170725/patch1 | Yger 2018 | 6 | 4.9–23.3 | 0.902 | +0.036 | +0.033 | +0.011 | +0.028 | 1.01 |
| 20170726/patch1 | Yger 2018 | 3 | 5.7–16.6 | 0.992 | -0.015 | -0.009 | -0.150 | -0.059 | 0.99 |
| 20170728/patch2 | Yger 2018 | 4 | 4.9–19.2 | 0.944 | +0.041 | +0.041 | +0.034 | +0.039 | 0.99 |
| 20170803/patch1 | Yger 2018 | 4 | 6.5–12.6 | 0.975 | +0.022 | +0.022 | +0.020 | +0.021 | 0.99 |

**Real-signal SNR sweep** (attenuated electrode + cross-recording noise; rows reuse the same cells, so not independent):

| measured SNR | conditions | mean label-free Δ MLP-8 | mean Δ logistic | mean Δ matched filter |
|---|---:|---:|---:|---:|
| 0–3.5 | 77 | +0.106 | +0.079 | +0.083 |
| 3.5–4.5 | 80 | +0.072 | +0.003 | +0.038 |
| 4.5–5.5 | 61 | +0.030 | -0.029 | +0.008 |
| 5.5–99 | 40 | +0.012 | -0.019 | +0.005 |

C pipeline bit-exact with Python on 77/77 native real electrodes. Cells with no electrode ≥ 4σ (excluded): 2 (cortex/c19, cortex/c24). Shortlisted electrodes dropped because the full recording was < 4σ: 7. Mixes dropped (SNR not below native): 0.


## detection

| Detector | ops/sample | M4 .rodata (B) | F1 @ SNR 2 (ptp 3.8) | F1 @ SNR 3 (ptp 5.5) | F1 @ SNR 4 (ptp 7.3) | F1 @ SNR 5 (ptp 9.0) | F1 @ SNR 6 (ptp 10.5) | F1 @ SNR 8 (ptp 14.0) |
|---|---:|---:|---|---|---|---|---|---|
| Threshold −4.5σ | 15.0 | 0 | 0.025 [0.02–0.03] | 0.209 [0.16–0.25] | 0.540 [0.47–0.61] | 0.757 [0.70–0.81] | 0.890 [0.86–0.92] | 0.938 [0.92–0.96] |
| Threshold −4.0σ (tuned) | 15.0 | 0 | 0.079 [0.06–0.10] | 0.381 [0.32–0.44] | 0.701 [0.64–0.76] | 0.840 [0.80–0.88] | 0.920 [0.90–0.94] | 0.904 [0.87–0.94] |
| Threshold −3.75σ, 2 ms dead (tuned) | 15.0 | 0 | 0.139 [0.11–0.17] | 0.486 [0.42–0.54] | 0.758 [0.71–0.80] | 0.867 [0.84–0.90] | 0.935 [0.92–0.95] | 0.950 [0.94–0.96] |
| NEO (tuned) | 24.0 | 0 | 0.138 [0.12–0.15] | 0.370 [0.32–0.42] | 0.662 [0.61–0.71] | 0.766 [0.73–0.80] | 0.848 [0.82–0.88] | 0.928 [0.91–0.95] |
| Gate only (−3σ) | 15.0 | 0 | 0.308 [0.27–0.34] | 0.561 [0.52–0.60] | 0.711 [0.68–0.74] | 0.710 [0.68–0.73] | 0.716 [0.68–0.75] | 0.690 [0.65–0.73] |
| Gated matched filter int8 (32 MACs) | 15.2 | 72 | 0.196 [0.14–0.25] | 0.630 [0.55–0.70] | 0.844 [0.79–0.89] | 0.928 [0.90–0.95] | 0.967 [0.96–0.97] | 0.978 [0.97–0.98] |
| Gated logistic regression int8 (32 MACs) | 15.2 | 72 | 0.224 [0.17–0.28] | 0.648 [0.57–0.71] | 0.852 [0.80–0.89] | 0.928 [0.91–0.94] | 0.963 [0.95–0.97] | 0.974 [0.97–0.98] |
| Gated MLP-8 int8 (264 MACs) | 15.9 | 428 | 0.213 [0.16–0.27] | 0.671 [0.59–0.74] | 0.874 [0.83–0.91] | 0.944 [0.93–0.96] | 0.972 [0.97–0.98] | 0.978 [0.97–0.98] |
| Gated CNN-4 int8 (592 MACs) | 17.2 | 284 | 0.232 [0.18–0.29] | 0.685 [0.61–0.75] | 0.879 [0.84–0.91] | 0.943 [0.93–0.96] | 0.970 [0.96–0.97] | 0.976 [0.97–0.98] |
| Gated CNN-8 int8 (1760 MACs) | 20.6 | 568 | 0.215 [0.16–0.27] | 0.673 [0.59–0.74] | 0.876 [0.83–0.91] | 0.945 [0.93–0.96] | 0.972 [0.97–0.98] | 0.978 [0.97–0.98] |

| Detector | P / R @ 2 | P / R @ 3 | P / R @ 4 | P / R @ 5 | P / R @ 6 | P / R @ 8 |
|---|---|---|---|---|---|---|
| Threshold −4.5σ | 0.78 / 0.01 | 0.97 / 0.12 | 0.99 / 0.37 | 0.98 / 0.62 | 0.98 / 0.82 | 0.91 / 0.97 |
| Threshold −4.0σ (tuned) | 0.69 / 0.04 | 0.91 / 0.24 | 0.96 / 0.55 | 0.95 / 0.75 | 0.94 / 0.91 | 0.84 / 0.98 |
| Threshold −3.75σ, 2 ms dead (tuned) | 0.60 / 0.08 | 0.85 / 0.34 | 0.94 / 0.63 | 0.94 / 0.80 | 0.95 / 0.92 | 0.95 / 0.95 |
| NEO (tuned) | 0.21 / 0.10 | 0.48 / 0.30 | 0.75 / 0.59 | 0.79 / 0.74 | 0.83 / 0.86 | 0.90 / 0.95 |
| Gate only (−3σ) | 0.32 / 0.29 | 0.49 / 0.66 | 0.61 / 0.86 | 0.57 / 0.94 | 0.56 / 0.98 | 0.53 / 0.98 |
| Gated matched filter int8 (32 MACs) | 0.79 / 0.11 | 0.93 / 0.48 | 0.97 / 0.75 | 0.97 / 0.89 | 0.97 / 0.96 | 0.97 / 0.98 |
| Gated logistic regression int8 (32 MACs) | 0.75 / 0.13 | 0.91 / 0.50 | 0.96 / 0.77 | 0.95 / 0.90 | 0.96 / 0.97 | 0.97 / 0.98 |
| Gated MLP-8 int8 (264 MACs) | 0.79 / 0.12 | 0.93 / 0.53 | 0.97 / 0.80 | 0.96 / 0.93 | 0.97 / 0.98 | 0.97 / 0.98 |
| Gated CNN-4 int8 (592 MACs) | 0.78 / 0.14 | 0.92 / 0.55 | 0.97 / 0.81 | 0.96 / 0.93 | 0.96 / 0.98 | 0.97 / 0.98 |
| Gated CNN-8 int8 (1760 MACs) | 0.79 / 0.12 | 0.93 / 0.53 | 0.97 / 0.80 | 0.96 / 0.93 | 0.97 / 0.98 | 0.97 / 0.98 |

| Gated classifier | MACs/inference | val F1 float | val F1 int8 |
|---|---:|---:|---:|
| tmpl | 32 | 0.9119 | 0.9119 |
| lin32 | 32 | 0.9104 | 0.9104 |
| mlp8 | 264 | 0.9165 | 0.9167 |
| cnn4 | 592 | 0.9164 | 0.9164 |
| cnn8 | 1760 | 0.9163 | 0.9164 |

## robustness

| Detector | false events/s, no unit on channel (mean / worst ch) | F1 per 60 s block over 300 s of drift (first block after calibration) | F1 with artifacts (0.5/s, SNR 5) | unmatched events within 3 ms of an artifact, per artifact |
|---|---|---|---:|---:|
| Threshold −4.5σ | 0.10 / 0.27 | 0.705 · 0.784 · 0.825 · 0.724 · 0.772 | 0.741 | 0.95 |
| Threshold −4.0σ (tuned) | 0.61 / 1.27 | 0.818 · 0.858 · 0.888 · 0.830 · 0.847 | 0.824 | 0.95 |
| Threshold −3.75σ, 2 ms dead (tuned) | 1.63 / 2.97 | 0.851 · 0.881 · 0.902 · 0.863 · 0.867 | 0.853 | 0.93 |
| NEO (tuned) | 12.34 / 14.23 | 0.763 · 0.800 · 0.821 · 0.762 · 0.816 | 0.746 | 2.37 |
| Gate only (−3σ) | 19.16 / 28.20 | 0.765 · 0.759 · 0.765 · 0.765 · 0.762 | 0.699 | 1.00 |
| Gated matched filter int8 (32 MACs) | 0.97 / 2.97 | 0.916 · 0.929 · 0.956 · 0.936 · 0.914 | 0.923 | 0.47 |
| Gated logistic regression int8 (32 MACs) | 1.46 / 3.67 | 0.917 · 0.929 · 0.951 · 0.934 · 0.920 | 0.923 | 0.48 |
| Gated MLP-8 int8 (264 MACs) | 1.12 / 2.93 | 0.939 · 0.947 · 0.964 · 0.950 · 0.939 | 0.937 | 0.97 |
| Gated CNN-4 int8 (592 MACs) | 1.36 / 3.57 | 0.940 · 0.948 · 0.962 · 0.951 · 0.940 | 0.936 | 0.96 |
| Gated CNN-8 int8 (1760 MACs) | 1.14 / 3.10 | 0.940 · 0.948 · 0.963 · 0.951 · 0.940 | 0.937 | 0.96 |

SNR-5 clean-test F1 for reference: Threshold −3.75σ, 2 ms dead 0.867, Gated logistic regression int8 0.928, Gated MLP-8 int8 0.944.


## reduction


**Background hash velocity-tuned (default)**: mean over 3 simulated 32-ch arrays. Counts are sent as 3-bit saturating symbols (max saturated-bin fraction 0.0003). Add ≈3.1 b/s/ch framing for a 1024-ch frame header (assumed 64 bit / 20 ms).

| Representation | fixed-width b/s/ch | Huffman b/s/ch (≥1 bit/bin) | adaptive arithmetic b/s/ch | decode R² mean (per seed) | ΔR² vs tuned threshold, 95% CI per seed |
|---|---:|---:|---:|---|---|
| raw broadband (off-chip TC3.5+TC4.5+SBP) | 200,000 | — | 92,080 (lossless est.) | 0.805 (0.811, 0.750, 0.852) | [+0.038, +0.083] [+0.048, +0.069] [+0.048, +0.067] |
| threshold counts −4.5σ | 150 | 62 | 33 | 0.709 (0.697, 0.655, 0.774) | [-0.075, -0.041] [-0.049, -0.021] [-0.038, -0.011] |
| threshold counts −3.5σ | 150 | 73 | 58 | 0.751 (0.751, 0.694, 0.809) | [-0.015, +0.016] [-0.011, +0.009] [+0.004, +0.017] |
| threshold counts, tuned | 150 | 67 | 45 | 0.748 (0.750, 0.696, 0.797) | — |
| threshold counts, rate-matched to learned (label-free) | 150 | 70 | 48 | 0.760 (0.762, 0.706, 0.811) | [+0.001, +0.026] [-0.001, +0.023] [+0.007, +0.019] |
| spike-band power (8-bit) | 400 | 268 | 270 | 0.797 (0.795, 0.743, 0.854) | [+0.020, +0.068] [+0.031, +0.063] [+0.046, +0.072] |
| learned gated mlp8 counts | 150 | 69 | 49 | 0.780 (0.789, 0.734, 0.818) | [+0.022, +0.058] [+0.025, +0.051] [+0.014, +0.025] |
| learned counts + SBP | 550 | 337 | 318 | 0.809 (0.817, 0.760, 0.850) | [+0.043, +0.092] [+0.053, +0.079] [+0.046, +0.062] |
| oracle unsorted GT counts | 150 | 78 | 60 | 0.871 (0.895, 0.834, 0.885) | [+0.115, +0.187] [+0.118, +0.160] [+0.079, +0.106] |
| oracle sorted GT counts (per unit) | 200 | 96 | 81 | 0.910 (0.934, 0.878, 0.917) | [+0.153, +0.229] [+0.163, +0.205] [+0.109, +0.145] |

**Control: same recordings, hash untuned**: mean over 3 simulated 32-ch arrays. Counts are sent as 3-bit saturating symbols (max saturated-bin fraction 0.0003). Add ≈3.1 b/s/ch framing for a 1024-ch frame header (assumed 64 bit / 20 ms).

| Representation | fixed-width b/s/ch | Huffman b/s/ch (≥1 bit/bin) | adaptive arithmetic b/s/ch | decode R² mean (per seed) | ΔR² vs tuned threshold, 95% CI per seed |
|---|---:|---:|---:|---|---|
| raw broadband (off-chip TC3.5+TC4.5+SBP) | 200,000 | — | 92,081 (lossless est.) | 0.789 (0.798, 0.739, 0.831) | [+0.024, +0.068] [+0.015, +0.047] [+0.030, +0.051] |
| threshold counts −4.5σ | 150 | 62 | 34 | 0.712 (0.697, 0.660, 0.778) | [-0.080, -0.035] [-0.066, -0.027] [-0.028, +0.002] |
| threshold counts −3.5σ | 150 | 74 | 59 | 0.754 (0.749, 0.712, 0.801) | [-0.017, +0.007] [-0.010, +0.011] [+0.003, +0.014] |
| threshold counts, tuned | 150 | 67 | 46 | 0.752 (0.751, 0.713, 0.792) | — |
| threshold counts, rate-matched to learned (label-free) | 150 | 70 | 49 | 0.764 (0.759, 0.726, 0.809) | [+0.000, +0.014] [+0.001, +0.022] [+0.013, +0.025] |
| spike-band power (8-bit) | 400 | 268 | 270 | 0.780 (0.787, 0.728, 0.825) | [+0.007, +0.060] [-0.005, +0.038] [+0.023, +0.043] |
| learned gated mlp8 counts | 150 | 69 | 49 | 0.783 (0.787, 0.749, 0.813) | [+0.025, +0.052] [+0.022, +0.052] [+0.014, +0.026] |
| learned counts + SBP | 550 | 338 | 319 | 0.791 (0.797, 0.754, 0.823) | [+0.019, +0.070] [+0.025, +0.062] [+0.022, +0.040] |
| oracle unsorted GT counts | 150 | 78 | 60 | 0.871 (0.895, 0.834, 0.885) | [+0.119, +0.183] [+0.101, +0.146] [+0.084, +0.113] |
| oracle sorted GT counts (per unit) | 200 | 96 | 81 | 0.910 (0.934, 0.878, 0.917) | [+0.157, +0.226] [+0.146, +0.193] [+0.116, +0.154] |


## embedded

| Model | MACs/inf | ops/sample (avg) | M4 .text (B) | M4 .rodata (B) | state/ch (B) | stack scratch (B) | host ns/sample | host ns/inference | bit-exact (events checked) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cnn4 | 592 | 17.2 | 1000 | 284 | 136 | 288 | 4.82 | 349 | yes (21,139) |
| cnn8 | 1760 | 20.6 | 1000 | 568 | 136 | 288 | 5.96 | 957 | yes (21,139) |
| lin32 | 32 | 15.2 | 1000 | 72 | 136 | 288 | 4.21 | 11 | yes (21,139) |
| mlp8 | 264 | 15.9 | 1000 | 428 | 136 | 288 | 4.09 | 30 | yes (21,139) |
| tmpl | 32 | 15.2 | 1000 | 72 | 136 | 288 | 3.91 | 7 | yes (21,139) |

**1024-channel budget** (mean candidate rate 49/s per channel). The energy column uses an **assumed** 0.1–1.0 pJ per integer op (not sourced). It scales linearly, so plug in your process.

| Pipeline | classifier MAC/s per ch | total Mops/s (1024 ch) | of which classifier MMAC/s | state SRAM (KiB) | shared weights (B) | mW at assumed pJ/op |
|---|---:|---:|---:|---:|---:|---|
| threshold only | 0 | 307 | 0 | — | 0 | 0.03–0.31 |
| gated cnn4 | 28,981 | 352 | 29.7 | 136 | 192 | 0.04–0.35 |
| gated cnn8 | 86,158 | 423 | 88.2 | 136 | 476 | 0.04–0.42 |
| gated lin32 | 1,567 | 312 | 1.6 | 136 | 36 | 0.03–0.31 |
| gated mlp8 | 12,924 | 325 | 13.2 | 136 | 364 | 0.03–0.32 |
| gated tmpl | 1,567 | 312 | 1.6 | 136 | 36 | 0.03–0.31 |

Secondary, Cortex-M4 **estimate** (cycles only; MCU power depends on the part, so it is not quoted):

| Model | kernel | MHz per channel |
|---|---|---|
| cnn4 | plain C (this code) | 0.46–0.92 |
| cnn4 | CMSIS-NN style SMLAD | 0.41–0.83 |
| cnn8 | plain C (this code) | 0.57–1.14 |
| cnn8 | CMSIS-NN style SMLAD | 0.44–0.89 |
| lin32 | plain C (this code) | 0.40–0.81 |
| lin32 | CMSIS-NN style SMLAD | 0.40–0.80 |
| mlp8 | plain C (this code) | 0.43–0.85 |
| mlp8 | CMSIS-NN style SMLAD | 0.41–0.81 |
| tmpl | plain C (this code) | 0.40–0.81 |
| tmpl | CMSIS-NN style SMLAD | 0.40–0.80 |
