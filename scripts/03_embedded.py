"""Embedded cost: C/Python bit-exactness, host latency, Cortex-M4 code size,
and an *estimated* M4 cycle/power budget. Writes results/embedded.json."""
import json, re, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spike_edge import FS, WIN, qnn
from spike_edge.datasets import split, windows, SNR_LEVELS
from spike_edge.export_c import write_header, build, run_harness, CDIR
from spike_edge.frontend import frontend_int, threshold_events

B = ROOT / "build"
OD = subprocess.run(["xcrun", "--find", "llvm-objdump"], capture_output=True, text=True).stdout.strip() or "llvm-objdump"

# --- analytic op counts per sample (integer ALU ops; 1 MAC counted as 1 op) ---
FRONTEND_OPS = 13       # 1 shift in, 3x(sub, shift, add), 2 subs, 1 shift out
TC_OPS = 2              # compare + running-min compare
NEO_OPS = 3 + 7 + 1     # 2 mul + 1 sub, 7-tap smoothing, compare
# assumptions for the Cortex-M4 *estimate* (not measured on hardware)
CYC_PER_MAC = {"plain C (this code)": (2.0, 4.0), "CMSIS-NN style SMLAD": (0.5, 1.0)}
CYC_FIXED = (20, 40)    # per-sample front-end + state machine incl. load/store, branch
N_ARRAY = 1024          # channels in the implant-scale budget
PJ_PER_OP = (0.1, 1.0)  # ASIC energy per 8-16 bit integer op incl. local SRAM access: an ASSUMPTION
                        # for illustration (not sourced); total energy scales linearly with it


def m4_sizes(name):
    tu = B / f"m4_{name}.c"
    tu.write_text(f'#include "model_{name}.h"\nconst se_model_t *se_model_ptr = &SE_MODEL;\n')
    sizes = {}
    for src, obj in [(CDIR / "spike_edge.c", B / "se_m4.o"), (tu, B / f"m4_{name}.o")]:
        subprocess.run(["clang", "--target=thumbv7em-none-eabihf", "-mcpu=cortex-m4", "-mthumb", "-ffreestanding",
                        "-Os", f"-I{CDIR}", f"-I{CDIR/'generated'}", "-c", str(src), "-o", str(obj)], check=True)
        for line in subprocess.run([OD, "-h", str(obj)], capture_output=True, text=True).stdout.splitlines():
            m = re.match(r"\s*\d+\s+(\.text|\.rodata\S*|\.data\S*|\.bss\S*)\s+([0-9a-f]+)", line)
            if m:
                key = m.group(1).split(".")[1]
                sizes[key] = sizes.get(key, 0) + int(m.group(2), 16)
    return sizes


def main():
    det = json.loads((ROOT / "results" / "detection.json").read_text())
    cand_rate = float(np.mean([det["levels"][str(l)]["candidates_per_s"] for l in SNR_LEVELS]))
    out = {"candidates_per_s_mean_over_test_levels": cand_rate, "assumptions": dict(
        cycles_per_mac=CYC_PER_MAC, cycles_fixed_per_sample=CYC_FIXED, pj_per_op=PJ_PER_OP, n_array=N_ARRAY, fs=FS), "models": {}}
    chs = [split("test_3", i) for i in range(4)] + [split("test_8", i) for i in range(4)]
    for path in sorted((ROOT / "models").glob("*.npz")):
        name, q, meta = qnn.load(path)
        write_header(name, q)
        exe = build(name)
        # bit-exactness over 8 test channels x 40 s
        n_ev = n_mis = n_samp = 0
        ok = True
        for c in chs:
            raw = c["raw"]; s = c["sigma"]
            gate, gain, thr = int(round(meta["gate_k"] * s)), qnn.input_gain(s), int(meta["logit_thr"])
            y = frontend_int(raw)
            idx, w = windows(y, threshold_events(y, gate))
            lg = qnn.run_int(q, qnn.quantize_input(w, gain))
            yc, ev = run_harness(exe, raw, gate, gain, thr, B / f"bx_{name}")
            same = (np.array_equal(yc.astype(np.int32), y) and np.array_equal(ev[:, 0], idx)
                    and np.array_equal(ev[:, 1], lg) and np.array_equal(ev[:, 2].astype(bool), lg > thr))
            ok &= same; n_ev += len(idx); n_samp += len(raw)
            n_mis += int(np.sum(ev[:, 1] != lg)) if len(ev) == len(lg) else len(lg)
        # host latency
        bench = B / f"bench_{name}"
        subprocess.run(["cc", "-std=c99", "-O2", f"-I{CDIR}", f"-I{CDIR/'generated'}", f'-DMODEL_HEADER="model_{name}.h"',
                        str(CDIR / "bench.c"), str(CDIR / "spike_edge.c"), "-o", str(bench)], check=True)
        c = chs[0]
        run_harness(exe, c["raw"], int(round(meta["gate_k"] * c["sigma"])), qnn.input_gain(c["sigma"]), int(meta["logit_thr"]), B / "bench_in")
        hb = json.loads(subprocess.run([str(bench), str(B / "bench_in" / "in.bin")], capture_output=True, text=True, check=True).stdout)
        mac = qnn.macs(qnn.ARCHS[name], WIN)
        n_out = sum(co * lo for (_, _, co, lo) in qnn.layer_shapes(qnn.ARCHS[name], WIN))
        per_inf = mac + WIN * 2 + 3 * n_out          # MACs + input quant (mul, shift) + requant
        ops = FRONTEND_OPS + TC_OPS + cand_rate / FS * per_inf
        sz = m4_sizes(name)
        est = {}
        for k, (lo_c, hi_c) in CYC_PER_MAC.items():
            cyc = [CYC_FIXED[0] * FS + cand_rate * mac * lo_c, CYC_FIXED[1] * FS + cand_rate * mac * hi_c]
            mhz = [x / 1e6 for x in cyc]
            est[k] = dict(cycles_per_s=cyc, mhz_per_channel=mhz)
        out["models"][name] = dict(
            bit_exact=bool(ok), bitexact_events=n_ev, bitexact_samples=n_samp, logit_mismatches=n_mis,
            macs_per_inference=mac, ops_per_inference=per_inf, ops_per_sample=ops,
            weights_bytes=qnn.flash_bytes(q), m4_sizes_bytes=sz,
            ram_bytes=dict(state=hb["sizeof_state"], activations_stack=2 * 128, window_stack=WIN),
            host=hb, m4_estimate=est,
            array_budget=dict(
                classifier_mac_per_s_per_channel=cand_rate * mac,
                frontend_and_gate_ops_per_s_per_channel=(FRONTEND_OPS + TC_OPS) * FS,
                total_mops_per_s=ops * FS * N_ARRAY / 1e6,
                classifier_mmac_per_s=cand_rate * mac * N_ARRAY / 1e6,
                state_sram_kib=hb["sizeof_state"] * N_ARRAY / 1024,
                weights_bytes_shared=qnn.flash_bytes(q),
                mw_at_assumed_pj_per_op=[ops * FS * N_ARRAY * pj * 1e-12 * 1e3 for pj in PJ_PER_OP]))
        print(name, json.dumps(out["models"][name]), flush=True)
    th_ops = FRONTEND_OPS + TC_OPS
    out["threshold_only_array_budget"] = dict(total_mops_per_s=th_ops * FS * N_ARRAY / 1e6,
                                              mw_at_assumed_pj_per_op=[th_ops * FS * N_ARRAY * pj * 1e-9 for pj in PJ_PER_OP])
    out["baselines"] = dict(frontend_ops=FRONTEND_OPS, threshold_ops_per_sample=FRONTEND_OPS + TC_OPS,
                            neo_ops_per_sample=FRONTEND_OPS + NEO_OPS)
    (ROOT / "results" / "embedded.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
