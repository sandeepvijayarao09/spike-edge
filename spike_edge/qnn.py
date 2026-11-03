"""Integer-only int8 inference (TFLite-Micro style, symmetric, zero-point 0).

Per layer: int8 activations x (C_in, L_in), int8 weights (per-output-channel
scale), int32 bias, int32 accumulator, then requantisation
    q = (acc * m + 2^(sh-1)) >> sh          (int64 product, arithmetic shift)
followed by ReLU/clamp to [0, 127]. The last layer is a dense unit whose int32
accumulator is the logit and is compared against an integer threshold.
Everything here is int64 numpy so it is exactly what c/spike_edge.c computes.
"""
import numpy as np

IN_SCALE = 1.0 / 16.0    # float value of one input LSB
TARGET_SIGMA = 10        # input normalisation: noise sigma -> ~10 int8 LSB

ARCHS = {
    # name: list of (type, out_channels, kernel, stride)
    "cnn8": [("conv", 8, 5, 2), ("conv", 8, 3, 2), ("dense", 1, 0, 0)],
    "cnn4": [("conv", 4, 5, 2), ("conv", 4, 3, 2), ("dense", 1, 0, 0)],
    "mlp8": [("dense", 8, 0, 0), ("dense", 1, 0, 0)],
    # linear baselines on the same gated window (32 MACs): logistic regression and a
    # fixed matched filter (mean training spike as the weight vector)
    "lin32": [("dense", 1, 0, 0)],
    "tmpl": [("dense", 1, 0, 0)],
}


def input_gain(sigma_y):
    """Integer gain g so that ((y*g + 128) >> 8) has noise sigma ~ TARGET_SIGMA."""
    return int(max(1, min(32767, round(256 * TARGET_SIGMA / sigma_y))))


def quantize_input(windows_y, g):
    x = (windows_y.astype(np.int64) * g + 128) >> 8
    return np.clip(x, -128, 127).astype(np.int8)


def layer_shapes(arch, win):
    """Returns per layer (c_in, l_in, c_out, l_out); dense layers flatten."""
    c, l, out = 1, win, []
    for typ, co, k, s in arch:
        if typ == "conv":
            lo = (l - k) // s + 1
            out.append((c, l, co, lo)); c, l = co, lo
        else:
            out.append((c * l, 1, co, 1)); c, l = co, 1
    return out


def macs(arch, win):
    tot = 0
    for (typ, co, k, s), (ci, li, co_, lo) in zip(arch, layer_shapes(arch, win)):
        tot += lo * co * ci * k if typ == "conv" else ci * co
    return tot


def _multiplier(real):
    m, e = np.frexp(real)                 # real = m * 2^e, m in [0.5, 1)
    qm = np.round(m * (1 << 31)).astype(np.int64)
    e = np.where(qm == (1 << 31), e + 1, e); qm = np.where(qm == (1 << 31), 1 << 30, qm)
    sh = (31 - e).astype(np.int64)
    assert np.all(sh >= 1) and np.all(sh < 63)
    return qm, sh


def quantize_model(float_layers, arch, act_max):
    """float_layers: list of (W, b) with W shaped (out, in, k) or (out, in).
    act_max: max calibrated activation after each hidden layer's ReLU."""
    q, s_in = [], IN_SCALE
    for i, ((w, b), (typ, co, k, st)) in enumerate(zip(float_layers, arch)):
        w2 = w.reshape(w.shape[0], -1)
        s_w = np.maximum(np.abs(w2).max(1), 1e-12) / 127.0
        wq = np.clip(np.round(w2 / s_w[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
        bq = np.round(b / (s_in * s_w)).astype(np.int64)
        layer = dict(type=typ, k=k, stride=st, w=wq, b=bq.astype(np.int32))
        if i < len(arch) - 1:
            s_out = act_max[i] / 127.0
            layer["m"], layer["sh"] = _multiplier(s_in * s_w / s_out)
            s_in = s_out
        else:
            layer["logit_scale"] = float(s_in * s_w[0])
        q.append(layer)
    return q


def _requant(acc, m, sh):
    r = (acc * m + (np.int64(1) << (sh - 1))) >> sh
    return np.clip(r, 0, 127)


def run_int(qlayers, x_int8):
    """x_int8: (B, WIN) int8 -> int32 logits (B,). Exact integer reference."""
    x = x_int8.astype(np.int64)[:, None, :]            # (B, C=1, L)
    for i, L in enumerate(qlayers):
        w = L["w"].astype(np.int64); b = L["b"].astype(np.int64)
        if L["type"] == "conv":
            k, s = L["k"], L["stride"]
            pt = np.lib.stride_tricks.sliding_window_view(x, k, axis=2)[:, :, ::s, :]
            acc = np.einsum("bclk,ock->bol", pt, w) + b[None, :, None]
            m, sh = L["m"][None, :, None], L["sh"][None, :, None]
        else:
            xf = x.reshape(len(x), -1)
            acc = xf @ w.reshape(w.shape[0], -1).T + b[None, :]
            if i == len(qlayers) - 1:
                return acc[:, 0].astype(np.int32)
            acc = acc[:, :, None]
            m, sh = L["m"][None, :, None], L["sh"][None, :, None]
        x = _requant(acc, m, sh)
    raise ValueError("last layer must be dense")


def flash_bytes(qlayers):
    n = 0
    for L in qlayers:
        n += L["w"].size + 4 * L["b"].size
        if "m" in L:
            n += 8 * L["m"].size       # int32 multiplier + int32 shift
    return n


def save(path, qlayers, arch_name, meta):
    d = {"arch": np.array(arch_name)}
    for i, L in enumerate(qlayers):
        for key, v in L.items():
            d[f"{i}_{key}"] = np.asarray(v)
    for key, v in meta.items():
        d[f"meta_{key}"] = np.asarray(v)
    np.savez(path, **d)


def load(path):
    z = np.load(path)
    name = str(z["arch"]); arch = ARCHS[name]
    q = []
    for i in range(len(arch)):
        L = {}
        for key in ("type", "k", "stride", "w", "b", "m", "sh", "logit_scale"):
            if f"{i}_{key}" in z:
                v = z[f"{i}_{key}"]
                L[key] = v.item() if v.ndim == 0 else v
        L["type"] = str(L["type"])
        q.append(L)
    meta = {k[5:]: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z if k.startswith("meta_")}
    return name, q, meta
