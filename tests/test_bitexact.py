"""C implementation must match the Python integer reference bit for bit."""
from pathlib import Path

import numpy as np
import pytest

from spike_edge import qnn
from spike_edge.datasets import split, windows
from spike_edge.export_c import write_header, build, run_harness
from spike_edge.frontend import frontend_int, threshold_events

ROOT = Path(__file__).resolve().parents[1]
MODELS = sorted((ROOT / "models").glob("*.npz"))


def reference(raw, gate_thr, gain, q, logit_thr):
    y = frontend_int(raw)
    idx, w = windows(y, threshold_events(y, gate_thr))
    logits = qnn.run_int(q, qnn.quantize_input(w, gain)) if q is not None else np.zeros(len(idx), np.int32)
    return y, idx, logits


def signals():
    rng = np.random.default_rng(7)
    yield "synthetic_test_ch", split("test_3", 0)["raw"][: 20_000 * 20]
    yield "full_scale_noise", rng.integers(-512, 512, 200_000).astype(np.int16)
    sq = np.repeat(np.tile([-512, 511], 2000), 37).astype(np.int16)  # rail-to-rail steps
    yield "square_wave", sq


@pytest.mark.parametrize("model_path", MODELS, ids=[p.stem for p in MODELS])
def test_c_matches_python(model_path, tmp_path):
    name, q, meta = qnn.load(model_path)
    write_header(name, q)
    exe = build(name)
    for sig_name, raw in signals():
        sigma = 10.0
        gate, gain, thr = int(round(3 * sigma)), qnn.input_gain(sigma), int(meta["logit_thr"])
        y_ref, idx_ref, lg_ref = reference(raw, gate, gain, q, thr)
        y_c, ev = run_harness(exe, raw, gate, gain, thr, tmp_path / sig_name)
        assert np.array_equal(y_c.astype(np.int32), y_ref), sig_name
        assert np.array_equal(ev[:, 0], idx_ref), sig_name
        assert np.array_equal(ev[:, 1], lg_ref), sig_name
        assert np.array_equal(ev[:, 2].astype(bool), lg_ref > thr), sig_name
        assert len(idx_ref) > 0 or sig_name != "synthetic_test_ch"


def test_threshold_only_mode(tmp_path):
    name, q, meta = qnn.load(MODELS[0])
    exe = build(name)
    raw = split("test_5", 1)["raw"][:200_000]
    y_ref, idx_ref, _ = reference(raw, 45, 256, None, 0)
    y_c, ev = run_harness(exe, raw, 45, 256, 0, tmp_path, no_model=True)
    assert np.array_equal(ev[:, 0], idx_ref) and ev[:, 2].all()
