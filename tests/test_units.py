import numpy as np

from spike_edge import qnn, FS
from spike_edge.frontend import frontend_int, frontend_float, threshold_events, robust_sigma
from spike_edge.metrics import match, prf
from spike_edge.reduce import entropy_bits, bin_counts, BIN
from spike_edge.decode import lagged


def _frontend_pure_python(x):
    s1 = int(x[0]) << 8; s2 = s3 = 0; out = []
    for v in x:
        xs = int(v) << 8
        s1 += (xs - s1) >> 4; h1 = xs - s1
        s2 += (h1 - s2) >> 4; h2 = h1 - s2
        s3 += (h2 - s3) >> 1; out.append(s3 >> 6)
    return np.array(out)


def test_frontend_matches_pure_python_and_float():
    x = np.random.default_rng(0).integers(-512, 512, 5000).astype(np.int16)
    yi = frontend_int(x)
    assert np.array_equal(yi, _frontend_pure_python(x))
    assert np.max(np.abs(yi - frontend_float(x.astype(float)))) < 3  # rounding only
    assert np.abs(yi).max() < 2 ** 15  # fits the int16 ring buffer in C


def test_frontend_rejects_dc_and_lfp():
    t = np.arange(FS) / FS
    x = (300 + 100 * np.sin(2 * np.pi * 5 * t)).astype(np.int16)
    y = frontend_int(x)[FS // 2:]
    assert np.abs(y).max() < 0.1 * 4 * 100  # >10x attenuation of 5 Hz LFP


def test_threshold_events_state_machine():
    y = np.zeros(200, np.int32); y[50] = -100; y[52] = -150; y[60] = -120; y[100] = -90
    ev = threshold_events(y, 80)
    assert list(ev) == [52, 100]  # 60 is inside the search/dead window of 52


def test_match_one_to_one():
    tp, hit, used = match([100, 105, 300], [102, 500], tol=10)
    assert tp == 1 and hit.tolist() == [True, False, False]
    assert prf(1, 3, 2) == (1 / 3, 0.5, 0.4)


def test_requant_multiplier_accuracy():
    for real in [1e-4, 0.0123, 0.5, 0.9999]:
        m, sh = qnn._multiplier(np.array([real]))
        assert abs(m[0] / 2.0 ** sh[0] - real) / real < 1e-8


def test_quantized_model_tracks_float_layer():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.3, (4, 32)); b = rng.normal(0, 0.1, 4)
    w2 = rng.normal(0, 0.3, (1, 4)); b2 = np.zeros(1)
    x = rng.integers(-60, 60, (500, 32)).astype(np.int8)
    xf = x * qnn.IN_SCALE
    h = np.maximum(xf @ w.T + b, 0)
    ref = h @ w2.T + b2
    q = qnn.quantize_model([(w, b), (w2, b2)], qnn.ARCHS["mlp8"][:1] + [("dense", 1, 0, 0)], [h.max()])
    out = qnn.run_int(q, x) * q[-1]["logit_scale"]
    assert np.corrcoef(out, ref[:, 0])[0, 1] > 0.999


def test_entropy_and_bins():
    assert entropy_bits([0, 0, 1, 1]) == 1.0
    c = bin_counts([0, BIN - 1, BIN, 5 * BIN + 3], 6 * BIN)
    assert c.tolist() == [2, 1, 0, 0, 0, 1]


def test_lagged_is_causal():
    F = np.arange(10.0)[:, None]
    L = lagged(F, 3)
    assert L[5].tolist() == [5, 4, 3] and L[0].tolist() == [0, 0, 0]


def test_robust_sigma():
    x = np.random.default_rng(1).normal(0, 7, 100_000)
    assert abs(robust_sigma(x) - 7) < 0.1
