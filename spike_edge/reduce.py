"""On-implant data-reduction schemes and their bit rates (per channel)."""
import numpy as np

from . import FS, ADC_BITS

BIN = FS // 50   # 20 ms bins -> 50 Hz


def bin_counts(events, n_samples):
    nb = n_samples // BIN
    return np.bincount(np.asarray(events) // BIN, minlength=nb)[:nb]


def sbp_int(y, shift):
    """Spike-band power: sum |y| per bin, >> shift, clamp to uint8 (2 ops/sample)."""
    nb = len(y) // BIN
    s = np.abs(y[:nb * BIN].astype(np.int64)).reshape(nb, BIN).sum(1)
    return np.minimum(s >> shift, 255)


def sbp_shift(y_cal):
    """Smallest shift putting the calibration median of the bin sum below 128."""
    med = int(np.median(np.abs(y_cal[:len(y_cal) // BIN * BIN].astype(np.int64)).reshape(-1, BIN).sum(1)))
    s = 0
    while (med >> s) >= 128:
        s += 1
    return s


def entropy_bits(symbols):
    """Zeroth-order empirical entropy (bits/symbol) = ideal static entropy-coder rate."""
    _, c = np.unique(np.asarray(symbols).ravel(), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())




def raw_rates(raw):
    """Uncompressed 10-bit rate and a lossless-compression estimate
    (entropy of the first difference, a cheap predictive coder)."""
    return dict(fixed=ADC_BITS * FS, lossless_est=entropy_bits(np.diff(raw.astype(np.int32))) * FS)


# --- realistic coder rates -----------------------------------------------------
# Zeroth-order entropy < 1 bit/symbol is not reachable with a per-symbol prefix code;
# these are rates real coders achieve.

COUNT_BITS = 3          # a-priori fixed width for spike counts: saturate at 7 per 20 ms bin
FRAME_HEADER_BITS = 64  # assumed per-frame sync + timestamp + CRC, one frame per bin for all channels


def saturate(counts, bits=COUNT_BITS):
    return np.minimum(counts, (1 << bits) - 1)


def huffman_lengths(train_symbols):
    """Code length (bits) per symbol of a static Huffman code fitted on `train_symbols`
    (unseen symbols get an escape of length max+1). Every code word is >= 1 bit."""
    import heapq
    vals, cnt = np.unique(np.asarray(train_symbols).ravel(), return_counts=True)
    if len(vals) == 1:
        return {int(vals[0]): 1}, 2
    heap = [(int(c), i, [int(v)]) for i, (v, c) in enumerate(zip(vals, cnt))]
    heapq.heapify(heap)
    L = {int(v): 0 for v in vals}
    k = len(heap)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap); c2, _, s2 = heapq.heappop(heap)
        for v in s1 + s2:
            L[v] += 1
        heapq.heappush(heap, (c1 + c2, k, s1 + s2)); k += 1
    return L, max(L.values()) + 1


def huffman_bits(train_symbols, test_symbols):
    L, esc = huffman_lengths(train_symbols)
    t = np.asarray(test_symbols).ravel()
    return float(np.mean([L.get(int(v), esc) for v in t]))


def adaptive_ac_bits(symbols, alphabet):
    """Bits/symbol of an adaptive (Krichevsky-Trofimov) zeroth-order arithmetic coder:
    an achievable rate (an actual arithmetic coder adds < 2 bits per stream)."""
    s = np.asarray(symbols).ravel().astype(np.int64)
    cnt = np.full(alphabet, 0.5)
    tot = 0.5 * alphabet
    bits = 0.0
    for v in s:
        bits -= np.log2(cnt[v] / tot)
        cnt[v] += 1; tot += 1
    return bits / len(s)


def framing_bits_per_s_per_channel(n_channels=1024, bins_per_s=50):
    return FRAME_HEADER_BITS * bins_per_s / n_channels
