"""Linear (Wiener-filter style) velocity decoding with blocked CV and block-bootstrap CIs."""
import numpy as np

LAGS = 5   # 100 ms of causal history at 20 ms bins


def lagged(F, lags=LAGS):
    T, C = F.shape
    out = np.zeros((T, C * lags))
    for k in range(lags):
        out[k:, k * C:(k + 1) * C] = F[:T - k]
    return out


def _ridge(X, Y, a):
    return np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ Y)


def r2(Y, P):
    return float(np.mean(1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)))


def fit_eval(F, V, train_frac=0.7, alphas=10.0 ** np.arange(-1, 5), n_folds=5,
             block=250, n_boot=2000, seed=0):
    """Contiguous train (first 70%) / test (last 30%) split; no shuffling, so no
    temporal leakage. Alpha chosen by contiguous K-fold CV inside train only.
    CI: bootstrap over 5 s test blocks (block=250 bins)."""
    X = lagged(F.astype(np.float64)); T = len(X); n_tr = int(train_frac * T)
    gap = LAGS  # drop bins whose lag window straddles the split
    Xtr, Ytr, Xte, Yte = X[:n_tr], V[:n_tr], X[n_tr + gap:], V[n_tr + gap:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]; Xte = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    folds = np.array_split(np.arange(n_tr), n_folds)
    cv = []
    for a in alphas:
        s = []
        for f in folds:
            m = np.ones(n_tr, bool); m[max(f[0] - gap, 0):f[-1] + gap + 1] = False
            s.append(r2(Ytr[f], Xtr[f] @ _ridge(Xtr[m], Ytr[m], a)))
        cv.append(np.mean(s))
    a = alphas[int(np.argmax(cv))]
    P = Xte @ _ridge(Xtr, Ytr, a)
    nb = len(P) // block
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        b = rng.integers(0, nb, nb)
        ii = (b[:, None] * block + np.arange(block)[None]).ravel()
        bs.append(r2(Yte[ii], P[ii]))
    return dict(r2=r2(Yte, P), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                alpha=float(a), pred=P, y=Yte)


def paired_diff_ci(Y, Pa, Pb, block=250, n_boot=2000, seed=0):
    """Block-bootstrap CI of R2(a) - R2(b) on the same test bins."""
    nb = len(Y) // block
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ii = (rng.integers(0, nb, nb)[:, None] * block + np.arange(block)[None]).ravel()
        d.append(r2(Y[ii], Pa[ii]) - r2(Y[ii], Pb[ii]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
