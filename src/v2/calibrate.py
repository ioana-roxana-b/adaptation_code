from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ALPHA = 0.20                 # Eq. 7
DIVERGENCE_THRESHOLD = -0.5  # channel z (in control SDs) below this counts as divergent
Z_CLIP = 5.0
RIDGE = 1.0                  # Additive: a level seen n times keeps n / (n + RIDGE) of its effect
N_BOOT = 1000
SEED = 0
EPS = 1e-9

Channels = Dict[str, Dict[str, int]]
References = Dict[str, List[str]]


def channelled(z: np.ndarray) -> Tuple[float, int, int]:
    """Eq. 7 on channel z-scores: (score, peak channel index, divergent channels)."""
    ok = ~np.isnan(z)
    if not ok.any():
        return np.nan, -1, 0
    peak = int(np.nanargmax(z))
    n_neg = int((z[ok] < DIVERGENCE_THRESHOLD).sum())
    return max(0.0, float(z[peak])) * (1 + ALPHA * n_neg), peak, n_neg


def block_length(starts: np.ndarray, ends: np.ndarray) -> int:
    """1 + median number of later windows that start inside a window."""
    order = np.argsort(starts)
    s, e = starts[order], ends[order]
    later = [int(np.sum(s[i + 1:] < e[i])) for i in range(len(s))]
    return 1 + int(np.median(later)) if later else 1


def pair_features(rows: pd.DataFrame, pair_col: str, pos_col: str, end_col: str,
                  features: List[str]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, int]]:
    """Per pair: mean features, the per-window matrix (for the bootstrap) and its block length."""
    means, windows, blocks = {}, {}, {}
    for pk, sub in rows.groupby(pair_col, sort=False):
        per_window = sub.groupby(pos_col)[features + [end_col]].mean().sort_index()
        windows[pk] = per_window[features].to_numpy(dtype=float)
        means[pk] = per_window[features].mean()
        blocks[pk] = block_length(per_window.index.to_numpy(dtype=float), per_window[end_col].to_numpy(dtype=float))
    return pd.DataFrame(means).T, windows, blocks


def all_controls(meta: pd.DataFrame, is_ctrl: pd.Series) -> References:
    """Matched controls of every pair: all controls of the cell except the pair itself."""
    ctrl = list(is_ctrl.index[is_ctrl])
    return {pk: [c for c in ctrl if c != pk] for pk in is_ctrl.index}


def _percentile(score: float, null: np.ndarray) -> Tuple[float, float]:
    null = null[~np.isnan(null)]
    if np.isnan(score) or not len(null):
        return np.nan, np.nan
    pct = (np.sum(null < score) + 0.5 * np.sum(null == score)) / len(null)
    p = (1 + np.sum(null >= score)) / (1 + len(null))
    return float(pct), float(p)


class MatchedMean:
    """Baseline location: the mean of the pair's matched controls."""

    def __init__(self, refs: References):
        self.refs = refs

    def n_ref(self, pk: str) -> int:
        return len(self.refs[pk])

    def __call__(self, pk: str, M: np.ndarray, pos: Dict[str, int]) -> np.ndarray:
        return np.nanmean(M[[pos[c] for c in self.refs[pk]]], axis=0)


class Additive:
    """Baseline location from an additive model fitted to the controls other than the pair.

    value = mean + sum of factor-level effects (e.g. a novel effect and a
    screenplay effect), with ridge-shrunk dummy effects. The prediction for an
    adaptation is what its novel and its screenplay score against unrelated
    partners, so a dialogue-heavy novel or a transcript-format screenplay does
    not pass for similarity.
    """

    def __init__(self, meta: pd.DataFrame, is_ctrl: pd.Series, factors: List[str]):
        self.levels = meta[factors].astype(str)
        self.D = pd.get_dummies(self.levels).to_numpy(dtype=float)
        self.row = {pk: i for i, pk in enumerate(meta.index)}
        self.ctrl = list(is_ctrl.index[is_ctrl])

    def n_ref(self, pk: str) -> int:
        """Controls sharing at least one factor level with the pair."""
        others = [c for c in self.ctrl if c != pk]
        L = self.levels.loc[others].to_numpy()
        return int((L == self.levels.loc[pk].to_numpy()).any(axis=1).sum())

    def __call__(self, pk: str, M: np.ndarray, pos: Dict[str, int]) -> np.ndarray:
        rows = [c for c in self.ctrl if c != pk]
        X = self.D[[self.row[c] for c in rows]]
        Y = M[[pos[c] for c in rows]]
        d = self.D[self.row[pk]]
        out = np.full(M.shape[1], np.nan)
        # Columns without missing values share one solve
        full = ~np.isnan(Y).any(axis=0)
        if full.any():
            mu = Y[:, full].mean(axis=0)
            b = np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ (Y[:, full] - mu))
            out[full] = mu + d @ b
        for j in np.flatnonzero(~full):
            ok = ~np.isnan(Y[:, j])
            if ok.sum() < 2:
                continue
            y, Xj = Y[ok, j], X[ok]
            mu = y.mean()
            b = np.linalg.solve(Xj.T @ Xj + RIDGE * np.eye(Xj.shape[1]), Xj.T @ (y - mu))
            out[j] = mu + d @ b
        return out


class _Baseline:
    """Location from the locator (matched controls or a model of them), scale from all controls' residuals."""

    def __init__(self, M: np.ndarray, pos: Dict[str, int], ctrl: List[str], locate):
        self.loc = {pk: locate(pk, M, pos) for pk in pos}
        resid = {c: M[pos[c]] - self.loc[c] for c in ctrl}
        self.scale = {pk: np.nanstd(np.stack([resid[c] for c in ctrl if c != pk]), axis=0, ddof=1)
                      for pk in pos}

    def z(self, pk: str, x: np.ndarray) -> np.ndarray:
        sd = self.scale[pk]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > EPS, (x - self.loc[pk]) / sd, np.nan)


def score_cell(pairs: pd.DataFrame, is_ctrl: pd.Series, locate, channels: Channels,
               windows: Dict[str, np.ndarray] = None, blocks: Dict[str, int] = None,
               n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Channel z-scores and scores for the pairs (rows, indexed by pair id) of one cell.

    locate gives each pair's baseline location (MatchedMean, Additive); a
    References dict is taken as MatchedMean.
    """
    if isinstance(locate, dict):
        locate = MatchedMean(locate)
    chans = {ch: {f: s for f, s in fs.items() if f in pairs.columns} for ch, fs in channels.items()}
    chans = {ch: fs for ch, fs in chans.items() if fs}
    names = list(chans)
    feats = [f for fs in chans.values() for f in fs]
    sgn = np.array([s for fs in chans.values() for s in fs.values()], dtype=float)
    cols = [[feats.index(f) for f in chans[ch]] for ch in names]
    X = pairs[feats].to_numpy(dtype=float)
    pos = {pk: i for i, pk in enumerate(pairs.index)}
    ctrl = list(pairs.index[is_ctrl])

    def to_channels(pk: str, F: np.ndarray) -> np.ndarray:
        Z = np.clip(sgn * f_base.z(pk, F), -Z_CLIP, Z_CLIP)
        with np.errstate(invalid="ignore"):
            return np.stack([np.nanmean(Z[..., c], axis=-1) for c in cols], axis=-1)

    # Stages 2-3: feature z -> mean per channel -> channel z
    f_base = _Baseline(X, pos, ctrl, locate)
    raw = np.stack([to_channels(pk, X[pos[pk]]) for pk in pairs.index])
    c_base = _Baseline(raw, pos, ctrl, locate)
    cz = np.stack([c_base.z(pk, raw[pos[pk]]) for pk in pairs.index])

    # Stage 4: scores against the other controls' scores
    stats = [channelled(row) for row in cz]
    with np.errstate(invalid="ignore"):
        profile = np.nanmean(cz, axis=1)
    out = pd.DataFrame(cz, index=pairs.index, columns=[f"z_{ch}" for ch in names])
    out["peak_channel"] = [names[s[1]] if s[1] >= 0 else None for s in stats]
    out["n_divergent"] = [s[2] for s in stats]
    for name, s in (("channelled", np.array([s[0] for s in stats])), ("profile", profile)):
        out[name] = s
        res = [_percentile(s[pos[pk]], s[[pos[c] for c in ctrl if c != pk]]) for pk in pairs.index]
        out[f"{name}_pct"] = [r[0] for r in res]
        out[f"{name}_p"] = [r[1] for r in res]
    out["n_matched"] = [locate.n_ref(pk) for pk in pairs.index]
    out["n_controls"] = len(ctrl)

    # Stage 5: block bootstrap for pairs that are not controls, baseline fixed
    if windows is not None and n_boot > 0:
        rng = np.random.default_rng(seed)
        lo_hi = {}
        for pk in pairs.index[~is_ctrl]:
            W = windows[pk][:, [list(pairs.columns).index(f) for f in feats]]
            n, L = len(W), min(blocks[pk], len(W))
            starts = rng.integers(0, n - L + 1, size=(n_boot, int(np.ceil(n / L))))
            idx = (starts[:, :, None] + np.arange(L)).reshape(n_boot, -1)[:, :n]
            with np.errstate(invalid="ignore"):
                CZ = c_base.z(pk, to_channels(pk, np.nanmean(W[idx], axis=1)))
                boot = {f"z_{ch}": CZ[:, i] for i, ch in enumerate(names)}
                boot["channelled"] = np.array([channelled(r)[0] for r in CZ])
                boot["profile"] = np.nanmean(CZ, axis=1)
            lo_hi[pk] = {f"{k}_{q}": float(np.nanpercentile(v, p))
                         for k, v in boot.items() for q, p in (("lo", 2.5), ("hi", 97.5))}
        if lo_hi:
            out = out.join(pd.DataFrame(lo_hi).T)
    return out


def additive(factors: List[str]):
    """Reference for crossed designs: an Additive model on the given pair-level columns."""
    return lambda meta, is_ctrl: Additive(meta, is_ctrl, factors)


def score(rows: pd.DataFrame, channels: Channels, cell_cols: List[str], pair_col: str = "pair_key",
          type_col: str = "pair_type", pos_col: str = "B_unit_start", end_col: str = "B_unit_end",
          keep_cols: List[str] = (), reference=all_controls, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Score every pair in every cell.

    keep_cols are pair-level columns carried to the output (and passed to
    reference); reference(meta, is_ctrl) returns each pair's matched controls
    or a locator (e.g. additive(["novel", "screenplay"])).
    """
    feats = [f for fs in channels.values() for f in fs if f in rows.columns]
    for f in feats:
        rows[f] = pd.to_numeric(rows[f], errors="coerce")
    results = []
    for cell, sub in rows.groupby(cell_cols):
        pairs, windows, blocks = pair_features(sub, pair_col, pos_col, end_col, feats)
        meta = sub.groupby(pair_col)[[type_col, *keep_cols]].first().loc[pairs.index]
        is_ctrl = meta[type_col] == "control"
        res = score_cell(pairs, is_ctrl, reference(meta, is_ctrl), channels, windows, blocks, n_boot=n_boot)
        res = meta.join(res, how="right")
        for col, val in zip(cell_cols, cell if isinstance(cell, tuple) else (cell,)):
            res[col] = val
        res["block_length"] = pd.Series(blocks)
        results.append(res.rename_axis(pair_col).reset_index())
    return pd.concat(results, ignore_index=True)
