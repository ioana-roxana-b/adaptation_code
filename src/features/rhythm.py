from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import math
import zlib
import numpy as np
import nltk

from preprocess import Window
from similarity import cosine_dict


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0.0 else 0.0


def _entropy_from_counts(c: Counter) -> float:
    total = float(sum(c.values()))
    if total <= 0.0:
        return 0.0
    ent = 0.0
    for v in c.values():
        p = float(v) / total
        if p > 0.0:
            ent -= p * math.log(p)
    return float(ent)


def _burstiness(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return float((sd - mu) / (sd + mu)) if (sd + mu) > 0.0 else 0.0


def _mad_adjacent(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(x))))


def _autocorr1(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    x0 = x[:-1]
    x1 = x[1:]
    s0 = float(np.std(x0))
    s1 = float(np.std(x1))
    if s0 == 0.0 or s1 == 0.0:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def _sent_token_lengths(raw: str) -> np.ndarray:
    text = raw or ""
    if not text.strip():
        return np.array([], dtype=float)
    sents = nltk.sent_tokenize(text)
    lens: List[int] = []
    for s in sents:
        toks = nltk.word_tokenize(s)
        lens.append(len(toks))
    return np.array(lens, dtype=float)


def _punct_per_sentence(raw: str) -> np.ndarray:
    text = raw or ""
    if not text.strip():
        return np.array([], dtype=float)
    punct_chars = set(list(".,;:!?\"'()[]{}-—"))
    sents = nltk.sent_tokenize(text)
    vals: List[int] = []
    for s in sents:
        vals.append(sum(1 for ch in s if ch in punct_chars))
    return np.array(vals, dtype=float)


def _bin_series(x: np.ndarray, n_bins: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros(n_bins, dtype=float)
    if x.size == 1:
        out = np.zeros(n_bins, dtype=float)
        out[0] = float(x[0])
        return out

    pos = np.linspace(0.0, 1.0, num=x.size)
    edges = np.linspace(0.0, 1.0, num=n_bins + 1)

    out = np.zeros(n_bins, dtype=float)
    cnt = np.zeros(n_bins, dtype=float)

    for p, v in zip(pos, x):
        idx = int(np.searchsorted(edges, p, side="right") - 1)
        idx = max(0, min(n_bins - 1, idx))
        out[idx] += float(v)
        cnt[idx] += 1.0

    cnt = np.where(cnt == 0.0, 1.0, cnt)
    return out / cnt


def _cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    da = float(np.dot(a, a))
    db = float(np.dot(b, b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(np.dot(a, b) / math.sqrt(da * db))


def _compress_ratio(raw: str) -> float:
    b = (raw or "").encode("utf-8")
    if len(b) == 0:
        return 0.0
    c = zlib.compress(b)
    return float(len(c) / float(len(b)))


def compute(
    wA: List[Window],
    wB: List[Window],
    aligned: List[Tuple[int, int]],
    ctx: Dict[str, Any],
) -> Dict[str, List[float]]:
    """
    Rhythmic and temporal features.

    Outputs
        punct_cosine                     cosine over Window.punct dicts
        sentlen_ts_cosine                cosine over binned sentence length trajectories
        punct_ts_cosine                  cosine over binned punctuation per sentence trajectories
        sentlen_mean_A, sentlen_std_A, sentlen_cv_A
        sentlen_burstiness_A, sentlen_mad_adj_A, sentlen_autocorr1_A
        and B analogs
        punct_density_tok_A, punct_entropy_A, punct_burstiness_A
        and B analogs
        compress_ratio_A, compress_ratio_B
    """
    n_bins = int(ctx.get("rhythm_bins", 10))

    out: Dict[str, List[float]] = {
        "punct_cosine": [],
        "sentlen_ts_cosine": [],
        "punct_ts_cosine": [],
        "sentlen_mean_A": [],
        "sentlen_std_A": [],
        "sentlen_cv_A": [],
        "sentlen_burstiness_A": [],
        "sentlen_mad_adj_A": [],
        "sentlen_autocorr1_A": [],
        "sentlen_mean_B": [],
        "sentlen_std_B": [],
        "sentlen_cv_B": [],
        "sentlen_burstiness_B": [],
        "sentlen_mad_adj_B": [],
        "sentlen_autocorr1_B": [],
        "punct_density_tok_A": [],
        "punct_entropy_A": [],
        "punct_burstiness_A": [],
        "punct_density_tok_B": [],
        "punct_entropy_B": [],
        "punct_burstiness_B": [],
        "compress_ratio_A": [],
        "compress_ratio_B": [],
    }

    for i, j in aligned:
        wa = wA[i]
        wb = wB[j]

        pa = wa.punct or {}
        pb = wb.punct or {}
        out["punct_cosine"].append(cosine_dict(pa, pb) if pa and pb else 0.0)

        la = _sent_token_lengths(wa.raw or "")
        lb = _sent_token_lengths(wb.raw or "")

        out["sentlen_ts_cosine"].append(_cosine_vec(_bin_series(la, n_bins), _bin_series(lb, n_bins)))

        muA = float(np.mean(la)) if la.size else 0.0
        sdA = float(np.std(la)) if la.size else 0.0
        out["sentlen_mean_A"].append(muA)
        out["sentlen_std_A"].append(sdA)
        out["sentlen_cv_A"].append(_safe_div(sdA, muA) if muA > 0.0 else 0.0)
        out["sentlen_burstiness_A"].append(_burstiness(la))
        out["sentlen_mad_adj_A"].append(_mad_adjacent(la))
        out["sentlen_autocorr1_A"].append(_autocorr1(la))

        muB = float(np.mean(lb)) if lb.size else 0.0
        sdB = float(np.std(lb)) if lb.size else 0.0
        out["sentlen_mean_B"].append(muB)
        out["sentlen_std_B"].append(sdB)
        out["sentlen_cv_B"].append(_safe_div(sdB, muB) if muB > 0.0 else 0.0)
        out["sentlen_burstiness_B"].append(_burstiness(lb))
        out["sentlen_mad_adj_B"].append(_mad_adjacent(lb))
        out["sentlen_autocorr1_B"].append(_autocorr1(lb))

        ppsA = _punct_per_sentence(wa.raw or "")
        ppsB = _punct_per_sentence(wb.raw or "")
        out["punct_ts_cosine"].append(_cosine_vec(_bin_series(ppsA, n_bins), _bin_series(ppsB, n_bins)))

        n_tokA = float(len(wa.tokens or []))
        n_tokB = float(len(wb.tokens or []))
        tot_punctA = float(sum(pa.values())) if pa else 0.0
        tot_punctB = float(sum(pb.values())) if pb else 0.0

        out["punct_density_tok_A"].append(_safe_div(tot_punctA, n_tokA))
        out["punct_density_tok_B"].append(_safe_div(tot_punctB, n_tokB))

        out["punct_entropy_A"].append(_entropy_from_counts(Counter(pa)) if pa else 0.0)
        out["punct_entropy_B"].append(_entropy_from_counts(Counter(pb)) if pb else 0.0)

        out["punct_burstiness_A"].append(_burstiness(ppsA))
        out["punct_burstiness_B"].append(_burstiness(ppsB))

        out["compress_ratio_A"].append(_compress_ratio(wa.raw or ""))
        out["compress_ratio_B"].append(_compress_ratio(wb.raw or ""))

    return out
