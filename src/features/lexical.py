from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from preprocess import Window
from similarity import (
    char_ngrams, 
    js_divergence_from_counters, 
    jaccard_set, 
    directional_kl_divergence
)


def ttr(tokens: List[str]) -> float:
    """
    Simple type token ratio.
    Returns 0.0 for empty.
    """
    n = len(tokens)
    return (len(set(tokens)) / float(n)) if n else 0.0


def mattr(tokens: List[str], window: int = 200) -> float:
    """
    Moving average type token ratio (MATTR).
    If tokens shorter than window, falls back to TTR.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / float(n)

    vals: List[float] = []
    for i in range(0, n - window + 1):
        seg = tokens[i : i + window]
        vals.append(len(set(seg)) / float(window))
    return sum(vals) / float(len(vals)) if vals else 0.0


def shannon_entropy_from_counts(c: Counter, eps: float = 1e-12) -> float:
    """
    Shannon entropy (nats) of a discrete distribution represented by counts.
    Returns 0.0 if empty.
    """
    total = float(sum(c.values()))
    if total <= 0.0:
        return 0.0
    ent = 0.0
    for v in c.values():
        p = float(v) / total
        p = max(p, eps)
        ent -= p * math.log(p)
    return float(ent)


def zipf_slope_from_counts(c: Counter, min_types: int = 10) -> float:
    """
    Estimates Zipf slope by linear regression on:
        x = log(rank), y = log(freq)
    Returns 0.0 if not enough distinct types.
    """
    if not c:
        return 0.0
    freqs = sorted((v for v in c.values() if v > 0), reverse=True)
    if len(freqs) < min_types:
        return 0.0

    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    x = np.log(ranks)
    y = np.log(np.array(freqs, dtype=float))

    x = np.clip(x, -1e9, 1e9)
    y = np.clip(y, -1e9, 1e9)

    a, _b = np.polyfit(x, y, deg=1)
    return float(a)


def _idf_map_from_docs(docs: List[List[str]]) -> Dict[str, float]:
    """
    Computes a simple IDF map over a list of documents (token lists).
        idf(t) = log((N+1)/(df+1)) + 1
    """
    N = len(docs)
    if N == 0:
        return {}

    df: Counter = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1

    idf: Dict[str, float] = {}
    for t, dft in df.items():
        idf[t] = math.log((N + 1.0) / (float(dft) + 1.0)) + 1.0
    return idf


def idf_weighted_jaccard(a: List[str], b: List[str], idf: Dict[str, float]) -> float:
    """
    IDF weighted Jaccard similarity: sum_idf(intersection) / sum_idf(union)
    Returns 0.0 if both empty.
    """
    A = set(a)
    B = set(b)
    if not A and not B:
        return 0.0
    inter = A & B
    uni = A | B
    num = sum(float(idf.get(t, 0.0)) for t in inter)
    den = sum(float(idf.get(t, 0.0)) for t in uni)
    return float(num / den) if den > 0.0 else 0.0


def rare_token_set(tokens: List[str], freq_all: Counter, q: float = 0.10) -> set[str]:
    """
    Returns the subset of tokens considered "rare" under the combined (A+B) distribution.
    """
    if not tokens or not freq_all:
        return set()

    freqs = np.array(list(freq_all.values()), dtype=float)
    if freqs.size == 0:
        return set()

    thr = float(np.quantile(freqs, q))
    thr = max(thr, 1.0)

    return {t for t in set(tokens) if float(freq_all.get(t, 0)) <= thr}


def compute(
    wA: List[Window],
    wB: List[Window],
    aligned: List[Tuple[int, int]],
    ctx: Dict[str, Any],
) -> Dict[str, List[float]]:
    """
    Computes lexical-level features.
    """
    mattr_window = int(ctx.get("lexical_mattr_window", 200))
    rare_q = float(ctx.get("lexical_rare_quantile", 0.10))

    docs_content_A = [w.content_tokens or [] for w in wA]
    docs_content_B = [w.content_tokens or [] for w in wB]
    idf = _idf_map_from_docs(docs_content_A + docs_content_B)

    freq_all_content = Counter()
    for d in (docs_content_A + docs_content_B):
        freq_all_content.update(d)

    out: Dict[str, List[float]] = {
        "lex_jaccard": [],
        "content_lex_jaccard": [],
        "rare_content_jaccard": [],
        "idf_content_jaccard": [],
        "lex_unigram_jsd": [],
        "char3_jsd": [],
        "char4_jsd": [],
        "ttr_A": [],
        "ttr_B": [],
        "mattr_A": [],
        "mattr_B": [],
        "entropy_A": [],
        "entropy_B": [],
        "zipf_slope_A": [],
        "zipf_slope_B": [],
        "zipf_slope_diff": [],
        "entropy_diff": [],
        "lex_kl_surprise": [], # NEW: Asymmetric Information Gain
    }

    for i, j in aligned:
        a_tokens = wA[i].tokens
        b_tokens = wB[j].tokens
        ac = wA[i].content_tokens or []
        bc = wB[j].content_tokens or []

        # Surface overlap (sets)
        out["lex_jaccard"].append(jaccard_set(a_tokens, b_tokens))
        out["content_lex_jaccard"].append(jaccard_set(ac, bc))

        # Rare overlap on content words (pair specific rarity)
        rareA = rare_token_set(ac, freq_all_content, q=rare_q)
        rareB = rare_token_set(bc, freq_all_content, q=rare_q)
        out["rare_content_jaccard"].append(
            (len(rareA & rareB) / float(len(rareA | rareB))) if (rareA or rareB) else 0.0
        )

        # IDF weighted overlap (content)
        out["idf_content_jaccard"].append(idf_weighted_jaccard(ac, bc, idf))

        # Distributional divergence (unigrams)
        out["lex_unigram_jsd"].append(
            js_divergence_from_counters(Counter(a_tokens), Counter(b_tokens))
        )

        # Asymmetric Information Surprise (KL Divergence)
        # Source (ac) is the prior, Derived (bc) is the new distribution
        out["lex_kl_surprise"].append(
            directional_kl_divergence(Counter(ac), Counter(bc))
        )

        # Character n-gram divergence (raw)
        ra = wA[i].raw or ""
        rb = wB[j].raw or ""
        out["char3_jsd"].append(
            js_divergence_from_counters(Counter(char_ngrams(ra, n=3)), Counter(char_ngrams(rb, n=3)))
        )
        out["char4_jsd"].append(
            js_divergence_from_counters(Counter(char_ngrams(ra, n=4)), Counter(char_ngrams(rb, n=4)))
        )

        # Per window lexical statistics
        ttrA = ttr(a_tokens)
        ttrB = ttr(b_tokens)
        out["ttr_A"].append(ttrA)
        out["ttr_B"].append(ttrB)

        out["mattr_A"].append(mattr(a_tokens, window=mattr_window))
        out["mattr_B"].append(mattr(b_tokens, window=mattr_window))

        entA = shannon_entropy_from_counts(Counter(a_tokens))
        entB = shannon_entropy_from_counts(Counter(b_tokens))
        out["entropy_A"].append(entA)
        out["entropy_B"].append(entB)
        out["entropy_diff"].append(float(entA - entB))

        slopeA = zipf_slope_from_counts(Counter(a_tokens))
        slopeB = zipf_slope_from_counts(Counter(b_tokens))
        out["zipf_slope_A"].append(slopeA)
        out["zipf_slope_B"].append(slopeB)
        out["zipf_slope_diff"].append(float(slopeA - slopeB))

    return out