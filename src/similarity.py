import math
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


def jaccard_set(a: Sequence[str], b: Sequence[str]) -> float:
    """
    Computes Jaccard similarity between two sequences treated as sets.
    """
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / float(len(A | B))


def cosine_counter(ca: Counter, cb: Counter) -> float:
    """
    Computes cosine similarity between two sparse non-negative vectors represented by Counters.
    """
    keys = set(ca) | set(cb)
    dot = sum(ca[k] * cb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def cosine_counts(a: Sequence[str], b: Sequence[str]) -> float:
    """
    Computes cosine similarity between unigram frequency vectors of two token sequences.
    """
    return cosine_counter(Counter(a), Counter(b))


def cosine_dict(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Computes cosine similarity between two dense small dict vectors.
    """
    return cosine_counter(Counter(a), Counter(b))


def js_divergence_from_counters(ca: Counter, cb: Counter, eps: float = 1e-12) -> float:
    """
    Computes Jensen-Shannon divergence between two discrete distributions.
    Symmetric measure of divergence.
    """
    keys = list(set(ca) | set(cb))
    if not keys:
        return 0.0

    pa = np.array([float(ca.get(k, 0.0)) for k in keys], dtype=float)
    pb = np.array([float(cb.get(k, 0.0)) for k in keys], dtype=float)

    suma = pa.sum()
    sumb = pb.sum()
    if suma > 0: pa = pa / suma
    if sumb > 0: pb = pb / sumb

    pa = pa + eps
    pb = pb + eps
    pa = pa / pa.sum()
    pb = pb / pb.sum()

    m = 0.5 * (pa + pb)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log(p / q)))

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def directional_kl_divergence(prior_counts: Counter, new_counts: Counter, eps: float = 1e-9) -> float:
    """
    Asymmetric Kullback-Leibler (KL) Divergence: KL(New || Prior).
    Measures the 'information surprise' or information gain of the Derived text (New) 
    given the Source text (Prior).
    
    FIXED for ICCC 2026: Applies log1p dampening to prevent mathematical explosions 
    when the derived text introduces massive amounts of entirely novel vocabulary.
    
    Formula: ln( 1 + sum( P(x) * log( P(x) / Q(x) ) ) )
    Where Q is the Prior (Source) and P is the New (Derived).
    """
    keys = list(set(prior_counts) | set(new_counts))
    if not keys: 
        return 0.0

    # Source is Q (Prior), Derived is P (New)
    q = np.array([float(prior_counts.get(k, 0.0)) for k in keys], dtype=float)
    p = np.array([float(new_counts.get(k, 0.0)) for k in keys], dtype=float)

    # Convert to probabilities with epsilon smoothing to prevent log(0) and division by 0
    q = (q / (q.sum() + eps)) + eps
    p = (p / (p.sum() + eps)) + eps
    
    # Re-normalize to strictly sum to 1.0
    q = q / q.sum()
    p = p / p.sum()

    # Calculate raw KL Divergence
    raw_kl = float(np.sum(p * np.log(p / q)))
    
    # Dampen the blowout using log1p (natural log of 1 + x)
    return float(np.log1p(max(0.0, raw_kl)))


def char_ngrams(s: str, n: int = 3) -> List[str]:
    """
    Extracts character n-grams from a string using a sliding window.
    """
    s2 = s.replace("\n", " ")
    if len(s2) < n:
        return []
    return [s2[i : i + n] for i in range(len(s2) - n + 1)]