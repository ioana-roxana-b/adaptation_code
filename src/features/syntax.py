from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import nltk

from preprocess import Window
from similarity import cosine_counter


# -----------------------------
# Utilities
# -----------------------------

_STOPWORDS: set[str] | None = None

def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is not None:
        return _STOPWORDS
    try:
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english"))
    except Exception:
        _STOPWORDS = set()
    return _STOPWORDS


# -----------------------------
# POS template features
# -----------------------------

def pos_ngram_vector(pos_list: List[str], n: int) -> Counter:
    if not pos_list or len(pos_list) < n:
        return Counter()
    grams = list(nltk.ngrams(pos_list, n))
    return Counter(grams)

def pos_adjacency_edges(pos_list: List[str]) -> Counter:
    edges = Counter()
    if not pos_list or len(pos_list) < 2:
        return edges
    for a, b in zip(pos_list, pos_list[1:]):
        edges[(a, b)] += 1
    return edges


# -----------------------------
# Clause and embedding proxies
# -----------------------------

_SUBORD_POS_MARKERS = {"IN", "WDT", "WP", "WRB"}
_SUBORDINATOR_TOKENS = {
    "because", "although", "though", "while", "whereas",
    "if", "unless", "since", "once", "until", "when", "whenever",
    "that", "whether", "as", "before", "after",
}
_RELATIVE_PRONOUNS = {"which", "that", "who", "whom", "whose", "where", "when"}

def subord_pos_rate(pos_list: List[str]) -> float:
    if not pos_list:
        return 0.0
    c = sum(1 for p in pos_list if p in _SUBORD_POS_MARKERS)
    return float(c / len(pos_list))

def token_rate(tokens: List[str], target_set: set[str]) -> float:
    if not tokens:
        return 0.0
    c = sum(1 for t in tokens if t in target_set)
    return float(c / len(tokens))


# -----------------------------
# Function word usage patterns
# -----------------------------

def function_word_profile(tokens: List[str]) -> Counter:
    sw = _get_stopwords()
    if not sw:
        return Counter()
    fw = [t for t in tokens if t in sw]
    return Counter(fw)

def function_word_rate(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    sw = _get_stopwords()
    if not sw:
        return 0.0
    fw = sum(1 for t in tokens if t in sw)
    return float(fw / len(tokens))


# -----------------------------
# Deep Dependency Features (Normalized via Meta)
# -----------------------------

def compute_dependency_stats(meta: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes shallow dependency depth and clause rate using preprocessed integers in Window.meta.
    """
    depths = meta.get("dep_depths", [])
    clause_hits = meta.get("clause_hits", 0)
    sent_len = meta.get("dep_valid_toks", 0)

    if not depths or sent_len == 0:
        return {
            "dep_depth_norm": 0.0,
            "dep_depth_var": 0.0,
            "dep_depth_max": 0.0,
            "dep_clause_rate": 0.0,
        }

    raw_depth_mean = float(np.mean(depths))
    normalized_depth = raw_depth_mean / float(np.log(sent_len + 2))

    return {
        "dep_depth_norm": normalized_depth,
        "dep_depth_var": float(np.var(depths)),
        "dep_depth_max": float(np.max(depths)),
        "dep_clause_rate": float(clause_hits / sent_len),
    }


# -----------------------------
# Main Compute Dispatcher
# -----------------------------

def compute(
    wA: List[Window],
    wB: List[Window],
    aligned: List[Tuple[int, int]],
    ctx: Dict[str, Any],
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {
        "pos_trigram_cosine": [],
        "pos4_cosine": [],
        "pos5_cosine": [],
        "pos_graph_cosine": [],
        "subord_pos_rate_A": [],
        "subord_pos_rate_B": [],
        "subordinator_token_rate_A": [],
        "subordinator_token_rate_B": [],
        "relative_pronoun_rate_A": [],
        "relative_pronoun_rate_B": [],
        "fw_rate_A": [],
        "fw_rate_B": [],
        "fw_unigram_cosine": [],
        "dep_depth_norm_A": [],
        "dep_depth_var_A": [],
        "dep_depth_max_A": [],
        "dep_clause_rate_A": [],
        "dep_depth_norm_B": [],
        "dep_depth_var_B": [],
        "dep_depth_max_B": [],
        "dep_clause_rate_B": [],
    }

    for i, j in aligned:
        posA, posB = wA[i].pos or [], wB[j].pos or []
        toksA, toksB = wA[i].tokens or [], wB[j].tokens or []

        out["pos_trigram_cosine"].append(float(cosine_counter(pos_ngram_vector(posA, 3), pos_ngram_vector(posB, 3))))
        out["pos4_cosine"].append(float(cosine_counter(pos_ngram_vector(posA, 4), pos_ngram_vector(posB, 4))))
        out["pos5_cosine"].append(float(cosine_counter(pos_ngram_vector(posA, 5), pos_ngram_vector(posB, 5))))
        out["pos_graph_cosine"].append(float(cosine_counter(pos_adjacency_edges(posA), pos_adjacency_edges(posB))))

        out["subord_pos_rate_A"].append(subord_pos_rate(posA))
        out["subord_pos_rate_B"].append(subord_pos_rate(posB))
        out["subordinator_token_rate_A"].append(token_rate(toksA, _SUBORDINATOR_TOKENS))
        out["subordinator_token_rate_B"].append(token_rate(toksB, _SUBORDINATOR_TOKENS))
        out["relative_pronoun_rate_A"].append(token_rate(toksA, _RELATIVE_PRONOUNS))
        out["relative_pronoun_rate_B"].append(token_rate(toksB, _RELATIVE_PRONOUNS))

        out["fw_rate_A"].append(function_word_rate(toksA))
        out["fw_rate_B"].append(function_word_rate(toksB))
        out["fw_unigram_cosine"].append(float(cosine_counter(function_word_profile(toksA), function_word_profile(toksB))))

        # Read dependency stats safely from the pre-calculated meta dictionary
        da = compute_dependency_stats(wA[i].meta)
        db = compute_dependency_stats(wB[j].meta)
        for k in ["dep_depth_norm", "dep_depth_var", "dep_depth_max", "dep_clause_rate"]:
            out[f"{k}_A"].append(da[k])
            out[f"{k}_B"].append(db[k])

    return out