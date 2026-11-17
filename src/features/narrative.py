from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from preprocess import Window


# =====================================================================
# 1. Narrative Texture Vectors
# =====================================================================

def _function_vector(w: Window) -> np.ndarray:
    """Extract a narrative TEXTURE vector from a Window.

    Characterises the COMPOSITIONAL STYLE of this passage — the balance
    of action, description, dialogue, complexity, etc. — without caring
    about WHAT it's specifically about.

    NOTE (v1.2): Named "function vector" for backward compatibility, but
    this measures TEXTURE (POS-level compositional characteristics), not
    abstract narrative FUNCTION (Proppian roles, event schemas). See
    module docstring for the distinction and its consequences.

    Uses only data already in the Window object — no new NLP models.
    All features are rate-based for length normalisation.
    """
    pos = w.pos or []
    tokens = w.tokens or []
    raw = w.raw or ""
    n_pos = max(1, len(pos))
    n_chars = max(1, len(raw))
    meta = w.meta or {}

    # 1. Character focus: proper noun density
    proper_noun_rate = sum(1 for p in pos if p in {"NNP", "NNPS"}) / n_pos

    # 2. Action intensity: verb density
    verb_rate = sum(1 for p in pos if p.startswith("VB")) / n_pos

    # 3. Description density: adjective density
    adj_rate = sum(1 for p in pos if p.startswith("JJ")) / n_pos

    # 4. Introspection / reflection: adverb + modal density
    reflect_rate = sum(1 for p in pos if p in {"MD", "RB", "RBR", "RBS"}) / n_pos

    # 5. Dialogue density: quotation mark proportion
    n_quotes = sum(raw.count(c) for c in ['"', '\u201c', '\u201d', '\u2018', '\u2019'])
    dialogue_density = n_quotes / n_chars

    # 6. Emotional intensity markers
    excl_rate = raw.count("!") / n_chars
    quest_rate = raw.count("?") / n_chars

    # 7. Concreteness: noun+verb rate (sensory content)
    concrete_tags = {
        "NN", "NNS", "NNP", "NNPS",
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    }
    concrete_rate = sum(1 for p in pos if p in concrete_tags) / n_pos

    # 8. Subordination complexity
    subord_rate = sum(1 for p in pos if p in {"IN", "WDT", "WP", "WRB"}) / n_pos

    # 9. Dependency depth (from preprocessed meta)
    dep_depths = meta.get("dep_depths", [])
    depth_mean = float(np.mean(dep_depths)) if dep_depths else 0.0

    # 10. Clause density
    clause_hits = meta.get("clause_hits", 0)
    dep_valid = meta.get("dep_valid_toks", 0)
    clause_rate = clause_hits / max(1, dep_valid)

    # 11. Coordination rate (compound structures)
    coord_rate = sum(1 for p in pos if p == "CC") / n_pos

    # 12. Past tense rate (narrative distance)
    past_rate = sum(1 for p in pos if p in {"VBD", "VBN"}) / n_pos

    return np.array([
        proper_noun_rate,   # 0
        verb_rate,          # 1
        adj_rate,           # 2
        reflect_rate,       # 3
        dialogue_density,   # 4
        excl_rate,          # 5
        quest_rate,         # 6
        concrete_rate,      # 7
        subord_rate,        # 8
        depth_mean,         # 9
        clause_rate,        # 10
        coord_rate,         # 11
        past_rate,          # 12
    ], dtype=float)


_FUNCTION_DIM = 13


# =====================================================================
# 2. Trajectory Binning and Comparison
# =====================================================================

def _bin_trajectory(vectors: List[np.ndarray], n_bins: int) -> np.ndarray:
    """Bin a sequence of vectors into n_bins positional buckets.

    Returns shape (n_bins, n_features).
    """
    n = len(vectors)
    if n == 0:
        return np.zeros((n_bins, _FUNCTION_DIM), dtype=float)

    dim = vectors[0].shape[0]
    positions = np.linspace(0.0, 1.0, n)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    binned = np.zeros((n_bins, dim), dtype=float)
    counts = np.zeros(n_bins, dtype=float)

    for idx, pos in enumerate(positions):
        b = int(np.searchsorted(edges, pos, side="right") - 1)
        b = max(0, min(n_bins - 1, b))
        binned[b] += vectors[idx]
        counts[b] += 1.0

    counts = np.where(counts == 0, 1.0, counts)
    return binned / counts[:, np.newaxis]


def _cosine_flat(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten(), b.flatten()
    na, nb = float(np.linalg.norm(af)), float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(af, bf) / (na * nb))


def _cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _per_track_cosines(tA: np.ndarray, tB: np.ndarray) -> np.ndarray:
    n_features = tA.shape[1]
    cosines = np.zeros(n_features, dtype=float)
    for f in range(n_features):
        cosines[f] = _cosine_vec(tA[:, f], tB[:, f])
    return cosines


def _per_track_correlations(tA: np.ndarray, tB: np.ndarray) -> np.ndarray:
    n_features = tA.shape[1]
    corrs = np.zeros(n_features, dtype=float)
    for f in range(n_features):
        corrs[f] = _pearson(tA[:, f], tB[:, f])
    return corrs


# =====================================================================
# 3. Texture Sequence DTW
# =====================================================================

def _function_dtw_cost(tA: np.ndarray, tB: np.ndarray) -> float:
    """Normalised DTW cost between texture vector sequences."""
    n, m = tA.shape[0], tB.shape[0]
    if n == 0 or m == 0:
        return 1.0

    C = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            C[i, j] = float(np.linalg.norm(tA[i] - tB[j]))

    D = np.full((n + 1, m + 1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(D[n, m]) / float(n + m)


# =====================================================================
# 4. Volatility and Rhythm Features
# =====================================================================

def _compute_volatility(vectors: List[np.ndarray]) -> np.ndarray:
    """Per-feature mean absolute successive difference."""
    if len(vectors) < 2:
        return np.zeros(_FUNCTION_DIM, dtype=float)
    mat = np.array(vectors)
    return np.mean(np.abs(np.diff(mat, axis=0)), axis=0)


def _compute_diversity(vectors: List[np.ndarray]) -> np.ndarray:
    """Per-feature std across all windows (episodic breadth)."""
    if len(vectors) < 2:
        return np.zeros(_FUNCTION_DIM, dtype=float)
    return np.std(np.array(vectors), axis=0)


def _autocorrelation_vector(vectors: List[np.ndarray], lag: int = 1) -> np.ndarray:
    """Per-feature lag-1 autocorrelation (narrative momentum)."""
    if len(vectors) < lag + 3:
        return np.zeros(_FUNCTION_DIM, dtype=float)
    mat = np.array(vectors)
    n_features = mat.shape[1]
    ac = np.zeros(n_features, dtype=float)
    for f in range(n_features):
        x0, x1 = mat[:-lag, f], mat[lag:, f]
        s0, s1 = float(np.std(x0)), float(np.std(x1))
        if s0 > 0.0 and s1 > 0.0:
            ac[f] = float(np.corrcoef(x0, x1)[0, 1])
    return ac


# =====================================================================
# 5. Sentiment Arc (global, uses VADER)
# =====================================================================

_VADER_CACHE: Dict[str, Any] = {}


def _get_vader():
    global _VADER_CACHE
    if "vader" not in _VADER_CACHE:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            _VADER_CACHE["vader"] = SentimentIntensityAnalyzer()
        except Exception:
            _VADER_CACHE["vader"] = None
    return _VADER_CACHE["vader"]


def _window_sentiment(w: Window) -> float:
    vader = _get_vader()
    if vader is None or not w.raw or not w.raw.strip():
        return 0.0
    try:
        import nltk
        sents = nltk.sent_tokenize(w.raw)
    except Exception:
        sents = [s.strip() for s in w.raw.split(". ") if s.strip()]
    if not sents:
        return 0.0
    return float(np.mean([vader.polarity_scores(s)["compound"] for s in sents]))


# =====================================================================
# 6. Composite Helpers
# =====================================================================

# Discrimination-weighted composite (primary)
_COMPOSITE_WEIGHTS = {
    "volatility": 0.25,
    "diversity": 0.20,
    "arc": 0.15,
    "corr": 0.15,
    "dtw": 0.15,
    "sentiment": 0.05,
    "rhythm": 0.05,
}

# Equal-weight composite (robustness check)
_COMPOSITE_WEIGHTS_EQ = {
    "volatility": 1.0 / 7.0,
    "diversity": 1.0 / 7.0,
    "arc": 1.0 / 7.0,
    "corr": 1.0 / 7.0,
    "dtw": 1.0 / 7.0,
    "sentiment": 1.0 / 7.0,
    "rhythm": 1.0 / 7.0,
}


def _global_composite(
    volatility_sim: float,
    diversity_sim: float,
    global_arc_cosine: float,
    mean_arc_corr: float,
    dtw_sim: float,
    global_sentiment_arc: float,
    rhythm_sim: float,
    weights: dict,
) -> float:
    """Compute weighted global narrative composite."""
    return float(np.clip(
        weights["volatility"] * volatility_sim
        + weights["diversity"] * diversity_sim
        + weights["arc"] * ((global_arc_cosine + 1.0) / 2.0)
        + weights["corr"] * ((mean_arc_corr + 1.0) / 2.0)
        + weights["dtw"] * dtw_sim
        + weights["sentiment"] * ((global_sentiment_arc + 1.0) / 2.0)
        + weights["rhythm"] * rhythm_sim,
        0.0, 1.0,
    ))


# =====================================================================
# 7. Main Compute
# =====================================================================

def compute(
    wA: List[Window],
    wB: List[Window],
    aligned: List[Tuple[int, int]],
    ctx: Dict[str, Any],
) -> Dict[str, List[float]]:
    """Narrative-level macro-structural texture features."""
    n_bins = int(ctx.get("narrative_bins", 20))
    n_aligned = len(aligned)

    if n_aligned == 0:
        return {}

    # --- Step 1: Texture vectors for all windows ---
    func_A_raw = [_function_vector(w) for w in wA]
    func_B_raw = [_function_vector(w) for w in wB]

    # Z-score normalise across the combined corpus so features are
    # on comparable scales for cosine / DTW computations
    all_vecs = func_A_raw + func_B_raw
    mat_all = np.array(all_vecs)
    mu = np.mean(mat_all, axis=0)
    sd = np.std(mat_all, axis=0)
    sd = np.where(sd == 0, 1.0, sd)

    func_A = [(v - mu) / sd for v in func_A_raw]
    func_B = [(v - mu) / sd for v in func_B_raw]

    # --- Step 2: Binned trajectories ---
    traj_A = _bin_trajectory(func_A, n_bins)
    traj_B = _bin_trajectory(func_B, n_bins)

    # --- Step 3: Global arc comparisons ---
    global_arc_cosine = _cosine_flat(traj_A, traj_B)

    track_cosines = _per_track_cosines(traj_A, traj_B)
    track_corrs = _per_track_correlations(traj_A, traj_B)

    # Named arc features (indices match _function_vector)
    character_arc = float(track_cosines[0])
    action_arc = float(np.mean([track_cosines[1], track_cosines[12]]))
    complexity_arc = float(np.mean([track_cosines[8], track_cosines[9], track_cosines[10]]))
    tone_arc = float(np.mean([track_cosines[4], track_cosines[5], track_cosines[6]]))
    register_arc = float(np.mean([track_cosines[2], track_cosines[3], track_cosines[7]]))

    mean_arc_corr = float(np.mean(track_corrs))

    # --- Step 4: Global sentiment arc ---
    sent_A = np.array([_window_sentiment(w) for w in wA], dtype=float)
    sent_B = np.array([_window_sentiment(w) for w in wB], dtype=float)

    sent_traj_A = _bin_trajectory([np.array([s]) for s in sent_A], n_bins).flatten()
    sent_traj_B = _bin_trajectory([np.array([s]) for s in sent_B], n_bins).flatten()
    global_sentiment_arc = _cosine_vec(sent_traj_A, sent_traj_B)
    sentiment_arc_corr = _pearson(sent_traj_A, sent_traj_B)

    # --- Step 5: Volatility similarity ---
    vol_A = _compute_volatility(func_A)
    vol_B = _compute_volatility(func_B)
    volatility_sim = _cosine_vec(vol_A, vol_B)

    # --- Step 6: Diversity similarity ---
    div_A = _compute_diversity(func_A)
    div_B = _compute_diversity(func_B)
    diversity_sim = _cosine_vec(div_A, div_B)

    # --- Step 7: Rhythm autocorrelation similarity ---
    ac_A = _autocorrelation_vector(func_A)
    ac_B = _autocorrelation_vector(func_B)
    rhythm_sim = _cosine_vec(ac_A, ac_B)

    # --- Step 8: Texture sequence DTW ---
    dtw_cost = _function_dtw_cost(traj_A, traj_B)

    # --- Step 9: Per-aligned-pair features ---
    nA = float(max(1, len(wA) - 1))
    nB = float(max(1, len(wB) - 1))

    func_cosines: List[float] = []
    pos_dists: List[float] = []

    for i, j in aligned:
        func_cosines.append(_cosine_vec(func_A[i], func_B[j]))
        pos_a = float(i) / nA
        pos_b = float(j) / nB
        pos_dists.append(abs(pos_a - pos_b))

    # --- Step 10: Composite narrative retention ---
    #
    # v1.2: LOCAL-GLOBAL BLEND (with equal-weight variant)
    #
    # Global signals (broadcast) capture macro-architectural similarity:
    # does the text share arc shape, episodic rhythm, narrative pacing?
    # Local signals (per-window) capture positional texture correspondence:
    # does THIS passage at THIS position exhibit similar compositional style?
    #
    # Blend: 55% global (macro-architecture) + 30% local texture
    #        + 15% positional alignment

    dtw_sim = 1.0 / (1.0 + dtw_cost)

    # Primary (discrimination-weighted)
    global_comp = _global_composite(
        volatility_sim, diversity_sim, global_arc_cosine, mean_arc_corr,
        dtw_sim, global_sentiment_arc, rhythm_sim, _COMPOSITE_WEIGHTS,
    )

    # Equal-weight robustness variant
    global_comp_eq = _global_composite(
        volatility_sim, diversity_sim, global_arc_cosine, mean_arc_corr,
        dtw_sim, global_sentiment_arc, rhythm_sim, _COMPOSITE_WEIGHTS_EQ,
    )

    # Per-window composites
    narr_composite_list: List[float] = []
    narr_composite_eq_list: List[float] = []
    narr_local_list: List[float] = []

    for k_idx in range(n_aligned):
        # Local: texture vector cosine mapped to [0, 1]
        local_func = (func_cosines[k_idx] + 1.0) / 2.0   # [-1,1] → [0,1]
        local_func = max(0.0, min(1.0, local_func))

        # Positional: penalise distant positional matches
        pos_alignment = max(0.0, 1.0 - pos_dists[k_idx])

        # Primary blend
        blended = float(np.clip(
            0.55 * global_comp
            + 0.30 * local_func
            + 0.15 * pos_alignment,
            0.0, 1.0,
        ))
        narr_composite_list.append(blended)

        # Equal-weight blend
        blended_eq = float(np.clip(
            0.55 * global_comp_eq
            + 0.30 * local_func
            + 0.15 * pos_alignment,
            0.0, 1.0,
        ))
        narr_composite_eq_list.append(blended_eq)

        narr_local_list.append(local_func)

    # --- Build output ---
    return {
        # Per-aligned-pair (column names retain "func" for backward compat)
        "narr_func_cosine": func_cosines,
        "narr_positional_dist": pos_dists,
        "narr_retention_local": narr_local_list,

        # Global (broadcast)
        "narr_global_arc_cosine": [global_arc_cosine] * n_aligned,
        "narr_sentiment_arc": [global_sentiment_arc] * n_aligned,
        "narr_sentiment_arc_corr": [sentiment_arc_corr] * n_aligned,
        "narr_character_arc": [character_arc] * n_aligned,
        "narr_action_arc": [action_arc] * n_aligned,
        "narr_complexity_arc": [complexity_arc] * n_aligned,
        "narr_tone_arc": [tone_arc] * n_aligned,
        "narr_register_arc": [register_arc] * n_aligned,
        "narr_mean_arc_corr": [mean_arc_corr] * n_aligned,
        "narr_volatility_sim": [volatility_sim] * n_aligned,
        "narr_diversity_sim": [diversity_sim] * n_aligned,
        "narr_rhythm_autocorr_sim": [rhythm_sim] * n_aligned,
        "narr_func_dtw_cost": [dtw_cost] * n_aligned,

        # Composite: varies per-window (global-local blend)
        "narr_retention_composite": narr_composite_list,
        "narr_retention_composite_eq": narr_composite_eq_list,
        "narr_global_composite": [global_comp] * n_aligned,
        "narr_global_composite_eq": [global_comp_eq] * n_aligned,
    }