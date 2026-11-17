from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _clip01(s: pd.Series) -> pd.Series:
    """Clamp to [0, 1] after cleaning infinities and NaNs."""
    return s.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def _median_exponential_normalise(raw: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Map non-negative values into [0, 1) via 1 - exp(-x / median).

    This is scale-free: it adapts to the pair's own distribution rather than
    requiring an external reference. The median is a robust centre, and the
    exponential saturates gracefully for outliers.
    """
    clean = raw.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    tau = float(np.nanmedian(clean.values)) if len(clean) > 0 else 1.0
    return _clip01(1.0 - np.exp(-clean / max(tau, eps)))


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------

def _creative_peak(x: pd.Series, baseline: float = 0.0) -> pd.Series:
    """Peaked activation function that rewards the MIDDLE zone of RETENTION.

    Applied to retention signals only (v5.0). The creative middle hypothesis:
        Creativity requires MODERATE retention of the source — enough to
        constitute genuine influence, not so much as to constitute reproduction.

    Mathematical model:
        excess(x) = clip(x - baseline, 0, ∞)  → zero below baseline
        mapped    = clip(2 × excess, 0, 1)     → saturates at 1 when excess = 0.5
        peak      = sin(π × mapped)            → peaks at mapped = 0.5 (the middle)

    The peak occurs when x is at baseline + 0.25 (e.g., at the 75th
    percentile if baseline = 0.5 in CDF-normalised space).
    """
    excess = (x.astype(float) - baseline).clip(lower=0.0)
    mapped = (excess * 2.0).clip(0.0, 1.0)
    return _clip01(np.sin(np.pi * mapped))


def _creative_peak_scalar(x: float, baseline: float = 0.0) -> float:
    """Scalar version of _creative_peak for single values."""
    excess = max(0.0, x - baseline)
    mapped = min(1.0, excess * 2.0)
    return float(np.sin(np.pi * mapped))


def _soft_gate(x: pd.Series, floor: float = 0.15) -> pd.Series:
    """Soft monotonic gate for DIVERGENCE signals (v5.0).

    Unlike _creative_peak, this does NOT penalise high values.
    The Tarde-Baldwin thesis says creative works can be VERY different
    from their source while still carrying traces at another level.

    Below `floor`: score ≈ 0 (pure copy, no creative transformation).
    Above `floor`: smoothly rises via sin(π/2 × normalised).
    Above 2×`floor`: score ≈ 1 (enough divergence — saturates).

    This ensures SOME novelty exists without penalising texts that
    diverge strongly from their source on a given representational level.
    """
    excess = (x.astype(float) - floor).clip(lower=0.0)
    mapped = (excess / max(floor, 0.01)).clip(0.0, 1.0)
    return _clip01(np.sin(np.pi / 2.0 * mapped))


def _soft_gate_scalar(x: float, floor: float = 0.15) -> float:
    """Scalar version of _soft_gate."""
    excess = max(0.0, x - floor)
    mapped = min(1.0, excess / max(floor, 0.01))
    return float(np.sin(np.pi / 2.0 * mapped))


# ---------------------------------------------------------------------------
# Building blocks: extract retention and divergence for each level
# ---------------------------------------------------------------------------

def _semantic_retention(df: pd.DataFrame) -> pd.Series:
    """How much of the source's meaning persists in the derived text.

    v4+: Pure per-window semantic retention only. Narrative arc is a
    SEPARATE retention axis (see _narrative_retention). This separation
    is critical: for Homer→Joyce, per-window semantics are at control
    baseline but narrative architecture is elevated. Blending them
    dilutes the narrative signal.
    """
    if "sem_retention_composite" in df.columns:
        return _clip01(df["sem_retention_composite"])
    if "sbert_retention" in df.columns:
        return _clip01(df["sbert_retention"])
    if "sbert_cosine" in df.columns:
        return _clip01((df["sbert_cosine"].astype(float) + 1.0) / 2.0)
    for fallback in ["topic_cosine", "lsa_cosine"]:
        if fallback in df.columns:
            return _clip01(df[fallback])
    return pd.Series(0.0, index=df.index, dtype=float)


def _semantic_divergence(df: pd.DataFrame) -> pd.Series:
    """How much new meaning the derived text introduces.

    LEGACY (v2): Uses sbert_displacement (trigonometric complement of SBERT
    cosine). This is mechanically coupled with sbert_retention (r ≈ -0.98).
    Retained for backward compatibility with raw TBI cross-products.

    For calibrated TBI, use indep_sem_divergence instead (see below).
    """
    if "sbert_displacement" in df.columns:
        return _clip01(df["sbert_displacement"])
    return _clip01(1.0 - _semantic_retention(df))


def _independent_semantic_divergence(df: pd.DataFrame) -> pd.Series:
    """How much new meaning the derived text introduces — measured from
    features that are NOT derived from the SBERT cosine angle.

    NEW in v3.0. This is the key fix for the anti-correlation artifact.

    Uses:
      - entity_type_jsd:       narrative fabric divergence (what KINDS of
                                entities populate the text: persons, places, times)
      - supersense_jsd:        experiential domain divergence (WordNet lexicographer
                                categories: verb.motion, noun.person, etc.)
      - concreteness_sim:      inverted → divergence in aesthetic register
                                (concrete sensory vs abstract philosophical)
      - sentiment_arc_cosine:  inverted → divergence in emotional trajectory shape

    These are genuinely independent of the SBERT embedding angle because they
    operate on categorical distributions (NER types, WordNet synsets, POS
    categories) rather than dense vector cosine. Empirically, the correlation
    between this measure and sem_retention_composite is moderate (|r| ≈ 0.3–0.5),
    not the mechanical r = -0.98 of the SBERT retention/displacement pair.

    The moderate negative correlation is EXPECTED and DESIRABLE: texts that
    retain more meaning should show less semantic divergence. The key is that
    the relationship is empirical, not trigonometric — it can vary across pairs,
    which is exactly what makes the cross-channel product informative.

    v5.0: Provides both weighted and equal-weight variants. The weighted
    version is primary; equal-weight is for robustness checks.
    """
    components: List[pd.Series] = []
    weights: List[float] = []

    # Entity type distribution divergence (JSD over NER category counts)
    if "entity_type_jsd" in df.columns:
        components.append(_median_exponential_normalise(
            df["entity_type_jsd"].astype(float).clip(lower=0.0)
        ))
        weights.append(0.30)

    # WordNet supersense distribution divergence (JSD over lexicographer files)
    if "supersense_jsd" in df.columns:
        components.append(_median_exponential_normalise(
            df["supersense_jsd"].astype(float).clip(lower=0.0)
        ))
        weights.append(0.30)

    # Concreteness profile divergence (1 - cosine similarity)
    # v5.0: de-weighted from 0.20 to 0.10 — saturates within tradition
    if "concreteness_sim" in df.columns:
        components.append(_clip01(1.0 - df["concreteness_sim"].astype(float)))
        weights.append(0.10)

    # Sentiment arc divergence (1 - mapped cosine)
    # v5.0: up from 0.20 to 0.25 — has high within-tradition variance
    if "sentiment_arc_cosine" in df.columns:
        arc_sim_01 = (df["sentiment_arc_cosine"].astype(float) + 1.0) / 2.0  # [-1,1] → [0,1]
        components.append(_clip01(1.0 - arc_sim_01))
        weights.append(0.25)

    if not components:
        print("  WARNING [transform]: No independent semantic divergence features available. "
              "Defaulting to 0.5 (neutral).")
        return pd.Series(0.5, index=df.index, dtype=float)

    total_w = sum(weights)
    result = sum(c * (w / total_w) for c, w in zip(components, weights))
    return _clip01(result)


def _independent_semantic_divergence_eq(df: pd.DataFrame) -> pd.Series:
    """Equal-weight variant for robustness checks (v5.0)."""
    components: List[pd.Series] = []

    if "entity_type_jsd" in df.columns:
        components.append(_median_exponential_normalise(
            df["entity_type_jsd"].astype(float).clip(lower=0.0)
        ))
    if "supersense_jsd" in df.columns:
        components.append(_median_exponential_normalise(
            df["supersense_jsd"].astype(float).clip(lower=0.0)
        ))
    if "concreteness_sim" in df.columns:
        components.append(_clip01(1.0 - df["concreteness_sim"].astype(float)))
    if "sentiment_arc_cosine" in df.columns:
        arc_sim_01 = (df["sentiment_arc_cosine"].astype(float) + 1.0) / 2.0
        components.append(_clip01(1.0 - arc_sim_01))

    if not components:
        return pd.Series(0.5, index=df.index, dtype=float)

    return _clip01(sum(components) / len(components))


def _structural_retention(df: pd.DataFrame) -> pd.Series:
    """How much of the source's formal organisation persists.

    Combines POS template similarity with dependency depth similarity
    and function word network topology. Each captures a different
    grain of structural inheritance.
    """
    components: List[pd.Series] = []
    weights: List[float] = []

    # POS n-gram template (surface syntax)
    for col, w in [("pos_trigram_cosine", 0.30), ("pos4_cosine", 0.15), ("pos5_cosine", 0.10)]:
        if col in df.columns:
            components.append(_clip01(df[col]))
            weights.append(w)

    # Function word network topology (deeper structural logic)
    if "wan_degree_jsd" in df.columns:
        components.append(_clip01(1.0 - df["wan_degree_jsd"]))
        weights.append(0.25)

    # Dependency depth profile similarity (clause embedding habits)
    dep_a = "dep_depth_norm_A" in df.columns
    dep_b = "dep_depth_norm_B" in df.columns
    if dep_a and dep_b:
        diff = (df["dep_depth_norm_A"] - df["dep_depth_norm_B"]).abs()
        components.append(_clip01(1.0 - _median_exponential_normalise(diff)))
        weights.append(0.20)

    if not components:
        return pd.Series(0.0, index=df.index, dtype=float)

    total_w = sum(weights)
    result = sum(c * (w / total_w) for c, w in zip(components, weights))
    return _clip01(result)


def _structural_retention_eq(df: pd.DataFrame) -> pd.Series:
    """Equal-weight variant for robustness checks (v5.0)."""
    components: List[pd.Series] = []

    for col in ["pos_trigram_cosine", "pos4_cosine", "pos5_cosine"]:
        if col in df.columns:
            components.append(_clip01(df[col]))
    if "wan_degree_jsd" in df.columns:
        components.append(_clip01(1.0 - df["wan_degree_jsd"]))
    if "dep_depth_norm_A" in df.columns and "dep_depth_norm_B" in df.columns:
        diff = (df["dep_depth_norm_A"] - df["dep_depth_norm_B"]).abs()
        components.append(_clip01(1.0 - _median_exponential_normalise(diff)))

    if not components:
        return pd.Series(0.0, index=df.index, dtype=float)

    return _clip01(sum(components) / len(components))


def _structural_divergence(df: pd.DataFrame) -> pd.Series:
    """How much the derived text departs from the source's formal habits.

    Uses POS divergence, function word profile divergence, and rhythm
    differences as complementary signals.

    NOTE: These are already independent of SBERT cosine (they come from
    syntax, graphs, and rhythm modules). This function therefore serves
    double duty as both the raw structural divergence AND the independent
    structural divergence for the calibrated model.
    """
    components: List[pd.Series] = []
    weights: List[float] = []

    # POS template divergence (invert cosine similarity)
    if "pos_trigram_cosine" in df.columns:
        components.append(_clip01(1.0 - df["pos_trigram_cosine"]))
        weights.append(0.30)

    # Function word network divergence
    if "wan_degree_jsd" in df.columns:
        components.append(_clip01(df["wan_degree_jsd"]))
        weights.append(0.25)

    # Function word usage rate difference
    if "fw_rate_A" in df.columns and "fw_rate_B" in df.columns:
        fw_diff = (df["fw_rate_A"] - df["fw_rate_B"]).abs()
        components.append(_median_exponential_normalise(fw_diff))
        weights.append(0.15)

    # Compression ratio difference (rhythmic/textural divergence)
    if "compress_ratio_A" in df.columns and "compress_ratio_B" in df.columns:
        cr_diff = (df["compress_ratio_A"] - df["compress_ratio_B"]).abs()
        components.append(_median_exponential_normalise(cr_diff))
        weights.append(0.15)

    # Dependency depth divergence
    if "dep_depth_norm_A" in df.columns and "dep_depth_norm_B" in df.columns:
        dep_diff = (df["dep_depth_norm_A"] - df["dep_depth_norm_B"]).abs()
        components.append(_median_exponential_normalise(dep_diff))
        weights.append(0.15)

    if not components:
        return pd.Series(0.0, index=df.index, dtype=float)

    total_w = sum(weights)
    result = sum(c * (w / total_w) for c, w in zip(components, weights))
    return _clip01(result)


# ---------------------------------------------------------------------------
# Narrative Retention (macro-structural)
# ---------------------------------------------------------------------------

def _narrative_retention(df: pd.DataFrame) -> pd.Series:
    """How much of the source's narrative ARCHITECTURE persists.

    This is a genuinely distinct signal from per-window semantic retention.
    Semantic retention measures "do aligned windows share meaning?"
    Narrative retention measures "do the texts share arc shape, episodic
    rhythm, and positional texture patterns?"

    NOTE (v5.0): The underlying features measure narrative TEXTURE
    (POS-level style characteristics at proportional positions), not
    abstract narrative FUNCTION (Proppian roles, hero's journey stages).
    The distinction matters: Homer→Joyce shares abstract narrative
    function (documented by Gilbert) but not narrative texture (Victorian
    translation vs modernist prose). This module captures the latter.
    See narrative.py for full discussion.

    For Homer→Joyce, per-window semantics are at control baseline
    (completely different vocabulary, imagery, cultural context). But
    for same-tradition pairs like James→Wharton, the narrative texture
    — the sequence of POS-level characteristics across proportional
    positions — preserves inherited pacing and compositional habits.
    """
    if "narr_retention_composite" in df.columns:
        return _clip01(df["narr_retention_composite"])
    return pd.Series(0.5, index=df.index, dtype=float)  # agnostic default


# ---------------------------------------------------------------------------
# Lexical Surprise Gate (v5.0: channel-specific)
# ---------------------------------------------------------------------------

def _lexical_gate(df: pd.DataFrame, eps: float = 1e-12) -> pd.Series:
    """Full lexical gate for semantic and structural channels.

    Pure copying (low KL surprise) scores zero. Pure noise (extreme KL)
    also decays via the sine envelope. The gate peaks for moderate
    lexical transformation — the derived text is neither a copy nor
    unrelated at the word level.
    """
    if "lex_kl_surprise" in df.columns:
        kl = df["lex_kl_surprise"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        D = _median_exponential_normalise(kl)
        return _clip01(np.sin(np.pi * D))

    if "lex_unigram_jsd" in df.columns:
        D = _clip01(df["lex_unigram_jsd"])
        return _clip01(4.0 * D * (1.0 - D))

    return pd.Series(0.5, index=df.index, dtype=float)


def _lexical_gate_narrative(df: pd.DataFrame, eps: float = 1e-12) -> pd.Series:
    """Attenuated lexical gate for the narrative channel (v5.0).

    For cross-tradition pairs (e.g., Homer→Joyce), lexical overlap is
    essentially random — the full peaked gate would suppress genuine
    narrative signals. The attenuated gate still requires the text not
    to be a verbatim copy (KL surprise > 0), but it doesn't penalise
    high lexical distance.

    Model: soft monotonic gate that rises from 0 and saturates at 1.
    Floor is very low (0.05) — only exact copies are suppressed.
    """
    if "lex_kl_surprise" in df.columns:
        kl = df["lex_kl_surprise"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        D = _median_exponential_normalise(kl)
        # Soft gate: rises from 0, saturates by D ≈ 0.2
        return _clip01(np.sin(np.pi / 2.0 * (D / 0.20).clip(0.0, 1.0)))

    if "lex_unigram_jsd" in df.columns:
        D = _clip01(df["lex_unigram_jsd"])
        # Soft gate: any non-trivial JSD passes
        return _clip01(np.sin(np.pi / 2.0 * (D / 0.10).clip(0.0, 1.0)))

    return pd.Series(0.5, index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def compute(
    wA: List[Any],
    wB: List[Any],
    aligned: List[Tuple[int, int]],
    ctx: Dict[str, Any],
) -> Dict[str, List[float]]:
    """
    Channelled Imitation Model (v5.0).

    Computes per-window influence transform scores that decompose the
    relationship between source (A) and derived (B) texts into:

    1. RETENTION: How much of the source persists at each representational
       level (semantic, structural, narrative).
    2. DIVERGENCE: How much the derived text departs at each level.
    3. CROSS-CHANNEL PRODUCTS: Where retention on one level co-occurs with
       divergence on another — the signature of creative transformation.

    Raw channels (backward compatible, known anti-correlation artifact):
        tbi_concept = semantic_retention × structural_divergence × lex_gate
        tbi_struct  = structural_retention × semantic_divergence × lex_gate
        tbi         = max(tbi_concept, tbi_struct)

    Channelled Imitation channels (v5.0 — asymmetric activation):
        creativity_concept   = peak(sem_ret) × gate(struct_div) × lex_gate
        creativity_struct    = peak(struct_ret) × gate(sem_div) × lex_gate
        creativity_narrative = peak(narr_ret) × gate(surface_div) × lex_gate_narr
        creativity_score     = L2_norm(concept, struct, narrative) / sqrt(3)

    The peaked function (retention) rewards the MIDDLE zone: enough influence
    to constitute genuine inheritance, not so much as to be reproduction.
    The soft gate (divergence) is MONOTONIC: it requires some divergence
    but does NOT penalise high divergence, because the Tarde-Baldwin thesis
    says creativity can be very different from its source while still
    carrying traces at another level.

    Independent divergence signals (for calibrated model in main.py):
        indep_sem_divergence   : entity type + supersense + concreteness + sentiment arc
        indep_struct_divergence: POS + WAN + dep depth + compression + fw rate
    """
    rows = ctx.get("rows_so_far", [])
    if not rows:
        return {
            "tbi": [], "tbi_concept": [], "tbi_struct": [],
            "tbi_dominance": [],
            "sem_retention": [], "sem_divergence": [],
            "struct_retention": [], "struct_divergence": [],
            "narr_retention": [],
            "indep_sem_divergence": [], "indep_struct_divergence": [],
            "lex_gate": [], "lex_gate_narr": [],
            "creativity_concept": [], "creativity_struct": [],
            "creativity_narrative": [], "creativity_score": [],
            "creative_zone": [], "creativity_dominance": [],
            # Equal-weight robustness variants
            "creativity_concept_eq": [], "creativity_struct_eq": [],
            "creativity_narrative_eq": [], "creativity_score_eq": [],
        }

    df = pd.DataFrame(rows)

    # --- Extract the four directional signals ---
    sem_ret    = _semantic_retention(df)
    sem_div    = _semantic_divergence(df)       # Legacy (SBERT-based, coupled)
    struct_ret = _structural_retention(df)
    struct_div = _structural_divergence(df)

    # --- Narrative architectural retention (third axis) ---
    narr_ret = _narrative_retention(df)

    # --- Independent divergence signals (v3.0+) ---
    indep_sem_div   = _independent_semantic_divergence(df)
    indep_struct_div = struct_div.copy()  # Already independent of SBERT

    # --- Equal-weight variants for robustness ---
    indep_sem_div_eq = _independent_semantic_divergence_eq(df)
    struct_ret_eq    = _structural_retention_eq(df)

    # --- Lexical gates (v5.0: channel-specific) ---
    G     = _lexical_gate(df)          # Full gate for semantic/structural
    G_nar = _lexical_gate_narrative(df)  # Attenuated gate for narrative

    # =================================================================
    # RAW TBI (v3.x backward compatible)
    # =================================================================

    # --- Channel 1: Conceptual Imitation → Structural Invention ---
    # "I inherited your ideas but built new forms"
    tbi_concept = _clip01(sem_ret * struct_div * G)

    # --- Channel 2: Structural Imitation → Conceptual Invention ---
    # "I inherited your forms but filled them with new meaning"
    tbi_struct = _clip01(struct_ret * sem_div * G)

    # --- Envelope and dominance ---
    tbi = _clip01(pd.concat([tbi_concept, tbi_struct], axis=1).max(axis=1))

    dominance_threshold = 0.10
    diff = tbi_concept - tbi_struct
    denom = tbi + 1e-12
    relative_diff = diff / denom

    dominance = pd.Series(0.0, index=df.index, dtype=float)
    dominance[relative_diff > dominance_threshold] = 1.0
    dominance[relative_diff < -dominance_threshold] = -1.0

    # =================================================================
    # v5.0: CHANNELLED IMITATION MODEL
    # =================================================================
    # ASYMMETRIC activation across axes:
    #
    # RETENTION: PEAKED — the creative middle requires moderate retention.
    #   Baseline = pair median (pair-internal diagnostic).
    #
    # DIVERGENCE: SOFT GATE — monotonic, requires SOME divergence but
    #   does NOT penalise high divergence. This is the key v5.0 fix.
    #   A text that retains source semantics at the 95th percentile while
    #   completely reinventing structure now scores HIGH (as it should
    #   under the channelled-imitation thesis).
    #
    # LEXICAL GATE: Channel-specific (v5.0). Full gate for channels 1-2
    #   (lexical surface matters for semantic/structural comparison).
    #   Attenuated gate for channel 3 (narrative architecture can be
    #   inherited even when lexical overlap is random).

    # Peaked retention signals (median of the pair is the natural baseline)
    sem_ret_peaked    = _creative_peak(sem_ret,    baseline=float(sem_ret.median()))
    struct_ret_peaked = _creative_peak(struct_ret,  baseline=float(struct_ret.median()))
    narr_ret_peaked   = _creative_peak(narr_ret,    baseline=float(narr_ret.median()))

    # Gated divergence signals (v5.0: monotonic, not peaked)
    struct_div_gated     = _soft_gate(struct_div,   floor=float(struct_div.median()))
    indep_sem_div_gated  = _soft_gate(indep_sem_div, floor=float(indep_sem_div.median()))

    # Channel 1: Conceptual creativity — source meaning retained, form reinvented
    # High when: meaning moderately persists AND form shows genuine novelty
    creativity_concept = _clip01(sem_ret_peaked * struct_div_gated * G)

    # Channel 2: Structural creativity — source form retained, meaning renewed
    # High when: structure moderately persists AND meaning shows genuine novelty
    creativity_struct = _clip01(struct_ret_peaked * indep_sem_div_gated * G)

    # Channel 3: Narrative creativity — arc architecture retained, surface reinvented
    # Uses geometric mean of structural + semantic divergence as surface gate
    # v5.0: uses attenuated lexical gate (G_nar) for cross-tradition robustness
    surface_div_gated = _clip01(
        np.sqrt(struct_div_gated * indep_sem_div_gated + 1e-12)
    )
    creativity_narrative = _clip01(narr_ret_peaked * surface_div_gated * G_nar)

    # Envelope: L2 norm across channels, normalised by sqrt(3)
    creativity_score = _clip01(np.sqrt(
        creativity_concept**2 + creativity_struct**2 + creativity_narrative**2
    ) / np.sqrt(3.0))

    # Creative zone flag (v5.0: fixed operator precedence)
    # At least one channel shows creative transformation
    creative_zone = (
        (creativity_concept > 0.05)
        | (creativity_struct > 0.05)
        | (creativity_narrative > 0.05)
    ).astype(float)

    # Directional asymmetry: which channel dominates the creative profile?
    # +1 = concept-dominant, -1 = structure-dominant, +2 = narrative-dominant
    creativity_channels = pd.DataFrame({
        "concept": creativity_concept,
        "struct": creativity_struct,
        "narrative": creativity_narrative,
    }, index=df.index)
    cr_max_chan = creativity_channels.idxmax(axis=1)
    cr_max_val = creativity_channels.max(axis=1)
    cr_second  = creativity_channels.apply(
        lambda r: sorted(r, reverse=True)[1] if len(r) > 1 else 0.0, axis=1
    )
    cr_margin = (cr_max_val - cr_second) / (creativity_score + 1e-12)
    cr_dominant = cr_margin > 0.15

    creativity_dominance = pd.Series(0.0, index=df.index, dtype=float)
    creativity_dominance[cr_dominant & (cr_max_chan == "concept")]   =  1.0
    creativity_dominance[cr_dominant & (cr_max_chan == "struct")]    = -1.0
    creativity_dominance[cr_dominant & (cr_max_chan == "narrative")] =  2.0

    # =================================================================
    # v5.0: EQUAL-WEIGHT ROBUSTNESS VARIANTS
    # =================================================================
    # Same model structure but with equal-weight composites for
    # semantic divergence and structural retention. For the paper,
    # report both and show results are robust to weight choice.

    struct_ret_eq_peaked    = _creative_peak(struct_ret_eq, baseline=float(struct_ret_eq.median()))
    indep_sem_div_eq_gated  = _soft_gate(indep_sem_div_eq, floor=float(indep_sem_div_eq.median()))

    creativity_concept_eq   = _clip01(sem_ret_peaked * struct_div_gated * G)
    creativity_struct_eq    = _clip01(struct_ret_eq_peaked * indep_sem_div_eq_gated * G)

    surface_div_eq_gated = _clip01(
        np.sqrt(struct_div_gated * indep_sem_div_eq_gated + 1e-12)
    )
    creativity_narrative_eq = _clip01(narr_ret_peaked * surface_div_eq_gated * G_nar)

    creativity_score_eq = _clip01(np.sqrt(
        creativity_concept_eq**2 + creativity_struct_eq**2 + creativity_narrative_eq**2
    ) / np.sqrt(3.0))

    return {
        # v3.x backward compatible (raw gated channels)
        "tbi":              tbi.tolist(),
        "tbi_concept":      tbi_concept.tolist(),
        "tbi_struct":       tbi_struct.tolist(),
        "tbi_dominance":    dominance.tolist(),
        "sem_retention":    sem_ret.tolist(),
        "sem_divergence":   sem_div.tolist(),
        "struct_retention":  struct_ret.tolist(),
        "struct_divergence": struct_div.tolist(),
        "narr_retention":   narr_ret.tolist(),
        "indep_sem_divergence":   indep_sem_div.tolist(),
        "indep_struct_divergence": indep_struct_div.tolist(),
        "lex_gate":         G.tolist(),
        "lex_gate_narr":    G_nar.tolist(),
        # v5.0: Channelled Imitation Model (primary)
        "creativity_concept":    creativity_concept.tolist(),
        "creativity_struct":     creativity_struct.tolist(),
        "creativity_narrative":  creativity_narrative.tolist(),
        "creativity_score":      creativity_score.tolist(),
        "creative_zone":         creative_zone.tolist(),
        "creativity_dominance":  creativity_dominance.tolist(),
        # v5.0: Equal-weight robustness variants
        "creativity_concept_eq":   creativity_concept_eq.tolist(),
        "creativity_struct_eq":    creativity_struct_eq.tolist(),
        "creativity_narrative_eq": creativity_narrative_eq.tolist(),
        "creativity_score_eq":     creativity_score_eq.tolist(),
    }