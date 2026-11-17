import gc
import os
import sys
from typing import Dict, List, Tuple

# Must be set before torch is imported (via features_runner)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Pipeline modules live in src/ and import each other by top-level name
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd
from scipy.stats import norm as normal_dist

from features_runner import run_pair_preprocessed
from preprocess import ensure_nltk, load_or_build_preprocessed


# =====================================================================
# Configuration
# =====================================================================

DATASET_DIR = "dataset"
OUT_DIR = "out_profiles"
RAW_DIR = os.path.join(OUT_DIR, "raw")

EPS = 1e-9

CONFIG = {
    "target_tokens": 2000,
    "step_tokens": 500,
    "lowercase": True,
    "remove_stopwords": False,
    "strategies": ["chapter", "paragraph"],
    "strip_gutenberg": True,
    "syntax_spacy_model": "en_core_web_sm",
}

FEATURE_FAMILIES = ["lexical", "syntax", "rhythm", "graphs", "semantic", "narrative", "transform"]

# key -> (file in DATASET_DIR, year of the English-language text used).
# Keys are "<author>_<work>"; the author prefix selects the tradition.
CORPUS: Dict[str, Tuple[str, int]] = {
    "james_portrait":        ("The_Portrait_of_a_Lady_Henry_James.txt", 1881),
    "james_daisy":           ("Daisy_Miller_Henry_James.txt", 1878),
    "wharton_mirth":         ("The_House_of_Mirth_Edith_Wharton.txt", 1905),
    "wharton_age":           ("The_Age_Edith_Wharton.txt", 1920),
    "eliot_middlemarch":     ("Middlemarch_George_Eliot.txt", 1871),
    "eliot_mill":            ("The_Mill_George_Eliot.txt", 1860),
    "austen_pride":          ("Pride_and_Prejudice_Jane_Austen.txt", 1813),
    "austen_emma":           ("Emma_Jane_Austen.txt", 1815),
    "hardy_tess":            ("Tess_of_the_dUrbervilles_Thomas_Hardy.txt", 1891),
    "wilde_dorian":          ("The_Picture_of_Dorian_Gray_Oscar_Wilde.txt", 1890),
    "dickens_coperfield":    ("David_Copperfield_Charles_Dickens.txt", 1850),
    "dickens_cities":        ("Two_Cities_Charles_Dickens.txt", 1859),
    "stoker_dracula":        ("Dracula_Bram_Stoker.txt", 1897),
    "flaubert_bovary":       ("Madame_Bovary_Gustave_Flaubert.txt", 1856),
    "flaubert_salammbo":     ("Salammbo_Gustave_Flaubert.txt", 1862),
    "joyce_dubliners":       ("Dubliners_James_Joyce.txt", 1914),
    "joyce_ulysses":         ("Ulysses_James_Joyce.txt", 1920),
    "homer_oddyssey_pope":   ("Odyssey_Homer_pope.txt", 1725),
    "homer_oddyssey_butler": ("Odyssey_Homer_butler.txt", 1900),
    "homer_iliad":           ("The_Iliad_Homer.txt", 1715),
    "virginia_voyage":       ("The_Voyage_Out_Virginia_Woolf.txt", 1915),
    "virginia_dalloway":     ("Dalloway_Virginia_Woolf.txt", 1925),
    "virginia_jacob":        ("Jacobs_Virginia_Woolf.txt", 1922),
    "shelley_frankenstein":  ("Frankenstein_Mary_Shelley.txt", 1818),
    "mansfield_prelude":     ("Prelude_Katherine_Mansfield.txt", 1918),
    "mansfield_bliss":       ("Bliss_Katherine_Mansfield.txt", 1918),
    "mansfield_bay":         ("Bay_Katherine_Mansfield.txt", 1922),
}

YEAR_MAP = {key: year for key, (_, year) in CORPUS.items()}

TRADITION_MAP = {
    "james":     "victorian_eng",
    "wharton":   "victorian_eng",
    "eliot":     "victorian_eng",
    "hardy":     "victorian_eng",
    "dickens":   "victorian_eng",
    "wilde":     "victorian_eng",
    "stoker":    "victorian_eng",
    "austen":    "romantic_eng",
    "shelley":   "romantic_eng",
    "virginia":  "modernist_eng",
    "joyce":     "modernist_eng",
    "mansfield": "modernist_eng",
    "homer":     "classical",
    "flaubert":  "continental",
}

# Historically documented source -> target relationships (reference pairs)
INFLUENCE_PAIRS = [
    ("james_portrait", "wharton_mirth"),
    ("austen_pride", "eliot_middlemarch"),
    ("eliot_middlemarch", "hardy_tess"),
    ("homer_oddyssey_pope", "joyce_ulysses"),
    ("homer_oddyssey_pope", "virginia_dalloway"),
    ("homer_oddyssey_butler", "joyce_ulysses"),
    ("homer_oddyssey_butler", "virginia_dalloway"),
    ("homer_oddyssey_pope", "homer_oddyssey_butler"),
    ("flaubert_bovary", "joyce_dubliners"),
    ("flaubert_bovary", "wharton_age"),
    ("mansfield_prelude", "virginia_jacob"),
    ("mansfield_prelude", "virginia_dalloway"),
    ("mansfield_bay", "virginia_dalloway"),
    ("mansfield_bliss", "virginia_dalloway"),
    ("mansfield_bliss", "virginia_jacob"),
    ("virginia_voyage", "mansfield_bliss"),
    ("virginia_voyage", "mansfield_prelude"),
]

# Source texts from which control pairs are built
ANCHORS = [
    "james_portrait",
    "austen_pride",
    "eliot_middlemarch",
    "homer_oddyssey_pope",
    "homer_oddyssey_butler",
    "flaubert_bovary",
    "virginia_dalloway",
    "mansfield_prelude",
    "mansfield_bliss",
    "mansfield_bay",
]

# Columns coerced to numeric (NaN -> 0) after loading the per-pair features
NUMERIC_COLS = [
    "lex_kl_surprise", "pos_trigram_cosine",
    "tbi", "tbi_concept", "tbi_struct",
    "sem_retention", "sem_divergence",
    "struct_retention", "struct_divergence",
    "narr_retention",
    "indep_sem_divergence", "indep_struct_divergence",
    "lex_gate", "lex_gate_narr",
    "sem_retention_composite",
    "sentiment_arc_cosine", "entity_type_jsd",
    "concreteness_sim", "supersense_cosine", "supersense_jsd",
    "narr_retention_composite", "narr_retention_composite_eq",
    "narr_global_arc_cosine",
    "narr_sentiment_arc", "narr_volatility_sim",
    "narr_diversity_sim", "narr_func_dtw_cost",
    "narr_func_cosine", "narr_mean_arc_corr",
    "narr_retention_local", "narr_global_composite",
    "narr_global_composite_eq",
    "creativity_concept", "creativity_struct",
    "creativity_narrative", "creativity_score",
    "creativity_concept_eq", "creativity_struct_eq",
    "creativity_narrative_eq", "creativity_score_eq",
    "creative_zone", "creativity_dominance",
]


# =====================================================================
# Text metadata helpers
# =====================================================================

def _get_tradition(text_key: str) -> str:
    author = text_key.split("_")[0]
    return TRADITION_MAP.get(author, "other")


def _get_pair_traditions(pair_key: str) -> Tuple[str, str]:
    a, b = pair_key.split("_vs_")
    return _get_tradition(a), _get_tradition(b)


def _get_year(text_key: str) -> int:
    if text_key not in YEAR_MAP:
        raise KeyError(f"Missing publication year for text key: {text_key}")
    return YEAR_MAP[text_key]


def _chronologically_possible(source_key: str, target_key: str) -> bool:
    """Source must be published no later than target."""
    return _get_year(source_key) <= _get_year(target_key)


def _same_author(a: str, b: str) -> bool:
    return a.split("_")[0] == b.split("_")[0]


# =====================================================================
# Control pair generation
# =====================================================================

def build_anchor_controls(
    keys: List[str],
    anchors: List[str],
    influence_pairs: List[Tuple[str, str]],
    chronological_only: bool = True,
) -> List[Tuple[str, str]]:
    """Directed (source, target) control pairs anchored on the given sources.

    Excludes same-author pairs, influence pairs in either direction and,
    by default, pairs where the source postdates the target.
    """
    excluded = set()
    for a, b in influence_pairs:
        excluded.add((a, b))
        excluded.add((b, a))

    pairs = set()
    for source in anchors:
        if source not in keys:
            continue
        for target in keys:
            if target == source or _same_author(source, target):
                continue
            if (source, target) in excluded:
                continue
            if chronological_only and not _chronologically_possible(source, target):
                continue
            pairs.add((source, target))

    return sorted(pairs, key=lambda p: (YEAR_MAP[p[0]], YEAR_MAP[p[1]], p[0], p[1]))


def safe_mean_std(s: pd.Series):
    """Mean and std of the numeric values in s; std is None if undefined or zero."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return None, None
    mu = float(s.mean())
    sd = float(s.std())
    if sd != sd or sd <= 0.0:
        sd = None
    return mu, sd


# =====================================================================
# DFW: five-channel discriminative feature weighting
# =====================================================================

DFW_CHANNELS = {
    "narrative": [
        "narr_mean_arc_corr", "sentiment_arc_cosine", "narr_retention_composite",
        "narr_volatility_sim", "narr_diversity_sim", "sentlen_ts_cosine",
    ],
    "structural": [
        "pos_trigram_cosine", "pos4_cosine", "wan_fw_cosine", "fw_unigram_cosine",
    ],
    "semantic": [
        "sbert_cosine", "topic_cosine", "lsa_cosine",
    ],
    "conceptual": [
        "supersense_cosine", "concreteness_sim", "entity_type_jsd",
    ],
    "lexical": [
        "lex_jaccard", "content_lex_jaccard", "content_tfidf_cosine",
        "idf_content_jaccard", "lex_unigram_jsd", "lex_kl_surprise", "punct_cosine",
    ],
}
DFW_CHANNEL_NAMES = list(DFW_CHANNELS)
ALL_DFW_FEATURES = [f for feats in DFW_CHANNELS.values() for f in feats]
_FEAT_TO_CHANNEL = {f: ch for ch, feats in DFW_CHANNELS.items() for f in feats}

# Features where higher = more different; their z-scores are negated
DIVERGENCE_FEATURES = {"entity_type_jsd", "lex_unigram_jsd", "lex_kl_surprise", "supersense_jsd"}

CHANNELLING_ALPHA = 0.20     # alpha in Eq. 7
DIVERGENCE_THRESHOLD = -0.5  # channel z below this counts as divergent
RETENTION_THRESHOLD = 0.5    # channel z above this counts as retained


def _dfw_weights(ctrl: pd.DataFrame, features: List[str], min_tradition_windows: int) -> Dict[str, float]:
    """Eq. 4: w_f = max(0.01, sigma_tradition(f) / (sigma_global(f) + eps)).

    sigma_tradition is the mean std over within-tradition control pairs,
    using only traditions with at least min_tradition_windows windows.
    """
    weights = {}
    for f in features:
        global_std = float(ctrl[f].std()) if len(ctrl) > 0 else 1.0

        trad_stds = []
        for trad in ctrl["trad_B"].unique():
            trad_ctrl = ctrl[(ctrl["trad_A"] == trad) & (ctrl["trad_B"] == trad)]
            if len(trad_ctrl) >= min_tradition_windows:
                trad_stds.append(float(trad_ctrl[f].std()))
        within_std = float(np.mean(trad_stds)) if trad_stds else global_std

        weights[f] = max(within_std / (global_std + EPS), 0.01)
    return weights


def compute_dfw_tbi(
    all_df: pd.DataFrame,
    strategy: str,
    min_tradition_windows: int = 100,
) -> pd.DataFrame:
    """Per-pair five-channel DFW profile and channelled score for one strategy."""
    df_strat = all_df[all_df["window_strategy"] == strategy].copy()
    df_strat["trad_A"] = df_strat["pair_key"].map(lambda pk: _get_pair_traditions(pk)[0])
    df_strat["trad_B"] = df_strat["pair_key"].map(lambda pk: _get_pair_traditions(pk)[1])
    ctrl = df_strat[df_strat["pair_type"] == "control"]

    available = [f for f in ALL_DFW_FEATURES if f in df_strat.columns]
    missing = [f for f in ALL_DFW_FEATURES if f not in df_strat.columns]
    if not available:
        print("  WARNING [DFW]: No pairwise features available. Skipping.")
        return pd.DataFrame()
    if missing:
        print(f"  WARNING [DFW]: Missing features (ignored): {', '.join(missing)}")

    weights = _dfw_weights(ctrl, available, min_tradition_windows)

    print(f"\n  DFW Weights (strategy: {strategy}):")
    for f in sorted(available, key=lambda x: -weights[x])[:10]:
        print(f"    w={weights[f]:.3f}  {f}")

    results = []
    for pk in sorted(df_strat["pair_key"].unique()):
        sub = df_strat[df_strat["pair_key"] == pk]
        trad_a = sub["trad_A"].iloc[0]
        trad_b = sub["trad_B"].iloc[0]

        # Tradition-matched control baseline, falling back to all controls
        trad_ctrl = ctrl[ctrl["trad_A"].isin([trad_a, trad_b]) & ctrl["trad_B"].isin([trad_a, trad_b])]
        if len(trad_ctrl) < min_tradition_windows:
            trad_ctrl = ctrl
            baseline_type = "global"
        else:
            baseline_type = f"{trad_a}+{trad_b}"

        feat_z = {}
        for f in available:
            mu = float(trad_ctrl[f].mean())
            sd = float(trad_ctrl[f].std())
            if not sd >= EPS:
                continue
            z = (float(sub[f].mean()) - mu) / sd
            feat_z[f] = -z if f in DIVERGENCE_FEATURES else z

        def weighted_mean(feats):
            w_sum = sum(weights[f] for f in feats)
            return sum(weights[f] * feat_z[f] for f in feats) / w_sum if w_sum > 0 else 0.0

        # Eq. 5: weighted mean z per channel
        channel_z = {
            ch: weighted_mean([f for f in feat_z if _FEAT_TO_CHANNEL[f] == ch])
            for ch in DFW_CHANNEL_NAMES
        }
        scores = list(channel_z.values())

        # Eqs. 6-7: peak channel and channelled score
        dfw_peak = max(scores)
        peak_channel = DFW_CHANNEL_NAMES[int(np.argmax(scores))]
        n_pos = sum(z > RETENTION_THRESHOLD for z in scores)
        n_neg = sum(z < DIVERGENCE_THRESHOLD for z in scores)
        dfw_channelled = max(0.0, dfw_peak) * (1.0 + CHANNELLING_ALPHA * n_neg)

        top_feat = max(feat_z, key=lambda f: weights[f] * feat_z[f]) if feat_z else "none"

        results.append({
            "pair_key": pk,
            "pair_type": sub["pair_type"].iloc[0],
            "window_strategy": strategy,
            "trad_A": trad_a,
            "trad_B": trad_b,
            "same_tradition": trad_a == trad_b,
            "baseline_type": baseline_type,
            # Weighted mean over all features, ignoring channels
            "dfw_tbi": weighted_mean(list(feat_z)),
            "dfw_z_narrative": channel_z["narrative"],
            "dfw_z_structural": channel_z["structural"],
            "dfw_z_semantic": channel_z["semantic"],
            "dfw_z_conceptual": channel_z["conceptual"],
            "dfw_z_lexical": channel_z["lexical"],
            "dfw_peak": dfw_peak,
            "dfw_peak_channel": peak_channel,
            "dfw_mean_pos": float(np.mean([max(0.0, z) for z in scores])),
            "dfw_channelled": dfw_channelled,
            # 0 = all active channels retained, 1 = all active channels divergent
            "dfw_channelling_idx": n_neg / max(n_pos + n_neg, 1),
            "dfw_breadth": n_pos,
            "dfw_asymmetry": channel_z["narrative"] - channel_z["structural"],
            "dfw_top_feature": top_feat,
            "dfw_top_feature_z": feat_z.get(top_feat, 0.0),
            "n_windows": len(sub),
        })

    res = pd.DataFrame(results)
    if not res.empty:
        res["dfw_rank"] = res["dfw_channelled"].rank(ascending=False, method="min").astype(int)
    return res


# =====================================================================
# Calibration helpers
# =====================================================================

_DOMINANCE_CODES = {"concept": 1.0, "struct": -1.0, "narrative": 2.0}


def _top_two_gap(channels: pd.DataFrame) -> pd.Series:
    """Per row: largest minus second-largest channel value."""
    vals = np.sort(channels.to_numpy(dtype=float), axis=1)
    return pd.Series(vals[:, -1] - vals[:, -2], index=channels.index)


def _dominance(channels: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Dominant channel code (concept=1, struct=-1, narrative=2) where mask, else 0."""
    codes = channels.idxmax(axis=1).map(_DOMINANCE_CODES)
    return codes.where(mask, 0.0).astype(float)


def _zscore_against_controls(all_df, col, idx, mask_control):
    """z-score of col over idx against control rows, or None if undefined."""
    mu, sd = safe_mean_std(all_df.loc[mask_control, col])
    if mu is None or sd is None:
        return None
    return (all_df.loc[idx, col] - mu) / (sd + EPS)


def _creative_peak_cal(x: pd.Series, baseline: float = 0.5) -> pd.Series:
    """Peaked activation for retention in CDF space (0.5 = control level).

    Zero at or below baseline, peaks at CDF ~0.75, returns to zero at 1.0:
    moderate retention indicates creative transformation.
    """
    excess = (x.astype(float) - baseline).clip(lower=0.0)
    mapped = (excess * 2.0).clip(0.0, 1.0)
    return np.sin(np.pi * mapped).clip(0.0, 1.0)


def _soft_gate_cal(x: pd.Series, floor: float = 0.20) -> pd.Series:
    """Monotonic gate for divergence; high divergence is not penalised.

    ~0 below floor (near copy), rises smoothly, saturates at 2 x floor.
    """
    excess = (x.astype(float) - floor).clip(lower=0.0)
    mapped = (excess / max(floor, 0.01)).clip(0.0, 1.0)
    return np.sin(np.pi / 2.0 * mapped).clip(0.0, 1.0)


# =====================================================================
# Pipeline stages
# =====================================================================

def extract_pair_features(pairs: List[Tuple[str, str, str]], paths: Dict[str, str]) -> List[str]:
    """Run feature extraction for each (source, target, type), checkpointed per pair."""
    os.makedirs(RAW_DIR, exist_ok=True)
    raw_paths = []

    for a_key, b_key, p_type in pairs:
        pair_label = f"{a_key}_vs_{b_key}"
        raw_path = os.path.join(RAW_DIR, f"{pair_label}.csv")
        raw_paths.append(raw_path)

        if os.path.exists(raw_path):
            print(f"Skipping: {pair_label} (checkpoint found).")
            continue

        print(f"Processing: {pair_label}...")
        pre_a = load_or_build_preprocessed(a_key, paths[a_key], CONFIG)
        pre_b = load_or_build_preprocessed(b_key, paths[b_key], CONFIG)

        df = run_pair_preprocessed(
            pre_a, pre_b, a_key, b_key, pair_label,
            strategies=CONFIG["strategies"],
            families=FEATURE_FAMILIES,
            semantic_enabled=True, semantic_cfg={},
            syntax_use_spacy=True,
            syntax_spacy_model=CONFIG["syntax_spacy_model"],
        )
        df["pair_type"] = p_type
        df["pair_key"] = pair_label
        df["trad_A"] = _get_tradition(a_key)
        df["trad_B"] = _get_tradition(b_key)

        # Write to a temp file first so an interrupted write is not taken as a checkpoint
        tmp_path = raw_path + ".tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, raw_path)
        print(f"  Saved raw features to {raw_path}")

        del pre_a, pre_b, df
        gc.collect()

    return raw_paths


def add_legacy_zscores(all_df: pd.DataFrame) -> None:
    """Control-calibrated z-scores of single legacy signals."""
    for strat in all_df["window_strategy"].dropna().unique():
        mask_strat = all_df["window_strategy"] == strat
        mask_control = mask_strat & (all_df["pair_type"] == "control")

        z = _zscore_against_controls(all_df, "lex_kl_surprise", mask_strat, mask_control)
        if z is not None:
            all_df.loc[mask_strat, "z_surprise"] = z
            all_df.loc[mask_strat, "z_imitation"] = -z

        for col, z_col in [
            ("pos_trigram_cosine", "z_struct"),
            ("tbi", "z_tbi"),
            ("tbi_concept", "z_tbi_concept"),
            ("tbi_struct", "z_tbi_struct"),
        ]:
            if col not in all_df.columns:
                continue
            z = _zscore_against_controls(all_df, col, mask_strat, mask_control)
            if z is not None:
                all_df.loc[mask_strat, z_col] = z


def add_calibrated_tbi(all_df: pd.DataFrame) -> None:
    """Three-channel gated TBI on CDF-normalised signals (0.5 = control mean).

    Also writes the norm_* columns used by add_creative_middle.
    """
    norm_cols = [
        "sem_retention", "struct_retention", "narr_retention",
        "indep_sem_divergence", "indep_struct_divergence",
    ]

    for strat in all_df["window_strategy"].dropna().unique():
        mask_strat = all_df["window_strategy"] == strat
        mask_control = mask_strat & (all_df["pair_type"] == "control")
        idx = all_df.index[mask_strat]

        for col in norm_cols:
            z = _zscore_against_controls(all_df, col, idx, mask_control) if col in all_df.columns else None
            norm = pd.Series(normal_dist.cdf(z), index=idx) if z is not None else pd.Series(0.5, index=idx)
            all_df.loc[idx, f"norm_{col}"] = norm.fillna(0.5)

        excess = lambda c: ((all_df.loc[idx, f"norm_{c}"] - 0.5) * 2.0).clip(lower=0.0)
        struct_div_gate = (all_df.loc[idx, "norm_indep_struct_divergence"] / 0.5).clip(0.0, 1.0)
        sem_div_gate = (all_df.loc[idx, "norm_indep_sem_divergence"] / 0.5).clip(0.0, 1.0)
        narr_div_gate = np.sqrt(sem_div_gate * struct_div_gate).clip(0.0, 1.0)

        tbi_concept = (excess("sem_retention") * struct_div_gate).clip(0.0, 1.0)
        tbi_struct = (excess("struct_retention") * sem_div_gate).clip(0.0, 1.0)
        tbi_narrative = (excess("narr_retention") * narr_div_gate).clip(0.0, 1.0)
        tbi_cal = (np.sqrt(tbi_concept**2 + tbi_struct**2 + tbi_narrative**2) / np.sqrt(3.0)).clip(0.0, 1.0)

        all_df.loc[idx, "tbi_concept_cal"] = tbi_concept
        all_df.loc[idx, "tbi_struct_cal"] = tbi_struct
        all_df.loc[idx, "tbi_narrative_cal"] = tbi_narrative
        all_df.loc[idx, "tbi_cal"] = tbi_cal

        z_tbi = _zscore_against_controls(all_df, "tbi_cal", idx, mask_control)
        if z_tbi is None:
            z_tbi = pd.Series(0.0, index=idx)
        all_df.loc[idx, "z_tbi_cal"] = z_tbi

        channels = pd.DataFrame(
            {"concept": tbi_concept, "struct": tbi_struct, "narrative": tbi_narrative}, index=idx
        )
        clearly_dominant = _top_two_gap(channels) / (tbi_cal + 1e-12) > 0.10
        all_df.loc[idx, "tbi_dominance_cal"] = _dominance(channels, (z_tbi >= 1.0) & clearly_dominant)


def add_creative_middle(all_df: pd.DataFrame) -> None:
    """Calibrated creative-middle model: peaked retention x gated divergence x lexical gate."""
    for strat in all_df["window_strategy"].dropna().unique():
        mask_strat = all_df["window_strategy"] == strat
        mask_control = mask_strat & (all_df["pair_type"] == "control")
        idx = all_df.index[mask_strat]

        def col_or(name, default):
            if name in all_df.columns:
                return all_df.loc[idx, name].fillna(default)
            return pd.Series(default, index=idx)

        # Retention: peaked (creative middle)
        peaked_sem_ret = _creative_peak_cal(col_or("norm_sem_retention", 0.5))
        peaked_str_ret = _creative_peak_cal(col_or("norm_struct_retention", 0.5))
        peaked_narr_ret = _creative_peak_cal(col_or("norm_narr_retention", 0.5))

        # Divergence: average of gates on calibrated and raw values
        sem_div_gate = ((
            _soft_gate_cal(col_or("norm_indep_sem_divergence", 0.5), floor=0.20)
            + _soft_gate_cal(col_or("indep_sem_divergence", 0.5), floor=0.15)
        ) / 2.0).clip(0.0, 1.0)
        str_div_gate = ((
            _soft_gate_cal(col_or("norm_indep_struct_divergence", 0.5), floor=0.20)
            + _soft_gate_cal(col_or("indep_struct_divergence", 0.4), floor=0.15)
        ) / 2.0).clip(0.0, 1.0)

        lex_gate = col_or("lex_gate", 0.5)
        if "lex_gate_narr" in all_df.columns:
            lex_gate_narr = all_df.loc[idx, "lex_gate_narr"].fillna(0.5)
        else:
            lex_gate_narr = np.sin(np.pi / 2.0 * (lex_gate / 0.30).clip(0.0, 1.0)).clip(0.0, 1.0)

        surface_gate = np.sqrt(str_div_gate * sem_div_gate + 1e-12).clip(0.0, 1.0)
        cm_concept = (peaked_sem_ret * str_div_gate * lex_gate).clip(0.0, 1.0)
        cm_struct = (peaked_str_ret * sem_div_gate * lex_gate).clip(0.0, 1.0)
        cm_narrative = (peaked_narr_ret * surface_gate * lex_gate_narr).clip(0.0, 1.0)
        cm_score = (np.sqrt(cm_concept**2 + cm_struct**2 + cm_narrative**2) / np.sqrt(3.0)).clip(0.0, 1.0)

        all_df.loc[idx, "creative_middle_concept"] = cm_concept
        all_df.loc[idx, "creative_middle_struct"] = cm_struct
        all_df.loc[idx, "creative_middle_narrative"] = cm_narrative
        all_df.loc[idx, "creative_middle_score"] = cm_score

        all_df.loc[idx, "peaked_sem_ret"] = peaked_sem_ret
        all_df.loc[idx, "peaked_str_ret"] = peaked_str_ret
        all_df.loc[idx, "peaked_narr_ret"] = peaked_narr_ret
        all_df.loc[idx, "gated_sem_div"] = sem_div_gate
        all_df.loc[idx, "gated_str_div"] = str_div_gate
        all_df.loc[idx, "lex_gate_narr_cal"] = lex_gate_narr

        for col, z_col in [
            ("creative_middle_score", "z_creative_middle"),
            ("creative_middle_concept", "z_cm_concept"),
            ("creative_middle_struct", "z_cm_struct"),
            ("creative_middle_narrative", "z_cm_narrative"),
        ]:
            z = _zscore_against_controls(all_df, col, idx, mask_control)
            all_df.loc[idx, z_col] = z if z is not None else 0.0

        channels = pd.DataFrame({
            "concept": all_df.loc[idx, "z_cm_concept"],
            "struct": all_df.loc[idx, "z_cm_struct"],
            "narrative": all_df.loc[idx, "z_cm_narrative"],
        }, index=idx)
        margin = _top_two_gap(channels) / (all_df.loc[idx, "z_creative_middle"].abs() + 1e-12)
        all_df.loc[idx, "cm_dominance"] = _dominance(channels, margin > 0.20)


DFW_MERGE_COLS = [
    "dfw_tbi",
    "dfw_z_narrative", "dfw_z_structural", "dfw_z_semantic", "dfw_z_conceptual", "dfw_z_lexical",
    "dfw_peak", "dfw_peak_channel", "dfw_mean_pos",
    "dfw_channelled", "dfw_channelling_idx", "dfw_breadth",
    "dfw_asymmetry", "dfw_top_feature", "dfw_rank",
]


def print_dfw_ranking(dfw_all: pd.DataFrame) -> None:
    for strat in dfw_all["window_strategy"].unique():
        ranked = dfw_all[dfw_all["window_strategy"] == strat].sort_values("dfw_channelled", ascending=False)
        print(f"\n  DFW Ranking ({strat}):")
        print(f"    {'pair':48s} {'trad':14s} {'type':10s} "
              f"{'DFW':>7s} {'CHANN':>7s} {'narr_z':>7s} {'str_z':>7s} "
              f"{'sem_z':>7s} {'con_z':>7s} {'lex_z':>7s} {'peak_ch':>11s}")
        print("  " + "-" * 140)
        for _, r in ranked.iterrows():
            marker = "*" if r["pair_type"] == "influence" else " "
            trad = f"[{r['trad_A'][:4]}->{r['trad_B'][:4]}]"
            print(
                f"  {marker} {r['pair_key']:48s} {trad:14s} {r['pair_type']:10s} "
                f"{r['dfw_tbi']:+7.3f} {r['dfw_channelled']:+7.3f} "
                f"{r['dfw_z_narrative']:+7.2f} {r['dfw_z_structural']:+7.2f} "
                f"{r['dfw_z_semantic']:+7.2f} {r['dfw_z_conceptual']:+7.2f} "
                f"{r['dfw_z_lexical']:+7.2f} {r['dfw_peak_channel']:>11s}"
            )


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    ensure_nltk()

    paths = {key: os.path.join(DATASET_DIR, fname) for key, (fname, _) in CORPUS.items()}
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing files: {', '.join(missing)}")

    influence_pairs = [(a, b) for a, b in INFLUENCE_PAIRS if a in paths and b in paths and a != b]
    control_pairs = build_anchor_controls(list(paths), ANCHORS, influence_pairs, chronological_only=True)

    print(f"\n{len(influence_pairs)} influence pairs, "
          f"{len(control_pairs)} chronologically plausible control pairs.")
    print("Sample controls:")
    for a, b in control_pairs[:10]:
        print(f"  {a} ({YEAR_MAP[a]}) -> {b} ({YEAR_MAP[b]})")

    all_pairs = (
        [(a, b, "influence") for a, b in influence_pairs]
        + [(a, b, "control") for a, b in control_pairs]
    )
    raw_paths = extract_pair_features(all_pairs, paths)

    print("\nLoading all processed pairs for global normalisation...")
    all_df = pd.concat([pd.read_csv(p) for p in raw_paths], ignore_index=True)
    for col in NUMERIC_COLS:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce").fillna(0.0)

    print("\nCalculating legacy z-scores against control baselines...")
    add_legacy_zscores(all_df)

    print("Calibrating TBI (three-channel gated model)...")
    add_calibrated_tbi(all_df)

    print("Calibrating creative-middle model...")
    add_creative_middle(all_df)

    print("\n" + "=" * 70)
    print("Computing DFW (five-channel discriminative feature weighting)...")
    print("=" * 70)

    dfw_results = [compute_dfw_tbi(all_df, strategy=s) for s in all_df["window_strategy"].dropna().unique()]
    dfw_results = [d for d in dfw_results if not d.empty]

    os.makedirs(OUT_DIR, exist_ok=True)
    if dfw_results:
        dfw_all = pd.concat(dfw_results, ignore_index=True)
        all_df = all_df.merge(
            dfw_all[["pair_key", "window_strategy"] + DFW_MERGE_COLS],
            on=["pair_key", "window_strategy"], how="left",
        )
        print_dfw_ranking(dfw_all)

        dfw_all.to_csv(os.path.join(OUT_DIR, "DFW_TBI_SUMMARY.csv"), index=False)
        print(f"\n  DFW summary saved to {OUT_DIR}/DFW_TBI_SUMMARY.csv")

    print("\nSaving per-pair CSVs...")
    for pair_label, pair_df in all_df.groupby("pair_key", sort=False):
        pair_df.to_csv(os.path.join(OUT_DIR, f"{pair_label}.csv"), index=False)
    all_df.to_csv(os.path.join(OUT_DIR, "ALL_PAIRS_ALL_STRATEGIES.csv"), index=False)

    print(f"Done. Outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
