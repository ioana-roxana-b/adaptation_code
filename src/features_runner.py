from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import Window
from align import align_windows, _window_text_for_embedding

try:
    from features import lexical, syntax, rhythm, graphs, semantic, narrative, transform
except Exception:
    import lexical, syntax, rhythm, graphs, semantic, narrative, transform


def _tfidf_pair(wA: List[Window], wB: List[Window], stop_words: str | None) -> Tuple[Any, Any]:
    docsA = [d if d.strip() else "__EMPTY__" for d in [" ".join(w.tokens) for w in wA]]
    docsB = [d if d.strip() else "__EMPTY__" for d in [" ".join(w.tokens) for w in wB]]
    vect = TfidfVectorizer(norm="l2", stop_words=stop_words, min_df=2, max_df=0.9)
    X = vect.fit_transform(docsA + docsB)
    return X[: len(docsA)], X[len(docsA) :]


def _tfidf_pair_content(wA: List[Window], wB: List[Window]) -> Tuple[Any, Any]:
    docsA = [d if d.strip() else "__EMPTY__" for d in [" ".join(w.content_tokens or []) for w in wA]]
    docsB = [d if d.strip() else "__EMPTY__" for d in [" ".join(w.content_tokens or []) for w in wB]]
    vect = TfidfVectorizer(norm="l2", min_df=2, max_df=0.9)
    X = vect.fit_transform(docsA + docsB)
    return X[: len(docsA)], X[len(docsA) :]


def _cosine_sparse_row(a_row, b_row) -> float:
    return float(a_row.multiply(b_row).sum())


def _encode_sbert_cached(wA: List[Window], wB: List[Window]) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Encode windows with SBERT once, to be shared between align and semantic.

    FIX #7: Returns the embedding matrices so they can be passed through ctx.
    Uses raw text (which SBERT was trained on) with fallback to joined tokens.
    """
    try:
        from semantic import get_sbert_model
    except ImportError:
        return None, None

    model = get_sbert_model("all-MiniLM-L6-v2")
    if model is None:
        return None, None

    docsA = [_window_text_for_embedding(w) for w in wA]
    docsB = [_window_text_for_embedding(w) for w in wB]

    embsA = model.encode(docsA, batch_size=8, show_progress_bar=False)
    embsB = model.encode(docsB, batch_size=8, show_progress_bar=False)
    return embsA, embsB


def compute_strategy_profiles(
    wA: List[Window],
    wB: List[Window],
    pair_label: str,
    labelA: str,
    labelB: str,
    strategy: str,
    families: List[str],
    semantic_enabled: bool,
    semantic_cfg: Dict[str, Any],
    syntax_use_spacy: bool,
    syntax_spacy_model: str,
) -> pd.DataFrame:
    # FIX #7: Encode SBERT once, pass to both align and semantic
    sbert_A, sbert_B = _encode_sbert_cached(wA, wB)

    # Alignment (reuses cached embeddings if available)
    aligned = align_windows(wA, wB, embeddings_A=sbert_A, embeddings_B=sbert_B)

    n_aligned = len(aligned)
    if n_aligned == 0:
        return pd.DataFrame()

    XA, XB = _tfidf_pair(wA, wB, stop_words="english")
    XcA, XcB = _tfidf_pair_content(wA, wB)

    base_rows: List[Dict[str, Any]] = []
    for k, (i, j) in enumerate(aligned):
        base_rows.append({
            "pair": pair_label, "A": labelA, "B": labelB,
            "window_strategy": strategy, "k": k,
            "A_unit_start": wA[i].unit_start, "A_unit_end": wA[i].unit_end,
            "B_unit_start": wB[j].unit_start, "B_unit_end": wB[j].unit_end,
            "A_n_tokens": len(wA[i].tokens), "B_n_tokens": len(wB[j].tokens),
            "lex_tfidf_cosine": _cosine_sparse_row(XA[i], XB[j]),
            "content_tfidf_cosine": _cosine_sparse_row(XcA[i], XcB[j]),
        })

    ctx: Dict[str, Any] = {
        "syntax_use_spacy": bool(syntax_use_spacy),
        "syntax_spacy_model": str(syntax_spacy_model),
    }
    if semantic_enabled:
        ctx.update(semantic_cfg)

    # FIX #7: Pass cached SBERT embeddings to semantic module via ctx
    if sbert_A is not None and sbert_B is not None:
        ctx["sbert_embeddings_A"] = sbert_A
        ctx["sbert_embeddings_B"] = sbert_B

    # Run feature families
    merged: Dict[str, List[float]] = {}
    family_modules = {
        "lexical": lexical,
        "syntax": syntax,
        "rhythm": rhythm,
        "graphs": graphs,
        "narrative": narrative,
    }
    if semantic_enabled:
        family_modules["semantic"] = semantic

    for fam_name, fam_module in family_modules.items():
        if fam_name not in families:
            continue
        result = fam_module.compute(wA, wB, aligned, ctx)

        # FIX #3: Validate output lengths before merging
        for col, vals in result.items():
            if len(vals) != n_aligned:
                print(
                    f"WARNING: {fam_name}.compute() returned {len(vals)} values "
                    f"for '{col}' but expected {n_aligned}. Skipping this column."
                )
                continue
            merged[col] = vals

    for col, vals in merged.items():
        for idx, v in enumerate(vals):
            base_rows[idx][col] = v

    if "transform" in families:
        ctx2 = dict(ctx)
        ctx2["rows_so_far"] = base_rows
        derived = transform.compute(wA, wB, aligned, ctx2)

        # FIX #3: Same length guard for transform
        for col, vals in derived.items():
            if len(vals) != n_aligned:
                print(
                    f"WARNING: transform.compute() returned {len(vals)} values "
                    f"for '{col}' but expected {n_aligned}. Skipping."
                )
                continue
            for idx, v in enumerate(vals):
                base_rows[idx][col] = v

    return pd.DataFrame(base_rows)


def run_pair_preprocessed(
    preA: Dict[str, List[Window]],
    preB: Dict[str, List[Window]],
    labelA: str,
    labelB: str,
    pair_label: str,
    strategies: List[str],
    families: List[str] | None = None,
    semantic_enabled: bool = False,
    semantic_cfg: Dict[str, Any] | None = None,
    syntax_use_spacy: bool = False,
    syntax_spacy_model: str = "en_core_web_sm",
) -> pd.DataFrame:
    if families is None:
        families = ["lexical", "syntax", "rhythm", "graphs", "narrative", "transform"]
    if semantic_cfg is None:
        semantic_cfg = {}

    dfs: List[pd.DataFrame] = []
    for strat in strategies:
        df = compute_strategy_profiles(
            preA[strat], preB[strat], pair_label, labelA, labelB, strat,
            families, semantic_enabled, semantic_cfg, syntax_use_spacy, syntax_spacy_model,
        )
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()