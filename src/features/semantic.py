from __future__ import annotations

import math
import os
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from preprocess import Window
from similarity import js_divergence_from_counters, cosine_counter

# =====================================================================
# Global caches
# =====================================================================

_SBERT_CACHE = {}
_VADER_CACHE = {}

def get_sbert_model(model_name="all-MiniLM-L6-v2"):
    global _SBERT_CACHE
    if model_name not in _SBERT_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(device)
            # Explicitly trust remote code for the nomic model
            _SBERT_CACHE[model_name] = SentenceTransformer(model_name, trust_remote_code=True, device=device)
            print(f"  [SBERT] Loaded {model_name} on {device}")
        except Exception as e:
            print(f"WARNING: SBERT failed to load: {e}")
            _SBERT_CACHE[model_name] = None
    return _SBERT_CACHE[model_name]


def _get_vader():
    """Lazy-load VADER sentiment analyzer."""
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
        except Exception as e:
            print(f"WARNING: VADER unavailable ({e}). Sentiment arc disabled.")
            _VADER_CACHE["vader"] = None
    return _VADER_CACHE["vader"]


def _get_spacy():
    """Lazy-load spaCy for NER."""
    try:
        from preprocess import get_spacy_model
        return get_spacy_model("en_core_web_sm")
    except Exception:
        return None


# =====================================================================
# 1. Sentiment Arc
# =====================================================================

# The per-window representations below depend only on the window, and a text
# recurs in many pairs, so they are cached for the life of the process
@lru_cache(maxsize=None)
def _sentiment_arc(raw: str, n_bins: int = 10) -> np.ndarray:
    """Compute binned sentiment trajectory over the raw text."""
    vader = _get_vader()
    if vader is None or not raw or not raw.strip():
        return np.zeros(n_bins, dtype=float)

    try:
        import nltk
        sents = nltk.sent_tokenize(raw)
    except Exception:
        sents = [s.strip() for s in raw.split(". ") if s.strip()]

    if not sents:
        return np.zeros(n_bins, dtype=float)

    scores = np.array(
        [vader.polarity_scores(s)["compound"] for s in sents], dtype=float
    )

    if len(scores) == 0:
        return np.zeros(n_bins, dtype=float)
    if len(scores) == 1:
        out = np.zeros(n_bins, dtype=float)
        out[0] = scores[0]
        return out

    positions = np.linspace(0.0, 1.0, num=len(scores))
    edges = np.linspace(0.0, 1.0, num=n_bins + 1)
    out = np.zeros(n_bins, dtype=float)
    cnt = np.zeros(n_bins, dtype=float)

    for p, v in zip(positions, scores):
        idx = int(np.searchsorted(edges, p, side="right") - 1)
        idx = max(0, min(n_bins - 1, idx))
        out[idx] += v
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


# =====================================================================
# 2. Entity Type Distribution
# =====================================================================

_NER_GROUPS = {
    "PERSON": "person",
    "NORP": "group",
    "FAC": "place",
    "ORG": "group",
    "GPE": "place",
    "LOC": "place",
    "PRODUCT": "thing",
    "EVENT": "event",
    "WORK_OF_ART": "thing",
    "LAW": "thing",
    "LANGUAGE": "thing",
    "DATE": "time",
    "TIME": "time",
    "PERCENT": "quantity",
    "MONEY": "quantity",
    "QUANTITY": "quantity",
    "ORDINAL": "quantity",
    "CARDINAL": "quantity",
}


@lru_cache(maxsize=None)
def _entity_type_distribution(raw: str, nlp) -> Counter:
    """Extract coarse entity type distribution from raw text."""
    if nlp is None or not raw or not raw.strip():
        return Counter()

    text = raw[:nlp.max_length] if len(raw) > nlp.max_length else raw
    doc = nlp(text)

    counts = Counter()
    for ent in doc.ents:
        group = _NER_GROUPS.get(ent.label_, "other")
        counts[group] += 1

    return counts


# =====================================================================
# 3. Concreteness Profile
# =====================================================================

_CONCRETE_POS = {
    "NN", "NNS", "NNP", "NNPS",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
}
_ABSTRACT_POS = {
    "MD",
    "RB", "RBR", "RBS",
    "IN",
    "CC",
    "WDT", "WP", "WRB",
}


def _concreteness_vector(pos_list: List[str]) -> np.ndarray:
    """Build a 3-dim concreteness profile: [concrete_rate, abstract_rate, other_rate]."""
    if not pos_list:
        return np.array([0.33, 0.33, 0.33], dtype=float)

    n = len(pos_list)
    n_concrete = sum(1 for p in pos_list if p in _CONCRETE_POS)
    n_abstract = sum(1 for p in pos_list if p in _ABSTRACT_POS)
    n_other = n - n_concrete - n_abstract

    return np.array([n_concrete / n, n_abstract / n, n_other / n], dtype=float)


# =====================================================================
# 4. WordNet Supersense Distribution
# =====================================================================

_POS_TO_WN = {}
for _tag in ("NN", "NNS", "NNP", "NNPS"):
    _POS_TO_WN[_tag] = "n"
for _tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
    _POS_TO_WN[_tag] = "v"
for _tag in ("JJ", "JJR", "JJS"):
    _POS_TO_WN[_tag] = "a"
for _tag in ("RB", "RBR", "RBS"):
    _POS_TO_WN[_tag] = "r"

_WN_CACHE: Dict[str, Optional[Any]] = {}


def _get_wordnet():
    """Lazy-load WordNet."""
    global _WN_CACHE
    if "wn" not in _WN_CACHE:
        try:
            import nltk
            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True)
            try:
                nltk.data.find("corpora/omw-1.4")
            except LookupError:
                nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet
            _WN_CACHE["wn"] = wordnet
        except Exception as e:
            print(f"WARNING: WordNet unavailable ({e}). Supersense disabled.")
            _WN_CACHE["wn"] = None
    return _WN_CACHE["wn"]


def _supersense_distribution(
    tokens: List[str], pos_tags: List[str]
) -> Counter:
    """Map each content token to its WordNet supersense (lexicographer file)."""
    return _supersense_cached(tuple(tokens), tuple(pos_tags))


@lru_cache(maxsize=None)
def _supersense_cached(tokens: Tuple[str, ...], pos_tags: Tuple[str, ...]) -> Counter:
    wn = _get_wordnet()
    if wn is None:
        return Counter()

    counts = Counter()
    for tok, pos in zip(tokens, pos_tags):
        wn_pos = _POS_TO_WN.get(pos)
        if wn_pos is None:
            continue

        synsets = wn.synsets(tok, pos=wn_pos)
        if not synsets:
            continue

        lexname = synsets[0].lexname()
        counts[lexname] += 1

    return counts


# =====================================================================
# SBERT (retained as complementary signal)
# =====================================================================

def _cosine_rows(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _orthogonal_displacement(
    source_vec: np.ndarray, derived_vec: np.ndarray
) -> Tuple[float, float]:
    na = float(np.linalg.norm(source_vec))
    nb = float(np.linalg.norm(derived_vec))
    if na == 0.0 or nb == 0.0:
        return 0.0, 0.0
    u = source_vec / na
    v = derived_vec / nb
    cos = float(np.clip(np.dot(u, v), -1.0, 1.0))
    retention_01 = (cos + 1.0) / 2.0
    displacement_01 = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return float(retention_01), float(displacement_01)


def _sbert_vectors_for_pair(
    wA: List[Window],
    wB: List[Window],
    model_name: str,
    ctx: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    cached_A = ctx.get("sbert_embeddings_A")
    cached_B = ctx.get("sbert_embeddings_B")
    if cached_A is not None and cached_B is not None:
        return cached_A, cached_B

    model = get_sbert_model(model_name)
    if model is None:
        return None, None

    docsA = [" ".join(w.content_tokens or []) for w in wA]
    docsB = [" ".join(w.content_tokens or []) for w in wB]
    docsA = [d if d.strip() else "__EMPTY__" for d in docsA]
    docsB = [d if d.strip() else "__EMPTY__" for d in docsB]

    return model.encode(docsA, batch_size=8), model.encode(docsB, batch_size=8)


# =====================================================================
# Topic models (retained)
# =====================================================================

def _topic_vectors_for_pair(wA, wB, n_topics, max_features, random_state):
    docsA = [" ".join(w.content_tokens or []) for w in wA]
    docsB = [" ".join(w.content_tokens or []) for w in wB]
    docsA = [d if d.strip() else "__EMPTY__" for d in docsA]
    docsB = [d if d.strip() else "__EMPTY__" for d in docsB]

    vect = CountVectorizer(
        lowercase=False, token_pattern=r"(?u)\b\w+\b",
        max_features=max_features, min_df=2, max_df=0.9,
    )
    try:
        X = vect.fit_transform(docsA + docsB)
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=random_state,
            learning_method="batch", max_iter=20,
        )
        Z = lda.fit_transform(X)
    except ValueError:
        Z = np.zeros((len(docsA) + len(docsB), n_topics))

    return Z[: len(docsA)], Z[len(docsA) :]


def _lsa_vectors_for_pair(wA, wB, n_components, random_state):
    docsA = [" ".join(w.content_tokens or []) for w in wA]
    docsB = [" ".join(w.content_tokens or []) for w in wB]
    docsA = [d if d.strip() else "__EMPTY__" for d in docsA]
    docsB = [d if d.strip() else "__EMPTY__" for d in docsB]

    vect = TfidfVectorizer(
        lowercase=False, tokenizer=str.split, preprocessor=None,
        token_pattern=None, norm="l2", min_df=2, max_df=0.9,
    )
    try:
        X = vect.fit_transform(docsA + docsB)
        max_k = max(2, min(X.shape) - 1)
        k = int(min(n_components, max_k))
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        Z = svd.fit_transform(X)
    except ValueError:
        Z = np.zeros((len(docsA) + len(docsB), n_components))

    return Z[: len(docsA)], Z[len(docsA) :]


# =====================================================================
# Main compute
# =====================================================================

# Module-level availability tracker (logged once per run)
_AVAILABILITY_LOGGED = False


def compute(
    wA: List[Window],
    wB: List[Window],
    aligned: List[Tuple[int, int]],
    ctx: Dict[str, Any],
) -> Dict[str, List[float]]:
    """
    Outputs:
      sentiment_arc_cosine   : emotional trajectory similarity
      entity_type_jsd        : narrative fabric divergence (lower = more similar)
      concreteness_sim       : register similarity
      supersense_cosine      : WordNet domain cosine similarity
      supersense_jsd         : WordNet domain divergence
      sbert_cosine           : dense contextual overlap
      sbert_retention        : mapped SBERT retention [0, 1]
      sbert_displacement     : orthogonal component [0, 1]
      topic_cosine           : LDA topic mixture cosine
      lsa_cosine             : LSA cosine
      sem_retention_composite: weighted blend for calibrated TBI
    """
    global _AVAILABILITY_LOGGED

    n_topics = int(ctx.get("semantic_n_topics", 20))
    max_features = int(ctx.get("semantic_max_features", 5000))
    random_state = int(ctx.get("semantic_random_state", 7))
    lsa_components = int(ctx.get("semantic_lsa_components", 100))
    arc_bins = int(ctx.get("semantic_arc_bins", 10))

    # --- Compute representations ---
    SA, SB = _sbert_vectors_for_pair(wA, wB, "all-MiniLM-L6-v2", ctx)
    ZA, ZB = _topic_vectors_for_pair(
        wA, wB, n_topics, max_features, random_state
    )
    LA, LB = _lsa_vectors_for_pair(
        wA, wB, n_components=lsa_components, random_state=random_state
    )

    # NER model (may be None)
    nlp = _get_spacy()

    # --- Log feature availability (once per run) ---
    if not _AVAILABILITY_LOGGED:
        avail = {
            "SBERT": SA is not None,
            "VADER (sentiment arc)": _get_vader() is not None,
            "spaCy NER (entity types)": nlp is not None,
            "WordNet (supersense)": _get_wordnet() is not None,
            "POS tags (concreteness)": len(wA) > 0 and wA[0].pos is not None,
        }
        print("  [semantic] Feature availability:")
        for feat, ok in avail.items():
            status = "OK" if ok else "UNAVAILABLE — composite will use fallback"
            print(f"    {feat}: {status}")

        # Composite weight breakdown (v5.0 weights)
        active_weights = {}
        if _get_wordnet() is not None:
            active_weights["supersense"] = 0.30
        if _get_vader() is not None:
            active_weights["sentiment_arc"] = 0.25
        if nlp is not None:
            active_weights["entity_types"] = 0.20
        if SA is not None:
            active_weights["sbert"] = 0.20
        if len(wA) > 0 and wA[0].pos is not None:
            active_weights["concreteness"] = 0.05

        if active_weights:
            total = sum(active_weights.values())
            print(f"    Composite blend: {active_weights} (sum={total:.2f})")
            if total < 1.0:
                print(f"    NOTE: {1.0 - total:.2f} weight missing — will be redistributed among active components")
        else:
            print("    WARNING: No composite components available!")

        _AVAILABILITY_LOGGED = True

    out: Dict[str, List[float]] = {
        "sentiment_arc_cosine": [],
        "entity_type_jsd": [],
        "concreteness_sim": [],
        "supersense_cosine": [],
        "supersense_jsd": [],
        "sbert_cosine": [],
        "sbert_retention": [],
        "sbert_displacement": [],
        "topic_cosine": [],
        "lsa_cosine": [],
        "sem_retention_composite": [],
    }

    for i, j in aligned:
        # --- 1. Sentiment arc ---
        arc_a = _sentiment_arc(wA[i].raw or "", n_bins=arc_bins)
        arc_b = _sentiment_arc(wB[j].raw or "", n_bins=arc_bins)
        arc_cos = _cosine_vec(arc_a, arc_b)
        out["sentiment_arc_cosine"].append(arc_cos)

        # --- 2. Entity type distribution ---
        ent_a = _entity_type_distribution(wA[i].raw or "", nlp)
        ent_b = _entity_type_distribution(wB[j].raw or "", nlp)
        ent_jsd = (
            js_divergence_from_counters(ent_a, ent_b)
            if (ent_a and ent_b) else 0.0
        )
        out["entity_type_jsd"].append(ent_jsd)

        # --- 3. Concreteness profile ---
        cv_a = _concreteness_vector(wA[i].pos or [])
        cv_b = _concreteness_vector(wB[j].pos or [])
        conc_sim = _cosine_vec(cv_a, cv_b)
        out["concreteness_sim"].append(conc_sim)

        # --- 4. WordNet supersense distribution ---
        ss_a = _supersense_distribution(wA[i].tokens or [], wA[i].pos or [])
        ss_b = _supersense_distribution(wB[j].tokens or [], wB[j].pos or [])
        if ss_a and ss_b:
            ss_cos = cosine_counter(ss_a, ss_b)
            ss_jsd = js_divergence_from_counters(ss_a, ss_b)
        else:
            ss_cos = 0.0
            ss_jsd = 0.0
        out["supersense_cosine"].append(ss_cos)
        out["supersense_jsd"].append(ss_jsd)

        # --- 5. SBERT (retained as one signal) ---
        if SA is not None and SB is not None:
            sbert_cos = _cosine_rows(SA[i], SB[j])
            ret, disp = _orthogonal_displacement(SA[i], SB[j])
        else:
            sbert_cos, ret, disp = 0.0, 0.0, 0.0
        out["sbert_cosine"].append(sbert_cos)
        out["sbert_retention"].append(ret)
        out["sbert_displacement"].append(disp)

        # --- 6. Topic & LSA ---
        out["topic_cosine"].append(_cosine_rows(ZA[i], ZB[j]))
        out["lsa_cosine"].append(_cosine_rows(LA[i], LB[j]))

        # --- 7. Composite semantic retention ---
        # Adaptive weighted blend: only includes available components,
        # renormalizing weights when some features are unavailable.
        components: List[Tuple[float, float]] = []  # (value, weight)

        # Supersense (primary signal)
        if ss_a and ss_b:
            ss_sim = max(0.0, ss_cos)
            components.append((ss_sim, 0.30))

        # Sentiment arc
        if _get_vader() is not None:
            arc_01 = (arc_cos + 1.0) / 2.0
            components.append((arc_01, 0.25))

        # Entity type similarity (inverted JSD)
        if ent_a and ent_b:
            entity_sim = max(0.0, 1.0 - ent_jsd) if ent_jsd < float("inf") else 0.0
            components.append((entity_sim, 0.20))

        # SBERT retention
        if SA is not None and SB is not None:
            components.append((ret, 0.20))

        # Concreteness (v5.0: de-weighted — saturates within tradition)
        if wA[i].pos is not None and wB[j].pos is not None:
            components.append((conc_sim, 0.05))

        if components:
            total_w = sum(w for _, w in components)
            composite = sum(v * (w / total_w) for v, w in components)
        else:
            composite = 0.0

        out["sem_retention_composite"].append(
            float(np.clip(composite, 0.0, 1.0))
        )

    return out