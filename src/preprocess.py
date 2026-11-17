import os
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import Counter

import networkx as nx

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


_SPACY_CACHE: Dict[str, Any] = {}


def get_spacy_model(model_name: str = "en_core_web_sm"):
    """Load a spaCy model if available.

    Paper safe behavior: no auto download.
    If the model is not installed, return None and let callers fall back.
    """
    global _SPACY_CACHE
    if model_name in _SPACY_CACHE:
        return _SPACY_CACHE[model_name]

    try:
        import spacy
        try:
            nlp = spacy.load(model_name)
        except OSError:
            nlp = None
    except Exception:
        nlp = None

    _SPACY_CACHE[model_name] = nlp
    return nlp


def ensure_nltk() -> None:
    """Ensure local NLTK resources exist."""
    resource_map = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "vader_lexicon": "sentiment/vader_lexicon",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for pkg, res in resource_map.items():
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(pkg, quiet=True)


@dataclass
class Window:
    unit_type: str
    unit_start: int
    unit_end: int
    raw: Optional[str] = None
    tokens: List[str] = field(default_factory=list)
    pos: Optional[List[str]] = None
    content_tokens: Optional[List[str]] = None
    punct: Optional[Dict[str, float]] = None
    fw_edges: Optional[Counter] = None
    wan_metrics: Optional[Dict[str, float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "unit_type": self.unit_type,
            "unit_start": self.unit_start,
            "unit_end": self.unit_end,
            "raw": self.raw,
            "tokens": self.tokens,
            "pos": self.pos,
            "content_tokens": self.content_tokens,
            "punct": self.punct,
            "fw_edges": [{"edge": list(k), "weight": v} for k, v in (self.fw_edges or Counter()).items()],
            "wan_metrics": self.wan_metrics,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Window":
        w = cls(
            unit_type=d["unit_type"],
            unit_start=int(d["unit_start"]),
            unit_end=int(d["unit_end"]),
            raw=d.get("raw"),
            tokens=list(d.get("tokens") or []),
        )
        w.pos = d.get("pos")
        w.content_tokens = d.get("content_tokens")
        w.punct = d.get("punct")
        w.fw_edges = Counter({tuple(item["edge"]): item["weight"] for item in (d.get("fw_edges") or [])})
        w.wan_metrics = d.get("wan_metrics")
        w.meta = d.get("meta", {})
        return w


def strip_gutenberg_boilerplate(raw: str) -> str:
    start_pat = r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    end_pat = r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    m_start = re.search(start_pat, raw, flags=re.IGNORECASE | re.DOTALL)
    if m_start:
        raw = raw[m_start.end():]
    m_end = re.search(end_pat, raw, flags=re.IGNORECASE | re.DOTALL)
    if m_end:
        raw = raw[:m_end.start()]
    raw = raw.replace("\r\n", "\n")
    return re.sub(r"\n{3,}", "\n\n", raw).strip()


def load_text(path: str, strip_gutenberg: bool = True) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    return strip_gutenberg_boilerplate(raw) if strip_gutenberg else raw


def normalize_whitespace(s: str) -> str:
    s = s.replace("\t", " ")
    s = re.sub(r"[ \u00A0]+", " ", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def tokenize_string(s: str, lowercase: bool = True, remove_stopwords: bool = False) -> List[str]:
    if lowercase:
        s = s.lower()
    tokens = word_tokenize(s)
    cleaned = [re.sub(r"[^a-zA-Z']", "", t) for t in tokens]
    cleaned = [t for t in cleaned if t]

    if remove_stopwords:
        sw = set(stopwords.words("english"))
        cleaned = [t for t in cleaned if t not in sw]

    return cleaned


def pos_tags(tokens: List[str]) -> List[str]:
    return [tag for _, tag in nltk.pos_tag(tokens)]


def build_content_tokens(tokens: List[str], pos: List[str]) -> List[str]:
    content_tags = {
        "NN", "NNS", "NNP", "NNPS",
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
        "JJ", "JJR", "JJS",
        "RB", "RBR", "RBS",
    }
    return [t for t, p in zip(tokens, pos) if p in content_tags]


def split_into_chapters(text: str, fallback_word_target: int = 5000) -> Dict[str, str]:
    text = normalize_whitespace(text)

    pat = re.compile(
        r"(?im)^(?P<h>\s*(?:(?:BOOK|PART|CHAPTER|ACT|PROLOGUE|EPILOGUE|PREFACE|PRELUDE)\b[^\n]*|[IVXLCDM]+\.?))\s*$"
    )
    matches = list(pat.finditer(text))

    chapters: Dict[str, str] = {}
    if matches:
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            title = re.sub(r"\s+", " ", m.group(0)).strip()
            body = text[start:end].strip()
            if len(body) >= 500:
                chapters[title] = body

    if not chapters:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        current_chunk: List[str] = []
        current_wc = 0
        chunk_idx = 1

        for p in paragraphs:
            wc = len(p.split())
            current_chunk.append(p)
            current_wc += wc
            if current_wc >= fallback_word_target:
                chapters[f"Chunk_{chunk_idx}"] = "\n\n".join(current_chunk)
                chunk_idx += 1
                current_chunk = []
                current_wc = 0

        if current_chunk:
            chapters[f"Chunk_{chunk_idx}"] = "\n\n".join(current_chunk)

    return chapters


def make_chapter_windows(text: str, lowercase: bool, remove_stopwords: bool) -> List[Window]:
    ch = split_into_chapters(text)
    out: List[Window] = []
    for i, (title, raw) in enumerate(ch.items()):
        out.append(
            Window(
                unit_type="chapter",
                unit_start=i,
                unit_end=i + 1,
                raw=raw,
                tokens=tokenize_string(raw, lowercase=lowercase, remove_stopwords=remove_stopwords),
                meta={"chapter_title": title},
            )
        )
    return out


def _make_sliding_windows(
    units: List[str],
    unit_toks: List[List[str]],
    unit_type: str,
    target_tokens: int,
    step_tokens: int,
    join_str: str = "\n\n",
) -> List[Window]:
    """Generic sliding window builder used by both paragraph and sentence strategies.

    FIX #2: Filters out units with zero tokens before windowing so that
    empty paragraphs/sentences cannot cause degenerate skip behaviour.
    Also guards against the step loop overshooting.
    """
    # Filter out empty units up front
    filtered = [(u, t) for u, t in zip(units, unit_toks) if t]
    if not filtered:
        return []
    units_clean, toks_clean = zip(*filtered)
    units_clean = list(units_clean)
    toks_clean = list(toks_clean)

    windows: List[Window] = []
    i = 0
    n = len(units_clean)

    while i < n:
        # Accumulate units until we hit target_tokens
        j = i
        count = 0
        while j < n and count < target_tokens:
            count += len(toks_clean[j])
            j += 1

        # Safety: always advance at least one unit
        if j <= i:
            j = i + 1
        j = min(j, n)

        toks = [t for seg in toks_clean[i:j] for t in seg]
        raw = join_str.join(units_clean[i:j])
        windows.append(Window(unit_type, i, j, raw, toks))

        # Step forward by step_tokens worth of units
        adv = 0
        tok_adv = 0
        while i + adv < n and tok_adv < step_tokens:
            tok_adv += len(toks_clean[i + adv])
            adv += 1
        i += max(1, adv)

    return windows


def make_paragraph_windows(
    text: str,
    target_tokens: int,
    step_tokens: int,
    lowercase: bool,
    remove_stopwords: bool,
) -> List[Window]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", normalize_whitespace(text)) if p.strip()]
    para_toks = [tokenize_string(p, lowercase=lowercase, remove_stopwords=remove_stopwords) for p in paras]
    return _make_sliding_windows(paras, para_toks, "paragraph", target_tokens, step_tokens, "\n\n")


def make_sentence_windows(
    text: str,
    target_tokens: int,
    step_tokens: int,
    lowercase: bool,
    remove_stopwords: bool,
) -> List[Window]:
    sents = [s.strip() for s in sent_tokenize(normalize_whitespace(text)) if s.strip()]
    sent_toks = [tokenize_string(s, lowercase=lowercase, remove_stopwords=remove_stopwords) for s in sents]
    return _make_sliding_windows(sents, sent_toks, "sentence", target_tokens, step_tokens, " ")


def punctuation_features(raw: str, n_tokens: int) -> Dict[str, float]:
    denom = max(1, int(n_tokens))
    per_k = 1000.0 / float(denom)

    chars = [",", ".", ";", ":", "!", "?", '"']
    feats = {f"{c}_per_k": raw.count(c) * per_k for c in chars}

    feats["apostrophe_per_k"] = raw.count("'") * per_k
    feats["comma_to_period"] = float(raw.count(",")) / float(max(1, raw.count(".")))
    feats["quote_density_per_k"] = raw.count('"') * per_k
    return feats


def wan_graph_metrics_from_edges(edges: Counter) -> Dict[str, float]:
    g = nx.DiGraph()
    for (a, b), w in edges.items():
        if a == b:
            continue
        g.add_edge(a, b, weight=float(w))

    n = g.number_of_nodes()
    if n == 0:
        return {"average_degree": 0.0, "density": 0.0, "n_nodes": 0.0}

    return {
        "average_degree": float(sum(dict(g.degree()).values())) / float(n),
        "density": float(nx.density(g)),
        "n_nodes": float(n),
    }


def _enrich_window_spacy(w: Window, nlp, lowercase: bool) -> None:
    """Enrich a window using spaCy, ensuring tokens and POS are aligned.

    FIX #1: Previously, w.tokens came from NLTK and w.pos came from spaCy,
    producing misaligned arrays. Now both are derived from the same spaCy doc,
    applying the same cleaning rules (alphabetic-only, non-empty) so that
    len(w.tokens) == len(w.pos) is guaranteed.
    """
    raw_for_spacy = w.raw or ""
    if not raw_for_spacy.strip():
        return

    if len(raw_for_spacy) > nlp.max_length:
        raw_for_spacy = raw_for_spacy[: nlp.max_length]

    doc = nlp(raw_for_spacy)

    # Build aligned tokens and POS from the SAME source (spaCy doc).
    # Apply the same cleaning as tokenize_string: alphabetic + apostrophe only.
    alpha_re = re.compile(r"[^a-zA-Z']")
    tokens: List[str] = []
    pos_list: List[str] = []
    content_tokens: List[str] = []

    # Content POS tags (same set as build_content_tokens)
    content_tags = {
        "NN", "NNS", "NNP", "NNPS",
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
        "JJ", "JJR", "JJS",
        "RB", "RBR", "RBS",
    }

    for t in doc:
        if t.is_space:
            continue
        cleaned = alpha_re.sub("", t.text)
        if not cleaned:
            continue

        tok = cleaned.lower() if lowercase else cleaned
        tokens.append(tok)
        pos_list.append(t.tag_)

        # Content tokens: not stopword, not punct (already filtered above)
        if not t.is_stop and t.tag_ in content_tags:
            content_tokens.append(tok)

    w.tokens = tokens
    w.pos = pos_list
    w.content_tokens = content_tokens

    # Dependency features (from the full doc, not the cleaned tokens)
    dep_clause_labels = {"advcl", "ccomp", "xcomp", "relcl", "acl"}
    depths: List[int] = []
    clause_hits = 0
    valid_toks = 0
    for t in doc:
        if t.is_space:
            continue
        valid_toks += 1
        if t.dep_ in dep_clause_labels:
            clause_hits += 1
        d = 0
        cur = t
        while cur.head != cur and cur.head is not None:
            d += 1
            cur = cur.head
        depths.append(d)

    w.meta["dep_depths"] = depths
    w.meta["clause_hits"] = clause_hits
    w.meta["dep_valid_toks"] = valid_toks


def preprocess_text(
    text: str,
    strategies: List[str],
    target_tokens: int,
    step_tokens: int,
    lowercase: bool,
    remove_stopwords: bool,
    compute_pos: bool = True,
    spacy_model: str = "en_core_web_sm",
):
    nlp = get_spacy_model(spacy_model) if spacy_model else None

    sw_set: Optional[set] = None
    try:
        sw_set = set(stopwords.words("english"))
    except Exception:
        sw_set = None

    out: Dict[str, List[Window]] = {}

    for strat in strategies:
        if strat == "chapter":
            ws = make_chapter_windows(text, lowercase, remove_stopwords)
        elif strat == "paragraph":
            ws = make_paragraph_windows(text, target_tokens, step_tokens, lowercase, remove_stopwords)
        elif strat == "sentence":
            ws = make_sentence_windows(text, target_tokens, step_tokens, lowercase, remove_stopwords)
        else:
            raise ValueError(f"Unknown window strategy: {strat}")

        for w in ws:
            # FIX #1: Use unified spaCy enrichment that keeps tokens/pos aligned
            if nlp is not None and w.raw:
                _enrich_window_spacy(w, nlp, lowercase)
            elif compute_pos:
                w.pos = pos_tags(w.tokens)
                w.content_tokens = build_content_tokens(w.tokens, w.pos)

            w.punct = punctuation_features(w.raw or "", len(w.tokens))

            if sw_set is not None:
                fw_seq = [t for t in (w.tokens or []) if t in sw_set]
                w.fw_edges = Counter(zip(fw_seq, fw_seq[1:]))
                w.wan_metrics = wan_graph_metrics_from_edges(w.fw_edges)
                w.meta["fw_stopwords_available"] = True
            else:
                w.fw_edges = Counter()
                w.wan_metrics = {"average_degree": 0.0, "density": 0.0, "n_nodes": 0.0}
                w.meta["fw_stopwords_available"] = False

        out[strat] = ws

    return out


def _cache_key(label: str, config: Dict[str, Any]) -> str:
    sg = ",".join(config["strategies"])
    spacy_model = str(config.get("syntax_spacy_model") or "none")
    # Bumped to v4 because token/POS alignment changed
    return (
        f"v4__{label}__tg{config['target_tokens']}__st{config['step_tokens']}"
        f"__lc{int(config['lowercase'])}__sw{int(config['remove_stopwords'])}__sg{sg}__sp{spacy_model}"
    )


def load_or_build_preprocessed(label: str, path: str, config: Dict[str, Any], cache_dir: str = "cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{_cache_key(label, config)}.json")

    if os.path.exists(cache_path):
        print(f"Loading {label} from local JSON cache...")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {strat: [Window.from_dict(w) for w in windows] for strat, windows in data.items()}

    print(f"Building {label} and saving to local JSON cache...")
    text = load_text(path, config.get("strip_gutenberg", True))
    pre = preprocess_text(
        text=text,
        strategies=config["strategies"],
        target_tokens=int(config["target_tokens"]),
        step_tokens=int(config["step_tokens"]),
        lowercase=bool(config["lowercase"]),
        remove_stopwords=bool(config["remove_stopwords"]),
        compute_pos=True,
        spacy_model=str(config.get("syntax_spacy_model") or ""),
    )

    # Written to a temporary file and renamed, so parallel runs never read a partial cache
    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json_data = {strat: [w.to_dict() for w in windows] for strat, windows in pre.items()}
        json.dump(json_data, f)
    os.replace(tmp_path, cache_path)

    return pre