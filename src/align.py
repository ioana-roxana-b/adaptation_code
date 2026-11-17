import numpy as np
from typing import List, Optional, Tuple

from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from preprocess import Window
from features import semantic


def _window_text_for_embedding(w: Window) -> str:
    """Prefer raw text for sentence embeddings.

    SBERT is trained on natural sentences, not token joined strings.
    We keep a safe fallback for cases where raw is missing.
    """
    if w.raw and w.raw.strip():
        return w.raw.strip()
    if w.tokens:
        return " ".join(w.tokens)
    return "__EMPTY__"


def align_windows(
    wA: List[Window],
    wB: List[Window],
    embeddings_A: Optional[np.ndarray] = None,
    embeddings_B: Optional[np.ndarray] = None,
    directional: bool = True,
    positional_weight: float = 0.15,
) -> List[Tuple[int, int]]:
    """Align windows using DTW over a dense semantic space.

    v2.0: Source-anchored directional alignment.

    When directional=True (default), the DTW cost includes a positional
    penalty that encourages the alignment to map proportional positions
    in A to corresponding positions in B. This makes the alignment
    asymmetric: it asks "how does the derived text (B) map onto the
    source (A)?" rather than treating both texts symmetrically.

    The positional penalty is:
        pos_cost(i, j) = |i/n - j/m|
    weighted by `positional_weight` (default 0.15).

    Additionally, the traceback favours diagonal steps when directional
    is True, biasing toward one-to-one coverage of the source text.
    This ensures every source position gets a meaningful match, which
    is essential for measuring WHERE influence persists.

    FIX #7: Accepts optional pre-computed embeddings to avoid redundant
    SBERT encoding when features_runner has already encoded them.
    """
    if not wA or not wB:
        return []

    if embeddings_A is not None and embeddings_B is not None:
        denseA = embeddings_A
        denseB = embeddings_B
    else:
        docsA = [_window_text_for_embedding(w) for w in wA]
        docsB = [_window_text_for_embedding(w) for w in wB]

        model = semantic.get_sbert_model("all-MiniLM-L6-v2")

        print(model)
        if model is not None:
            
            denseA = model.encode(docsA, batch_size=8, show_progress_bar=False)
            denseB = model.encode(docsB, batch_size=8, show_progress_bar=False)
        else:
            print("WARNING: SBERT not available. Falling back to TF IDF for DTW alignment.")
            try:
                vect = TfidfVectorizer(lowercase=True, stop_words="english", min_df=2)
                tfidf = vect.fit_transform(docsA + docsB)

                n_comp = min(100, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
                if n_comp > 1:
                    svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    dense = svd.fit_transform(tfidf)
                else:
                    dense = tfidf.toarray()

                denseA = dense[: len(docsA)]
                denseB = dense[len(docsA):]
            except ValueError:
                return [(i, min(i, len(wB) - 1)) for i in range(len(wA))]

    dist_matrix = cosine_distances(denseA, denseB)
    n, m = dist_matrix.shape

    # v2.0: Add positional cost for directional alignment
    if directional and positional_weight > 0:
        pos_i = np.arange(n, dtype=np.float32) / max(n - 1, 1)
        pos_j = np.arange(m, dtype=np.float32) / max(m - 1, 1)
        pos_cost = np.abs(pos_i[:, None] - pos_j[None, :])
        dist_matrix = dist_matrix + positional_weight * pos_cost

    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw_matrix[0, 0] = 0.0

    band = int(0.10 * max(n, m))
    band = max(band, 5)
    band = max(band, abs(n - m) + 2)

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            cost = dist_matrix[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        # v2.0: Directional traceback bias — prefer diagonal steps
        # to ensure one-to-one source coverage where possible
        candidates = [
            dtw_matrix[i - 1, j - 1],  # diagonal (covers both)
            dtw_matrix[i - 1, j],       # advance source only
            dtw_matrix[i, j - 1],       # advance derived only
        ]

        if directional:
            # Slight bias toward diagonal: subtract epsilon from diagonal cost
            # This breaks ties in favour of covering both texts
            diag_bias = 1e-6
            candidates[0] = candidates[0] - diag_bias

        step = int(np.argmin(candidates))

        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1

    path.reverse()
    return path