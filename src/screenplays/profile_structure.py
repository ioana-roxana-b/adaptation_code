import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

import os

# Main version uses pooled window embeddings; DH_SUFFIX="" for the first-256-word-piece version
SCORES = f"out_v2/screenplays_full_masked{os.environ.get('DH_SUFFIX', '_pooled')}.csv"
COLS = ["z_lexical", "z_semantic", "z_conceptual", "z_structural", "narr_order", "narr_curves"]
N = 2000
SEED = 0


def eig(X):
    return np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]


def shuffled(X, rng):
    return np.column_stack([rng.permutation(c) for c in X.T])


def ward(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = pd.read_csv(SCORES)
    ff = d[(d.pair_type != "control") & (d.novel_part == "full") & (d.script_part == "full")].set_index("screenplay")
    X = ((ff[COLS] - ff[COLS].mean()) / ff[COLS].std()).to_numpy()
    rng = np.random.default_rng(SEED)
    print(f"{len(X)} films, {len(COLS)} scores")

    obs = eig(X)
    null = np.array([eig(shuffled(X, rng)) for _ in range(N)])
    print("\nPCA, parallel analysis")
    for i, (e, q) in enumerate(zip(obs, np.percentile(null, 95, axis=0)), 1):
        print(f"  PC{i}: eigenvalue {e:.2f} ({e / len(COLS):.0%} of variance), null 95th pct {q:.2f}, {'retained' if e > q else 'not retained'}")
    vals, vecs = np.linalg.eigh(np.corrcoef(X, rowvar=False))
    vecs = vecs[:, np.argsort(vals)[::-1]]
    print(pd.DataFrame(vecs[:, :3], index=COLS, columns=["PC1", "PC2", "PC3"]).round(2).to_string())

    print("\nWard clustering: silhouette against shuffled columns; stability over resampled films")
    for k in range(2, 7):
        lab = ward(X, k)
        s = silhouette_score(X, lab)
        ns = np.array([silhouette_score(Y, ward(Y, k)) for Y in (shuffled(X, rng) for _ in range(500))])
        aris = []
        for _ in range(500):
            i = rng.integers(0, len(X), len(X))
            if len(set(i)) > k:
                aris.append(adjusted_rand_score(lab[i], ward(X[i], k)))
        sizes = sorted(np.bincount(lab), reverse=True)
        print(f"  k={k}: silhouette {s:.3f} (null mean {ns.mean():.3f}, p {(1 + (ns >= s).sum()) / 501:.3f}); "
              f"stability ARI {np.mean(aris):.2f}; sizes {sizes}")

    print("\nk=2 membership")
    lab = ward(X, 2)
    for c in sorted(set(lab)):
        print(f"  group {c} ({(lab == c).sum()}): mean " + ", ".join(f"{n[2:] if n.startswith('z_') else n} {v:+.2f}" for n, v in ff[COLS][lab == c].mean().items()))
        print("    " + "; ".join(ff.index[lab == c]))


if __name__ == "__main__":
    main()
