import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

from v2 import calibrate

from . import corpus, experiment, novel, parser

OUT_DIR = os.path.join(experiment.OUT_DIR, "sw_baseline")
MODEL = "msmarco-distilbert-cos-v5"
Z_MEAN, Z_STD = 0.097, 0.099   # their constants for this model
Z_THRESHOLD = 0.6
GAP = -0.7
BOOK_UNIT = 7
SEED = 0
N_BOOT = 2000
# Retention % reported by Pial et al. (Table E.1) for films in both corpora
PUBLISHED = {"Anna Karenina (2012 film)": 22.1, "The Three Musketeers (1993 film)": 29.3,
             "Mary Shelleys Frankenstein (film)": 69.1, "The Wizard of Oz (1939 film)": 72.9,
             "Pride & Prejudice (2005 film)": 89.4}


def align(sim: np.ndarray):
    """(book aligned?, script aligned?, (book, script) index pairs of the matches)."""
    S = (2 / (1 + np.exp(-((sim - Z_MEAN) / Z_STD - Z_THRESHOLD))) - 1).astype(np.float32)
    n, m = S.shape
    H = np.zeros((n + 1, m + 1), dtype=np.float32)
    P = np.zeros((n + 1, m + 1), dtype=np.uint8)   # 0 stop, 1 diagonal, 2 up, 3 left
    for i in range(1, n + 1):
        s = S[i - 1]
        step = GAP + np.maximum(0, s)
        diag, up = H[i - 1, :-1] + s, H[i - 1, 1:] + step
        c = np.maximum(0, np.maximum(diag, up))
        # H[j] = max(c[j], H[j-1] + step[j]), which unrolls to a running maximum over a cumulative sum
        A = np.cumsum(step, dtype=np.float64)
        best = np.maximum.accumulate(c - A)
        row = (A + best).astype(np.float32)
        H[i, 1:] = row
        from_left = np.empty(m, dtype=bool)
        from_left[0] = False
        from_left[1:] = row[1:] > c[1:] + 1e-6
        P[i, 1:] = np.where(from_left, 3, np.where(c == 0, 0, np.where(diag >= up, 1, 2)))

    idx = np.flatnonzero(H[1:, 1:] > 0)
    idx = idx[np.argsort(-H[1:, 1:].ravel()[idx])]
    rows, cols = np.unravel_index(idx, (n, m))
    used_row, used_col = np.zeros(n + 1, bool), np.zeros(m + 1, bool)
    book, script = np.zeros(n, bool), np.zeros(m, bool)
    matches = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        i, j = r + 1, c + 1
        if used_row[i] or used_col[j]:
            continue
        while i > 0 and j > 0 and H[i, j] > 0 and not (used_row[i] or used_col[j]):
            move = P[i, j]
            if move == 0:
                break
            if S[i - 1, j - 1] > 0:
                book[i - 1] = script[j - 1] = True
                matches.append((i - 1, j - 1))
            if move in (1, 2):
                used_row[i] = True
            if move in (1, 3):
                used_col[j] = True
            i, j = (i - 1, j - 1) if move == 1 else (i - 1, j) if move == 2 else (i, j - 1)
    return book, script, matches


def book_paragraphs(path: str, names: List[str]) -> List[str]:
    return [p for p in (experiment.mask(x, names) for x in novel.load(path).text().split("\n\n")) if p.strip()]


def script_paragraphs(film: str, names: List[str]) -> List[str]:
    return [experiment.mask(e.text, names) for e in parser.load(film).elements if e.text.strip()]


def run(variant: str) -> None:
    from sentence_transformers import SentenceTransformer

    pairs = experiment.full_pairs()
    names = (experiment.character_names(sorted({p[1].film for p in pairs}), sorted({p[0].novel_path for p in pairs}))
             if variant == "masked" else [])
    model = SentenceTransformer(MODEL)
    enc = lambda t: model.encode(t, batch_size=128, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
    books = {p[0].novel_key: enc(book_paragraphs(p[0].novel_path, names)) for p in {q[0].novel_key: q for q in pairs}.values()}
    scripts = {p[1].film: enc(script_paragraphs(p[1].film, names)) for p in {q[1].film: q for q in pairs}.values()}
    print("embedded", flush=True)
    rows = []
    for src, scr, kind in pairs:
        b, s = books[src.novel_key], scripts[scr.film]
        bk, sc, m = align(b @ s.T)
        units = [bk[i:i + BOOK_UNIT] for i in range(0, len(bk), BOOK_UNIT)]
        rho = spearmanr(*zip(*sorted((j, i) for i, j in m)))[0] if len(m) > 2 else np.nan
        rows.append({"novel": src.novel_key, "screenplay": scr.film, "pair_type": kind, "n_book": len(bk), "n_script": len(sc),
                     "retention": 100 * float(np.mean([u.mean() >= 0.5 for u in units])), "para_retention": 100 * float(bk.mean()),
                     "script_aligned": 100 * float(sc.mean()), "sw_order": rho})
        print(f"{kind} {src.novel_key} -> {scr.film}: retention {rows[-1]['retention']:.1f}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"{variant}.csv"), index=False)


def boot_auc(y, a, b, rng):
    """Bootstrap of AUC(a), and of AUC(b) - AUC(a) on the same resamples."""
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    da, dd = [], []
    for _ in range(N_BOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        ya = roc_auc_score(y[i], a[i])
        da.append(ya)
        dd.append(roc_auc_score(y[i], b[i]) - ya)
    return np.percentile(da, [2.5, 97.5]), np.percentile(dd, [2.5, 97.5])


def score() -> None:
    v2 = {"masked": "full_masked", "unmasked": "full_unmasked"}
    rng = np.random.default_rng(SEED)
    for variant, name in v2.items():
        path = os.path.join(OUT_DIR, f"{variant}.csv")
        if not os.path.exists(path):
            continue
        sw = pd.read_csv(path)
        ours = pd.read_csv(f"out_v2/screenplays_{name}.csv")
        ours = ours[(ours.novel_part == "full") & (ours.script_part == "full")]
        d = sw.merge(ours, on=["novel", "screenplay"], how="inner", suffixes=("", "_v2"))
        d["pair_key"] = d.novel + "_vs_" + d.screenplay
        d = d.set_index("pair_key")
        is_ctrl = d.pair_type == "control"
        ref = calibrate.Additive(d, is_ctrl, ["novel", "screenplay"])
        chans = {"retention": {"retention": +1}, "para": {"para_retention": +1}, "script": {"script_aligned": +1}, "order": {"sw_order": +1}}
        z = calibrate.score_cell(d[["retention", "para_retention", "script_aligned", "sw_order"]], is_ctrl, ref, chans, n_boot=0)
        d = d.join(z[[f"z_{c}" for c in chans]], rsuffix="_sw")
        d = d[is_ctrl | (d.fidelity != corpus.PARODY)]
        y = (d.pair_type != "control").to_numpy()
        print(f"\n===== {variant}: {y.sum()} adaptations, {(~y).sum()} controls =====")
        ad = d[y]
        print("Mean, adaptations / controls: retention "
              f"{ad.retention.mean():.1f} / {d[~y].retention.mean():.1f}; paragraphs aligned {ad.para_retention.mean():.1f} / {d[~y].para_retention.mean():.1f}")
        for label, col in (("SW retention, raw", "retention"), ("SW paragraphs aligned, raw", "para_retention"),
                           ("SW script share aligned, raw", "script_aligned"),
                           ("SW retention, our calibration", "z_retention"), ("SW paragraphs, our calibration", "z_para"),
                           ("our profile", "profile"), ("our lexical", "z_lexical"), ("our semantic", "z_semantic"), ("our channelled", "channelled")):
            print(f"  {label:32s} AUC {roc_auc_score(y, d[col]):.3f}")
        ci, diff = boot_auc(y, d.retention.to_numpy(), d.profile.to_numpy(), rng)
        print(f"  raw retention AUC 95% {ci[0]:.3f}-{ci[1]:.3f}; profile minus raw retention 95% {diff[0]:+.3f} to {diff[1]:+.3f}")
        ci, diff = boot_auc(y, d.z_retention.to_numpy(), d.profile.to_numpy(), rng)
        print(f"  calibrated retention AUC 95% {ci[0]:.3f}-{ci[1]:.3f}; profile minus calibrated 95% {diff[0]:+.3f} to {diff[1]:+.3f}")

        print("Spearman, retention (our calibration) with channels, adaptations:")
        print("  " + ", ".join(f"{c[2:]} {spearmanr(ad.z_retention, ad[c])[0]:+.2f}"
                               for c in ["z_lexical", "z_semantic", "z_conceptual", "z_structural", "z_narrative", "narr_order", "narr_curves", "profile"]))
        print(f"Order, adaptations: SW {roc_auc_score(y, d.z_order):.3f} (our calibration), raw {roc_auc_score(y, d.sw_order.fillna(0)):.3f}; "
              f"ours order {roc_auc_score(y, d.narr_order):.3f}; Spearman SW order, our order {spearmanr(ad.z_order, ad.narr_order)[0]:+.2f}")
        rank = ad.fidelity.map({"close": 3, "loose": 2, "transposed": 1})
        for c in ("retention", "z_retention", "profile", "z_order", "narr_order"):
            print(f"  Spearman with fidelity rank, {c:12s} {spearmanr(rank, ad[c])[0]:+.2f} (p = {spearmanr(rank, ad[c])[1]:.3f})")
        close, loose = ad[ad.fidelity == "close"], ad[ad.fidelity == "loose"]
        print(f"  close vs loose, SW order z: p = {mannwhitneyu(close.z_order, loose.z_order).pvalue:.4f}; our order: p = {mannwhitneyu(close.narr_order, loose.narr_order).pvalue:.4f}")
        pub = pd.Series(PUBLISHED)
        got = ad.set_index("screenplay").retention.reindex(pub.index)
        print("Check against Pial et al.'s published retention:", {k.split(" (")[0]: (pub[k], round(got[k], 1)) for k in pub.index})
        print(f"  Spearman published vs ours: {spearmanr(pub, got)[0]:+.2f}")
        ad_sort = ad.sort_values("retention", ascending=False)
        print(ad_sort[["screenplay", "fidelity", "n_book", "n_script", "retention", "para_retention", "z_retention", "profile", "sw_order"]].round(2).to_string(index=False))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["masked", "unmasked"])
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    score() if args.score else run(args.variant)


if __name__ == "__main__":
    main()
