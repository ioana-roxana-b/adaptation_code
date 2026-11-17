import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path[:0] = [".", "src"]
import preprocess
from screenplays import corpus

OUT = "paper"
# Main version: windows embedded by pooling 180-word pieces (paper/sensitivity.py).
# DH_SUFFIX="" reproduces the version that embeds only the first 256 word pieces.
SUFFIX = os.environ.get("DH_SUFFIX", "_pooled")
CHANNELS = ["z_lexical", "z_semantic", "z_conceptual", "z_structural", "z_narrative"]
PARTS = ["narr_order", "narr_curves"]
HEADINGS = ["Lexical", "Semantic", "Conceptual", "Structural", "Narrative", "Order", "Curves"]
SHORT = {
    "Pride & Prejudice (2005 film)": "Pride & Prejudice (2005)", "Mary Shelleys Frankenstein (film)": "Mary Shelley's Frankenstein (1994)",
    "little_women_2019": "Little Women (2019)", "little_women_1994": "Little Women (1994)",
    "Sense and Sensibility (film)": "Sense and Sensibility (1995)", "Jane Eyre (2011 film)": "Jane Eyre (2011)",
    "Barry Lyndon": "Barry Lyndon (1975)", "The Age of Innocence (1993 film)": "The Age of Innocence (1993)",
    "The Four Feathers (2002 film)": "The Four Feathers (2002)", "The Time Machine (1960 film)": "The Time Machine (1960)",
    "The Wizard of Oz (1939 film)": "The Wizard of Oz (1939)", "Anna Karenina (2012 film)": "Anna Karenina (2012)",
    "The Three Musketeers (1993 film)": "The Three Musketeers (1993)", "The Man in the Iron Mask (1998 film)": "The Man in the Iron Mask (1998)",
    "The Last of the Mohicans (1992 film)": "The Last of the Mohicans (1992)", "The Time Machine (2002 film)": "The Time Machine (2002)",
    "Sleepy Hollow (film)": "Sleepy Hollow (1999)", "Apocalypse Now": "Apocalypse Now (1979)",
    "War of the Worlds (2005 film)": "War of the Worlds (2005)", "Easy A": "Easy A (2010)",
    "O Brother, Where Art Thou_": "O Brother, Where Art Thou? (2000)", "Young Frankenstein": "Young Frankenstein (1974)",
}


def holm(p):
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty(len(p))
    out[order] = adj
    return out


def paired(a, b, label):
    print(f"\n{label} (n = {len(a)}; Wilcoxon, Holm over the five channels)")
    rows = []
    for c in CHANNELS:
        x = (a[c] - b[c].reindex(a.index)).dropna()
        rows.append((c[2:], x.mean(), wilcoxon(x).pvalue))
    adj = holm([r[2] for r in rows])
    for (c, m, p), h in zip(rows, adj):
        print(f"  {c:11s} diff {m:+.2f}  p {p:.4f}  Holm {h:.3f}")
    for c in ["profile", *PARTS]:
        x = (a[c] - b[c].reindex(a.index)).dropna()
        print(f"  {c:11s} diff {x.mean():+.2f}  p {wilcoxon(x).pvalue:.4f}")


def words(pattern):
    paths = [p for p in glob.glob(os.path.join(corpus.EXTERNAL_DIR, "components", pattern)) if "__masked__" not in p]
    # Tokens as the pipeline counts them: NLTK word tokens with punctuation stripped
    return len(preprocess.tokenize_string(open(max(paths, key=os.path.getmtime), encoding="utf-8").read())) if paths else None


def figure(ff):
    t = ff.sort_values("profile", ascending=False)
    M = t[CHANNELS + PARTS].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("dv", ["#2a78d6", "#f0efec", "#e34948"])
    norm = TwoSlopeNorm(vcenter=0, vmin=-4, vmax=4)
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v + 0.0 if round(v, 1) else 0.0:.1f}".replace("-", "\u2212"), ha="center", va="center",
                fontsize=6.5, color="white" if abs(v) > 2.6 else "#222222")
    ax.set_xticks(range(M.shape[1]), HEADINGS, fontsize=6.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(t)), [SHORT[s] for s in t.screenplay])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(4.5, color="white", lw=3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_profiles.{ext}"), dpi=200)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = pd.read_csv(f"out_v2/screenplays_full_masked{SUFFIX}.csv")
    u = pd.read_csv(f"out_v2/screenplays_full_unmasked{SUFFIX}.csv")
    a = d[d.pair_type != "control"]
    ff = a[(a.novel_part == "full") & (a.script_part == "full")]
    figure(ff)
    ad = a
    print("AUC, all 22 adaptations vs all 110 controls of the cell")
    for (n, s), g in d.groupby(["novel_part", "script_part"]):
        y = g.pair_type != "control"
        print(f"  {n:9s} x {s:8s} n_ctrl {(~y).sum()}  channelled {roc_auc_score(y, g.channelled):.3f}  profile {roc_auc_score(y, g.profile):.3f}  | "
              + "  ".join(f"{c[2:]} {roc_auc_score(y, g[c]):.3f}" for c in CHANNELS))
    cell = lambda df, n, s: df[(df.novel_part == n) & (df.script_part == s)].set_index("screenplay")
    paired(cell(ad, "dialogue", "dialogue"), cell(ad, "narration", "action"), "Dialogue x dialogue minus narration x action")
    ua = u[u.pair_type != "control"]
    paired(cell(ua, "full", "full"), cell(ad, "full", "full"), "Unmasked minus masked, full x full")
    by_source = lambda df: df.groupby("novel")[CHANNELS + PARTS + ["profile"]].mean()
    paired(by_source(cell(ad, "dialogue", "dialogue").reset_index()), by_source(cell(ad, "narration", "action").reset_index()),
           "Same, one value per literary source")
    paired(by_source(cell(ua, "full", "full").reset_index()), by_source(cell(ad, "full", "full").reset_index()),
           "Unmasked minus masked, one value per literary source")

    print("\nAUC with 95% interval from resampling literary sources (all pairs of a source move together)")
    rng = np.random.default_rng(0)
    for (n, s), g in d.groupby(["novel_part", "script_part"]):
        groups = {k: v.to_numpy() for k, v in g.reset_index(drop=True).groupby("novel").groups.items()}
        keys = list(groups)
        y, s_p = (g.pair_type != "control").to_numpy(), g.profile.to_numpy()
        vals = []
        while len(vals) < 2000:
            i = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
            if 0 < y[i].sum() < len(i):
                vals.append(roc_auc_score(y[i], s_p[i]))
        print(f"  {n:9s} x {s:8s} {roc_auc_score(y, s_p):.3f} ({np.percentile(vals, 2.5):.3f}-{np.percentile(vals, 97.5):.3f})")

    print("\nMeans, full x full")
    print(ff[CHANNELS + PARTS + ["profile"]].mean().round(2).to_string())

    print("\nSpearman between channels across films, full x full")
    fa = ff
    print(fa[CHANNELS + PARTS].corr(method="spearman").round(2).to_string())
    dd, na = cell(ad, "dialogue", "dialogue"), cell(ad, "narration", "action")
    delta = dd[CHANNELS] - na[CHANNELS].reindex(dd.index)
    r, p = spearmanr(delta.z_lexical, delta.z_structural)
    print(f"Route gain, lexical vs structural: Spearman {r:.2f} (p {p:.3f}); structural higher in {(delta.z_structural > 0).sum()} of {len(delta)} films")
    print("Mean semantic score by route:", round(dd.z_semantic.mean(), 2), round(na.z_semantic.mean(), 2))

    m = pd.read_csv(os.path.join("out_screenplays", "full_masked", "sequence_matches.csv"))
    m = m[(m.novel_part == "full") & (m.script_part == "full") & (m.pair_type != "control")]
    lw = m[m.screenplay == "little_women_2019"].reset_index(drop=True)
    print(f"\nLittle Women 2019: order over the whole film {spearmanr(lw.index, lw.novel_pos)[0]:.2f}")
    for tl, g in lw.groupby("timeline"):
        print(f"  within timeline {tl}: rho {spearmanr(range(len(g)), g.novel_pos)[0]:.2f} ({len(g)} scenes)")
    oz = m[m.screenplay == "The Wizard of Oz (1939 film)"]
    print(f"Oz: {(oz.novel_pos < 0.1).sum()} of {len(oz)} scenes match the novel's first tenth")

    films = corpus.included()
    key = lambda s: re.sub(r"\W+", "_", s).strip("_").lower()
    nw = {f.novel_key: words(f"novel_{f.novel_key}__full__*.txt") for f in films}
    sw = {f.film: words(f"script_{key(f.film)}__full__*.txt") for f in films}
    pd.DataFrame([{"film": SHORT[f.film], "source_tokens": nw[f.novel_key], "script_tokens": sw[f.film]} for f in films]).to_csv(
        os.path.join(OUT, "corpus_tokens.csv"), index=False)
    print(f"\n{len(films)} films, {len(nw)} novels; novel tokens {sum(v for v in nw.values() if v):,}; "
          f"screenplay tokens {sum(v for v in sw.values() if v):,}; missing counts: "
          f"{[k for k, v in {**nw, **sw}.items() if v is None]}")


if __name__ == "__main__":
    main()
