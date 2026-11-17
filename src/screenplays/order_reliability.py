import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MATCHES = os.path.join("out_screenplays", "full_masked", "sequence_matches.csv")
OUT = os.path.join("out_screenplays", "full_masked", "order_reliability.csv")
MIN_SCENES = 10


def rho(g):
    return spearmanr(g.scene, g.novel_pos)[0] if len(g) >= MIN_SCENES and g.novel_pos.nunique() > 1 else np.nan


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    m = pd.read_csv(MATCHES)
    m = m[(m.novel_part == "full") & (m.script_part == "full")]

    rows, ctrl = [], []
    for film, g in m.groupby("screenplay"):
        wide = g.pivot_table(index="scene", columns="novel", values="sim")
        src = g[g.pair_type != "control"].novel.iloc[0]
        for novel, h in g.groupby("novel"):
            others = wide.drop(columns=novel)
            ok = h.scene.map(wide[novel] > others.max(axis=1)).astype(bool)
            kept = h[ok]
            row = {"screenplay": film, "novel": novel, "n": len(h), "n_reliable": len(kept), "share": ok.mean(),
                   "rho_all": rho(h), "rho_reliable": rho(kept),
                   "p_reliable": (spearmanr(kept.scene, kept.novel_pos)[1] if len(kept) >= MIN_SCENES else np.nan)}
            (rows if novel == src else ctrl).append(row)

    a = pd.DataFrame(rows)
    c = pd.DataFrame(ctrl)
    chance = c.share.mean()
    a["est_correct"] = ((a.share - chance) / (1 - chance)).clip(lower=0)
    a.sort_values("rho_reliable").to_csv(OUT, index=False)

    print(f"Pass rate of control novels: mean {chance:.3f}, 95th pct {c.share.quantile(.95):.3f}")
    print(f"Control rho on passing scenes: mean {c.rho_reliable.mean():+.3f}, "
          f"2.5-97.5 pct {c.rho_reliable.quantile(.025):+.2f} to {c.rho_reliable.quantile(.975):+.2f}, defined for {c.rho_reliable.notna().sum()}/{len(c)}")
    print(f"Control rho on all scenes: mean {c.rho_all.mean():+.3f}, 2.5-97.5 pct {c.rho_all.quantile(.025):+.2f} to {c.rho_all.quantile(.975):+.2f}")
    print(a.sort_values("rho_reliable")[["screenplay", "n", "n_reliable", "share", "est_correct", "rho_all", "rho_reliable", "p_reliable"]]
          .round(3).to_string(index=False))
    print(f"\nReliable share across films: mean {a.share.mean():.3f}, median {a.share.median():.3f}, min {a.share.min():.3f}, max {a.share.max():.3f}")
    print(f"Films with at least 10 reliable scenes: {a.rho_reliable.notna().sum()} of {len(a)}")

    corr = a.dropna(subset=["rho_reliable"])
    print(f"\nrho_all vs rho_reliable across films: Spearman {spearmanr(corr.rho_all, corr.rho_reliable)[0]:.2f}")
    print(f"share vs rho_all across films: Spearman {spearmanr(a.share, a.rho_all)[0]:.2f} (p {spearmanr(a.share, a.rho_all)[1]:.3f})")

    tl = m[(m.screenplay == "little_women_2019") & (m.pair_type != "control")]
    kept = tl[tl.scene.isin(a[a.screenplay == "little_women_2019"].pipe(lambda _: tl.scene))]
    wide = m[m.screenplay == "little_women_2019"].pivot_table(index="scene", columns="novel", values="sim")
    src = tl.novel.iloc[0]
    ok = wide.index[wide[src] > wide.drop(columns=src).max(axis=1)]
    kept = tl[tl.scene.isin(ok)]
    print("\nLittle Women 2019, reliable scenes by timeline")
    for t, g in kept.groupby("timeline"):
        print(f"  {t}: n {len(g)}, mean novel pos {g.novel_pos.mean():.2f}, rho {spearmanr(g.scene, g.novel_pos)[0]:.2f}")


if __name__ == "__main__":
    main()
