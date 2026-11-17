import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import calibrate
from screenplays import corpus
from screenplays.channels import CHANNELS, NARRATIVE_PARTS

OUT_DIR = "out_v2"
INFLUENCE_DIR = "out_influence"
ICCC_ROWS = os.path.join("out_profiles", "ALL_PAIRS_ALL_STRATEGIES.csv")
SCREENPLAY_DIR = "out_screenplays"
SCREENPLAY_KEYS = ["novel", "screenplay", "novel_part", "script_part"]
# Target window position: pairs are averaged per target window (v2.calibrate, step 1)
POSITION_COLS = ["B_unit_start", "B_unit_end"]

FEATURES = [f for fs in CHANNELS.values() for f in fs]
# Narrative channel split into order and curves, reported next to z_narrative for screenplays
PART_COLS = [f"narr_{p}" for p in NARRATIVE_PARTS]
NOVEL_KEYS = ["source", "target", "trad_A", "trad_B", "movement"]
BASELINES = {
    "author": calibrate.additive(["source", "target"]),
    "movement": calibrate.additive(["source", "target", "movement"]),
}


def add_movement(rows: pd.DataFrame) -> pd.DataFrame:
    """The movement of a pair: the shared tradition of its two texts, or "cross"."""
    rows["movement"] = np.where(rows.trad_A == rows.trad_B, rows.trad_A, "cross")
    return rows


def score_novels(rows: pd.DataFrame, cell_cols) -> dict:
    """Scores under every baseline, by name."""
    results = {}
    for name, ref in BASELINES.items():
        res = calibrate.score(rows.copy(), CHANNELS, cell_cols=cell_cols, keep_cols=NOVEL_KEYS, reference=ref)
        res["baseline"] = name
        results[name] = res
    return results


def _zcols(df):
    return [c for c in df if c.startswith("z_") and not c.endswith(("_lo", "_hi"))]


def _auc(df, col):
    y = df["pair_type"] != "control"
    ok = df[col].notna()
    return round(roc_auc_score(y[ok], df.loc[ok, col]), 3)


def iccc() -> None:
    import main as v1_pipeline

    cols = pd.read_csv(ICCC_ROWS, nrows=0).columns
    feats = sorted(set(FEATURES) | set(v1_pipeline.ALL_DFW_FEATURES))
    usecols = ["pair_key", "pair_type", "window_strategy", *POSITION_COLS] + [f for f in feats if f in cols]
    rows = pd.read_csv(ICCC_ROWS, usecols=usecols)
    # Traditions from the current map (the stored trad_* columns may predate it)
    trads = {pk: v1_pipeline._get_pair_traditions(pk) for pk in rows.pair_key.unique()}
    rows["trad_A"] = rows.pair_key.map(lambda pk: trads[pk][0])
    rows["trad_B"] = rows.pair_key.map(lambda pk: trads[pk][1])
    rows["source"] = rows.pair_key.str.split("_vs_").str[0]
    rows["target"] = rows.pair_key.str.split("_vs_").str[1]
    add_movement(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    results = score_novels(rows, ["window_strategy"])
    res = pd.concat(results.values(), ignore_index=True)
    res.to_csv(os.path.join(OUT_DIR, "iccc_scores.csv"), index=False)
    print(f"Saved {OUT_DIR}/iccc_scores.csv ({res.pair_key.nunique()} pairs, baselines {', '.join(results)})")

    # v1 scores (the paper's tradition-matched method), recomputed with the ICCC code on the same rows
    v1 = pd.concat([v1_pipeline.compute_dfw_tbi(rows, strat) for strat in rows.window_strategy.unique()],
                   ignore_index=True)
    zc = _zcols(res)
    print("\nAUC, influence vs control pairs")
    for strat in v1.window_strategy.unique():
        o = v1[v1.window_strategy == strat]
        print(f"  {strat:9s} v1 channelled {_auc(o, 'dfw_channelled')}  v1 mean {_auc(o, 'dfw_tbi')}")
        for name, r in results.items():
            r = r[r.window_strategy == strat]
            print(f"  {'':9s} v2 [{name:9s}] channelled {_auc(r, 'channelled')}  profile {_auc(r, 'profile')}  |  "
                  + "  ".join(f"{c[2:]} {_auc(r, c)}" for c in zc))

    main_res = results["movement"]
    print("\nPeak channel among control pairs (v1 is biased toward high-variance channels)")
    for strat in v1.window_strategy.unique():
        o = v1[(v1.window_strategy == strat) & (v1.pair_type == "control")]
        r = main_res[(main_res.window_strategy == strat) & (main_res.pair_type == "control")]
        print(f"  {strat:9s} v1 {o.dfw_peak_channel.value_counts().to_dict()}")
        print(f"  {'':9s} v2 {r.peak_channel.value_counts().to_dict()}")

    print("\nSegmentation stability, Spearman rho between chapter and paragraph scores")
    for name, df, col in (("v1 channelled", v1, "dfw_channelled"), ("v2 channelled", main_res, "channelled"),
                          ("v2 profile", main_res, "profile")):
        p = df.pivot_table(index=["pair_key", "pair_type"], columns="window_strategy", values=col).reset_index()
        inf = p[p.pair_type != "control"]
        print(f"  {name:14s} all {spearmanr(p.chapter, p.paragraph)[0]:.2f}  "
              f"influence {spearmanr(inf.chapter, inf.paragraph)[0]:.2f}")

    print("\nInfluence pairs, paragraph strategy, movement baseline (channel z in control SDs)")
    inf = main_res[(main_res.pair_type != "control") & (main_res.window_strategy == "paragraph")]
    show = inf.sort_values("profile", ascending=False)[
        ["pair_key"] + zc + ["peak_channel", "channelled", "channelled_p", "profile", "profile_lo", "profile_hi",
                             "profile_p"]]
    print(show.round(2).to_string(index=False))


def screenplays(variants) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for variant in variants:
        raw = sorted(glob.glob(os.path.join(SCREENPLAY_DIR, variant, "raw", "*.csv")))
        if not raw:
            print(f"No raw outputs for {variant}")
            continue
        keep = ["pair_key", "pair_type", *SCREENPLAY_KEYS, *POSITION_COLS]
        rows = pd.concat([pd.read_csv(p, usecols=lambda c: c in keep + FEATURES) for p in raw], ignore_index=True)
        seq_path = os.path.join(SCREENPLAY_DIR, variant, "sequence.csv")
        if os.path.exists(seq_path):
            seq = pd.read_csv(seq_path)[SCREENPLAY_KEYS + ["order_rho"]]
            rows = rows.merge(seq, on=SCREENPLAY_KEYS, how="left")
        crossed = variant.startswith("full_")
        reference = calibrate.additive(["novel", "screenplay"]) if crossed else calibrate.all_controls
        cells = ["novel_part", "script_part"]
        res = calibrate.score(rows, CHANNELS, cell_cols=cells, keep_cols=["novel", "screenplay"], reference=reference)
        parts = calibrate.score(rows, NARRATIVE_PARTS, cell_cols=cells, keep_cols=["novel", "screenplay"],
                                reference=reference, n_boot=0)
        parts = parts.rename(columns={f"z_{p}": f"narr_{p}" for p in NARRATIVE_PARTS})
        res = res.merge(parts[["pair_key", *cells, *PART_COLS]], on=["pair_key", *cells], how="left")
        res["fidelity"] = res.screenplay.map(lambda f: corpus.get(f).fidelity if corpus.get(f) else None)
        out = os.path.join(OUT_DIR, f"screenplays_{variant}.csv")
        res.to_csv(out, index=False)
        print(f"\n===== {variant} =====  saved {out}")

        zc = _zcols(res)
        inf = res[res.pair_type != "control"]
        print(f"\nControls per cell: {int(res.n_controls.iloc[0])}; smallest attainable p = "
              f"{1 / (1 + res.n_controls.iloc[0]):.3f} for an adaptation")
        if crossed:
            _full_design_report(res, zc)
            continue
        print("\nChannelled score percentile among controls (rows: novel part, cols: screenplay part)")
        print(inf.pivot_table(index="screenplay", columns=["novel_part", "script_part"],
                              values="channelled_pct").round(2).to_string())
        print("\nProfile score percentile among controls")
        print(inf.pivot_table(index="screenplay", columns=["novel_part", "script_part"],
                              values="profile_pct").round(2).to_string())
        full = inf[(inf.novel_part == "full") & (inf.script_part == "full")]
        print("\nFull novel x full screenplay (channel z in control SDs)")
        print(full[["screenplay"] + zc + PART_COLS + ["peak_channel", "channelled", "channelled_lo", "channelled_hi",
                                                     "profile", "profile_lo", "profile_hi"]].round(2).to_string(index=False))
        print("\nMean channel z of the adaptations (parody excluded), per cell")
        ref = inf[inf.screenplay != "Young Frankenstein"]
        print(ref.groupby(["novel_part", "script_part"])[zc].mean().round(2).to_string())


def _full_design_report(res: pd.DataFrame, zc) -> None:
    """AUC per cell and channel, adaptations by fidelity, and the full x full table."""
    print("\nAUC, adaptations vs controls (parody excluded)")
    for (n_part, s_part), cell in res[res.fidelity != corpus.PARODY].groupby(["novel_part", "script_part"]):
        print(f"  {n_part:9s} x {s_part:8s} channelled {_auc(cell, 'channelled')}  profile {_auc(cell, 'profile')}"
              f"  |  " + "  ".join(f"{c[2:]} {_auc(cell, c)}" for c in zc))
    inf = res[res.pair_type != "control"]
    print("\nMean channel z and profile by fidelity, per cell")
    print(inf.groupby(["novel_part", "script_part", "fidelity"])[zc + PART_COLS + ["profile"]].mean().round(2).to_string())
    full = inf[(inf.novel_part == "full") & (inf.script_part == "full")].sort_values("profile", ascending=False)
    print("\nFull novel x full screenplay (channel z in control SDs; narr_order and narr_curves split z_narrative)")
    print(full[["screenplay", "fidelity"] + zc + PART_COLS + ["profile", "profile_lo", "profile_hi", "profile_p",
                                                 "peak_channel", "channelled_p"]].round(2).to_string(index=False))


INFLUENCE_KEYS = ["source", "target", "trad_A", "trad_B"]


def load_influence(variant: str, features=FEATURES, reversed_pairs: bool = False):
    """Window rows of the influence study (with order_rho, movement and cell), or None if absent.

    reversed_pairs adds the influence pairs scored target -> source (pair type "reversed").
    """
    parts = [("raw", "sequence.csv")] + ([("raw_reversed", "sequence_reversed.csv")] if reversed_pairs else [])
    keep = ["pair_key", "pair_type", *INFLUENCE_KEYS, *POSITION_COLS]
    frames = []
    for raw_dir, seq_name in parts:
        raw = sorted(glob.glob(os.path.join(INFLUENCE_DIR, variant, raw_dir, "*.csv")))
        if not raw:
            continue
        rows = pd.concat([pd.read_csv(p, usecols=lambda c: c in keep + list(features)) for p in raw], ignore_index=True)
        seq_path = os.path.join(INFLUENCE_DIR, variant, seq_name)
        if os.path.exists(seq_path):
            rows = rows.merge(pd.read_csv(seq_path)[["pair_key", "order_rho"]], on="pair_key", how="left")
        frames.append(rows)
    if not frames:
        return None
    rows = add_movement(pd.concat(frames, ignore_index=True))
    rows["cell"] = "full"
    return rows


def influence(variants) -> None:
    """Influence, translation and circle pairs against controls, with and without a movement effect."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for variant in variants:
        rows = load_influence(variant)
        if rows is None:
            print(f"No raw outputs for {variant}")
            continue
        results = score_novels(rows, ["cell"])
        res = pd.concat(results.values(), ignore_index=True).drop(columns="cell")
        out = os.path.join(OUT_DIR, f"influence_{variant}.csv")
        res.to_csv(out, index=False)
        print(f"\n===== influence {variant} =====  saved {out}")
        _influence_report(results)


def _influence_report(results) -> None:
    from influence.corpus import CONTROL, INFLUENCE

    any_res = next(iter(results.values()))
    print("\nPairs:", any_res.pair_type.value_counts().to_dict(),
          f"| smallest attainable p = {1 / (1 + any_res.n_controls.iloc[0]):.3f}")
    zc = _zcols(any_res)
    print("\nAUC, influence vs control pairs")
    for name, res in results.items():
        sub = res[res.pair_type.isin([INFLUENCE, CONTROL])]
        print(f"  baseline {name:8s} channelled {_auc(sub, 'channelled')}  profile {_auc(sub, 'profile')}  |  "
              + "  ".join(f"{c[2:]} {_auc(sub, c)}" for c in zc))

    print("\nMean profile by pair type and movement (same-movement pairs vs cross-movement pairs)")
    for name, res in results.items():
        t = res.assign(same=np.where(res.movement == "cross", "cross", "same"))
        print(f"  baseline {name}")
        print(t.pivot_table(index="pair_type", columns="same", values="profile", aggfunc="mean").round(2).to_string())

    cols = ["pair_key", "pair_type"] + zc + ["profile", "profile_lo", "profile_hi", "profile_p", "peak_channel",
                                             "channelled_p"]
    for name, res in results.items():
        print(f"\nInfluence, translation and circle pairs, baseline {name} (channel z in control SDs)")
        show = res[res.pair_type != CONTROL].sort_values(["pair_type", "profile"], ascending=[False, False])
        print(show[[c for c in cols if c in show]].round(2).to_string(index=False))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2 or sys.argv[1] not in ("iccc", "screenplays", "influence"):
        sys.exit("usage: python -m src.v2.run {iccc|screenplays|influence} [variant ...]")
    if sys.argv[1] == "iccc":
        iccc()
    elif sys.argv[1] == "influence":
        influence(sys.argv[2:] or ["masked"])
    else:
        screenplays(sys.argv[2:] or ["unmasked", "masked"])


if __name__ == "__main__":
    main()
