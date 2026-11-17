import os
import sys

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "src")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402

import control_seeds as cs  # noqa: E402
from src.v2 import run  # noqa: E402
from src.screenplays import experiment  # noqa: E402


def main(n_seeds=20):
    pairs, names = cs.setup()
    comps = experiment.build_components(pairs, names)
    seqs = [os.path.join(cs.MAIN, "sequence.csv"), os.path.join(cs.POOL, "sequence.csv")]
    seq = pd.concat([pd.read_csv(s) for s in seqs if os.path.exists(s)])
    seq = seq.drop_duplicates(subset=run.SCREENPLAY_KEYS)[run.SCREENPLAY_KEYS + ["order_rho"]]
    full = dict(run.CHANNELS)
    no_order = {k: ({f: s for f, s in v.items() if f != "order_rho"} if k == "narrative" else v) for k, v in full.items()}
    rows = []
    for seed in range(n_seeds + 1):
        for label, ch in (("with order", full), ("without order", no_order)):
            run.CHANNELS = ch
            res = cs.route_test(cs.score_seed(seed, comps, seq))
            rows.append({"seed": seed, "narrative_channel": label, **res})
            print(seed, label, {k: round(float(v), 3) for k, v in res.items() if k.startswith(("narrative", "profile"))}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("paper", "narrative_no_order.csv"), index=False)
    lines = []
    for label, g in df.groupby("narrative_channel", sort=False):
        s0, alt = g[g.seed == 0].iloc[0], g[g.seed > 0]
        for c in ("narrative", "profile"):
            lines.append(f"{label:14s} {c:9s} seed 0 {s0[c]:+.2f} (p {s0[c + '_p']:.3f}) | other seeds {alt[c].min():+.2f} to "
                         f"{alt[c].max():+.2f}, positive {(alt[c] > 0).sum()}/{len(alt)}, p<.05 {(alt[c + '_p'] < .05).sum()}/{len(alt)}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join("paper", "narrative_no_order_results.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
