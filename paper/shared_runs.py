import csv
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from src.screenplays import corpus, novel, parser  # noqa: E402

K = 6
OUT = os.path.join("paper", "shared_runs.csv")
WORD = re.compile(r"[a-z']+")


def words(text):
    return WORD.findall(text.lower().replace("’", "'"))


def coverage(source, target):
    """Share of target words inside a run of at least K words that occurs in source."""
    grams = {tuple(source[i:i + K]) for i in range(len(source) - K + 1)}
    covered = set()
    for i in range(len(target) - K + 1):
        if tuple(target[i:i + K]) in grams:
            covered.update(range(i, i + K))
    return len(covered) / max(len(target), 1)


def main():
    rows = []
    for a in corpus.included():
        n = novel.load(a.novel_path)
        sp = parser.load(a.film)
        rows.append({
            "film": a.film,
            "dialogue": round(coverage(words(n.text((novel.DIALOGUE,))), words(sp.text((parser.DIALOGUE,)))), 3),
            "action": round(coverage(words(n.text((novel.NARRATION,))), words(sp.text((parser.ACTION,)))), 3),
        })
    rows.sort(key=lambda r: -r["dialogue"])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["film", "dialogue", "action"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['dialogue']:.3f}  {r['action']:.3f}  {r['film']}")


if __name__ == "__main__":
    main()
