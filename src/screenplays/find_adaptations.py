import csv
import html
import json
import os
import re
import sys
import tarfile

from . import corpus

OUT_CSV = os.path.join(corpus.EXTERNAL_DIR, "scriptbase_gutenberg_matches.csv")


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", s.lower().replace("&", "and")).strip()
    return re.sub(r"^(the|a|an) ", "", s)


def _strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def load_pg_titles(path: str):
    """Normalised main title -> list of (id, title, authors, subjects), English texts only."""
    titles = {}
    with open(path, encoding="utf-8") as f:
        for b in csv.DictReader(f):
            if b["Language"] != "en" or b["Type"] != "Text":
                continue
            key = _norm(re.split(r"[;:]|\bor,", b["Title"])[0])
            if len(key) >= 4:
                titles.setdefault(key, []).append(
                    (b["Text#"], b["Title"].replace("\n", " "), b["Authors"], b["Subjects"])
                )
    return titles


def film_sources(archive: str):
    """(based_on, source categories, script size in KB) from a ScriptBase archive."""
    with tarfile.open(archive) as t:
        members = {m.name.rsplit("/", 1)[-1]: m for m in t.getmembers()}
        page = t.extractfile(members["wiki.html"]).read().decode("utf-8", "replace") if "wiki.html" in members else ""
        script_kb = members["script.txt"].size // 1024 if "script.txt" in members else 0

    m = re.search(r'"wgCategories":(\[.*?\])', page)
    cats = json.loads(m.group(1)) if m else []
    m = re.search(r"<th[^>]*>\s*Based on\s*</th>\s*<td[^>]*>(.*?)</td>", page, re.S | re.I)
    based_on = _strip_tags(m.group(1)) if m else ""
    source_cats = [c for c in cats if re.search(r"based on|adaptations of", c, re.I)]
    return based_on, source_cats, script_kb


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pg = load_pg_titles(corpus.PG_CATALOG)

    rows = []
    for fn in sorted(os.listdir(corpus.SCRIPTBASE_DIR)):
        if not fn.endswith(".tar.gz"):
            continue
        film = fn[: -len(".tar.gz")]
        based_on, source_cats, script_kb = film_sources(os.path.join(corpus.SCRIPTBASE_DIR, fn))
        if not (based_on or source_cats):
            continue

        # Candidate source titles: film title, infobox title, category titles
        phrases = {re.sub(r"\s*\(.*$", "", film)}
        if based_on:
            phrases.add(re.split(r"\s+by\s+|\s*\(", based_on)[0])
        phrases |= {re.sub(r"^Films based on (works by )?", "", c) for c in source_cats}
        context = _norm(based_on + " " + " ".join(source_cats))

        hits = []
        for p in phrases:
            for pg_id, title, authors, subjects in pg.get(_norm(p.strip(" \"'")), []):
                confirmed = _norm(authors.split(",")[0]) in context
                hits.append((not confirmed, int(pg_id), title, authors, subjects))
        if not hits:
            continue
        unconfirmed, pg_id, title, authors, subjects = min(hits)
        rows.append({
            "film": film, "script_kb": script_kb, "based_on": based_on,
            "source_cats": "; ".join(source_cats),
            "author_confirmed": not unconfirmed,
            "pg_id": pg_id, "pg_title": title, "pg_author": authors,
            "pg_fiction": bool(re.search(r"fiction|novel|stories", subjects, re.I)),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    confirmed = [r for r in rows if r["author_confirmed"]]
    print(f"{len(rows)} Gutenberg title matches, {len(confirmed)} author-confirmed -> {OUT_CSV}")
    for r in confirmed:
        print(f"  {r['film'][:40]:40s} PG {r['pg_id']:>6}  {r['pg_title'][:45]}")


if __name__ == "__main__":
    main()
