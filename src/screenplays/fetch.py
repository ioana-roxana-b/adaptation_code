import argparse
import json
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import corpus

SCRIPTBASE_TREE = "https://api.github.com/repos/EdinburghNLP/scriptbase/git/trees/master?recursive=1"
SCRIPTBASE_RAW = "https://raw.githubusercontent.com/EdinburghNLP/scriptbase/master/"
PG_CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
PG_TEXT_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def _download(url: str, dest: str, retries: int = 3) -> None:
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest + ".part", "wb") as f:
                f.write(r.read())
            os.replace(dest + ".part", dest)
            return
        except Exception as e:
            last = e
    raise last


def safe_filename(name: str) -> str:
    """ScriptBase titles may contain characters Windows does not allow."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def fetch_scriptbase(workers: int = 16) -> None:
    os.makedirs(corpus.SCRIPTBASE_DIR, exist_ok=True)
    with urllib.request.urlopen(SCRIPTBASE_TREE, timeout=60) as r:
        tree = json.load(r)["tree"]
    items = [x for x in tree if x["path"].startswith("scriptbase_alpha/") and x["path"].endswith(".tar.gz")]

    def fetch(item):
        name = item["path"].split("/", 1)[1]
        dest = os.path.join(corpus.SCRIPTBASE_DIR, safe_filename(name))
        if os.path.exists(dest) and os.path.getsize(dest) == item["size"]:
            return "cached"
        # Non-ASCII titles are stored decomposed on GitHub
        for form in ("NFC", "NFD"):
            url = SCRIPTBASE_RAW + urllib.parse.quote(unicodedata.normalize(form, item["path"]))
            try:
                _download(url, dest, retries=2)
                return "downloaded"
            except Exception:
                continue
        print(f"  FAILED: {name}")
        return "failed"

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(fetch, items))
    print(f"ScriptBase-alpha: {results.count('downloaded')} downloaded, "
          f"{results.count('cached')} already present, {results.count('failed')} failed")


def fetch_gutenberg() -> None:
    os.makedirs(corpus.GUTENBERG_DIR, exist_ok=True)
    if not os.path.exists(corpus.PG_CATALOG):
        _download(PG_CATALOG_URL, corpus.PG_CATALOG)
        print(f"Saved {corpus.PG_CATALOG}")

    for pg_id in sorted({a.pg_id for a in corpus.ADAPTATIONS}):
        dest = os.path.join(corpus.GUTENBERG_DIR, f"pg{pg_id}.txt")
        if os.path.exists(dest):
            continue
        _download(PG_TEXT_URL.format(id=pg_id), dest)
        print(f"Saved {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-scriptbase", action="store_true")
    ap.add_argument("--skip-gutenberg", action="store_true")
    args = ap.parse_args()

    if not args.skip_gutenberg:
        fetch_gutenberg()
    if not args.skip_scriptbase:
        fetch_scriptbase()

    missing = [a.film for a in corpus.included() if not os.path.exists(a.script_path)]
    if missing:
        print("Screenplays still missing (obtain manually):")
        for film in missing:
            print(f"  {film}  ->  {corpus.get(film).script_path}")


if __name__ == "__main__":
    main()
