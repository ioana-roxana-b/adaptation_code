import os
from dataclasses import dataclass
from typing import Optional

EXTERNAL_DIR = "external"
SCRIPTBASE_DIR = os.path.join(EXTERNAL_DIR, "scriptbase_alpha")
GUTENBERG_DIR = os.path.join(EXTERNAL_DIR, "gutenberg")
MANUAL_DIR = os.path.join(EXTERNAL_DIR, "manual_screenplays")
PG_CATALOG = os.path.join(EXTERNAL_DIR, "pg_catalog.csv")
DATASET_DIR = "dataset"

# Fidelity categories, from closest to furthest from the source
CLOSE, LOOSE, TRANSPOSED, PARODY = "close", "loose", "transposed", "parody"


@dataclass(frozen=True)
class Adaptation:
    novel_key: str            # "<author>_<work>", as in main.py
    pg_id: int                # Project Gutenberg ebook number of the source
    film: str                 # ScriptBase archive name (without .tar.gz) or manual file stem
    year: int
    fidelity: str
    include: bool = True
    source: str = "scriptbase"  # "scriptbase" or "manual"
    note: str = ""

    @property
    def novel_path(self) -> str:
        """Cleaned text in dataset/ if present, else the raw Gutenberg download."""
        local = DATASET_NOVELS.get(self.novel_key)
        if local and os.path.exists(os.path.join(DATASET_DIR, local)):
            return os.path.join(DATASET_DIR, local)
        return os.path.join(GUTENBERG_DIR, f"pg{self.pg_id}.txt")

    @property
    def script_path(self) -> str:
        if self.source == "manual":
            return os.path.join(MANUAL_DIR, f"{self.film}.txt")
        return os.path.join(SCRIPTBASE_DIR, f"{self.film}.tar.gz")


# Novels already cleaned by hand in dataset/
DATASET_NOVELS = {
    "austen_pride": "Pride_and_Prejudice_Jane_Austen.txt",
    "shelley_frankenstein": "Frankenstein_Mary_Shelley.txt",
    "wharton_age": "The_Age_Edith_Wharton.txt",
    "homer_oddyssey_butler": "Odyssey_Homer_butler.txt",
}

# Body of a Gutenberg download where novel.py's automatic start is wrong or
# back matter follows: (regex of the first body paragraph, regex of the first
# paragraph after the body); None keeps the default
NOVEL_BODY = {
    4558: (r"Since the days of Adam, there has been hardly a mischief", None),   # contents fooled the start
    18883: (r"Lieutenant Sutch was the first of General Feversham", r"By A\. E\. W\. Mason$"),  # publisher's ads
    2759: (None, r"End of The Man in the Iron Mask\."),                          # translator's notes
    33: (r"A throng of bearded men", None),     # skips "The Custom-House", the introductory sketch
    35: (r"The Time Traveller \(for so it will be convenient", None),            # title and contents
    41: (r"In the bosom of one of those spacious coves", None),                  # title and epigraph
    # Controls of the influence study (src/influence)
    15408: (r"THE GOOD ANNA$", None),                                            # contents
    58340: (r"PART I$", r"Transcriber’s Note"),                                  # author's prologue; notes
    217: (r"“The Bottoms” succeeded", None),                                       # contents
    3019: (None, r"_?The Mayflower Press"),                                      # printer and errata list
    416: (r"The writer, an old man with a white mustache", None),                # contents fooled the start
    1424: (r"Having, out of friendship for the family", r"GLOSSARY$"),           # Ritchie's introduction
    1237: (r"Mme\. Vauquer \(_?nee_? de Conflans\)", r"ADDENDUM$"),              # title page; character index
    # Direction test set (influence.corpus.TEST_TEXTS)
    131: (r"(\{\d+\} )?As I walk.d through the wilderness of this world", None),  # transcriber's notes, apology
    26: (r"Of Man’s first disobedience", r"Transcriber’s Notes"),               # contents; PG introduction
    1251: (r"It befell in the days of Uther Pendragon", r"GLOSSARY$"),          # contents, introduction; glossary
    121: (None, r"Northanger Abbey was written in 1797"),                      # editor's note
    9611: (r"As it is possible the mere English reader", None),                # contents
    6593: (r"An author ought to consider himself, not as a gentleman", None),  # contents
}

PILOT = ["austen_pride", "shelley_frankenstein", "alcott_women"]

ADAPTATIONS = [
    # --- Pilot ---
    Adaptation("austen_pride", 1342, "Pride & Prejudice (2005 film)", 2005, CLOSE),
    Adaptation("shelley_frankenstein", 84, "Mary Shelleys Frankenstein (film)", 1994, CLOSE,
               note="2nd revised draft, Feb 1993"),
    Adaptation("alcott_women", 514, "little_women_2019", 2019, LOOSE, source="manual",
               note="Gerwig; from PDF via pdf_to_text; non-linear, red = 1861 timeline"),
    Adaptation("alcott_women", 514, "little_women_1994", 1994, CLOSE, source="manual",
               note="Swicord, first draft rev. May 1993; scanned PDF, via ocr_to_text"),

    # --- Close ---
    Adaptation("austen_sense", 161, "Sense and Sensibility (film)", 1995, CLOSE),
    Adaptation("bronte_jane", 1260, "Jane Eyre (2011 film)", 2011, CLOSE),
    Adaptation("thackeray_barry", 4558, "Barry Lyndon", 1975, CLOSE),
    Adaptation("wharton_age", 541, "The Age of Innocence (1993 film)", 1993, CLOSE,
               note="transcript format: flush-left cues, [bracketed] action"),
    Adaptation("mason_feathers", 18883, "The Four Feathers (2002 film)", 2002, CLOSE,
               note="inconsistent cue capitalisation"),
    Adaptation("wells_time", 35, "The Time Machine (1960 film)", 1960, CLOSE),
    Adaptation("baum_oz", 55, "The Wizard of Oz (1939 film)", 1939, CLOSE),

    # --- Loose / stylised ---
    Adaptation("tolstoy_anna", 1399, "Anna Karenina (2012 film)", 2012, LOOSE,
               note="source is an English translation"),
    Adaptation("dumas_musketeers", 1257, "The Three Musketeers (1993 film)", 1993, LOOSE,
               note="source is an English translation"),
    Adaptation("dumas_mask", 2759, "The Man in the Iron Mask (1998 film)", 1998, LOOSE,
               note="source is an English translation"),
    Adaptation("cooper_mohicans", 940, "The Last of the Mohicans (1992 film)", 1992, LOOSE,
               note="also draws on the 1936 film"),
    Adaptation("wells_time", 35, "The Time Machine (2002 film)", 2002, LOOSE),
    Adaptation("irving_hollow", 41, "Sleepy Hollow (film)", 1999, LOOSE, note="short story source"),

    # --- Transposed / modernised ---
    Adaptation("conrad_darkness", 219, "Apocalypse Now", 1979, TRANSPOSED, note="novella source"),
    Adaptation("wells_worlds", 36, "War of the Worlds (2005 film)", 2005, TRANSPOSED),
    Adaptation("hawthorne_letter", 33, "Easy A", 2010, TRANSPOSED),
    Adaptation("homer_oddyssey_butler", 1727, "O Brother, Where Art Thou_", 2000, TRANSPOSED,
               note="links to the ICCC Odyssey pairs"),
    Adaptation("austen_emma", 158, "Clueless (film)", 1995, TRANSPOSED, include=False,
               note="file starts with credits; check whether it is a usable script"),

    # --- Parody ---
    Adaptation("shelley_frankenstein", 84, "Young Frankenstein", 1974, PARODY),

    # --- Excluded after inspection ---
    Adaptation("stoker_dracula", 345, "Bram Stokers Dracula", 1992, CLOSE, include=False,
               note="only ~9k words; fragment"),
    Adaptation("stoker_dracula", 345, "Dracula (1931 film)", 1931, LOOSE, include=False,
               note="~6k words; via the stage play"),
    Adaptation("hugo_miserables", 135, "Les Miserables (2012 film)", 2012, LOOSE, include=False,
               note="adapts the musical, not the novel"),
    Adaptation("hawthorne_letter", 33, "The Scarlet Letter (1926 film)", 1926, CLOSE, include=False,
               note="silent film; intertitles only"),
]


def included(pilot_only: bool = False):
    items = [a for a in ADAPTATIONS if a.include]
    if pilot_only:
        items = [a for a in items if a.novel_key in PILOT]
    return items


def get(film: str) -> Optional[Adaptation]:
    return next((a for a in ADAPTATIONS if a.film == film), None)
