# Sense and Similarity: code

Code for *Sense and Similarity: A Computational Study of Novel-to-Screenplay Adaptation* (DH2027).
It compares 22 screenplays with the 19 literary works they adapt across five channels of similarity
(lexical, semantic, conceptual, structural, narrative), calibrated against unrelated control pairs,
and compares the dialogue route (novel dialogue vs. screenplay dialogue) with the route from
narration to action.

All commands are run from this directory.

## Layout

| Path | Contents |
|---|---|
| `main.py`, `src/features_runner.py`, `src/features/`, `src/preprocess.py`, `src/align.py`, `src/similarity.py` | Feature pipeline: windows, DTW alignment, feature extraction |
| `src/screenplays/` | Corpus, screenplay parser, dialogue/narration split, experiment design, narrative order, analyses |
| `src/v2/` | Calibration against the controls (additive ridge model) and channel scores |
| `paper/` | Scripts for the paper's figure, tests and robustness checks |

## Setup

```
python -m venv venv
venv/bin/pip install -r requirements.txt
```

NLTK data is downloaded on first use. OCR of the 1994 *Little Women* screenplay needs the
`tesseract` binary (`sudo apt install tesseract-ocr`).

## Data

Texts are not included. ScriptBase screenplays and Project Gutenberg sources are downloaded into
`external/`:

```
python -m src.screenplays.fetch
```

The two *Little Women* screenplays are not in ScriptBase and must be obtained separately as PDFs:

```
python -m src.screenplays.pdf_to_text little-women-by-greta-gerwig.pdf little_women_2019
python -m src.screenplays.ocr_to_text "LITTLE+WOMEN_+Robin+Swicord.pdf" little_women_1994
```

`python -m src.screenplays.find_adaptations` lists the ScriptBase/Gutenberg candidates from which
the corpus (`src/screenplays/corpus.py`) was curated by hand.

## Reproducing the paper

Feature runs are slow; the experiment scripts accept `--shard i/n` to run in parallel processes.

1. Base runs (windows represented by their first 256 word pieces; also provide the text components
   and narrative order):

   ```
   python -m src.screenplays.experiment --design full --mask-names
   python -m src.screenplays.experiment --design full
   python -m src.screenplays.sequence --design full --mask-names
   python -m src.screenplays.sequence --design full
   ```

2. Main configuration (pooled window embeddings) and the order-free alignment:

   ```
   python paper/sensitivity.py --variant pooled --shard 0/1
   python paper/sensitivity.py --variant pooled --shard 0/1 --unmasked
   python paper/sensitivity.py --variant free --shard 0/1
   ```

   (`--shard 0/1` runs everything in one process; use `0/4` ... `3/4` for four.)

3. Calibrated scores (written to `out_v2/`):

   ```
   python -m src.v2.run screenplays full_masked full_unmasked full_masked_pooled full_unmasked_pooled full_masked_free
   ```

4. Figure, paired tests and patterns across adaptations:

   ```
   python paper/make_results.py
   python paper/sensitivity.py --compare
   python -m src.screenplays.profile_structure
   python -m src.screenplays.order_reliability
   ```

5. Robustness checks:

   ```
   python paper/shared_runs.py                       # share of verbatim runs per screenplay
   NORUNS_POOLED=1 python paper/noruns.py --stats    # then --preprocess, --shard i/n, --sequence,
                                                     # src.v2.run on full_masked_noruns_pooled, --compare
   python paper/control_seeds.py --shard 0/4         # ... 3/4, then --sequence, then --score --seeds 20
   python paper/narrative_no_order.py                # route comparison without narrative order
   python -m src.screenplays.sw_baseline --variant masked && python -m src.screenplays.sw_baseline --score
   ```

Results of the scripts in `paper/` are written to `paper/`; intermediate outputs go to
`out_screenplays/` and `out_v2/`, and preprocessing caches to `cache/`.
