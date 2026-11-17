from typing import Dict

CHANNELS: Dict[str, Dict[str, int]] = {
    # Trajectories over the ordered windows, and the order in which the material is presented
    "narrative": {
        "narr_sentiment_arc_corr": +1,
        "narr_tone_arc": +1,
        "narr_character_arc": +1,
        "narr_action_arc": +1,
        "narr_complexity_arc": +1,
        "narr_register_arc": +1,
        "narr_global_arc_cosine": +1,
        "narr_func_dtw_cost": -1,
        "order_rho": +1,
    },
    # Sentence construction and function-word style
    "structural": {
        "pos_trigram_cosine": +1,
        "pos4_cosine": +1,
        "fw_unigram_cosine": +1,
        "wan_fw_cosine": +1,
    },
    # Meaning independent of wording
    "semantic": {
        "sbert_cosine": +1,
    },
    # Kinds of things talked about: semantic classes, concreteness, entity types
    "conceptual": {
        "supersense_cosine": +1,
        "concreteness_sim": +1,
        "entity_type_jsd": -1,
    },
    # Wording carried over: shared content vocabulary, weighted and unweighted
    "lexical": {
        "content_lex_jaccard": +1,
        "content_tfidf_cosine": +1,
        "idf_content_jaccard": +1,
        "lex_unigram_jsd": -1,
        "char4_jsd": -1,
        "lsa_cosine": +1,
    },
}

# The narrative channel split for reporting only (not a channel, not in any score):
# whether the material comes in the source's order, and whether the trajectories
# match position by position, which they do not when a part is expanded or cut
NARRATIVE_PARTS: Dict[str, Dict[str, int]] = {
    "order": {"order_rho": +1},
    "curves": {f: s for f, s in CHANNELS["narrative"].items() if f != "order_rho"},
}

# ICCC channel features left out, and why
EXCLUDED: Dict[str, str] = {
    "topic_cosine": "near 1e-4 for almost all pairs (per-pair LDA separates the two texts); heavy-tailed",
    "rare_content_jaccard": "zero for almost every window pair",
    "lex_jaccard": "dominated by function words; duplicates the syntactic channel",
    "lex_kl_surprise": "same quantity as lex_unigram_jsd, asymmetric",
    "punct_cosine": "measures the medium (screenplay punctuation), not retained material",
    "supersense_jsd": "same distribution as supersense_cosine",
    "narr_mean_arc_corr": "mean of the arc correlations, which are in the channel individually",
    "narr_retention_composite": "composite of other narrative features",
    "sentiment_arc_cosine": "same sentiment arc as narr_sentiment_arc_corr",
    "narr_volatility_sim": "compares one summary number per text (window-to-window change), not trajectories",
    "narr_diversity_sim": "compares one summary number per text (spread of window profiles), not trajectories",
    "narr_rhythm_autocorr_sim": "compares one summary number per text (autocorrelation), not trajectories",
    "sentlen_ts_cosine": "uncentred cosine: compares mean sentence length rather than the shape of its trajectory",
    "order_local": "scene-to-scene jitter of best matches; near chance (0.50 vs 0.56) in the pilot",
}

FEATURES = [f for fs in CHANNELS.values() for f in fs]
