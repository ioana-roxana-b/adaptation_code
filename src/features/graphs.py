import math
from collections import Counter
from typing import Any, Dict, List, Tuple
import numpy as np

from preprocess import Window
from similarity import cosine_counter, js_divergence_from_counters

def compute_degree_distribution(edges: Counter) -> Counter:
    """
    Converts a sequence of directed edges into an undirected node degree distribution.
    This captures the 'hub' structure of the logic (e.g., how heavily the text 
    relies on central function words like 'that' or 'which').
    """
    degrees = Counter()
    for (u, v), weight in edges.items():
        degrees[u] += weight
        degrees[v] += weight
    return degrees

def compute(
    wA: List[Window],
    wB: List[Window],
    aligned: List[Tuple[int, int]],
    ctx: Dict[str, Any],
) -> Dict[str, List[float]]:
    """
    Computes graph-based topological features.
    
    Fixed for ICCC 2026:
    Shifts from "Exact Edge Matching" to "Macroscopic Topology Matching".
    """
    out: Dict[str, List[float]] = {
        "wan_fw_cosine": [],       # Legacy surface overlap
        "wan_degree_jsd": [],      # NEW: Topological hub divergence
        "wan_density_diff": []     # NEW: Network density difference
    }

    for i, j in aligned:
        fwA = wA[i].fw_edges or Counter()
        fwB = wB[j].fw_edges or Counter()
        
        # 1. Surface Edge Overlap (Will likely cluster near the mean for all English texts)
        out["wan_fw_cosine"].append(cosine_counter(fwA, fwB) if fwA and fwB else 0.0)

        # 2. Topological Hub Structure (The crucial Tardean structural metric)
        degA = compute_degree_distribution(fwA)
        degB = compute_degree_distribution(fwB)
        
        # We want to know if the SHAPE of the network is the same.
        # Lower JSD means higher structural similarity.
        out["wan_degree_jsd"].append(js_divergence_from_counters(degA, degB))

        # 3. Macroscopic Topology (Density)
        wmA = wA[i].wan_metrics or {}
        wmB = wB[j].wan_metrics or {}
        
        densA = wmA.get("density", 0.0)
        densB = wmB.get("density", 0.0)
        # Difference in network density (how inter-connected the function words are)
        out["wan_density_diff"].append(abs(densA - densB))

    return out