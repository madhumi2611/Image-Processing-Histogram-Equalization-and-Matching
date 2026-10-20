"""
core_engine.py - Custom Mathematical Engine for Histogram Processing
Implements:
1. Histogram Equalization from scratch
2. Histogram Matching (Specification) from scratch
"""

from typing import Dict, Any, Tuple
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def histogram_equalization(counts: np.ndarray, L: int) -> Dict[str, Any]:
    """
    Computes Histogram Equalization following standard textbook steps:
    1. Total Pixels (n): Sum all pixel counts, n = sum(n_k).
    2. PDF: P(r_k) = n_k / n.
    3. CDF: S_k = sum_{j=0}^k P(r_j).
    4. Scaling: (L - 1) * S_k.
    5. Rounding: s_k = round((L - 1) * S_k).
    6. Reassignment: Combine pixel counts mapping to same level.
    
    Returns a dictionary with:
    - n: total pixel count
    - r_k: array of gray levels 0 .. L-1
    - n_k: original counts
    - pr_k: PDF array
    - cdf: cumulative sum array S_k
    - scaled_cdf: (L-1) * S_k
    - s_k: mapped rounded levels (int)
    - equalized_counts: new counts per gray level 0 .. L-1
    - mapping: array mapping each input level k to equalized level s_k
    - df_steps: pandas DataFrame of calculation steps
    - df_result: pandas DataFrame of equalized histogram
    """
    counts = np.asarray(counts, dtype=np.int64)
    if len(counts) != L:
        raise ValueError(f"Counts length ({len(counts)}) must match L ({L})")

    n = int(np.sum(counts))
    r_k = np.arange(L, dtype=np.int32)

    if n == 0:
        pr_k = np.zeros(L, dtype=np.float64)
        cdf = np.zeros(L, dtype=np.float64)
        scaled_cdf = np.zeros(L, dtype=np.float64)
        s_k = np.zeros(L, dtype=np.int32)
        equalized_counts = np.zeros(L, dtype=np.int64)
    else:
        pr_k = counts / n
        cdf = np.cumsum(pr_k)
        scaled_cdf = (L - 1) * cdf
        
        # Rounding rule: round half up / standard rounding
        # np.floor(x + 0.5) ensures consistent round-half-up
        s_k = np.floor(scaled_cdf + 0.5).astype(np.int32)
        s_k = np.clip(s_k, 0, L - 1)

        # Reassignment
        equalized_counts = np.zeros(L, dtype=np.int64)
        for k in range(L):
            target_level = s_k[k]
            equalized_counts[target_level] += counts[k]

    # Build calculation DataFrame
    if HAS_PANDAS:
        df_steps = pd.DataFrame({
            "Level (k)": r_k,
            "Gray Level (r_k)": r_k,
            "Pixel Count (n_k)": counts,
            "PDF P(r_k)": np.round(pr_k, 5),
            "CDF (S_k)": np.round(cdf, 5),
            f"(L-1) * S_k ({L-1}*S_k)": np.round(scaled_cdf, 4),
            "Equalized Level (s_k)": s_k
        })

        df_result = pd.DataFrame({
            "Gray Level (s)": r_k,
            "Equalized Count (n_s)": equalized_counts,
            "Equalized PDF P(s)": np.round(equalized_counts / n, 5) if n > 0 else np.zeros(L)
        })
    else:
        df_steps = None
        df_result = None

    return {
        "n": n,
        "L": L,
        "r_k": r_k,
        "n_k": counts,
        "pr_k": pr_k,
        "cdf": cdf,
        "scaled_cdf": scaled_cdf,
        "s_k": s_k,
        "equalized_counts": equalized_counts,
        "mapping": s_k,
        "df_steps": df_steps,
        "df_result": df_result
    }


def histogram_matching(
    source_counts: np.ndarray,
    target_counts: np.ndarray,
    L: int,
    mapping_rule: str = "nearest"
) -> Dict[str, Any]:
    """
    Computes Histogram Matching / Specification:
    1. Equalize Source: find s_k for source histogram.
    2. Equalize Target: find v_q for target/specified histogram.
    3. Inverse Mapping (q = G^{-1}(s_k)):
       - "nearest": argmin_q |v_q - s_k| (Gonzalez & Woods standard)
       - "nearest_le": closest v_q such that v_q <= s_k (Floor match)
       - "smallest_ge": smallest q such that v_q >= s_k (Ceiling match)
    4. Final Redistribution: Map original pixel counts to target gray levels q.
    """
    source_res = histogram_equalization(source_counts, L)
    target_res = histogram_equalization(target_counts, L)

    s_k = source_res["s_k"]  # Equalized source values for each k in 0..L-1
    v_q = target_res["s_k"]  # Equalized target values for each q in 0..L-1

    inverse_mapping = np.zeros(L, dtype=np.int32)
    chosen_vq = np.zeros(L, dtype=np.int32)
    abs_diffs = np.zeros(L, dtype=np.int32)

    for k in range(L):
        curr_sk = s_k[k]

        if mapping_rule == "nearest_le":
            # Find closest v_q such that v_q <= curr_sk
            valid_mask = (v_q <= curr_sk)
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                diffs_le = curr_sk - v_q[valid_indices]
                # Pick valid index that minimizes (curr_sk - v_q)
                best_sub_idx = int(np.argmin(diffs_le))
                best_q = int(valid_indices[best_sub_idx])
            else:
                # If no v_q <= curr_sk, fall back to index 0
                best_q = 0
        elif mapping_rule == "smallest_ge":
            # Smallest q such that v_q >= curr_sk
            valid_mask = (v_q >= curr_sk)
            if np.any(valid_mask):
                best_q = int(np.where(valid_mask)[0][0])
            else:
                best_q = L - 1
        else:
            # Standard nearest: argmin |v_q - curr_sk|
            diffs = np.abs(v_q - curr_sk)
            best_q = int(np.argmin(diffs))

        inverse_mapping[k] = best_q
        chosen_vq[k] = v_q[best_q]
        abs_diffs[k] = abs(int(v_q[best_q]) - int(curr_sk))

    # Reassign source counts to matched levels
    matched_counts = np.zeros(L, dtype=np.int64)
    for k in range(L):
        matched_q = inverse_mapping[k]
        matched_counts[matched_q] += source_counts[k]

    n_source = source_res["n"]

    # Routing DataFrame
    if HAS_PANDAS:
        df_routing = pd.DataFrame({
            "Source Gray Level (r_k)": np.arange(L),
            "Source Count (n_k)": source_counts,
            "Source Equalized (s_k)": s_k,
            "Matched Target Level (q)": inverse_mapping,
            "Target Equalized Value (v_q)": chosen_vq,
            "Distance |v_q - s_k|": abs_diffs
        })

        # Final result DataFrame
        df_result = pd.DataFrame({
            "Gray Level (z_q)": np.arange(L),
            "Original Source Count": source_counts,
            "Target Specified Count": target_counts,
            "Final Matched Count": matched_counts,
            "Matched PDF P(z_q)": np.round(matched_counts / n_source, 5) if n_source > 0 else np.zeros(L)
        })
    else:
        df_routing = None
        df_result = None

    return {
        "L": L,
        "source_res": source_res,
        "target_res": target_res,
        "inverse_mapping": inverse_mapping,
        "matched_counts": matched_counts,
        "df_routing": df_routing,
        "df_result": df_result
    }
