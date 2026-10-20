"""
test_engine.py - Rigorous unit tests for core math engine and image processing pipeline
"""

import numpy as np


def test_histogram_equalization_textbook():
    from core_engine import histogram_equalization
    counts = np.array([790, 1023, 850, 656, 329, 245, 122, 81])
    L = 8
    res = histogram_equalization(counts, L)
    
    assert res["n"] == 4096, f"Expected n=4096, got {res['n']}"
    expected_sk = np.array([1, 3, 5, 6, 6, 7, 7, 7])
    assert np.array_equal(res["s_k"], expected_sk), f"Expected s_k={expected_sk}, got {res['s_k']}"
    
    expected_eq_counts = np.array([0, 790, 0, 1023, 0, 850, 985, 448])
    assert np.array_equal(res["equalized_counts"], expected_eq_counts), (
        f"Expected {expected_eq_counts}, got {res['equalized_counts']}"
    )
    assert np.sum(res["equalized_counts"]) == 4096, "Sum of equalized counts must equal total n"
    print("✓ test_histogram_equalization_textbook passed!")


def test_histogram_matching_logic():
    from core_engine import histogram_matching
    src_counts = np.array([790, 1023, 850, 656, 329, 245, 122, 81])
    # Target distribution concentrated in high values
    tgt_counts = np.array([0, 0, 0, 0, 1024, 2048, 1024, 0])
    L = 8

    # Test standard nearest rule
    res_nearest = histogram_matching(src_counts, tgt_counts, L, mapping_rule="nearest")
    assert np.sum(res_nearest["matched_counts"]) == 4096, "Matched count sum must match source sum"
    assert len(res_nearest["inverse_mapping"]) == 8

    # Test nearest <= rule (Floor match)
    res_le = histogram_matching(src_counts, tgt_counts, L, mapping_rule="nearest_le")
    assert np.sum(res_le["matched_counts"]) == 4096, "Matched count sum must match source sum"
    assert len(res_le["inverse_mapping"]) == 8

    print("✓ test_histogram_matching_logic (nearest & nearest_le) passed!")


def test_color_space_and_downsampling():
    from image_processor import (
        rgb_to_ycrcb,
        ycrcb_to_rgb,
        downsample_channel,
        upsample_channel
    )
    # Test random RGB image
    test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    y, cr, cb = rgb_to_ycrcb(test_img)
    recon = ycrcb_to_rgb(y, cr, cb)
    
    diff = np.abs(test_img.astype(int) - recon.astype(int))
    max_err = np.max(diff)
    assert max_err <= 3, f"Color space round-trip max error too high: {max_err}"

    # Downsampling & Upsampling
    for L in [2, 4, 8, 16, 32, 64, 128, 256]:
        y_q = downsample_channel(y, L)
        assert np.min(y_q) >= 0 and np.max(y_q) < L, f"Quantized values out of [0, {L-1}]"
        y_up = upsample_channel(y_q, L)
        assert np.min(y_up) >= 0 and np.max(y_up) <= 255
    print("✓ test_color_space_and_downsampling passed!")


def test_image_processing_pipeline():
    from image_processor import process_image_equalization, process_image_matching
    from samples import generate_sample_low_contrast, generate_sample_high_contrast

    low_c = generate_sample_low_contrast()
    high_c = generate_sample_high_contrast()

    # Equalization
    eq_res = process_image_equalization(low_c, L=8)
    assert eq_res["output_image"].shape == low_c.shape
    assert np.sum(eq_res["equalized_counts"]) == low_c.shape[0] * low_c.shape[1]

    # Matching with different dimensions
    src = low_c
    tgt = high_c[:180, :200]  # different dimensions
    match_res = process_image_matching(src, tgt, L=16)
    assert match_res["output_image"].shape == src.shape
    assert np.sum(match_res["matched_counts"]) == src.shape[0] * src.shape[1]
    print("✓ test_image_processing_pipeline passed!")


if __name__ == "__main__":
    test_histogram_equalization_textbook()
    test_histogram_matching_logic()
    test_color_space_and_downsampling()
    test_image_processing_pipeline()
    print("All tests passed successfully!")
