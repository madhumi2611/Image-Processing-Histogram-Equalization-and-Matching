"""
image_processor.py - Image Processing Pipeline for Histogram Equalization & Matching
Handles:
- RGB to YCrCb color space conversion
- Luminance (Y channel) extraction and downsampling to L gray levels
- Integration with custom core math engine
- Reassignment, rescaling, and color reconstruction
- Grayscale and RGB image support
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from core_engine import histogram_equalization, histogram_matching


def rgb_to_ycrcb(rgb_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts RGB uint8 image to Y, Cr, Cb channels.
    Uses OpenCV if available, or precise BT.601 formula.
    """
    if HAS_OPENCV:
        ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
        return ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    
    # Pure NumPy BT.601 implementation
    r = rgb_img[:, :, 0].astype(np.float64)
    g = rgb_img[:, :, 1].astype(np.float64)
    b = rgb_img[:, :, 2].astype(np.float64)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0

    y = np.clip(np.round(y), 0, 255).astype(np.uint8)
    cr = np.clip(np.round(cr), 0, 255).astype(np.uint8)
    cb = np.clip(np.round(cb), 0, 255).astype(np.uint8)
    return y, cr, cb


def ycrcb_to_rgb(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """
    Reconstructs RGB uint8 image from Y, Cr, Cb channels.
    """
    if HAS_OPENCV:
        ycrcb = cv2.merge([y, cr, cb])
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        return rgb

    y_f = y.astype(np.float64)
    cr_f = cr.astype(np.float64) - 128.0
    cb_f = cb.astype(np.float64) - 128.0

    r = y_f + 1.402 * cr_f
    g = y_f - 0.714136 * cr_f - 0.344136 * cb_f
    b = y_f + 1.772 * cb_f

    rgb = np.stack([
        np.clip(np.round(r), 0, 255).astype(np.uint8),
        np.clip(np.round(g), 0, 255).astype(np.uint8),
        np.clip(np.round(b), 0, 255).astype(np.uint8)
    ], axis=-1)
    return rgb


def downsample_channel(channel: np.ndarray, L: int) -> np.ndarray:
    """
    Downsamples an 8-bit channel (0..255) to L discrete levels (0..L-1).
    Formula: floor(channel * L / 256), clipped to [0, L-1].
    """
    scaled = np.floor((channel.astype(np.float64) * L) / 256.0).astype(np.int32)
    return np.clip(scaled, 0, L - 1)


def upsample_channel(channel_quantized: np.ndarray, L: int) -> np.ndarray:
    """
    Upsamples a quantized channel (0..L-1) back to 8-bit range (0..255).
    Formula: round(channel_quantized * 255 / (L - 1)).
    """
    if L <= 1:
        return np.zeros_like(channel_quantized, dtype=np.uint8)
    scaled = np.round((channel_quantized.astype(np.float64) * 255.0) / (L - 1))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def get_channel_histogram(quantized_channel: np.ndarray, L: int) -> np.ndarray:
    """
    Computes frequency counts of gray levels 0..L-1.
    """
    counts = np.bincount(quantized_channel.ravel(), minlength=L)
    return counts[:L].astype(np.int64)


def process_image_equalization(image: np.ndarray, L: int) -> Dict[str, Any]:
    """
    Executes Histogram Equalization on an image:
    - If RGB: Converts to YCrCb, extracts Y, downsamples to L, equalizes, scales up, merges.
    - If Grayscale: Extracts 2D array, downsamples to L, equalizes, scales up.
    
    Returns processed image, original & equalized counts, metadata, and calculation tables.
    """
    is_rgb = (image.ndim == 3 and image.shape[2] >= 3)
    orig_h, orig_w = image.shape[:2]
    total_pixels = orig_h * orig_w

    if is_rgb:
        rgb_input = image[:, :, :3]
        y, cr, cb = rgb_to_ycrcb(rgb_input)
        target_luminance = y
    else:
        target_luminance = image if image.ndim == 2 else image[:, :, 0]
        cr, cb = None, None

    # Step 2: Downsample luminance to L levels
    y_quant = downsample_channel(target_luminance, L)
    original_counts = get_channel_histogram(y_quant, L)

    # Step 3: Run custom math engine
    eq_results = histogram_equalization(original_counts, L)
    mapping = eq_results["mapping"]

    # Step 4: Map pixels and scale back up
    y_eq_quant = mapping[y_quant]
    y_eq_upscaled = upsample_channel(y_eq_quant, L)

    # Reconstruct final image
    if is_rgb:
        output_image = ycrcb_to_rgb(y_eq_upscaled, cr, cb)
    else:
        output_image = y_eq_upscaled

    eq_results["original_image"] = image
    eq_results["output_image"] = output_image
    eq_results["is_rgb"] = is_rgb
    eq_results["total_pixels"] = total_pixels
    eq_results["shape"] = (orig_h, orig_w)
    return eq_results


def process_image_matching(
    source_image: np.ndarray,
    target_image: np.ndarray,
    L: int,
    mapping_rule: str = "nearest"
) -> Dict[str, Any]:
    """
    Executes Histogram Matching / Specification between source and target images:
    - Extracts luminance for both (handling different pixel dimensions).
    - Downsamples both to L levels.
    - Computes matching mapping from custom math engine using specified mapping rule.
    - Reassigns source luminance, scales back up, and merges with source color channels.
    """
    src_rgb = (source_image.ndim == 3 and source_image.shape[2] >= 3)
    tgt_rgb = (target_image.ndim == 3 and target_image.shape[2] >= 3)

    src_h, src_w = source_image.shape[:2]
    tgt_h, tgt_w = target_image.shape[:2]

    # Source luminance
    if src_rgb:
        src_rgb_input = source_image[:, :, :3]
        y_src, cr_src, cb_src = rgb_to_ycrcb(src_rgb_input)
    else:
        y_src = source_image if source_image.ndim == 2 else source_image[:, :, 0]
        cr_src, cb_src = None, None

    # Target luminance
    if tgt_rgb:
        tgt_rgb_input = target_image[:, :, :3]
        y_tgt, _, _ = rgb_to_ycrcb(tgt_rgb_input)
    else:
        y_tgt = target_image if target_image.ndim == 2 else target_image[:, :, 0]

    # Downsample both
    y_src_quant = downsample_channel(y_src, L)
    y_tgt_quant = downsample_channel(y_tgt, L)

    source_counts = get_channel_histogram(y_src_quant, L)
    target_counts = get_channel_histogram(y_tgt_quant, L)

    # Custom math engine matching
    match_results = histogram_matching(source_counts, target_counts, L, mapping_rule=mapping_rule)
    inverse_mapping = match_results["inverse_mapping"]

    # Map source luminance and upsample
    y_matched_quant = inverse_mapping[y_src_quant]
    y_matched_upscaled = upsample_channel(y_matched_quant, L)

    # Reconstruct source with matched luminance
    if src_rgb:
        output_image = ycrcb_to_rgb(y_matched_upscaled, cr_src, cb_src)
    else:
        output_image = y_matched_upscaled

    match_results["source_image"] = source_image
    match_results["target_image"] = target_image
    match_results["output_image"] = output_image
    match_results["src_is_rgb"] = src_rgb
    match_results["tgt_is_rgb"] = tgt_rgb
    match_results["src_pixels"] = src_h * src_w
    match_results["tgt_pixels"] = tgt_h * tgt_w
    match_results["src_shape"] = (src_h, src_w)
    match_results["tgt_shape"] = (tgt_h, tgt_w)
    return match_results
