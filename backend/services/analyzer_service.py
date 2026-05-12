"""
analyzer_service.py — Real OpenCV-based Kolam image analysis.

Full pipeline:
  1.  Decode bytes → BGR image
  2.  Resize to ≤ 800 px
  3.  Grayscale conversion
  4.  Noise reduction — Gaussian + median blur (preprocess_image)
  5.  Auto-detect background type (dark chalk / light pigment)
  6.  Isolate Kolam pattern (HSV chalk mask or adaptive threshold)
  7.  Dynamic Canny edge detection (sigma-based thresholds)
  8.  Line count — Probabilistic Hough on clean edges
  9.  Filtered contours — area + aspect-ratio + approxPolyDP
  10. Loop count — RETR_CCOMP enclosed-region method
  11. Dot count  — HoughCircles + distance-based clustering
  12. Symmetry   — bilateral (H+V) + rotational (90°, 180°), with score
  13. Kolam type — rule-based classifier (includes Mandala type)
  14. Complexity — weighted score
"""
import logging

from ..utils.image_utils import (
    bytes_to_image,
    resize_to_standard,
    to_grayscale,
    preprocess_image,
    isolate_pattern,
    detect_edges,
    detect_lines,
    extract_contours_filtered,
    detect_loop_contours,
    detect_dots,
)
from ..utils.math_utils import (
    check_symmetry,
    estimate_loops,
    classify_kolam_type,
    calculate_complexity,
)
from ..models.response_models import AnalysisData

log = logging.getLogger(__name__)


def analyze_image(file_bytes: bytes) -> AnalysisData:
    """
    Run the full Kolam analysis pipeline on uploaded image bytes.

    Args:
        file_bytes: Raw bytes of the uploaded image file.

    Returns:
        AnalysisData with real computer-vision results plus a
        symmetry_score reflecting the strength of the detected symmetry.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    # ── 1. Decode ──────────────────────────────────────────────────────────
    img = bytes_to_image(file_bytes)
    log.debug("Decoded: %s px", img.shape[:2])

    # ── 2. Resize ──────────────────────────────────────────────────────────
    img      = resize_to_standard(img, max_dim=800)
    h, w     = img.shape[:2]
    img_area = h * w
    log.debug("Resized to %d × %d (area=%d)", w, h, img_area)

    # ── 3. Grayscale ───────────────────────────────────────────────────────
    gray = to_grayscale(img)

    # ── 4. Noise reduction ─────────────────────────────────────────────────
    # Combined Gaussian + median blur for robust denoising
    preprocessed = preprocess_image(gray)
    log.debug("Preprocessing done")

    # ── 5 & 6. Detect background + isolate pattern ─────────────────────────
    binary, bg_type = isolate_pattern(img, preprocessed)
    log.debug("BG type: %s | pattern px: %d", bg_type, int(binary.sum() // 255))

    # ── 7. Edges — dynamic Canny (sigma-based thresholds) ──────────────────
    edges = detect_edges(binary)

    # ── 8. Line count ──────────────────────────────────────────────────────
    line_count = detect_lines(edges)
    log.debug("Lines: %d", line_count)

    # ── 9. Filtered contours (aspect-ratio + approxPolyDP) ─────────────────
    filtered_contours = extract_contours_filtered(binary, img_area)
    log.debug("Filtered contours: %d", len(filtered_contours))

    # ── 10. Loop count — RETR_CCOMP enclosed-region method ─────────────────
    loop_contours = detect_loop_contours(binary)
    loop_count    = estimate_loops(loop_contours, img_area)
    log.debug("Loops: %d (from %d enclosed regions)", loop_count, len(loop_contours))

    # ── 11. Dot count — HoughCircles + SimpleBlobDetector ──────────────────
    # Pass the isolated binary so the blob detector has a clean pattern mask.
    # Denoised grayscale drives HoughCircles; binary drives SimpleBlobDetector.
    dot_count = detect_dots(img, preprocessed, bg_type, binary)
    log.debug("Dots: %d", dot_count)

    # ── 12. Symmetry — bilateral + rotational, with score ──────────────────
    symmetry_type, symmetry_score = check_symmetry(img, gray)
    log.debug("Symmetry: %s (score=%.3f)", symmetry_type, symmetry_score)

    # ── 13. Kolam type ─────────────────────────────────────────────────────
    kolam_type = classify_kolam_type(dot_count, loop_count, line_count, symmetry_type)
    log.debug("Type: %s", kolam_type)

    # ── 14. Complexity ─────────────────────────────────────────────────────
    complexity = calculate_complexity(dot_count, loop_count, line_count)
    log.debug("Complexity: %s", complexity)

    return AnalysisData(
        lines=line_count,
        loops=loop_count,
        symmetry=symmetry_type,
        symmetry_score=symmetry_score,
        dots=dot_count,
        kolam_type=kolam_type,
        complexity=complexity,
        note=(
            "All values are approximate estimates produced by classical image processing. "
            "Results are fully explainable and deterministic — no machine learning is used. "
            "Accuracy improves with high-contrast, clean Kolam images on a plain background."
        ),
    )
