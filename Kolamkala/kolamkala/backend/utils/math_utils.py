"""
math_utils.py — Analysis helpers for KolamKala.

Covers:
  - Grid spacing           (generator)
  - Symmetry detection     (bilateral + rotational, with score)
  - Loop estimation        (circularity + area filter)
  - Kolam type classification
  - Complexity scoring
"""
import cv2
import numpy as np
from typing import Tuple


# ────────────────────────────────────────────────────────
# Generator utility
# ────────────────────────────────────────────────────────

def grid_spacing(rows: int, cols: int, pad: float = 0.10) -> Tuple[float, float]:
    """Normalised dot spacing within [0, 1] coordinate space."""
    range_x = 1.0 - 2 * pad
    range_y = 1.0 - 2 * pad
    sx = range_x / (cols - 1) if cols > 1 else range_x
    sy = range_y / (rows - 1) if rows > 1 else range_y
    return sx, sy


# ────────────────────────────────────────────────────────
# Symmetry detection — bilateral + rotational
# ────────────────────────────────────────────────────────

def _prepare_comparison_image(img: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """
    Shared preprocessing for all symmetry checks.

    1. Zero out skin/hand pixels so a hand in one corner doesn't break
       an otherwise symmetric pattern.
       Safe guard: if the skin mask covers > 40% of the image (e.g. warm
       wooden floor triggers false-positive skin detection), the mask is
       skipped entirely so the full image is used for comparison.
    2. Resize to 256×256 for resolution-independent comparison.
    3. Apply Gaussian blur (σ=3) to tolerate minor misalignments and
       small photographic distortions.
    """
    from .image_utils import build_skin_mask
    h, w       = gray.shape[:2]
    img_area   = h * w
    skin_mask  = build_skin_mask(img)
    clean_gray = gray.copy()
    if int(skin_mask.sum() // 255) <= int(img_area * 0.40):
        clean_gray[skin_mask > 0] = 0
    compare = cv2.resize(clean_gray, (256, 256))
    compare = cv2.GaussianBlur(compare, (9, 9), 3)
    return compare.astype(np.float32)


def _pixel_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalised pixel-difference similarity in [0, 1].
    score = 1 means identical; score = 0 means maximum difference.
    Uses vectorised NumPy operations for speed.
    """
    return float(1.0 - np.mean(np.abs(a - b)) / 255.0)


def check_symmetry(
    img: np.ndarray,
    gray: np.ndarray,
    threshold: float = 0.72,
) -> Tuple[str, float]:
    """
    Determine symmetry type and score for a Kolam image.

    Tests four symmetry types:
      - Horizontal (top/bottom halves match after vertical flip)
      - Vertical   (left/right halves match after horizontal flip)
      - Rotational — 90° rotation similarity
      - Rotational — 180° rotation similarity

    Decision priority:
      1. Rotational (180°) — characteristic of complex Kolam; tested first
         because a 4-fold pattern also passes bilateral tests.
      2. Both bilateral (h+v)
      3. Horizontal only
      4. Vertical only
      5. None

    Args:
        img:       Original BGR image (for skin masking).
        gray:      Grayscale version of img.
        threshold: Minimum similarity score to declare a symmetry present.

    Returns:
        Tuple of (symmetry_type, symmetry_score):
          symmetry_type  — 'horizontal' | 'vertical' | 'both' |
                           'rotational' | 'none'
          symmetry_score — highest score among all tested axes (0–1 float,
                           rounded to 3 decimal places).
    """
    compare = _prepare_comparison_image(img, gray)
    h, w    = compare.shape

    # ── Horizontal symmetry (top vs flipped bottom) ──────────────────────
    top         = compare[: h // 2, :]
    bottom_flip = np.flipud(compare[h - h // 2 :, :])
    min_h       = min(top.shape[0], bottom_flip.shape[0])
    h_score     = _pixel_similarity(top[:min_h], bottom_flip[:min_h])

    # ── Vertical symmetry (left vs flipped right) ─────────────────────────
    left       = compare[:, : w // 2]
    right_flip = np.fliplr(compare[:, w - w // 2 :])
    min_w      = min(left.shape[1], right_flip.shape[1])
    v_score    = _pixel_similarity(left[:, :min_w], right_flip[:, :min_w])

    # ── Rotational symmetry ───────────────────────────────────────────────
    # 180° rotation: most common in traditional Kolam
    rot180       = np.rot90(compare, k=2)
    r180_score   = _pixel_similarity(compare, rot180)

    # 90° rotation: 4-fold patterns (e.g., full mandala Kolam)
    rot90        = np.rot90(compare, k=1)
    r90_score    = _pixel_similarity(compare, rot90)

    # ── Decision logic ────────────────────────────────────────────────────
    h_sym  = h_score  >= threshold
    v_sym  = v_score  >= threshold
    r90    = r90_score  >= threshold
    r180   = r180_score >= threshold

    # Gather all scores for the reported "best score"
    all_scores = [h_score, v_score, r90_score, r180_score]
    best_score = round(float(max(all_scores)), 3)

    if r90 or r180:
        return "rotational", best_score
    if h_sym and v_sym:
        return "both", round(float((h_score + v_score) / 2), 3)
    if h_sym:
        return "horizontal", round(h_score, 3)
    if v_sym:
        return "vertical", round(v_score, 3)
    return "none", round(float(max(h_score, v_score)), 3)


# ────────────────────────────────────────────────────────
# Loop estimation
# ────────────────────────────────────────────────────────

def estimate_loops(contours, img_area: int) -> int:
    """
    Count closed loop-like contours using circularity and area filters.

    min_area = max(300, img_area × 0.2%) — excludes isolated dot outlines.
    max_area = 25% of image — filters out the full border contour.
    min_circ = 0.20 — accepts teardrop and petal shapes, not just circles.

    Circularity = 4π·area / perimeter²  (1.0 = perfect circle).
    """
    min_area = max(300.0, img_area * 0.002)
    max_area = img_area * 0.25
    min_circ = 0.20

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim < 10:
            continue
        circularity = 4.0 * np.pi * area / (perim ** 2)
        if circularity >= min_circ:
            count += 1
    return count


# ────────────────────────────────────────────────────────
# Kolam type classification
# ────────────────────────────────────────────────────────

def classify_kolam_type(dots: int, loops: int, lines: int, symmetry: str = "none") -> str:
    """
    Classify the Kolam into one of four traditional types.

    Sikku Kolam  — thread / knot patterns; many loops woven around dots.
    Kambi Kolam  — line-based; straight or curved strokes dominate.
    Pulli Kolam  — pure dot-grid; minimal connecting line structure.
    Mandala Kolam— rotationally symmetric patterns with many elements.

    The symmetry parameter upgrades the classification when rotational
    symmetry is detected (a feature of mandala-style Kolam).
    """
    if symmetry == "rotational" and (loops >= 6 or dots >= 16):
        return "Mandala Kolam"

    if loops >= 8 or (loops >= 5 and lines >= 15):
        return "Sikku Kolam"

    if lines >= 15 or (lines >= 8 and loops >= 3):
        return "Kambi Kolam"

    return "Pulli Kolam"


# ────────────────────────────────────────────────────────
# Complexity scoring
# ────────────────────────────────────────────────────────

def calculate_complexity(dots: int, loops: int, lines: int) -> str:
    """
    Return a human-readable complexity label from a weighted score.

    Loops carry the most weight because they represent the intricate
    winding characteristic of complex Kolam art.

    Score bands:
      < 15  → Simple
      < 40  → Moderate
      < 90  → Complex
      ≥ 90  → Highly Complex
    """
    score = dots * 0.5 + loops * 2.0 + lines * 1.0

    if score < 15:
        return "Simple"
    if score < 40:
        return "Moderate"
    if score < 90:
        return "Complex"
    return "Highly Complex"
