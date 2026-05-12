"""
image_utils.py — OpenCV image processing pipeline for KolamKala.

Pipeline overview:
  1. bytes_to_image / resize_to_standard
  2. detect_background_type (dark chalk vs light-bg)
  3. preprocess_image      (Gaussian + median blur combo)
  4. isolate_pattern       (HSV chalk mask or adaptive threshold)
  5. detect_edges          (dynamic Canny with sigma-based thresholds)
  6. extract_contours_filtered (area + aspect-ratio + approxPolyDP)
  7. detect_lines          (Probabilistic Hough)
  8. detect_loop_contours  (RETR_CCOMP hole detection)
  9. detect_dots           (HoughCircles + distance-based clustering)
 10. build_skin_mask       (hand / arm removal)
"""
import cv2
import numpy as np


# ────────────────────────────────────────────────────────
# Decode & resize
# ────────────────────────────────────────────────────────

def bytes_to_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw bytes → OpenCV BGR array (uint8)."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Could not decode the uploaded file. "
            "Please upload a valid JPG, PNG, or WebP image."
        )
    return img


def resize_to_standard(img: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """
    Resize so the longest side ≤ max_dim, preserving aspect ratio.
    Very small images are scaled up to at least 400 px so Hough
    transforms have sufficient resolution.
    """
    h, w = img.shape[:2]
    longest  = max(h, w)
    shortest = min(h, w)

    if longest > max_dim:
        scale = max_dim / longest
    elif shortest < 400:
        scale = 400 / shortest
    else:
        return img

    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_LANCZOS4)


# ────────────────────────────────────────────────────────
# Background-type detection
# ────────────────────────────────────────────────────────

def detect_background_type(gray: np.ndarray) -> str:
    """
    Decide whether the Kolam sits on a dark or light background.
    Samples the four corners (always background, not pattern) and
    returns 'dark' if mean corner brightness < 110, else 'light'.
    """
    h, w   = gray.shape
    cs     = max(20, min(h, w) // 8)
    sample = np.concatenate([
        gray[:cs, :cs].flatten(),
        gray[:cs, w - cs:].flatten(),
        gray[h - cs:, :cs].flatten(),
        gray[h - cs:, w - cs:].flatten(),
    ])
    return 'dark' if float(sample.mean()) < 110 else 'light'


# ────────────────────────────────────────────────────────
# Preprocessing — noise reduction
# ────────────────────────────────────────────────────────

def preprocess_image(gray: np.ndarray) -> np.ndarray:
    """
    Combined Gaussian + Median blur for robust noise reduction.

    Steps:
      1. Gaussian blur (5×5, σ=1.5) — suppresses Gaussian noise
         from sensor / JPEG compression.
      2. Median blur (kernel=3) — removes salt-and-pepper noise and
         small isolated bright specks without blurring edges.

    The two-stage approach preserves edge sharpness better than a
    single large Gaussian while still removing grain.
    """
    gauss = cv2.GaussianBlur(gray, (5, 5), 1.5)
    med   = cv2.medianBlur(gauss, 3)
    return med


# ────────────────────────────────────────────────────────
# Skin-tone mask (hand / arm removal)
# ────────────────────────────────────────────────────────

def _raw_skin_mask(hsv: np.ndarray) -> np.ndarray:
    lower_skin = np.array([0,   25,  60], dtype=np.uint8)
    upper_skin = np.array([25, 170, 230], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower_skin, upper_skin)
    lower2 = np.array([160, 25,  60], dtype=np.uint8)
    upper2 = np.array([180, 170, 230], dtype=np.uint8)
    mask   = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))
    return mask


def build_skin_mask(img: np.ndarray, dilation_px: int = 20) -> np.ndarray:
    """
    Binary mask covering skin-coloured pixels (HSV-based).
    dilation_px expands the mask to catch boundary pixels:
      20 (default): aggressive — for dark-bg chalk isolation.
       5          : light — for light-bg images (hand shadows only).
    """
    hsv       = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skin_mask = _raw_skin_mask(hsv)
    kernel    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px, dilation_px)
    )
    return cv2.dilate(skin_mask, kernel, iterations=2)


# ────────────────────────────────────────────────────────
# Kolam pattern isolation
# ────────────────────────────────────────────────────────

def isolate_pattern(img: np.ndarray, gray: np.ndarray) -> tuple:
    """
    Produce a clean binary mask containing ONLY Kolam lines and dots.

    DARK background: extract high-brightness, low-saturation pixels
      (white chalk) then remove skin regions.

    LIGHT background: adaptive Gaussian threshold on denoised gray,
      then remove hand shadow edges.

    Morphological pipeline:
      - Opening (3×3, 1 iter): removes isolated noise pixels
      - Closing (3×3, 2 iter): fills hairline gaps in chalk strokes

    Returns (binary_mask, bg_type).
    """
    bg_type = detect_background_type(gray)
    hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_img, w_img = gray.shape[:2]
    img_total    = h_img * w_img

    def _safe_skin_mask(dilation_px: int) -> np.ndarray:
        """
        Return the skin mask only when it is plausible (covers ≤ 40% of the
        image).  Images with overall warm tones — colour Kolam photos, stylised
        hero shots — produce false-positive skin masks that would wipe out the
        entire pattern.  In those cases we return an all-zero mask so no pixels
        are suppressed.
        """
        raw = build_skin_mask(img, dilation_px=dilation_px)
        if int(raw.sum() // 255) > int(img_total * 0.40):
            return np.zeros_like(raw)
        return raw

    if bg_type == 'dark':
        chalk_mask = cv2.inRange(hsv,
                                 np.array([0,   0, 155], dtype=np.uint8),
                                 np.array([180, 65, 255], dtype=np.uint8))
        skin_mask = _safe_skin_mask(dilation_px=20)
        binary    = cv2.bitwise_and(chalk_mask, cv2.bitwise_not(skin_mask))

        # Fallback: if the chalk-colour mask is essentially empty (< 0.3% coverage),
        # the image likely has a coloured or stylised Kolam — use Otsu's
        # binarization on the denoised grayscale to separate bright foreground
        # from dark background without relying on colour heuristics.
        if int(binary.sum() // 255) < int(img_total * 0.003):
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_and(binary, cv2.bitwise_not(skin_mask))
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary  = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=4,
        )
        skin_mask = _safe_skin_mask(dilation_px=5)
        binary    = cv2.bitwise_and(binary, cv2.bitwise_not(skin_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return binary, bg_type


# ────────────────────────────────────────────────────────
# Colour conversion helper
# ────────────────────────────────────────────────────────

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """BGR → grayscale (single-channel uint8)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ────────────────────────────────────────────────────────
# Edge detection — dynamic Canny (sigma trick)
# ────────────────────────────────────────────────────────

def detect_edges(binary: np.ndarray) -> np.ndarray:
    """
    Canny edge detection with dynamic thresholds based on image statistics.

    Algorithm (the "sigma trick"):
      1. Normalize contrast so the full 0–255 range is used.
      2. Compute the median pixel intensity v of the normalized image.
      3. Set lower = max(0, (1 - σ) · v) and upper = min(255, (1 + σ) · v).
         σ = 0.33 is the standard value that works well across image types.

    This adapts the thresholds to the actual intensity distribution of
    each image instead of hardcoding numbers.
    """
    sigma  = 0.33
    norm   = cv2.normalize(binary, None, 0, 255, cv2.NORM_MINMAX)
    v      = float(np.median(norm[norm > 0])) if np.any(norm > 0) else 128.0
    low    = int(max(0,   (1.0 - sigma) * v))
    high   = int(min(255, (1.0 + sigma) * v))
    low    = max(low, 10)
    high   = max(high, low + 20)

    blurred = cv2.GaussianBlur(norm.astype(np.uint8), (3, 3), 0)
    return cv2.Canny(blurred, low, high)


# ────────────────────────────────────────────────────────
# Line detection
# ────────────────────────────────────────────────────────

def detect_lines(edges: np.ndarray) -> int:
    """
    Count line segments with Probabilistic Hough Transform.
    min_line_length scales to 3% of the image diagonal.
    """
    h, w    = edges.shape[:2]
    diag    = int((h ** 2 + w ** 2) ** 0.5)
    min_len = max(20, int(diag * 0.03))
    max_gap = max(8,  int(diag * 0.012))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    return int(len(lines)) if lines is not None else 0


# ────────────────────────────────────────────────────────
# Contour extraction with filtering
# ────────────────────────────────────────────────────────

def extract_contours_filtered(binary: np.ndarray, img_area: int):
    """
    Extract and filter contours for robustness.

    Filtering criteria:
      - Area: min = max(50, 0.01% of image area), max = 60% of image area.
        Removes tiny noise specks and the full image border.
      - Aspect ratio: bounding-box w/h must be in [0.05, 20].
        Removes extremely elongated artefacts (scratch marks, borders).
      - Approximation: cv2.approxPolyDP with ε = 1% of perimeter.
        Reduces the number of control points for stability and faster
        processing while preserving overall shape fidelity.

    Returns list of approximated contours (each is an ndarray).
    """
    min_area = max(50.0, img_area * 0.0001)
    max_area = img_area * 0.60

    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if aspect > 20.0 or aspect < 0.05:
            continue

        eps   = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        filtered.append(approx)

    return filtered


def detect_contours(binary: np.ndarray):
    """
    Compatibility wrapper — returns all contours without filtering.
    Used internally when the full contour list is needed.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


# ────────────────────────────────────────────────────────
# Loop detection — RETR_CCOMP hierarchy
# ────────────────────────────────────────────────────────

def detect_loop_contours(binary: np.ndarray):
    """
    Find the HOLES enclosed by chalk lines using contour hierarchy.

    A Kolam loop is a region completely surrounded by chalk strokes.
    In the binary mask these appear as black holes inside white rings.

    Algorithm:
      1. Close micro-gaps with a 5×5 kernel (2 iters) to seal hairline
         breaks at line junctions.
      2. RETR_CCOMP: level-0 = outer chalk boundary, level-1 = holes.
      3. Return only level-1 (hole) contours.

    Returns list of hole contours (each is an ndarray).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sealed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(
        sealed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or len(contours) == 0:
        return []

    return [
        cnt
        for cnt, h in zip(contours, hierarchy[0])
        if h[3] >= 0
    ]


# ────────────────────────────────────────────────────────
# Dot detection with distance-based clustering
# ────────────────────────────────────────────────────────

def _cluster_circles(circles: np.ndarray, dist_thresh: float) -> np.ndarray:
    """
    Remove duplicate nearby detections using a distance threshold.

    Two detected circles are considered duplicates when their centre
    distance is less than dist_thresh pixels.  Of each duplicate pair,
    the second (lower accumulator index) is discarded.

    This is a simple O(n²) sweep which is fast for the small number of
    circles HoughCircles typically returns (< 200 for a Kolam image).

    Args:
        circles:    (N, 3) float32 array — [cx, cy, r] per row.
        dist_thresh: Minimum allowed distance between two circle centres.

    Returns:
        Subset of circles with duplicates removed.
    """
    if len(circles) == 0:
        return circles

    centers = circles[:, :2]
    kept    = np.ones(len(circles), dtype=bool)

    for i in range(len(circles)):
        if not kept[i]:
            continue
        dists = np.linalg.norm(centers[i + 1:] - centers[i], axis=1)
        close = np.where(dists < dist_thresh)[0] + (i + 1)
        kept[close] = False

    return circles[kept]


def detect_dots(
    img: np.ndarray,
    gray: np.ndarray,
    bg_type: str,
    binary: np.ndarray = None,
) -> int:
    """
    Detect Kolam dot-marks using two complementary methods.

    Method A — HoughCircles on preprocessed grayscale:
      Radius bounds scale with the shorter image dimension
      (min_r ≈ 0.5%, max_r ≈ 3.0% of short side).

    Method B — SimpleBlobDetector on the isolated binary pattern:
      Finds small, isolated, circular blobs in the chalk-pattern mask.
      Effective even when HoughCircles misses low-contrast dots.

    Both methods are merged and deduplicated with _cluster_circles.

    Safe skin mask:
      build_skin_mask() is called but ONLY applied when it covers ≤ 40%
      of the image.  Warm-toned backgrounds (wooden floors, colour Kolam
      photos) trigger false-positive skin detection that would otherwise
      zero out the entire target, returning 0 dots.
    """
    h, w      = gray.shape[:2]
    short     = min(h, w)
    img_area  = h * w

    min_r     = max(3,  int(short * 0.005))
    max_r     = max(min_r + 4, int(short * 0.030))
    min_dist  = max(min_r * 2, int(short * 0.018))

    min_dot_area = float(max(20, int(np.pi * min_r ** 2 * 0.5)))
    max_dot_area = float(int(np.pi * max_r ** 2 * 2.0))

    # ── Safe skin mask ────────────────────────────────────────────────────────
    # Skip masking when the skin detector fires on warm-toned backgrounds
    # (e.g. wooden floor covers > 40 % → do not zero out the whole image).
    raw_skin  = build_skin_mask(img)
    skin_ok   = int(raw_skin.sum() // 255) <= int(img_area * 0.40)
    skin_mask = raw_skin if skin_ok else np.zeros_like(raw_skin)

    # ── Build grayscale target for HoughCircles ───────────────────────────────
    # Dark bg: keep only bright-enough pixels (chalk is high-luminance);
    # lower threshold (120 vs old 140) to catch slightly dim or photographed
    # chalk dots without pulling in the dark wooden floor.
    if bg_type == 'dark':
        _, bright = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        target    = cv2.bitwise_and(gray, bright)
    else:
        target = gray.copy()

    target = cv2.bitwise_and(target, cv2.bitwise_not(skin_mask))

    # ── Method A: HoughCircles ────────────────────────────────────────────────
    blurred   = cv2.GaussianBlur(target, (7, 7), 2)
    circles_a = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=50,
        param2=15,
        minRadius=min_r,
        maxRadius=max_r,
    )

    centers_a: list = []
    if circles_a is not None:
        clustered = _cluster_circles(circles_a[0], dist_thresh=float(min_r * 2.5))
        centers_a = clustered.tolist()

    # ── Method B: SimpleBlobDetector on isolated binary ───────────────────────
    # Works on the pattern mask already separated from background.
    # Invert so chalk (white in mask) becomes dark blobs on a light field —
    # SimpleBlobDetector finds dark blobs by design.
    centers_b: list = []
    if binary is not None and binary.size > 0:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea        = True
        params.minArea             = min_dot_area
        params.maxArea             = max_dot_area
        params.filterByCircularity = True
        params.minCircularity      = 0.40
        params.filterByConvexity   = True
        params.minConvexity        = 0.55
        params.filterByInertia     = True
        params.minInertiaRatio     = 0.35
        params.minDistBetweenBlobs = float(min_dist)

        detector   = cv2.SimpleBlobDetector_create(params)
        inverted   = cv2.bitwise_not(binary)
        keypoints  = detector.detect(inverted)
        centers_b  = [
            [float(kp.pt[0]), float(kp.pt[1]), float(kp.size) / 2.0]
            for kp in keypoints
        ]

    # ── Merge & deduplicate ───────────────────────────────────────────────────
    all_centers = centers_a + centers_b
    if not all_centers:
        return 0

    arr     = np.array(all_centers, dtype=np.float32)
    deduped = _cluster_circles(arr, dist_thresh=float(max(min_r * 2, min_dist)))
    return int(len(deduped))
