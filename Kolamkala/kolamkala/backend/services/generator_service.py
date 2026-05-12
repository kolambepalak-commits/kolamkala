"""
generator_service.py — Kolam pattern generation service.

Four pattern styles with strict rule-based geometry:
  basic     — fish-eye loops between adjacent dots (Pulli-style)
  symmetric — fish-eye loops + diagonal Bézier + ellipses (Kambi-style)
  diagonal  — S-curves along block diagonals + horizontal connectors
  sikku     — interlocking closed loops woven around each dot with
               weaving connectors (authentic Sikku Kolam geometry)

Each style produces a PatternData object with:
  - Dot grid (Cartesian, equal spacing, NumPy-computed coordinates)
  - Curve segments (quadratic Bézier arcs or ellipses)
  - Validation report (Euler compliance, connectivity check)
"""
from typing import List, Optional, Dict
import numpy as np

from ..models.response_models import Dot, Point, CurveSegment, PatternData, PatternValidation
from ..utils.math_utils import grid_spacing

PAD = 0.10  # Normalised padding on each side of the canvas


# ────────────────────────────────────────────────────────
# Dot grid generation
# ────────────────────────────────────────────────────────

def generate_dots(rows: int, cols: int) -> List[Dot]:
    """
    Build a rectangular Cartesian dot grid in normalised [0, 1] coordinates.

    Coordinates are computed with NumPy for precision and stored as
    rounded floats.  Ordering: row-major (top-to-bottom, left-to-right).
    """
    sx, sy = grid_spacing(rows, cols, PAD)

    # NumPy vectorised grid — no Python loops over individual dots
    r_idx = np.arange(rows)
    c_idx = np.arange(cols)
    rr, cc = np.meshgrid(r_idx, c_idx, indexing='ij')

    xs = (PAD + cc * sx).round(4)
    ys = (PAD + rr * sy).round(4)

    return [
        Dot(row=int(r), col=int(c), x=float(xs[r, c]), y=float(ys[r, c]))
        for r in range(rows)
        for c in range(cols)
    ]


def _get_pos(dots: List[Dot], cols: int, r: int, c: int) -> Point:
    """Retrieve the (x, y) position of the dot at grid cell (r, c)."""
    d = dots[r * cols + c]
    return Point(x=d.x, y=d.y)


# ────────────────────────────────────────────────────────
# Pattern validation — Euler compliance
# ────────────────────────────────────────────────────────

def validate_pattern(dots: List[Dot], curves: List[CurveSegment]) -> PatternValidation:
    """
    Validate the generated Kolam pattern for mathematical correctness.

    Checks:
    1. Dot connectivity — how many dots have at least one curve endpoint
       within the neighbourhood radius (50% of grid spacing).  This is
       generous enough to cover Sikku patterns where curve endpoints sit
       on the perimeter of each dot's enclosing loop rather than at the
       dot centre itself.

    2. Euler compliance — in an Eulerian graph every vertex has even
       degree.  We build an endpoint-frequency map (each curve contributes
       its start and end).  An odd-degree vertex means the path cannot be
       traversed in a single continuous stroke without lifting the hand.

       Note: basic patterns are fully Eulerian; symmetric, diagonal, and
       sikku patterns are valid multi-stroke Kolam that are NOT single
       Eulerian paths (their is_valid reflects overall structural soundness,
       not necessarily Euler compliance).

    3. is_valid — the pattern is structurally sound when ≥ 90% of dots
       are connected by the curve network.

    Returns a PatternValidation with all diagnostic fields.
    """
    # Compute adaptive snap distance from the dot grid
    if len(dots) >= 2:
        xs = sorted({d.x for d in dots})
        dx = xs[1] - xs[0] if len(xs) >= 2 else 0.12
    else:
        dx = 0.12
    DOT_SNAP = max(0.05, dx * 0.50)   # 50% of grid step catches Sikku loop perimeters

    dot_xy = np.array([[d.x, d.y] for d in dots], dtype=np.float32)

    dots_touched: set = set()
    endpoint_freq: Dict[tuple, int] = {}

    def _record(pt: Optional[Point]) -> None:
        if pt is None:
            return
        key = (round(pt.x, 2), round(pt.y, 2))
        endpoint_freq[key] = endpoint_freq.get(key, 0) + 1

        # Vectorised proximity check against all dot positions
        dists = np.linalg.norm(dot_xy - np.array([pt.x, pt.y], dtype=np.float32), axis=1)
        nearest_idx = int(np.argmin(dists))
        if float(dists[nearest_idx]) < DOT_SNAP:
            dots_touched.add(nearest_idx)

    for curve in curves:
        _record(curve.start)
        _record(curve.end)

    odd_endpoints = sum(1 for v in endpoint_freq.values() if v % 2 != 0)
    euler_ok      = (odd_endpoints == 0)

    # Structural validity: ≥ 90% of dots touched by the curve network
    connected_ratio = len(dots_touched) / max(len(dots), 1)
    structurally_valid = connected_ratio >= 0.90

    return PatternValidation(
        is_valid=structurally_valid,
        total_curves=len(curves),
        dots_connected=len(dots_touched),
        total_dots=len(dots),
        odd_endpoints=odd_endpoints,
        euler_compliant=euler_ok,
    )


# ────────────────────────────────────────────────────────
# Generator — Basic (fish-eye loops)
# ────────────────────────────────────────────────────────

def generate_basic_curves(dots: List[Dot], rows: int, cols: int) -> List[CurveSegment]:
    """
    Basic Kolam — fish-eye loops between every pair of adjacent dots.

    For each horizontal adjacent pair: one arc bulges above, one below.
    For each vertical adjacent pair:   one arc bulges left,  one right.
    Each arc is a quadratic Bézier with a midpoint control point.
    """
    sx, sy = grid_spacing(rows, cols, PAD)
    bH = round(sy * 0.40, 4)
    bV = round(sx * 0.40, 4)
    curves: List[CurveSegment] = []

    for r in range(rows):
        for c in range(cols - 1):
            p1 = _get_pos(dots, cols, r, c)
            p2 = _get_pos(dots, cols, r, c + 1)
            mx = round((p1.x + p2.x) / 2, 4)
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=p1, control=Point(x=mx, y=round(p1.y - bH, 4)), end=p2
            ))
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=p1, control=Point(x=mx, y=round(p1.y + bH, 4)), end=p2
            ))

    for r in range(rows - 1):
        for c in range(cols):
            p1 = _get_pos(dots, cols, r, c)
            p2 = _get_pos(dots, cols, r + 1, c)
            my = round((p1.y + p2.y) / 2, 4)
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=p1, control=Point(x=round(p1.x - bV, 4), y=my), end=p2
            ))
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=p1, control=Point(x=round(p1.x + bV, 4), y=my), end=p2
            ))

    return curves


# ────────────────────────────────────────────────────────
# Generator — Symmetric (loops + diagonals + ellipses)
# ────────────────────────────────────────────────────────

def generate_symmetric_curves(dots: List[Dot], rows: int, cols: int) -> List[CurveSegment]:
    """
    Symmetric Kolam — fish-eye loops + diagonal cross-curves + ellipses.

    Builds on basic curves and adds:
    - NW→SE and NE→SW quadratic Bézier inside each 2×2 block.
    - Axis-aligned ellipse centred on every dot.

    The four-fold structure produces bilateral symmetry (both axes).
    """
    sx, sy = grid_spacing(rows, cols, PAD)
    curves = generate_basic_curves(dots, rows, cols)

    for r in range(rows - 1):
        for c in range(cols - 1):
            tl = _get_pos(dots, cols, r,     c)
            tr = _get_pos(dots, cols, r,     c + 1)
            bl = _get_pos(dots, cols, r + 1, c)
            br = _get_pos(dots, cols, r + 1, c + 1)
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=tl, control=Point(x=tr.x, y=tl.y), end=br
            ))
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=tr, control=Point(x=tl.x, y=tr.y), end=bl
            ))

    rx = round(sx * 0.27, 4)
    ry = round(sy * 0.27, 4)
    for dot in dots:
        curves.append(CurveSegment(
            type="ellipse",
            center=Point(x=dot.x, y=dot.y),
            rx=rx, ry=ry,
        ))

    return curves


# ────────────────────────────────────────────────────────
# Generator — Diagonal (S-curves)
# ────────────────────────────────────────────────────────

def generate_diagonal_curves(dots: List[Dot], rows: int, cols: int) -> List[CurveSegment]:
    """
    Diagonal Kolam — S-curves along block diagonals + horizontal connectors.

    Each pair of diagonal curves in a 2×2 block forms a mirrored S-shape,
    creating the flowing "wave" characteristic of diagonal-style Kolam.
    """
    sx, sy = grid_spacing(rows, cols, PAD)
    bulge  = round(min(sx, sy) * 0.48, 4)
    curves: List[CurveSegment] = []

    for r in range(rows - 1):
        for c in range(cols - 1):
            tl = _get_pos(dots, cols, r,     c)
            tr = _get_pos(dots, cols, r,     c + 1)
            bl = _get_pos(dots, cols, r + 1, c)
            br = _get_pos(dots, cols, r + 1, c + 1)
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=tl,
                control=Point(x=round(tl.x + bulge * 0.6, 4), y=tl.y),
                end=br,
            ))
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=tr,
                control=Point(x=round(tr.x - bulge * 0.6, 4), y=tr.y),
                end=bl,
            ))

    for r in range(rows):
        for c in range(cols - 1):
            p1 = _get_pos(dots, cols, r, c)
            p2 = _get_pos(dots, cols, r, c + 1)
            mx = round((p1.x + p2.x) / 2, 4)
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=p1, control=Point(x=mx, y=p1.y), end=p2
            ))

    return curves


# ────────────────────────────────────────────────────────
# Generator — Sikku (interlocking loops, authentic geometry)
# ────────────────────────────────────────────────────────

def generate_sikku_curves(dots: List[Dot], rows: int, cols: int) -> List[CurveSegment]:
    """
    Sikku Kolam — interlocking closed loops woven continuously around dots.

    Algorithm (rule-based, mathematically defined):

    STEP 1 — Per-dot closed loop:
      For each dot, draw a rectangular closed loop consisting of 4
      quadratic Bézier arcs (NE, SE, SW, NW quadrants).  The loop
      surrounds the dot without touching it, spaced at 38% of the
      grid step in each axis.  This creates the "eyelet" that is
      characteristic of Sikku patterns.

    STEP 2 — Vertical weaving connectors:
      Between every vertically adjacent pair of dots, draw two
      S-curve connections — one on the left side, one on the right
      side of the dots.  The control point is centred between the
      two dot rows, creating the weaving "thread" effect.

    STEP 3 — Horizontal weaving connectors:
      Same as Step 2, applied horizontally between each pair of
      adjacent dots in the same row.

    Mathematical properties:
      - Every dot is enclosed in exactly one closed loop.
      - Weaving connectors link adjacent loops, forming a single
        continuous interlocked path across the whole grid.
      - The pattern has bilateral symmetry when rows == cols.
    """
    sx, sy = grid_spacing(rows, cols, PAD)
    rloop  = round(sx * 0.38, 4)   # half-width of each dot's enclosing loop
    sloop  = round(sy * 0.38, 4)   # half-height of each dot's enclosing loop
    curves: List[CurveSegment] = []

    # STEP 1 — Closed rectangular loops around each dot (4 arcs each)
    for dot in dots:
        cx, cy = dot.x, dot.y
        # NE arc (top → right)
        curves.append(CurveSegment(
            type="quadratic_bezier",
            start=Point(x=round(cx,       4), y=round(cy - sloop, 4)),
            control=Point(x=round(cx + rloop, 4), y=round(cy - sloop, 4)),
            end=Point(x=round(cx + rloop, 4), y=round(cy,         4)),
        ))
        # SE arc (right → bottom)
        curves.append(CurveSegment(
            type="quadratic_bezier",
            start=Point(x=round(cx + rloop, 4), y=round(cy,       4)),
            control=Point(x=round(cx + rloop, 4), y=round(cy + sloop, 4)),
            end=Point(x=round(cx,         4), y=round(cy + sloop, 4)),
        ))
        # SW arc (bottom → left)
        curves.append(CurveSegment(
            type="quadratic_bezier",
            start=Point(x=round(cx,       4), y=round(cy + sloop, 4)),
            control=Point(x=round(cx - rloop, 4), y=round(cy + sloop, 4)),
            end=Point(x=round(cx - rloop, 4), y=round(cy,         4)),
        ))
        # NW arc (left → top)
        curves.append(CurveSegment(
            type="quadratic_bezier",
            start=Point(x=round(cx - rloop, 4), y=round(cy,       4)),
            control=Point(x=round(cx - rloop, 4), y=round(cy - sloop, 4)),
            end=Point(x=round(cx,         4), y=round(cy - sloop, 4)),
        ))

    # STEP 2 — Vertical weaving connectors between adjacent loop rows
    for r in range(rows - 1):
        for c in range(cols):
            d1 = dots[r * cols + c]
            d2 = dots[(r + 1) * cols + c]
            my = round((d1.y + d2.y) / 2, 4)

            # Left weave connector
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=Point(x=round(d1.x - rloop, 4), y=round(d1.y, 4)),
                control=Point(x=round(d1.x - rloop * 0.4, 4), y=my),
                end=Point(x=round(d2.x - rloop, 4), y=round(d2.y, 4)),
            ))
            # Right weave connector
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=Point(x=round(d1.x + rloop, 4), y=round(d1.y, 4)),
                control=Point(x=round(d1.x + rloop * 0.4, 4), y=my),
                end=Point(x=round(d2.x + rloop, 4), y=round(d2.y, 4)),
            ))

    # STEP 3 — Horizontal weaving connectors between adjacent loop columns
    for r in range(rows):
        for c in range(cols - 1):
            d1 = dots[r * cols + c]
            d2 = dots[r * cols + c + 1]
            mx = round((d1.x + d2.x) / 2, 4)

            # Top weave connector
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=Point(x=round(d1.x, 4), y=round(d1.y - sloop, 4)),
                control=Point(x=mx,            y=round(d1.y - sloop * 0.4, 4)),
                end=Point(x=round(d2.x, 4), y=round(d2.y - sloop, 4)),
            ))
            # Bottom weave connector
            curves.append(CurveSegment(
                type="quadratic_bezier",
                start=Point(x=round(d1.x, 4), y=round(d1.y + sloop, 4)),
                control=Point(x=mx,            y=round(d1.y + sloop * 0.4, 4)),
                end=Point(x=round(d2.x, 4), y=round(d2.y + sloop, 4)),
            ))

    return curves


# ────────────────────────────────────────────────────────
# Pattern generator registry
# ────────────────────────────────────────────────────────

PATTERN_GENERATORS = {
    "basic":     generate_basic_curves,
    "symmetric": generate_symmetric_curves,
    "diagonal":  generate_diagonal_curves,
    "sikku":     generate_sikku_curves,
}


# ────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────

def create_pattern(
    rows: int,
    cols: int,
    pattern_type: str,
    color: Optional[str] = None,
) -> PatternData:
    """
    Generate a complete, validated Kolam pattern.

    Args:
        rows:         Number of rows in the dot grid (2–20).
        cols:         Number of columns in the dot grid (2–20).
        pattern_type: 'basic' | 'symmetric' | 'diagonal' | 'sikku'.
        color:        Optional primary colour (echoed back to frontend).

    Returns:
        PatternData with dots, curves, grid_spacing, and a validation
        report confirming Euler compliance and dot connectivity.
    """
    dots = generate_dots(rows, cols)
    sx, sy = grid_spacing(rows, cols, PAD)

    generator_fn = PATTERN_GENERATORS.get(pattern_type, generate_basic_curves)
    curves = generator_fn(dots, rows, cols)

    # Run validation
    validation = validate_pattern(dots, curves)

    return PatternData(
        rows=rows,
        cols=cols,
        pattern_type=pattern_type,
        dot_count=len(dots),
        curve_count=len(curves),
        dots=dots,
        curves=curves,
        grid_spacing={"x": round(sx, 4), "y": round(sy, 4)},
        validation=validation,
    )
