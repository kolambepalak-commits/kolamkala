"""
generator.py — FastAPI route handler for the Kolam pattern generator.

Endpoint: POST /generate
"""
from fastapi import APIRouter, HTTPException
from ..models.request_models import GenerateRequest
from ..models.response_models import GenerateResponse
from ..services.generator_service import create_pattern

router = APIRouter()

VALID_PATTERN_TYPES = {"basic", "symmetric", "diagonal", "sikku"}


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate a Kolam Pattern",
    description=(
        "Generate structured dot-and-curve coordinate data for a Kolam pattern "
        "using rule-based geometric algorithms. "
        "Choose from four pattern styles:\n\n"
        "- **basic** — fish-eye loops between adjacent dots (Pulli-style)\n"
        "- **symmetric** — loops + diagonal Bézier curves + ellipses (Kambi-style)\n"
        "- **diagonal** — S-curves along block diagonals\n"
        "- **sikku** — interlocking closed loops woven around every dot (authentic Sikku geometry)\n\n"
        "The response contains normalised [0, 1] coordinates ready for canvas rendering, "
        "plus a validation report confirming Euler compliance and dot connectivity."
    ),
)
def generate_kolam(request: GenerateRequest):
    """
    Validates the request, calls the generator service, and returns pattern data.

    - **rows** / **cols**: Dot grid dimensions (2–20 each).
    - **pattern_type**: One of 'basic', 'symmetric', 'diagonal', or 'sikku'.
    - **color**: Optional hex colour string (echoed back, used by the frontend).
    """
    if request.pattern_type not in VALID_PATTERN_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid pattern_type '{request.pattern_type}'. "
                f"Choose from: {', '.join(sorted(VALID_PATTERN_TYPES))}."
            ),
        )

    try:
        pattern = create_pattern(
            rows=request.rows,
            cols=request.cols,
            pattern_type=request.pattern_type,
            color=request.color,
        )
        return GenerateResponse(status="success", pattern=pattern)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pattern generation failed: {str(exc)}",
        )
