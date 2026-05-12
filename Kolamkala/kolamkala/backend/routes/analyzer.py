"""
analyzer.py — FastAPI route handler for the Kolam image analyzer.

Endpoint: POST /analyze
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from ..models.response_models import AnalyzeResponse
from ..services.analyzer_service import analyze_image

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a Kolam Image",
    description=(
        "Upload a Kolam image (JPG, PNG, or WebP) and receive a detailed pattern analysis. "
        "OpenCV extracts approximate counts of lines, loops, and dots, "
        "and checks horizontal/vertical symmetry. "
        "Results are estimates — accuracy depends on image quality."
    ),
)
async def analyze_kolam(
    file: UploadFile = File(..., description="Kolam image to analyze (JPG, PNG, or WebP, max 10 MB)")
):
    """
    Validates the uploaded file, passes its bytes to the analyzer service,
    and returns a structured analysis report.

    Analysis fields:
    - **lines**:    Number of detected line segments.
    - **loops**:    Approximate number of closed loop curves.
    - **symmetry**: Detected symmetry ('horizontal', 'vertical', 'both', or 'none').
    - **dots**:     Estimated number of circular dot marks.
    - **note**:     Accuracy disclaimer.
    """
    # Validate MIME type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                "Please upload a JPG, PNG, or WebP image."
            ),
        )

    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 10 MB.",
        )

    try:
        analysis = analyze_image(contents)
        return AnalyzeResponse(status="success", analysis=analysis)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(exc)}",
        )
