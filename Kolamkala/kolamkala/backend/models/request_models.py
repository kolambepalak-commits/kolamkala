"""
request_models.py — Pydantic input schemas for KolamKala API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    rows: int = Field(default=7, ge=2, le=20, description="Number of rows in the dot grid")
    cols: int = Field(default=7, ge=2, le=20, description="Number of columns in the dot grid")
    pattern_type: str = Field(
        default="basic",
        description="Pattern style: 'basic' (fish-eye loops), 'symmetric' (loops + diagonals + ellipses), 'diagonal' (S-curves)"
    )
    color: Optional[str] = Field(
        default=None,
        description="Primary colour as hex string, e.g. '#7B1818'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "rows": 7,
                "cols": 7,
                "pattern_type": "basic",
                "color": "#7B1818"
            }
        }
