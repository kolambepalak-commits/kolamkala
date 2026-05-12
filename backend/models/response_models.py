"""
response_models.py — Pydantic output schemas for KolamKala API responses.
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Point(BaseModel):
    x: float
    y: float


class Dot(BaseModel):
    row: int
    col: int
    x: float
    y: float


class CurveSegment(BaseModel):
    type: str
    start:   Optional[Point] = None
    control: Optional[Point] = None
    end:     Optional[Point] = None
    center:  Optional[Point] = None
    rx:      Optional[float] = None
    ry:      Optional[float] = None


class PatternValidation(BaseModel):
    is_valid:        bool
    total_curves:    int
    dots_connected:  int
    total_dots:      int
    odd_endpoints:   int
    euler_compliant: bool


class PatternData(BaseModel):
    rows:         int
    cols:         int
    pattern_type: str
    dot_count:    int
    curve_count:  int
    dots:         List[Dot]
    curves:       List[CurveSegment]
    grid_spacing: Dict[str, float]
    validation:   Optional[PatternValidation] = None


class GenerateResponse(BaseModel):
    status:  str
    pattern: PatternData


class AnalysisData(BaseModel):
    lines:          int
    loops:          int
    symmetry:       str
    symmetry_score: float = 0.0
    dots:           int
    kolam_type:     str
    complexity:     str
    note:           Optional[str] = None


class AnalyzeResponse(BaseModel):
    status:   str
    analysis: AnalysisData
