"""
FastAPI Backend Schema Definitions
===================================
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class ContainerPredictionRequest(BaseModel):
    """Single container prediction request."""
    container_id: str
    declared_value: float
    declared_weight: float
    measured_weight: float
    origin_country: str
    destination_country: str
    destination_port: str
    hs_code: str
    dwell_time_hours: float
    shipping_line: str
    trade_regime: str
    declaration_date: Optional[str] = None
    declaration_time: Optional[str] = None
    clearance_status: Optional[str] = None


class ContainerPredictionResponse(BaseModel):
    """Single container prediction response."""
    container_id: str
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: str = Field(..., pattern="^(Critical|High|Medium|Low)$")
    explanation_summary: str
    confidence: float = Field(..., ge=0, le=1)
    classifier_score: float
    anomaly_score: float


class UploadResponse(BaseModel):
    """Batch upload response."""
    status: str
    message: str
    file_id: str
    rows_received: int
    rows_valid: int
    rows_failed: int
    errors: List[str] = []


class BatchPredictionsResponse(BaseModel):
    """Batch predictions response."""
    status: str
    file_id: str
    total_containers: int
    predictions: List[ContainerPredictionResponse]
    risk_groups: Dict[str, List[ContainerPredictionResponse]] = Field(default_factory=dict)
    summary: dict


class RiskGroupedPredictionsResponse(BaseModel):
    """Predictions grouped by risk level for dashboard display."""
    file_id: str
    total_containers: int
    risk_groups: Dict[str, List[ContainerPredictionResponse]]


class SummaryStatistics(BaseModel):
    """Dashboard summary statistics."""
    total_containers_processed: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    anomaly_count: int
    risk_distribution: dict
    top_risk_factors: List[str]
    average_risk_score: float
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: datetime
