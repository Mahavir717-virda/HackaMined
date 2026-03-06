"""
FastAPI Backend Service
=======================
Main API service for SmartContainer Risk Engine
"""

import os
import logging
import re
from typing import List, Dict, Any
from datetime import datetime
import uuid
from io import StringIO

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import ML modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.preprocessing.data_cleaner import DataCleaner
from ml.features.feature_engineer import FeatureEngineer
from ml.core.ml_models import RiskDetectionModel, RiskScorer
from ml.core.explainability import RiskExplainer

# Import backend schemas using absolute path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.schemas.models import (
    ContainerPredictionResponse,
    UploadResponse,
    BatchPredictionsResponse,
    RiskGroupedPredictionsResponse,
    SummaryStatistics,
    HealthResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SmartContainer Risk Engine API",
    description="AI/ML-based container shipment risk analysis and prediction",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    def __init__(self):
        self.model = None
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.explainer = RiskExplainer()
        self.predictions_cache = {}
        self.model_version = "1.0.0"
        self.model_path = os.getenv("MODEL_PATH", "./models/risk_model.joblib")

app_state = AppState()
RISK_LEVEL_ORDER = ("Critical", "High", "Medium", "Low")
RISK_LEVEL_PRIORITY = {level: idx for idx, level in enumerate(RISK_LEVEL_ORDER)}


def _prediction_to_dict(prediction: ContainerPredictionResponse) -> Dict[str, Any]:
    if hasattr(prediction, "model_dump"):
        return prediction.model_dump()
    return prediction.dict()


def _group_predictions_by_risk(
    predictions: List[ContainerPredictionResponse],
) -> Dict[str, List[ContainerPredictionResponse]]:
    grouped: Dict[str, List[ContainerPredictionResponse]] = {
        level: [] for level in RISK_LEVEL_ORDER
    }
    for prediction in predictions:
        grouped.setdefault(prediction.risk_level, []).append(prediction)
    return grouped


def _build_prediction_summary(
    predictions: List[ContainerPredictionResponse],
    risk_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    source_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    counts = {
        level.lower(): sum(1 for p in predictions if p.risk_level == level)
        for level in RISK_LEVEL_ORDER
    }
    anomaly_count = int(np.sum(anomaly_scores > 0.7))
    avg_risk_score = float(np.mean(risk_scores) * 100) if len(risk_scores) else 0.0

    summary = {
        "total_containers": len(predictions),
        "critical": counts["critical"],
        "high": counts["high"],
        "medium": counts["medium"],
        "low": counts["low"],
        "anomalies": anomaly_count,
        "risk_distribution": counts,
        "average_risk_score": avg_risk_score,
        "processed_at": datetime.now().isoformat(),
    }
    if source_df is not None and "Clearance_Status" in source_df.columns:
        clearance_counts = (
            source_df["Clearance_Status"]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .to_dict()
        )
        summary["clearance_status_distribution"] = clearance_counts

    return summary


def _normalize_upload_header_name(name: str) -> str:
    text = str(name).replace("\ufeff", "").strip()
    text = re.sub(r"\s*\([^)]*\)\s*", "", text)
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def _normalize_upload_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_lookup = {
        _normalize_upload_header_name(required): required
        for required in app_state.cleaner.REQUIRED_FIELDS.keys()
    }
    column_mapping = {}
    for col in df.columns:
        normalized = _normalize_upload_header_name(col)
        mapped = required_lookup.get(normalized)
        if mapped:
            column_mapping[col] = mapped
    if column_mapping:
        df = df.rename(columns=column_mapping)
    return df


def _read_uploaded_csv(contents: bytes) -> pd.DataFrame:
    """
    Parse uploaded CSV defensively.

    `index_col=False` prevents pandas from auto-using the first column as index
    when rows include a trailing comma (common when optional last column is blank).
    """
    text = contents.decode("utf-8-sig")
    # Some exports append trailing "," for absent optional tail columns.
    # Strip only end-of-line empty fields to preserve core column alignment.
    sanitized_lines = [re.sub(r"(,\s*)+$", "", line) for line in text.splitlines()]
    sanitized_text = "\n".join(sanitized_lines)

    df = pd.read_csv(StringIO(sanitized_text), index_col=False)

    # Drop parser-generated empty placeholder columns (e.g., "Unnamed: 15").
    empty_unnamed_columns: List[str] = []
    for col in df.columns:
        col_name = str(col).strip().lower()
        if not col_name.startswith("unnamed"):
            continue
        values = df[col].fillna("").astype(str).str.strip()
        if (values == "").all():
            empty_unnamed_columns.append(col)
    if empty_unnamed_columns:
        df = df.drop(columns=empty_unnamed_columns)

    return df


def _filter_legacy_optional_schema_errors(errors: List[str]) -> List[str]:
    """
    Backward-compatible safety:
    Some older clients/models treated Clearance_Status as required.
    We now allow prediction without that column.
    """
    return [
        err for err in errors
        if "Missing required field: Clearance_Status" not in err
    ]


def load_model():
    """Load ML model at startup."""
    try:
        if os.path.exists(app_state.model_path):
            app_state.model = RiskDetectionModel()
            app_state.model.load(app_state.model_path)
            logger.info(f"Model loaded successfully from {app_state.model_path}")
        else:
            logger.warning(f"Model file not found at {app_state.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize app on startup."""
    load_model()
    logger.info("SmartContainer Risk Engine API started")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=(app_state.model is not None),
        model_version=app_state.model_version,
        timestamp=datetime.now()
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload CSV file for batch processing.
    
    Returns file_id for use in /predict endpoint.
    """
    try:
        # Generate file ID
        file_id = str(uuid.uuid4())[:8]
        
        # Read uploaded file
        contents = await file.read()
        df = _read_uploaded_csv(contents)
        df = _normalize_upload_columns(df)
        
        rows_received = len(df)
        logger.info(f"Uploaded file {file_id}: {rows_received} rows")
        
        # Validate schema
        valid, schema_errors = app_state.cleaner.validate_schema(df)
        schema_errors = _filter_legacy_optional_schema_errors(schema_errors)
        valid = len(schema_errors) == 0
        rows_failed = 0 if valid else rows_received
        
        if not valid:
            return UploadResponse(
                status="partial_success",
                message="File uploaded but validation failed",
                file_id=file_id,
                rows_received=rows_received,
                rows_valid=0,
                rows_failed=rows_failed,
                errors=schema_errors
            )
        
        # Clean data
        df_clean, stats = app_state.cleaner.clean(df, strict=False)
        rows_valid = len(df_clean)
        rows_failed = rows_received - rows_valid

        if rows_valid == 0:
            return UploadResponse(
                status="partial_success",
                message="File uploaded but no valid rows remained after cleaning",
                file_id=file_id,
                rows_received=rows_received,
                rows_valid=0,
                rows_failed=rows_failed,
                errors=[
                    (
                        "No valid rows found after preprocessing. "
                        "Ensure required columns are correctly aligned and numeric fields are valid."
                    )
                ],
            )
        
        # Cache cleaned data
        app_state.predictions_cache[file_id] = {
            'data': df_clean,
            'original_rows': rows_received,
            'timestamp': datetime.now()
        }
        
        return UploadResponse(
            status="success",
            message="File uploaded and validated successfully",
            file_id=file_id,
            rows_received=rows_received,
            rows_valid=rows_valid,
            rows_failed=rows_failed,
            errors=[]
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=BatchPredictionsResponse)
async def predict(file_id: str = None, use_cached: bool = True):
    """
    Generate risk predictions for uploaded dataset.
    """
    try:
        if app_state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable."
            )
        
        # Get data from cache
        if file_id not in app_state.predictions_cache:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found. Upload first with /upload endpoint."
            )
        
        df = app_state.predictions_cache[file_id]['data']
        logger.info(f"Processing predictions for {file_id}: {len(df)} containers")
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No valid rows available for prediction. "
                    "Upload a CSV with required columns and at least one valid row."
                ),
            )
        
        # Feature engineering
        df_features = app_state.engineer.engineer_features(df)
        
        # Prepare features
        feature_cols = app_state.engineer.get_available_features(df_features)
        X = df_features[feature_cols].fillna(0)
        
        # Make predictions
        clf_scores, anom_scores = app_state.model.predict(X)
        risk_scores, risk_levels = RiskScorer.score_batch(clf_scores, anom_scores)
        
        # Generate explanations
        risk_scores_normalized = risk_scores
        explanations = app_state.explainer.generate_batch_explanations(
            df_features, risk_levels, risk_scores_normalized
        )
        
        # Build response predictions
        predictions = []
        for idx in range(len(df)):
            pred = ContainerPredictionResponse(
                container_id=str(df.iloc[idx].get('Container_ID', f'C{idx}')),
                risk_score=float(risk_scores[idx] * 100),
                risk_level=risk_levels[idx],
                explanation_summary=explanations[idx],
                confidence=float(np.max([clf_scores[idx], anom_scores[idx]])),
                classifier_score=float(clf_scores[idx]),
                anomaly_score=float(anom_scores[idx])
            )
            predictions.append(pred)

        # Keep highest-risk containers first for dashboard display.
        predictions.sort(
            key=lambda p: (
                RISK_LEVEL_PRIORITY.get(p.risk_level, len(RISK_LEVEL_PRIORITY)),
                -p.risk_score,
            )
        )

        risk_groups = _group_predictions_by_risk(predictions)
        summary = _build_prediction_summary(
            predictions,
            risk_scores,
            anom_scores,
            source_df=df,
        )

        app_state.predictions_cache[file_id].update(
            {
                "predictions": [_prediction_to_dict(p) for p in predictions],
                "risk_groups": {
                    level: [_prediction_to_dict(p) for p in group]
                    for level, group in risk_groups.items()
                },
                "summary": summary,
            }
        )
        
        logger.info(f"Predictions complete: {summary}")
        
        return BatchPredictionsResponse(
            status="success",
            file_id=file_id,
            total_containers=len(predictions),
            predictions=predictions,
            risk_groups=risk_groups,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary", response_model=SummaryStatistics)
async def get_summary(file_id: str = None):
    """Get summary statistics for processed batch."""
    try:
        cache_entry = app_state.predictions_cache.get(file_id) if file_id else None

        if not cache_entry:
            return SummaryStatistics(
                total_containers_processed=0,
                critical_count=0,
                high_count=0,
                medium_count=0,
                low_count=0,
                anomaly_count=0,
                risk_distribution={
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                },
                top_risk_factors=["weight_discrepancy", "unusual_route", "high_value_density"],
                average_risk_score=0.0,
                timestamp=datetime.now(),
            )

        summary = cache_entry.get("summary")
        total = len(cache_entry.get("data", []))

        if summary:
            total = int(summary.get("total_containers", total))
            critical_count = int(summary.get("critical", 0))
            high_count = int(summary.get("high", 0))
            medium_count = int(summary.get("medium", 0))
            low_count = int(summary.get("low", 0))
            anomaly_count = int(summary.get("anomalies", 0))
            average_risk_score = float(summary.get("average_risk_score", 0.0))
            risk_distribution = summary.get("risk_distribution", {})
        else:
            critical_count = 0
            high_count = 0
            medium_count = 0
            low_count = 0
            anomaly_count = 0
            average_risk_score = 0.0
            risk_distribution = {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            }

        return SummaryStatistics(
            total_containers_processed=total,
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            anomaly_count=anomaly_count,
            risk_distribution={
                "critical": int(risk_distribution.get("critical", critical_count)),
                "high": int(risk_distribution.get("high", high_count)),
                "medium": int(risk_distribution.get("medium", medium_count)),
                "low": int(risk_distribution.get("low", low_count)),
            },
            top_risk_factors=["weight_discrepancy", "unusual_route", "high_value_density"],
            average_risk_score=average_risk_score,
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions-by-risk", response_model=RiskGroupedPredictionsResponse)
async def get_predictions_by_risk(file_id: str):
    """Return predictions separated by risk level for dashboard tables/charts."""
    try:
        if not file_id or file_id not in app_state.predictions_cache:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found. Upload and run /predict first.",
            )

        cache_entry = app_state.predictions_cache[file_id]
        risk_groups_payload = cache_entry.get("risk_groups")

        if not risk_groups_payload:
            predictions_payload = cache_entry.get("predictions")
            if not predictions_payload:
                raise HTTPException(
                    status_code=404,
                    detail=f"No predictions found for {file_id}. Run /predict first.",
                )

            predictions_models = [
                p if isinstance(p, ContainerPredictionResponse) else ContainerPredictionResponse(**p)
                for p in predictions_payload
            ]
            grouped_models = _group_predictions_by_risk(predictions_models)
            risk_groups_payload = {
                level: [_prediction_to_dict(p) for p in grouped]
                for level, grouped in grouped_models.items()
            }

        total_containers = sum(len(items) for items in risk_groups_payload.values())
        normalized_payload = {
            level: risk_groups_payload.get(level, [])
            for level in RISK_LEVEL_ORDER
        }

        return RiskGroupedPredictionsResponse(
            file_id=file_id,
            total_containers=total_containers,
            risk_groups=normalized_payload,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Predictions-by-risk error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-single")
async def predict_single(container: dict):
    """
    Predict risk for a single container.
    """
    try:
        if app_state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([container])
        df = _normalize_upload_columns(df)
        
        # Clean and engineer features
        df_clean, _ = app_state.cleaner.clean(df, strict=False)
        if df_clean.empty:
            raise HTTPException(
                status_code=400,
                detail="Input row is invalid after preprocessing. Check required field values.",
            )
        df_features = app_state.engineer.engineer_features(df_clean)
        
        # Prepare features
        feature_cols = app_state.engineer.get_available_features(df_features)
        X = df_features[feature_cols].fillna(0)
        
        # Predict
        clf_scores, anom_scores = app_state.model.predict(X)
        risk_scores, risk_levels = RiskScorer.score_batch(clf_scores, anom_scores)
        
        explanation = app_state.explainer.generate_explanation(
            df_features.iloc[0], risk_levels[0], risk_scores[0]
        )
        
        return {
            'container_id': container.get('Container_ID', 'UNKNOWN'),
            'risk_score': float(risk_scores[0] * 100),
            'risk_level': risk_levels[0],
            'explanation': explanation,
            'classifier_score': float(clf_scores[0]),
            'anomaly_score': float(anom_scores[0])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000))
    )
