import numpy as np

from backend.api.main import _build_prediction_summary, _group_predictions_by_risk
from backend.schemas.models import ContainerPredictionResponse


def _prediction(level: str, score: float, anomaly: float = 0.2) -> ContainerPredictionResponse:
    return ContainerPredictionResponse(
        container_id=f"{level}-{score}",
        risk_score=score,
        risk_level=level,
        explanation_summary="test",
        confidence=0.9,
        classifier_score=0.8,
        anomaly_score=anomaly,
    )


def test_group_predictions_keeps_levels_separate():
    predictions = [
        _prediction("Low", 10.0),
        _prediction("Critical", 95.0),
        _prediction("Low", 20.0),
        _prediction("Medium", 55.0),
    ]

    grouped = _group_predictions_by_risk(predictions)

    assert len(grouped["Critical"]) == 1
    assert len(grouped["High"]) == 0
    assert len(grouped["Medium"]) == 1
    assert len(grouped["Low"]) == 2


def test_build_prediction_summary_returns_real_counts():
    predictions = [
        _prediction("Critical", 95.0, anomaly=0.9),
        _prediction("High", 70.0, anomaly=0.8),
        _prediction("Low", 20.0, anomaly=0.1),
    ]
    risk_scores = np.array([0.95, 0.70, 0.20])
    anomaly_scores = np.array([0.90, 0.80, 0.10])

    summary = _build_prediction_summary(predictions, risk_scores, anomaly_scores)

    assert summary["total_containers"] == 3
    assert summary["critical"] == 1
    assert summary["high"] == 1
    assert summary["medium"] == 0
    assert summary["low"] == 1
    assert summary["anomalies"] == 2
    assert summary["risk_distribution"] == {
        "critical": 1,
        "high": 1,
        "medium": 0,
        "low": 1,
    }
