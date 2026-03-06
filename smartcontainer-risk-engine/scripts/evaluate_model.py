#!/usr/bin/env python3
"""
Evaluate model quality on labeled CSV data.

Usage example:
  python scripts/evaluate_model.py --data ./data/historical_data_processed.csv --label-column is_risky
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.data_cleaner import DataCleaner
from ml.features.feature_engineer import FeatureEngineer
from ml.core.ml_models import RiskDetectionModel, RiskScorer

POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "y",
    "risk",
    "risky",
    "high",
    "high risk",
    "critical",
    "flagged",
    "seized",
    "rejected",
    "anomaly",
    "fraud",
}
NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "n",
    "normal",
    "safe",
    "low",
    "low risk",
    "clear",
    "cleared",
}


def normalize_label(value) -> float:
    """Convert many common label formats into binary 0/1."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(float(value) > 0)

    text = str(value).strip().lower()
    if not text:
        return np.nan
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0
    if "critical" in text or "high" in text or "risky" in text:
        return 1
    if "low" in text or "clear" in text or "safe" in text:
        return 0
    return np.nan


def _metric(value: float) -> float:
    if value is None or np.isnan(value):
        return float("nan")
    return float(value)


def evaluate_model(
    data_path: str,
    model_path: str,
    label_column: str,
    threshold: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    raw_df = pd.read_csv(data_path)
    if label_column not in raw_df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in CSV. "
            f"Available columns: {', '.join(raw_df.columns)}"
        )

    y_binary = raw_df[label_column].apply(normalize_label)
    valid_label_mask = y_binary.notna()
    if valid_label_mask.sum() == 0:
        raise ValueError(
            f"No valid labels found in column '{label_column}'. "
            "Use 0/1, true/false, low/critical, clear/flagged, etc."
        )

    df_with_labels = raw_df.loc[valid_label_mask].copy()
    y = y_binary.loc[valid_label_mask].astype(int)

    cleaner = DataCleaner()
    df_clean, _ = cleaner.clean(df_with_labels, strict=False)
    y = y.loc[df_clean.index]

    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df_clean)
    feature_cols = engineer.get_available_features(df_features)
    X = df_features[feature_cols].fillna(0)

    model = RiskDetectionModel()
    model.load(model_path)

    clf_scores, anomaly_scores = model.predict(X)
    risk_scores, risk_levels = RiskScorer.score_batch(clf_scores, anomaly_scores)
    y_pred = (risk_scores >= threshold).astype(int)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, risk_scores) if len(np.unique(y)) > 1 else float("nan")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    report = {
        "total_rows": int(len(raw_df)),
        "rows_with_valid_labels": int(valid_label_mask.sum()),
        "rows_without_valid_labels": int(len(raw_df) - valid_label_mask.sum()),
        "rows_after_cleaning": int(len(df_clean)),
        "threshold": float(threshold),
        "accuracy": _metric(accuracy),
        "precision": _metric(precision),
        "recall": _metric(recall),
        "f1_score": _metric(f1),
        "roc_auc": _metric(auc),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
    }

    result_df = pd.DataFrame(
        {
            "container_id": df_clean.get("Container_ID", pd.Series(range(len(df_clean)))).astype(str),
            "actual_label": y.astype(int).values,
            "predicted_label": y_pred.astype(int),
            "risk_score": (risk_scores * 100).round(4),
            "risk_level": risk_levels,
            "classifier_score": clf_scores.round(6),
            "anomaly_score": anomaly_scores.round(6),
            "is_correct": (y.values == y_pred).astype(int),
        },
        index=df_clean.index,
    )

    return report, result_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate model on labeled CSV data.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to labeled CSV data.",
    )
    parser.add_argument(
        "--label-column",
        default="is_risky",
        help="Ground-truth label column (default: is_risky).",
    )
    parser.add_argument(
        "--model",
        default="./models/risk_model.joblib",
        help="Path to trained model (default: ./models/risk_model.joblib).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary decision threshold on risk score in [0,1] (default: 0.5).",
    )
    parser.add_argument(
        "--output-predictions",
        default="",
        help="Optional CSV path to save row-level predictions.",
    )
    parser.add_argument(
        "--classifier-weight",
        type=float,
        default=0.7,
        help="Classifier score weight (default: 0.7).",
    )
    parser.add_argument(
        "--anomaly-weight",
        type=float,
        default=0.3,
        help="Anomaly score weight (default: 0.3).",
    )
    args = parser.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be between 0 and 1.")
    RiskScorer.set_weights(args.classifier_weight, args.anomaly_weight)

    report, result_df = evaluate_model(
        data_path=args.data,
        model_path=args.model,
        label_column=args.label_column,
        threshold=args.threshold,
    )
    report["risk_scorer_weights"] = {
        "classifier_weight": float(RiskScorer.CLASSIFIER_WEIGHT),
        "anomaly_weight": float(RiskScorer.ANOMALY_WEIGHT),
    }

    print("Model evaluation report:")
    print(json.dumps(report, indent=2))

    if args.output_predictions:
        output_path = os.path.abspath(args.output_predictions)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"Saved row-level predictions to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
