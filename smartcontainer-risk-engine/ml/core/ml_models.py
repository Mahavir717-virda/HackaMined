"""
ML Core Models
==============
Production-grade machine learning models for risk prediction.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib

logger = logging.getLogger(__name__)


class LogisticRegression:
    """Custom logistic regression model from scratch."""

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, 
                 regularization: float = 0.0):
        self.lr = learning_rate
        self.iterations = iterations
        self.reg = regularization
        self.weights = None
        self.bias = None
        self.losses = []

    @staticmethod
    def sigmoid(z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        m = len(y_true)
        loss = -np.mean(y_true * np.log(y_pred + 1e-15) + 
                       (1 - y_true) * np.log(1 - y_pred + 1e-15))
        # L2 regularization
        loss += (self.reg / (2 * m)) * np.sum(self.weights ** 2)
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train logistic regression model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            Training history
        """
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for iteration in range(self.iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Compute loss
            loss = self.binary_cross_entropy(y, predictions)
            self.losses.append(loss)

            # Backward pass
            dz = predictions - y
            dw = (1 / m) * np.dot(X.T, dz) + (self.reg / m) * self.weights
            db = (1 / m) * np.sum(dz)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if (iteration + 1) % 100 == 0:
                logger.info(f"Iteration {iteration + 1}: Loss = {loss:.4f}")

        return {
            'losses': self.losses,
            'final_loss': self.losses[-1],
            'iterations': self.iterations
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions."""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return class predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


class RiskDetectionModel:
    """Production risk detection model combining classifier and anomaly detection."""

    def __init__(
        self,
        use_custom_lr: bool = False,
        anomaly_n_jobs: Optional[int] = None,
        use_balanced_weights: bool = True,
    ):
        self.use_custom_lr = use_custom_lr
        self.use_balanced_weights = use_balanced_weights
        self.classifier = None
        self.scaler = StandardScaler()
        if anomaly_n_jobs is None:
            try:
                anomaly_n_jobs = int(os.getenv("ANOMALY_N_JOBS", "1"))
            except ValueError:
                anomaly_n_jobs = 1
        self.anomaly_n_jobs = anomaly_n_jobs
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            n_jobs=self.anomaly_n_jobs
        )
        self.feature_cols = None
        self.training_stats = {}

    def _prepare_data(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare and scale feature data."""
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train risk detection model.

        Args:
            X: Feature DataFrame
            y: Target labels
            test_size: Test set fraction

        Returns:
            Training metrics and history
        """
        logger.info("Training RiskDetectionModel...")

        self.feature_cols = X.columns.tolist()

        # Handle class imbalance
        unique_classes = y.unique()
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")

        if len(unique_classes) < 2:
            logger.warning("Only one class in labels. Creating synthetic minority class...")
            n_minority = max(int(len(y) * 0.1), 1)
            minority_indices = np.random.choice(len(y), size=n_minority, replace=False)
            y_synthetic = y.copy()
            y_synthetic.iloc[minority_indices] = 1 - y.iloc[minority_indices]
            y = y_synthetic
            logger.info(f"Updated: {dict(y.value_counts())}")

        # Split data
        stratify_arg = y if len(y.unique()) == 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_arg
        )

        # Fit scaler
        self.scaler.fit(X_train)
        X_train_scaled = self._prepare_data(X_train)
        X_test_scaled = self._prepare_data(X_test)

        # Train classifier
        if self.use_custom_lr:
            logger.info("Using custom logistic regression...")
            self.classifier = LogisticRegression(learning_rate=0.01, iterations=500)
            history = self.classifier.fit(X_train_scaled, y_train.values)
            y_pred_proba = self.classifier.predict_proba(X_test_scaled)
            y_pred = self.classifier.predict(X_test_scaled)
        else:
            logger.info("Using GradientBoostingClassifier...")
            self.classifier = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42
            )
            fit_kwargs = {}
            if self.use_balanced_weights:
                fit_kwargs["sample_weight"] = compute_sample_weight(
                    class_weight="balanced",
                    y=y_train,
                )
            self.classifier.fit(X_train_scaled, y_train, **fit_kwargs)
            y_pred_proba = self.classifier.predict_proba(X_test_scaled)[:, 1]
            y_pred = self.classifier.predict(X_test_scaled)

        # Train anomaly detector
        try:
            self.anomaly_detector.fit(X_train_scaled)
        except (PermissionError, OSError) as e:
            if getattr(self.anomaly_detector, "n_jobs", 1) != 1:
                logger.warning(
                    "IsolationForest parallel fit failed (%s). Retrying with n_jobs=1.",
                    str(e),
                )
                self.anomaly_detector.set_params(n_jobs=1)
                self.anomaly_detector.fit(X_train_scaled)
            else:
                raise

        # Compute metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        self.training_stats = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(self.feature_cols),
            'anomaly_n_jobs': getattr(self.anomaly_detector, "n_jobs", self.anomaly_n_jobs),
        }

        logger.info(f"Training complete. AUC: {auc:.4f}")
        logger.info(
            "Validation metrics: precision=%.4f recall=%.4f f1=%.4f",
            precision,
            recall,
            f1,
        )
        logger.info(
            f"Classification report:\n{classification_report(y_test, y_pred, zero_division=0)}"
        )

        return self.training_stats

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions and anomaly scores.

        Args:
            X: Feature DataFrame

        Returns:
            (classification_scores, anomaly_scores)
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self._prepare_data(X)

        # Classification probabilities
        if isinstance(self.classifier, LogisticRegression):
            clf_scores = self.classifier.predict_proba(X_scaled)
        else:
            clf_scores = self.classifier.predict_proba(X_scaled)[:, 1]

        # Anomaly scores (normalized to 0-1)
        anom_raw = self.anomaly_detector.score_samples(X_scaled)
        anom_scores = 1 - (anom_raw - anom_raw.min()) / (anom_raw.max() - anom_raw.min() + 1e-9)
        anom_scores = np.clip(anom_scores, 0, 1)

        return np.clip(clf_scores, 0, 1), anom_scores

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        artifact = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'anomaly_detector': self.anomaly_detector,
            'feature_cols': self.feature_cols,
            'training_stats': self.training_stats,
        }
        joblib.dump(artifact, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load trained model from disk."""
        artifact = joblib.load(path)
        self.classifier = artifact['classifier']
        self.scaler = artifact['scaler']
        self.anomaly_detector = artifact['anomaly_detector']
        self.feature_cols = artifact['feature_cols']
        self.training_stats = artifact.get('training_stats', {})
        self.anomaly_n_jobs = getattr(self.anomaly_detector, "n_jobs", 1)
        logger.info(f"Model loaded from {path}")


class RiskScorer:
    """Risk scoring and classification logic."""

    # Risk thresholds
    SCORE_THRESHOLDS = {
        'critical': 0.75,
        'high': 0.50,
        'medium': 0.25,
        'low': 0.0
    }

    # Weighting
    CLASSIFIER_WEIGHT = float(os.getenv("CLASSIFIER_WEIGHT", "0.70"))
    ANOMALY_WEIGHT = float(os.getenv("ANOMALY_WEIGHT", "0.30"))

    @classmethod
    def set_weights(cls, classifier_weight: float, anomaly_weight: float) -> None:
        """Set score-combination weights safely."""
        if classifier_weight < 0 or anomaly_weight < 0:
            raise ValueError("Weights must be non-negative.")
        if classifier_weight + anomaly_weight == 0:
            raise ValueError("At least one weight must be > 0.")
        cls.CLASSIFIER_WEIGHT = float(classifier_weight)
        cls.ANOMALY_WEIGHT = float(anomaly_weight)

    @classmethod
    def compute_risk_score(cls, clf_score: float, anom_score: float) -> float:
        """
        Combine classifier and anomaly scores.

        Args:
            clf_score: Classification probability (0-1)
            anom_score: Anomaly score (0-1)

        Returns:
            Combined risk score (0-1)
        """
        total_weight = cls.CLASSIFIER_WEIGHT + cls.ANOMALY_WEIGHT
        if total_weight <= 0:
            classifier_weight = 0.7
            anomaly_weight = 0.3
        else:
            classifier_weight = cls.CLASSIFIER_WEIGHT / total_weight
            anomaly_weight = cls.ANOMALY_WEIGHT / total_weight
        risk_score = (
            classifier_weight * clf_score +
            anomaly_weight * anom_score
        )
        return np.clip(risk_score, 0, 1)

    @classmethod
    def classify_risk(cls, risk_score: float) -> str:
        """Classify risk level."""
        if risk_score >= cls.SCORE_THRESHOLDS['critical']:
            return 'Critical'
        elif risk_score >= cls.SCORE_THRESHOLDS['high']:
            return 'High'
        elif risk_score >= cls.SCORE_THRESHOLDS['medium']:
            return 'Medium'
        else:
            return 'Low'

    @classmethod
    def score_batch(cls, clf_scores: np.ndarray, anom_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score batch of predictions.

        Returns:
            (risk_scores, risk_levels)
        """
        risk_scores = np.array([
            cls.compute_risk_score(c, a) for c, a in zip(clf_scores, anom_scores)
        ])
        risk_levels = np.array([cls.classify_risk(s) for s in risk_scores])

        return risk_scores, risk_levels
