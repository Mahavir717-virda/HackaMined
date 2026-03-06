"""
Configuration Module
=====================
Centralized configuration management for the application.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""

    # API
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))

    # Model
    MODEL_PATH = os.getenv('MODEL_PATH', './models/risk_model.joblib')
    TRAINING_DATA_PATH = os.getenv('TRAINING_DATA_PATH', './data/sample_data.csv')

    # Feature Engineering
    USE_CUSTOM_LR = os.getenv('USE_CUSTOM_LOGISTIC_REGRESSION', 'false').lower() == 'true'
    ANOMALY_CONTAMINATION = float(os.getenv('ANOMALY_CONTAMINATION', 0.1))
    CLASSIFIER_WEIGHT = float(os.getenv('CLASSIFIER_WEIGHT', 0.70))
    ANOMALY_WEIGHT = float(os.getenv('ANOMALY_WEIGHT', 0.30))

    # Risk Thresholds
    RISK_CRITICAL_THRESHOLD = float(os.getenv('RISK_CRITICAL_THRESHOLD', 0.75))
    RISK_HIGH_THRESHOLD = float(os.getenv('RISK_HIGH_THRESHOLD', 0.50))
    RISK_MEDIUM_THRESHOLD = float(os.getenv('RISK_MEDIUM_THRESHOLD', 0.25))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'application.log')

    # Database (for future use)
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./predictions.db')

    @classmethod
    def validate(cls):
        """Validate configuration."""
        assert os.path.exists(cls.MODEL_PATH) or os.path.exists(
            os.path.dirname(cls.MODEL_PATH)
        ), f"Model directory must exist: {os.path.dirname(cls.MODEL_PATH)}"

        assert 0 <= cls.CLASSIFIER_WEIGHT <= 1, "CLASSIFIER_WEIGHT must be between 0 and 1"
        assert 0 <= cls.ANOMALY_WEIGHT <= 1, "ANOMALY_WEIGHT must be between 0 and 1"
        assert abs(
            (cls.CLASSIFIER_WEIGHT + cls.ANOMALY_WEIGHT) - 1.0
        ) < 1e-6, "Classifier and Anomaly weights must sum to 1.0"


# Initialize config
config = Config()
