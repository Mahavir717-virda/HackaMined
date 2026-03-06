"""
Model Training Script
======================
End-to-end ML pipeline for training the risk detection model
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.data_cleaner import DataCleaner
from ml.features.feature_engineer import FeatureEngineer
from ml.core.ml_models import RiskDetectionModel, RiskScorer
from ml.core.explainability import RiskExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 500, save_path: str = './data/sample_data.csv'):
    """
    Generate synthetic container shipment data for training.
    """
    logger.info(f"Generating {n_samples} synthetic container samples...")
    
    np.random.seed(42)
    
    countries = ['US', 'CN', 'IN', 'BR', 'DE', 'JP', 'GB', 'FR', 'KP', 'IR', 'NG', 'PK']
    ports = ['PORT_LA', 'PORT_NYC', 'PORT_SH', 'PORT_DXB', 'PORT_SG', 'PORT_HK']
    trade_regimes = ['FREE', 'BOND', 'TRANSIT', 'WAREHOUSE']
    shipping_lines = ['MAERSK', 'MSC', 'CMA_CGM', 'EVERGREEN', 'COSCO']
    
    data = {
        'Container_ID': [f'C{10000+i}' for i in range(n_samples)],
        'Declaration_Date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'Declaration_Time': [f'{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}' 
                            for _ in range(n_samples)],
        'Trade_Regime': np.random.choice(trade_regimes, n_samples),
        'Origin_Country': np.random.choice(countries, n_samples),
        'Destination_Country': np.random.choice(countries, n_samples),
        'Destination_Port': np.random.choice(ports, n_samples),
        'HS_Code': [f'{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
        'Importer_ID': [f'IMP_{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
        'Exporter_ID': [f'EXP_{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
        'Declared_Value': np.random.uniform(1000, 1000000, n_samples),
        'Declared_Weight': np.random.uniform(10, 1000, n_samples),
        'Measured_Weight': np.random.uniform(10, 1000, n_samples),
        'Shipping_Line': np.random.choice(shipping_lines, n_samples),
        'Dwell_Time_Hours': np.random.uniform(1, 200, n_samples),
    }
    
    # Create synthetic labels based on heuristics
    df = pd.DataFrame(data)
    df['is_risky'] = 0
    
    # High-risk countries
    high_risk_countries = {'KP', 'IR', 'NG', 'PK'}
    df.loc[df['Origin_Country'].isin(high_risk_countries), 'is_risky'] = 1
    
    # Weight discrepancies
    weight_diff = (df['Measured_Weight'] - df['Declared_Weight']).abs()
    df.loc[weight_diff > 200, 'is_risky'] = 1
    
    # Value anomalies
    value_per_kg = df['Declared_Value'] / (df['Declared_Weight'] + 1e-9)
    df.loc[(value_per_kg > 5000) | (value_per_kg < 10), 'is_risky'] = 1
    
    # Ensure some positive samples
    positive_count = (df['is_risky'] == 1).sum()
    logger.info(f"Synthetic data class distribution - Risk: {positive_count}, Normal: {len(df) - positive_count}")
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Synthetic data saved to {save_path}")
    
    return df


def train_pipeline(
    data_path: str,
    model_output_path: str = './models/risk_model.joblib',
    use_custom_lr: bool = False,
    label_column: str = "is_risky",
    test_size: float = 0.2,
    classifier_weight: float = 0.7,
    anomaly_weight: float = 0.3,
    report_output_path: str = "",
):
    """
    Execute complete training pipeline.
    """
    logger.info("=" * 60)
    logger.info("SMARTCONTAINER RISK ENGINE - TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Load data
    logger.info("\n[Step 1] Loading data...")
    if not os.path.exists(data_path):
        logger.info(f"Data file not found. Generating synthetic data...")
        df = generate_synthetic_data(save_path=data_path)
    else:
        df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(df)} records")
    
    # Step 2: Data cleaning
    logger.info("\n[Step 2] Data cleaning...")
    cleaner = DataCleaner()
    df, clean_stats = cleaner.clean(df, strict=False)
    logger.info(f"After cleaning: {len(df)} records")
    logger.info(f"Cleaning stats: {clean_stats}")
    
    # Step 3: Feature engineering
    logger.info("\n[Step 3] Feature engineering...")
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    features = engineer.get_available_features(df)
    logger.info(f"Engineered {len(features)} features")
    
    # Prepare X and y
    X = df[features].fillna(0)
    y_col = label_column if label_column in df.columns else None
    
    if y_col is None:
        # Create synthetic labels
        logger.warning(
            "No risk label column '%s' found. Creating heuristic labels...",
            label_column,
        )
        y = (df['flag_weight_mismatch'] | df['flag_high_value_density'] | 
             (df['origin_country_risk'] > 1)).astype(int)
    else:
        y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    # Configure risk score blending (classifier + anomaly)
    RiskScorer.set_weights(classifier_weight, anomaly_weight)
    logger.info(
        "Risk score weights set: classifier=%.3f anomaly=%.3f",
        RiskScorer.CLASSIFIER_WEIGHT,
        RiskScorer.ANOMALY_WEIGHT,
    )
    
    # Step 4: Model training
    logger.info("\n[Step 4] Training risk detection model...")
    model = RiskDetectionModel(use_custom_lr=use_custom_lr)
    training_stats = model.train(X, y, test_size=test_size)
    logger.info(f"Training completed. Stats: {training_stats}")
    
    # Step 5: Save model
    logger.info("\n[Step 5] Saving model...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)
    logger.info(f"Model saved to {model_output_path}")
    
    # Step 6: Generate sample predictions
    logger.info("\n[Step 6] Generating sample predictions...")
    clf_scores, anom_scores = model.predict(X.iloc[:5])
    risk_scores, risk_levels = RiskScorer.score_batch(clf_scores, anom_scores)
    
    explainer = RiskExplainer()
    explanations = explainer.generate_batch_explanations(
        df.iloc[:5], risk_levels, risk_scores
    )
    
    logger.info("\nSample predictions:")
    for i in range(5):
        logger.info(f"  Container {df.iloc[i].get('Container_ID', f'C{i}')}: "
                   f"Risk={risk_levels[i]} (Score={risk_scores[i]:.2f}) - {explanations[i][:60]}...")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)

    if report_output_path:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_path": os.path.abspath(data_path),
            "model_output_path": os.path.abspath(model_output_path),
            "label_column": y_col if y_col else "heuristic_generated",
            "rows_after_cleaning": int(len(df)),
            "feature_count": int(len(features)),
            "risk_scorer_weights": {
                "classifier_weight": float(RiskScorer.CLASSIFIER_WEIGHT),
                "anomaly_weight": float(RiskScorer.ANOMALY_WEIGHT),
            },
            "validation_metrics": training_stats,
        }
        report_dir = os.path.dirname(report_output_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(report_output_path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        logger.info("Training report saved to %s", report_output_path)
    
    return model, training_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train SmartContainer Risk Engine')
    parser.add_argument('--data', type=str, default='./data/sample_data.csv',
                       help='Path to training data CSV')
    parser.add_argument('--model-output', type=str, default='./models/risk_model.joblib',
                       help='Path to save trained model')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate synthetic training data')
    parser.add_argument('--custom-lr', action='store_true',
                       help='Use custom logistic regression instead of GradientBoosting')
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--label-column', type=str, default='is_risky',
                       help='Ground-truth label column (default: is_risky)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Validation split ratio in (0,1), default: 0.2')
    parser.add_argument('--classifier-weight', type=float, default=0.7,
                       help='Classifier weight in final risk score (default: 0.7)')
    parser.add_argument('--anomaly-weight', type=float, default=0.3,
                       help='Anomaly score weight in final risk score (default: 0.3)')
    parser.add_argument('--report-output', type=str, default='',
                       help='Optional path to save training report JSON')
    
    args = parser.parse_args()
    
    if args.generate_data:
        generate_synthetic_data(n_samples=args.samples, save_path=args.data)
    
    train_pipeline(
        data_path=args.data,
        model_output_path=args.model_output,
        use_custom_lr=args.custom_lr,
        label_column=args.label_column,
        test_size=args.test_size,
        classifier_weight=args.classifier_weight,
        anomaly_weight=args.anomaly_weight,
        report_output_path=args.report_output,
    )


if __name__ == '__main__':
    main()
