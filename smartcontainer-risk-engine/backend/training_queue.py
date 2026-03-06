"""
Async Training Queue Manager
=============================
Manages background model retraining with progress tracking.
"""

import os
import logging
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import traceback

import pandas as pd
import numpy as np
from ml.preprocessing.data_cleaner import DataCleaner
from ml.features.feature_engineer import FeatureEngineer
from ml.core.ml_models import RiskDetectionModel

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    CLEANING = "cleaning"
    ENGINEERING = "engineering"
    TRAINING = "training"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingQueue:
    """Manages background model training with progress tracking."""
    
    def __init__(self):
        self.training_jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.model_path = os.getenv("MODEL_PATH", "./models/risk_model.joblib")
        self.backup_path = os.getenv("MODEL_BACKUP_PATH", "./models/risk_model_backup.joblib")
        
    def queue_training(self, job_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Queue a new training job."""
        with self.lock:
            self.training_jobs[job_id] = {
                'status': TrainingStatus.QUEUED,
                'progress': 0,
                'message': 'Training queued',
                'rows_loaded': len(df),
                'rows_valid': 0,
                'started_at': datetime.now().isoformat(),
                'completed_at': None,
                'error': None,
                'metrics': {}
            }
        
        # Start training in background thread
        thread = threading.Thread(
            target=self._train_model,
            args=(job_id, df),
            daemon=True
        )
        thread.start()
        
        return self.training_jobs[job_id]
    
    def _train_model(self, job_id: str, df: pd.DataFrame):
        """Run model training in background."""
        try:
            # Step 1: Data Cleaning
            with self.lock:
                self.training_jobs[job_id]['status'] = TrainingStatus.CLEANING
                self.training_jobs[job_id]['progress'] = 10
                self.training_jobs[job_id]['message'] = 'Cleaning data...'
            
            cleaner = DataCleaner()
            df_clean, clean_stats = cleaner.clean(df, strict=False)
            rows_valid = len(df_clean)
            
            if rows_valid == 0:
                raise ValueError("No valid rows remaining after data cleaning")
            
            with self.lock:
                self.training_jobs[job_id]['rows_valid'] = rows_valid
            
            # Step 2: Feature Engineering
            with self.lock:
                self.training_jobs[job_id]['status'] = TrainingStatus.ENGINEERING
                self.training_jobs[job_id]['progress'] = 30
                self.training_jobs[job_id]['message'] = f'Engineering features ({rows_valid} rows)...'
            
            engineer = FeatureEngineer()
            df_features = engineer.engineer_features(df_clean)
            feature_cols = engineer.get_available_features(df_features)
            
            X = df_features[feature_cols].fillna(0)
            
            # Create risk labels if not present
            if 'risk_flag' not in df_clean.columns:
                # Use heuristics to label high-risk containers
                risk_flags = self._generate_risk_labels(df_clean)
            else:
                risk_flags = df_clean['risk_flag'].astype(int).values
            
            with self.lock:
                self.training_jobs[job_id]['progress'] = 50
            
            # Step 3: Model Training
            with self.lock:
                self.training_jobs[job_id]['status'] = TrainingStatus.TRAINING
                self.training_jobs[job_id]['progress'] = 60
                self.training_jobs[job_id]['message'] = 'Training risk model...'
            
            model = RiskDetectionModel()
            metrics = model.train(X, risk_flags)
            
            with self.lock:
                self.training_jobs[job_id]['metrics'] = metrics
                self.training_jobs[job_id]['progress'] = 85
            
            # Step 4: Save Model
            with self.lock:
                self.training_jobs[job_id]['status'] = TrainingStatus.SAVING
                self.training_jobs[job_id]['progress'] = 90
                self.training_jobs[job_id]['message'] = 'Saving trained model...'
            
            # Backup old model
            if os.path.exists(self.model_path):
                try:
                    os.rename(self.model_path, self.backup_path)
                    logger.info(f"Backed up old model to {self.backup_path}")
                except Exception as e:
                    logger.warning(f"Could not backup old model: {str(e)}")
            
            # Save new model
            model.save(self.model_path)
            
            # Mark as completed
            with self.lock:
                self.training_jobs[job_id]['status'] = TrainingStatus.COMPLETED
                self.training_jobs[job_id]['progress'] = 100
                self.training_jobs[job_id]['message'] = 'Training completed successfully'
                self.training_jobs[job_id]['completed_at'] = datetime.now().isoformat()
            
            logger.info(f"Training job {job_id} completed. Metrics: {metrics}")
            # Print accuracy explicitly to terminal for immediate visibility
            try:
                acc = metrics.get('accuracy') if isinstance(metrics, dict) else None
                if acc is not None:
                    logger.info(f"Training job {job_id} accuracy: {acc:.4f}")
                else:
                    logger.info(f"Training job {job_id} metrics: {metrics}")
            except Exception:
                logger.info(f"Training job {job_id} completed. Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {str(e)}\n{traceback.format_exc()}")
            with self.lock:
                self.training_jobs[job_id]['status'] = TrainingStatus.FAILED
                self.training_jobs[job_id]['progress'] = 0
                self.training_jobs[job_id]['message'] = f'Training failed: {str(e)}'
                self.training_jobs[job_id]['error'] = str(e)
                self.training_jobs[job_id]['completed_at'] = datetime.now().isoformat()
    
    def _generate_risk_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate risk labels using heuristics."""
        labels = np.zeros(len(df), dtype=int)
        
        # Weight discrepancy > 20%
        if 'weight_diff_pct' in df.columns:
            labels |= (abs(df['weight_diff_pct']) > 20).astype(int)
        
        # High value density
        if 'flag_high_value_density' in df.columns:
            labels |= df['flag_high_value_density'].astype(int)
        
        # High-risk country pairing
        if 'origin_country_risk' in df.columns and 'dest_country_risk' in df.columns:
            labels |= ((df['origin_country_risk'] == 2) | (df['dest_country_risk'] == 2)).astype(int)
        
        # Excessive dwell time
        if 'flag_excessive_dwell' in df.columns:
            labels |= df['flag_excessive_dwell'].astype(int)
        
        return labels
    
    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training job status."""
        with self.lock:
            return self.training_jobs.get(job_id)
    
    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        """List all training jobs."""
        with self.lock:
            return dict(self.training_jobs)
