"""
Unit Tests for ML Models
=========================
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.core.ml_models import LogisticRegression, RiskDetectionModel, RiskScorer


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    
    # Create binary labels
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y


class TestLogisticRegression:
    """Test custom LogisticRegression implementation."""

    def test_sigmoid(self):
        """Test sigmoid activation function."""
        lr = LogisticRegression()
        
        assert abs(lr.sigmoid(0) - 0.5) < 1e-10  # sigmoid(0) = 0.5
        assert lr.sigmoid(100) > 0.99  # large positive
        assert lr.sigmoid(-100) < 0.01  # large negative

    def test_fit(self, sample_features):
        """Test model training."""
        X, y = sample_features
        X_train = X.values[:80]
        y_train = y.values[:80]
        
        lr = LogisticRegression(learning_rate=0.01, iterations=100)
        history = lr.fit(X_train, y_train)
        
        assert 'losses' in history
        assert 'final_loss' in history
        assert len(history['losses']) == 100
        # Loss should generally decrease
        assert history['losses'][-1] <= history['losses'][0]

    def test_predict(self, sample_features):
        """Test predictions."""
        X, y = sample_features
        X_train = X.values[:80]
        y_train = y.values[:80]
        X_test = X.values[80:]
        
        lr = LogisticRegression(learning_rate=0.01, iterations=100)
        lr.fit(X_train, y_train)
        
        proba = lr.predict_proba(X_test)
        pred = lr.predict(X_test)
        
        assert proba.shape == (len(X_test),)
        assert pred.shape == (len(X_test),)
        assert np.all((proba >= 0) & (proba <= 1))
        assert np.all((pred == 0) | (pred == 1))


class TestRiskDetectionModel:
    """Test RiskDetectionModel class."""

    def test_initialization(self):
        """Test model initialization."""
        model = RiskDetectionModel()
        assert model.classifier is None
        assert model.feature_cols is None

    def test_train(self, sample_features):
        """Test model training."""
        X, y = sample_features
        
        model = RiskDetectionModel(use_custom_lr=False)
        stats = model.train(X, y, test_size=0.2)
        
        assert 'auc' in stats
        assert 'train_size' in stats
        assert 'test_size' in stats
        assert 'n_features' in stats
        assert model.classifier is not None
        assert model.feature_cols is not None

    def test_predict(self, sample_features):
        """Test predictions."""
        X, y = sample_features
        
        model = RiskDetectionModel()
        model.train(X, y)
        
        clf_scores, anom_scores = model.predict(X)
        
        assert clf_scores.shape == (len(X),)
        assert anom_scores.shape == (len(X),)
        assert np.all((clf_scores >= 0) & (clf_scores <= 1))
        assert np.all((anom_scores >= 0) & (anom_scores <= 1))

    def test_save_load(self, sample_features):
        """Test model serialization."""
        X, y = sample_features
        
        model = RiskDetectionModel()
        model.train(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            model.save(temp_path)
            assert os.path.exists(temp_path)
            
            # Load
            model2 = RiskDetectionModel()
            model2.load(temp_path)
            
            # Verify predictions match
            scores1 = model.predict(X)
            scores2 = model2.predict(X)
            
            assert np.allclose(scores1[0], scores2[0])
            assert np.allclose(scores1[1], scores2[1])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestRiskScorer:
    """Test RiskScorer class."""

    def test_compute_risk_score(self):
        """Test risk score computation."""
        score = RiskScorer.compute_risk_score(0.8, 0.6)
        
        expected = 0.7 * 0.8 + 0.3 * 0.6
        assert abs(score - expected) < 1e-10
        assert 0 <= score <= 1

    def test_classify_risk(self):
        """Test risk classification."""
        assert RiskScorer.classify_risk(0.9) == 'Critical'
        assert RiskScorer.classify_risk(0.6) == 'High'
        assert RiskScorer.classify_risk(0.4) == 'Medium'
        assert RiskScorer.classify_risk(0.1) == 'Low'

    def test_score_batch(self):
        """Test batch scoring."""
        clf_scores = np.array([0.2, 0.5, 0.8, 0.95])
        anom_scores = np.array([0.3, 0.4, 0.6, 0.7])
        
        risk_scores, risk_levels = RiskScorer.score_batch(clf_scores, anom_scores)
        
        assert risk_scores.shape == (4,)
        assert risk_levels.shape == (4,)
        assert all(level in ['Critical', 'High', 'Medium', 'Low'] for level in risk_levels)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
