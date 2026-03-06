"""
Unit Tests for Feature Engineering
===================================
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.features.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return pd.DataFrame({
        'Container_ID': ['C001', 'C002', 'C003'],
        'Declaration_Date': pd.to_datetime(['2024-01-01', '2024-01-07', '2024-01-15']),
        'Declaration_Time': ['10:30', '22:45', '03:15'],
        'Origin_Country': ['US', 'KP', 'IN'],  # KP is high-risk
        'Destination_Country': ['GB', 'CN', 'NG'],  # NG is medium-risk
        'Destination_Port': ['PORT1', 'PORT2', 'PORT1'],
        'HS_Code': ['2710', '8401', '6201'],  # Energy, machinery, textiles
        'Trade_Regime': ['FREE', 'BOND', 'FREE'],
        'Shipping_Line': ['SL1', 'SL2', 'SL1'],
        'Clearance_Status': ['Cleared', 'Flagged', 'Under Review'],
        'Declared_Value': [10000, 50000, 5000],
        'Declared_Weight': [1000, 500, 100],
        'Measured_Weight': [1200, 510, 95],  # First has discrepancy
        'Dwell_Time_Hours': [24, 100, 2],  # Second excessive, third minimal
    })


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    def test_country_risk_scoring(self):
        """Test country risk score assignment."""
        engineer = FeatureEngineer()
        
        high_risk = engineer._country_risk_score('KP')  # North Korea
        med_risk = engineer._country_risk_score('NG')   # Nigeria
        low_risk = engineer._country_risk_score('US')   # US
        
        assert high_risk == 2
        assert med_risk == 1
        assert low_risk == 0

    def test_hs_code_risk_scoring(self):
        """Test HS code risk score."""
        engineer = FeatureEngineer()
        
        high_risk = engineer._hs_code_risk_score('2710')  # Energy
        med_risk = engineer._hs_code_risk_score('6201')   # Textiles
        low_risk = engineer._hs_code_risk_score('0201')   # Meat
        
        assert high_risk == 2
        assert med_risk == 1
        assert low_risk == 0

    def test_weight_features(self, sample_data):
        """Test weight-related feature creation."""
        engineer = FeatureEngineer()
        df = engineer.create_weight_features(sample_data)
        
        assert 'weight_diff' in df.columns
        assert 'weight_diff_abs' in df.columns
        assert 'weight_diff_pct' in df.columns
        assert 'weight_ratio' in df.columns
        
        # Check weight discrepancy calculation
        expected_diff = sample_data.loc[0, 'Measured_Weight'] - sample_data.loc[0, 'Declared_Weight']
        assert abs(df.loc[0, 'weight_diff'] - expected_diff) < 1e-6

    def test_value_features(self, sample_data):
        """Test value-related features."""
        engineer = FeatureEngineer()
        df = engineer.create_weight_features(sample_data)
        df = engineer.create_value_features(df)
        
        assert 'value_per_kg' in df.columns
        assert 'log_value' in df.columns
        
        # Check value per kg calculation
        expected_vpk = sample_data.loc[0, 'Declared_Value'] / sample_data.loc[0, 'Declared_Weight']
        assert abs(df.loc[0, 'value_per_kg'] - expected_vpk) < 1e-6

    def test_route_features(self, sample_data):
        """Test route-based features."""
        engineer = FeatureEngineer()
        df = engineer.create_route_features(sample_data)
        
        assert 'origin_country_risk' in df.columns
        assert 'dest_country_risk' in df.columns
        assert 'route_risk_total' in df.columns
        
        # KP (N. Korea) should be high-risk
        assert df.loc[1, 'origin_country_risk'] == 2

    def test_time_features(self, sample_data):
        """Test time-based features."""
        engineer = FeatureEngineer()
        df = engineer.create_time_features(sample_data)
        
        assert 'day_of_week' in df.columns
        assert 'month' in df.columns
        assert 'hour_of_day' in df.columns
        assert 'is_night' in df.columns
        assert 'is_business_hours' in df.columns
        
        # Check night flag (22:45 should be night)
        assert df.loc[1, 'is_night'] == 1
        # Check business hours (10:30 should be business hours)
        assert df.loc[0, 'is_business_hours'] == 1

    def test_dwell_time_features(self, sample_data):
        """Test dwell time features."""
        engineer = FeatureEngineer()
        df = engineer.create_dwell_time_features(sample_data)
        
        assert 'dwell_time_log' in df.columns
        assert 'flag_excessive_dwell' in df.columns
        assert 'flag_minimal_dwell' in df.columns
        
        # 100 hours should be excessive
        assert df.loc[1, 'flag_excessive_dwell'] == 1
        # 2 hours should be minimal
        assert df.loc[2, 'flag_minimal_dwell'] == 1

    def test_full_feature_engineering(self, sample_data):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer()
        df = engineer.engineer_features(sample_data)
        
        feature_list = engineer.get_feature_list()
        available = engineer.get_available_features(df)
        
        assert len(available) > 0
        assert all(f in df.columns for f in available)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
