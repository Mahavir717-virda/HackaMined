"""
Unit Tests for Data Preprocessing
==================================
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.preprocessing.data_cleaner import DataCleaner


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return pd.DataFrame({
        'Container_ID': ['C001', 'C002', 'C003'],
        'Declaration_Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Declaration_Time': ['10:30', '14:45', '22:15'],
        'Trade_Regime': ['FREE', 'BOND', 'FREE'],
        'Origin_Country': ['US', 'CN', 'IN'],
        'Destination_Country': ['GB', 'DE', 'FR'],
        'Destination_Port': ['PORT1', 'PORT2', 'PORT1'],
        'HS_Code': ['2710', '2709', '2711'],
        'Importer_ID': ['IMP001', 'IMP002', 'IMP001'],
        'Exporter_ID': ['EXP001', 'EXP002', 'EXP003'],
        'Declared_Value': [10000, 50000, 25000],
        'Declared_Weight': [1000, 500, 750],
        'Measured_Weight': [1050, 510, 740],
        'Shipping_Line': ['SL1', 'SL2', 'SL1'],
        'Dwell_Time_Hours': [24, 48, 12],
        'Clearance_Status': ['Cleared', 'Under Review', 'Cleared'],
    })


class TestDataCleaner:
    """Test DataCleaner class."""

    def test_validate_schema_valid(self, sample_data):
        """Test schema validation with valid data."""
        cleaner = DataCleaner()
        valid, errors = cleaner.validate_schema(sample_data)
        assert valid is True
        assert len(errors) == 0

    def test_validate_schema_missing_field(self, sample_data):
        """Test schema validation with missing field."""
        cleaner = DataCleaner()
        df = sample_data.drop('Container_ID', axis=1)
        valid, errors = cleaner.validate_schema(df)
        assert valid is False
        assert any('Container_ID' in err for err in errors)

    def test_remove_duplicates(self, sample_data):
        """Test duplicate removal."""
        cleaner = DataCleaner()
        df = pd.concat([sample_data, sample_data.iloc[0:1]], ignore_index=True)
        assert len(df) == 4
        
        df_clean = cleaner.remove_duplicates(df)
        assert len(df_clean) == 3
        assert cleaner.cleaning_stats['duplicates_removed'] == 1

    def test_handle_missing_values(self):
        """Test missing value handling."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            'Container_ID': ['C001', 'C002', 'C003'],
            'Declared_Value': [1000, np.nan, 3000],
            'Declared_Weight': [100, 200, np.nan],
            'Measured_Weight': [105, 210, np.nan],
        })
        
        df_clean = cleaner.handle_missing_values(df)
        assert df_clean['Declared_Value'].isnull().sum() == 0
        assert df_clean['Declared_Weight'].isnull().sum() == 0

    def test_validate_value_ranges(self):
        """Test value range validation."""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            'Container_ID': ['C001', 'C002', 'C003'],
            'Declared_Value': [-1000, 5000, 1000000000],  # Invalid: negative and too large
            'Declared_Weight': [100, 0.05, 2000],  # Invalid: too small and too large
            'Measured_Weight': [105, 50, 100],
        })
        
        df_clean = cleaner.validate_value_ranges(df)
        # Should keep only rows with valid values
        assert len(df_clean) < len(df)

    def test_full_cleaning_pipeline(self, sample_data):
        """Test complete cleaning pipeline."""
        cleaner = DataCleaner()
        df_clean, stats = cleaner.clean(sample_data)
        
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(stats, dict)
        assert len(df_clean) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
