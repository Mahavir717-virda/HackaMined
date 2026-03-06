"""
Feature Engineering Module
===========================
Creates and transforms features for ML model training and prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Production-grade feature engineering pipeline."""

    # Risk country classifications
    HIGH_RISK_COUNTRIES = {
        'KP', 'IR', 'SY', 'LY', 'YE', 'VE', 'MM', 'CU', 'BY', 'ZW', 'HT', 'SD'
    }
    MED_RISK_COUNTRIES = {
        'NG', 'PK', 'AF', 'SD', 'IQ', 'SO', 'ML', 'CD', 'CF', 'SS', 'BA', 'KE'
    }

    # HS Code risk categories (first 2 digits)
    HIGH_RISK_HS_CODES = {'27', '28', '39', '84', '85', '86'}  # Energy, chem, plastics, machinery
    MED_RISK_HS_CODES = {'62', '63', '64', '65', '66', '67'}   # Textiles, footwear

    def __init__(self):
        self.feature_stats = {}
        self.scaler_params = {}

    def _country_risk_score(self, country_code: str) -> int:
        """Map country to risk score: 0=low, 1=medium, 2=high."""
        if pd.isna(country_code):
            return 0
        country_code = str(country_code).upper()
        if country_code in self.HIGH_RISK_COUNTRIES:
            return 2
        elif country_code in self.MED_RISK_COUNTRIES:
            return 1
        return 0

    def _hs_code_risk_score(self, hs_code: str) -> int:
        """Map HS code (first 2 digits) to risk score."""
        if pd.isna(hs_code):
            return 0
        hs_prefix = str(hs_code)[:2]
        if hs_prefix in self.HIGH_RISK_HS_CODES:
            return 2
        elif hs_prefix in self.MED_RISK_HS_CODES:
            return 1
        return 0

    def create_weight_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weight-related anomaly features."""
        df = df.copy()

        # Ensure numeric types
        df['Declared_Weight'] = pd.to_numeric(df['Declared_Weight'], errors='coerce').fillna(1)
        df['Measured_Weight'] = pd.to_numeric(df['Measured_Weight'], errors='coerce').fillna(df['Declared_Weight'])

        # Weight discrepancy features
        df['weight_diff'] = df['Measured_Weight'] - df['Declared_Weight']
        df['weight_diff_abs'] = df['weight_diff'].abs()
        df['weight_diff_pct'] = (df['weight_diff'] / (df['Declared_Weight'] + 1e-9)) * 100
        df['weight_ratio'] = df['Measured_Weight'] / (df['Declared_Weight'] + 1e-9)
        df['log_weight'] = np.log1p(df['Declared_Weight'])

        # Flags
        w_std = df['weight_diff_abs'].std() + 1e-9
        w_mean = df['weight_diff_abs'].mean()
        df['flag_weight_mismatch'] = (df['weight_diff_abs'] > w_mean + 2 * w_std).astype(int)
        df['flag_zero_declared_weight'] = (df['Declared_Weight'] <= 0).astype(int)

        return df

    def create_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create value-related features."""
        df = df.copy()

        # Ensure numeric types
        df['Declared_Value'] = pd.to_numeric(df['Declared_Value'], errors='coerce').fillna(0)
        df['Declared_Weight'] = pd.to_numeric(df['Declared_Weight'], errors='coerce').fillna(1)

        # Value features
        df['value_per_kg'] = df['Declared_Value'] / (df['Declared_Weight'] + 1e-9)
        df['log_value'] = np.log1p(df['Declared_Value'])
        df['value_to_weight_ratio'] = df['value_per_kg']

        # Flags
        v_mean = df['value_per_kg'].mean()
        v_std = df['value_per_kg'].std() + 1e-9
        df['flag_high_value_density'] = (df['value_per_kg'] > v_mean + 2 * v_std).astype(int)
        df['flag_low_value_density'] = (
            (df['value_per_kg'] < v_mean - 1.5 * v_std) & (df['value_per_kg'] > 0)
        ).astype(int)
        df['flag_zero_declared_value'] = (df['Declared_Value'] <= 0).astype(int)

        return df

    def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create route-based features."""
        df = df.copy()

        # Risk scores by country
        df['origin_country_risk'] = df['Origin_Country'].apply(self._country_risk_score)
        df['dest_country_risk'] = df['Destination_Country'].apply(self._country_risk_score)
        df['route_risk_total'] = df['origin_country_risk'] + df['dest_country_risk']

        # Route frequency (synthetic for now)
        route_counts = df.groupby(['Origin_Country', 'Destination_Port']).size()
        df['route_frequency'] = df.apply(
            lambda r: route_counts.get((r['Origin_Country'], r['Destination_Port']), 1),
            axis=1
        )
        df['route_frequency_log'] = np.log1p(df['route_frequency'])

        # Unusual route flags
        df['flag_high_risk_route'] = (df['route_risk_total'] >= 3).astype(int)

        return df

    def create_hs_code_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create HS code-based features."""
        df = df.copy()

        df['hs_code_risk'] = df['HS_Code'].apply(self._hs_code_risk_score)
        df['hs_code_first_digit'] = df['HS_Code'].astype(str).str[0]

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df = df.copy()

        # Date features
        if 'Declaration_Date' in df.columns:
            df['Declaration_Date'] = pd.to_datetime(df['Declaration_Date'], errors='coerce')
            df['day_of_week'] = df['Declaration_Date'].dt.dayofweek
            df['month'] = df['Declaration_Date'].dt.month
            df['day_of_month'] = df['Declaration_Date'].dt.day
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        else:
            df['day_of_week'] = 0
            df['month'] = 1
            df['day_of_month'] = 1
            df['is_weekend'] = 0
            df['is_month_end'] = 0

        # Time features
        if 'Declaration_Time' in df.columns:
            def extract_hour(t):
                try:
                    return int(str(t).split(':')[0])
                except:
                    return 12
            df['hour_of_day'] = df['Declaration_Time'].apply(extract_hour)
        else:
            df['hour_of_day'] = 12

        df['is_night'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] >= 22)).astype(int)
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)

        return df

    def create_dwell_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dwell time-based features."""
        df = df.copy()

        df['Dwell_Time_Hours'] = pd.to_numeric(df['Dwell_Time_Hours'], errors='coerce').fillna(24)
        df['dwell_time_log'] = np.log1p(df['Dwell_Time_Hours'])

        # Flags
        dwell_mean = df['Dwell_Time_Hours'].mean()
        dwell_std = df['Dwell_Time_Hours'].std() + 1e-9
        df['flag_excessive_dwell'] = (df['Dwell_Time_Hours'] > dwell_mean + 1.5 * dwell_std).astype(int)
        df['flag_minimal_dwell'] = (df['Dwell_Time_Hours'] < 2).astype(int)

        return df

    def create_categorical_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create encoded categorical features."""
        df = df.copy()

        # Use only pre-clearance fields to avoid leakage.
        categorical_cols = ['Trade_Regime', 'Shipping_Line']

        for col in categorical_cols:
            if col in df.columns:
                # Target encoding (mean of risk)
                risk_means = df.groupby(col).size()
                df[f'{col}_encoded'] = df[col].map(risk_means).fillna(0)

        return df

    def get_feature_list(self) -> List[str]:
        """Return complete list of engineered features."""
        return [
            # Weight features
            'weight_diff', 'weight_diff_abs', 'weight_diff_pct', 'weight_ratio', 'log_weight',
            'flag_weight_mismatch', 'flag_zero_declared_weight',
            # Value features
            'value_per_kg', 'log_value', 'value_to_weight_ratio',
            'flag_high_value_density', 'flag_low_value_density', 'flag_zero_declared_value',
            # Route features
            'origin_country_risk', 'dest_country_risk', 'route_risk_total',
            'route_frequency_log', 'flag_high_risk_route',
            # HS code
            'hs_code_risk',
            # Time features
            'day_of_week', 'month', 'day_of_month', 'is_weekend', 'is_month_end',
            'hour_of_day', 'is_night', 'is_business_hours',
            # Dwell time
            'dwell_time_log', 'flag_excessive_dwell', 'flag_minimal_dwell',
            # Encoded categoricals
            'Trade_Regime_encoded', 'Shipping_Line_encoded',
        ]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline...")

        df = self.create_weight_features(df)
        df = self.create_value_features(df)
        df = self.create_route_features(df)
        df = self.create_hs_code_features(df)
        df = self.create_time_features(df)
        df = self.create_dwell_time_features(df)
        df = self.create_categorical_encodings(df)

        logger.info(f"Feature engineering complete. Features created: {len(self.get_feature_list())}")

        return df

    def get_available_features(self, df: pd.DataFrame) -> List[str]:
        """Return list of features that exist in the DataFrame."""
        all_features = self.get_feature_list()
        return [f for f in all_features if f in df.columns]
