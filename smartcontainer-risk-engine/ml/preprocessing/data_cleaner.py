"""
Data Cleaning Module
====================
Handles data validation, cleaning, and standardization.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Production-grade data cleaning and validation."""

    # Required field mapping
    REQUIRED_FIELDS = {
        'Container_ID': 'string',
        'Declaration_Date': 'datetime',
        'Declaration_Time': 'time',
        'Trade_Regime': 'string',
        'Origin_Country': 'string',
        'Destination_Country': 'string',
        'Destination_Port': 'string',
        'HS_Code': 'string',
        'Importer_ID': 'string',
        'Exporter_ID': 'string',
        'Declared_Value': 'float',
        'Declared_Weight': 'float',
        'Measured_Weight': 'float',
        'Shipping_Line': 'string',
        'Dwell_Time_Hours': 'float',
    }

    FIELD_CONSTRAINTS = {
        'Declared_Value': {'min': 0, 'max': 1e8},
        'Declared_Weight': {'min': 0.1, 'max': 1000},
        'Measured_Weight': {'min': 0.1, 'max': 1000},
        'Dwell_Time_Hours': {'min': 0, 'max': 1000},
    }

    def __init__(self):
        self.cleaning_stats = {}

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        """
        Normalize header names for flexible matching.
        Example:
        - "Declaration_Date (YYYY-MM-DD)" -> "declaration_date"
        - "Trade_Regime (Import / Export / Transit)" -> "trade_regime"
        """
        text = str(name).replace("\ufeff", "").strip()
        text = re.sub(r"\s*\([^)]*\)\s*", "", text)
        text = text.replace("-", "_").replace(" ", "_")
        text = re.sub(r"_+", "_", text).strip("_")
        return text.lower()

    def _build_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        required_lookup = {
            self._normalize_column_name(required): required
            for required in self.REQUIRED_FIELDS.keys()
        }

        mapping = {}
        for col in columns:
            normalized = self._normalize_column_name(col)
            if normalized in required_lookup:
                mapping[col] = required_lookup[normalized]
        return mapping

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input DataFrame has required fields."""
        column_mapping = self._build_column_mapping(df.columns.tolist())
        mapped_columns = set(column_mapping.values()) | {
            col for col in df.columns if col in self.REQUIRED_FIELDS
        }

        errors = []
        for field in self.REQUIRED_FIELDS:
            if field not in mapped_columns:
                errors.append(f"Missing required field: {field}")
        return len(errors) == 0, errors

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate containers by Container_ID."""
        before = len(df)
        if 'Container_ID' in df.columns:
            df = df.drop_duplicates(subset=['Container_ID'], keep='first')
        after = len(df)
        removed = before - after
        logger.info(f"Removed {removed} duplicate containers")
        self.cleaning_stats['duplicates_removed'] = removed
        return df

    def handle_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing values.
        - Drop rows where critical fields are missing.
        - Fill numeric columns with median.
        - Fill categorical columns with mode.
        """
        critical_cols = ['Container_ID', 'Declared_Value', 'Declared_Weight', 'Measured_Weight']
        
        # Drop rows with missing critical fields
        before = len(df)
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        after = len(df)
        rows_dropped = before - after
        logger.info(f"Dropped {rows_dropped} rows with missing critical fields")
        self.cleaning_stats['rows_with_missing_critical'] = rows_dropped

        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median = df[col].median()
                df[col].fillna(median, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median}")

        # Fill categorical columns with 'UNKNOWN'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna('UNKNOWN', inplace=True)

        return df

    def validate_value_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with impossible/invalid values."""
        before = len(df)

        for col, constraints in self.FIELD_CONSTRAINTS.items():
            if col not in df.columns:
                continue
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            
            if min_val is not None:
                df = df[df[col] >= min_val]
            if max_val is not None:
                df = df[df[col] <= max_val]

        after = len(df)
        removed = before - after
        logger.info(f"Removed {removed} rows with invalid value ranges")
        self.cleaning_stats['invalid_value_ranges'] = removed

        return df

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types."""
        # Normalize column names and map to canonical required field names.
        df.columns = pd.Index([str(col).strip() for col in df.columns])
        column_mapping = self._build_column_mapping(df.columns.tolist())
        df = df.rename(columns=column_mapping)

        # Convert date columns
        if 'Declaration_Date' in df.columns:
            df['Declaration_Date'] = pd.to_datetime(df['Declaration_Date'], errors='coerce')

        # Convert numeric columns
        numeric_cols = [col for col in df.columns 
                       if any(x in col.lower() for x in ['value', 'weight', 'dwell', 'hours'])]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def clean(self, df: pd.DataFrame, strict: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full cleaning pipeline.
        
        Args:
            df: Input DataFrame
            strict: If True, validate schema before cleaning
            
        Returns:
            Cleaned DataFrame and cleaning statistics
        """
        logger.info("Starting data cleaning pipeline...")
        
        # Validate schema
        if strict:
            valid, errors = self.validate_schema(df)
            if not valid:
                raise ValueError(f"Schema validation failed: {errors}")

        # Execute cleaning steps
        df = self.standardize_columns(df)
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.validate_value_ranges(df)

        logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        logger.info(f"Cleaning statistics: {self.cleaning_stats}")

        return df, self.cleaning_stats
