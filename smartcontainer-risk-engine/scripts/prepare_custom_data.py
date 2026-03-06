"""
Custom Dataset Preparation
============================
Standardizes column names and creates risk labels for custom datasets.
"""

import pandas as pd
import numpy as np
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_historical_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Prepare custom historical data by standardizing columns and creating risk labels.
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save processed CSV
        
    Returns:
        Processed DataFrame with standardized schema
    """
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Column mapping: raw column names -> standardized names
    column_mapping = {
        'Declaration_Date (YYYY-MM-DD)': 'Declaration_Date',
        'Trade_Regime (Import / Export / Transit)': 'Trade_Regime',
        'Destination_Country': 'Destination_Country',
        'Destination_Port': 'Destination_Port',
    }
    
    # Apply mapping for columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            logger.info(f"Mapped column: {old_name} -> {new_name}")
    
    # Standardize data types
    logger.info("Standardizing data types...")
    df['Container_ID'] = df['Container_ID'].astype(str)
    df['Declaration_Date'] = pd.to_datetime(df['Declaration_Date'], errors='coerce')
    df['Declared_Value'] = pd.to_numeric(df['Declared_Value'], errors='coerce')
    df['Declared_Weight'] = pd.to_numeric(df['Declared_Weight'], errors='coerce')
    df['Measured_Weight'] = pd.to_numeric(df['Measured_Weight'], errors='coerce')
    df['Dwell_Time_Hours'] = pd.to_numeric(df['Dwell_Time_Hours'], errors='coerce')
    df['HS_Code'] = df['HS_Code'].astype(str)
    
    # Create risk labels based on heuristics
    logger.info("Creating risk labels...")
    df['is_risky'] = 0
    
    # High-risk countries
    high_risk_countries = {'KP', 'IR', 'NG', 'PK', 'SY', 'LB', 'MY', 'TH'}
    df.loc[df['Origin_Country'].isin(high_risk_countries), 'is_risky'] = 1
    logger.info(f"Flagged {(df['is_risky']==1).sum()} records from high-risk countries")
    
    # Weight discrepancies (>50% difference)
    weight_diff_pct = ((df['Measured_Weight'] - df['Declared_Weight']).abs() / 
                       (df['Declared_Weight'] + 1e-9) * 100)
    df.loc[weight_diff_pct > 50, 'is_risky'] = 1
    logger.info(f"Flagged {(weight_diff_pct > 50).sum()} records with weight discrepancies")
    
    # High dwell times (>100 hours)
    df.loc[df['Dwell_Time_Hours'] > 100, 'is_risky'] = 1
    logger.info(f"Flagged {(df['Dwell_Time_Hours'] > 100).sum()} records with high dwell times")
    
    # Suspicious clearance statuses
    suspicious_statuses = {'Flagged', 'Seized', 'Rejected', 'Held', 'Under Review'}
    if 'Clearance_Status' in df.columns:
        df.loc[df['Clearance_Status'].isin(suspicious_statuses), 'is_risky'] = 1
        logger.info(f"Flagged {df['Clearance_Status'].isin(suspicious_statuses).sum()} records with suspicious statuses")
    
    # High-value items (>$100k)
    df.loc[df['Declared_Value'] > 100000, 'is_risky'] = 1
    
    # Convert to binary integer
    df['is_risky'] = (df['is_risky'] >= 0.5).astype(int)
    
    # Display risk distribution
    risk_dist = df['is_risky'].value_counts().sort_index()
    logger.info(f"Risk distribution:\n{risk_dist}")
    
    # Save processed data
    logger.info(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Dataset ready! {len(df)} records saved")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Prepare custom dataset for model training')
    parser.add_argument('--input', type=str, default='data/historical_data.csv',
                       help='Path to raw historical data CSV')
    parser.add_argument('--output', type=str, default='data/historical_data_processed.csv',
                       help='Path to save processed data')
    
    args = parser.parse_args()
    prepare_historical_data(args.input, args.output)


if __name__ == '__main__':
    main()
