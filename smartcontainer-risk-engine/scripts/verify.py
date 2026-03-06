#!/usr/bin/env python3
"""
Verification Script
===================
Comprehensive system verification and diagnostics.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SystemVerifier:
    """Verify SmartContainer Risk Engine setup."""

    def __init__(self, root_dir='.'):
        self.root_dir = Path(root_dir)
        self.checks_passed = 0
        self.checks_failed = 0

    def check(self, name: str, condition: bool, fix: str = None):
        """Record check result."""
        if condition:
            logger.info(f"✓ {name}")
            self.checks_passed += 1
        else:
            logger.error(f"✗ {name}")
            if fix:
                logger.info(f"  Fix: {fix}")
            self.checks_failed += 1

    def verify_python_packages(self):
        """Verify Python dependencies."""
        logger.info("\n=== Python Packages ===")
        packages = {
            'numpy': 'Core numerical computing',
            'pandas': 'Data manipulation',
            'sklearn': 'Machine learning',
            'fastapi': 'REST API framework',
            'pytest': 'Testing framework'
        }

        for pkg, desc in packages.items():
            try:
                __import__(pkg)
                self.check(f"✓ {pkg}: {desc}", True)
            except ImportError:
                self.check(f"✗ {pkg}: {desc}", False, 
                          f"pip install {pkg}")

    def verify_directory_structure(self):
        """Verify required directories exist."""
        logger.info("\n=== Directory Structure ===")
        required_dirs = [
            'ml/preprocessing',
            'ml/features',
            'ml/core',
            'backend/api',
            'backend/schemas',
            'data',
            'models',
            'tests'
        ]

        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            self.check(f"Directory: {dir_path}", full_path.exists(),
                      f"mkdir -p {dir_path}")

    def verify_required_files(self):
        """Verify required files exist."""
        logger.info("\n=== Required Files ===")
        required_files = [
            'train.py',
            'requirements.txt',
            'README.md',
            'ml/preprocessing/data_cleaner.py',
            'ml/features/feature_engineer.py',
            'ml/core/ml_models.py',
            'ml/core/explainability.py',
            'backend/api/main.py',
            'backend/schemas/models.py',
            'tests/test_preprocessing.py',
            'Dockerfile.backend',
            'Dockerfile.frontend',
            'docker-compose.yml',
            'API_DOCUMENTATION.md'
        ]

        for file_path in required_files:
            full_path = self.root_dir / file_path
            self.check(f"File: {file_path}", full_path.exists(),
                      f"Create {file_path}")

    def verify_ml_modules(self):
        """Verify ML modules can be imported."""
        logger.info("\n=== ML Module Imports ===")
        
        sys.path.insert(0, str(self.root_dir))
        
        try:
            from ml.preprocessing.data_cleaner import DataCleaner
            self.check("Import DataCleaner", True)
        except Exception as e:
            self.check("Import DataCleaner", False, str(e))

        try:
            from ml.features.feature_engineer import FeatureEngineer
            self.check("Import FeatureEngineer", True)
        except Exception as e:
            self.check("Import FeatureEngineer", False, str(e))

        try:
            from ml.core.ml_models import RiskDetectionModel
            self.check("Import RiskDetectionModel", True)
        except Exception as e:
            self.check("Import RiskDetectionModel", False, str(e))

        try:
            from ml.core.explainability import RiskExplainer
            self.check("Import RiskExplainer", True)
        except Exception as e:
            self.check("Import RiskExplainer", False, str(e))

    def verify_api_module(self):
        """Verify FastAPI module."""
        logger.info("\n=== API Module ===")
        
        try:
            import fastapi
            self.check("FastAPI installation", True)
        except ImportError:
            self.check("FastAPI installation", False, 
                      "pip install fastapi uvicorn")

    def verify_model(self):
        """Verify trained model exists."""
        logger.info("\n=== Model Files ===")
        model_path = self.root_dir / 'models' / 'risk_model.joblib'
        self.check("Trained model exists", model_path.exists(),
                  "Run: python train.py --generate-data")

    def verify_test_suite(self):
        """Verify tests can be discovered."""
        logger.info("\n=== Test Suite ===")
        test_dir = self.root_dir / 'tests'
        test_files = list(test_dir.glob('test_*.py')) if test_dir.exists() else []
        
        self.check(f"Test files found: {len(test_files)}", len(test_files) > 0,
                  f"Found {len(test_files)} test files")

    def verify_docker(self):
        """Verify Docker is available."""
        logger.info("\n=== Docker Setup ===")
        
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            self.check("Docker installed", result.returncode == 0)
        except FileNotFoundError:
            self.check("Docker installed", False, 
                      "Install Docker from https://www.docker.com")

        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            self.check("Docker Compose installed", result.returncode == 0)
        except FileNotFoundError:
            self.check("Docker Compose installed", False, 
                      "Install Docker Compose")

    def verify_sample_data(self):
        """Check if sample data can be generated."""
        logger.info("\n=== Sample Data ===")
        data_dir = self.root_dir / 'data'
        sample_file = data_dir / 'sample_data.csv'
        
        has_sample = sample_file.exists() and sample_file.stat().st_size > 0
        self.check("Sample data exists", has_sample,
                  "Run: python train.py --generate-data --samples 500")

    def run_quick_test(self):
        """Run quick integration test."""
        logger.info("\n=== Quick Integration Test ===")
        
        try:
            import pandas as pd
            from ml.preprocessing.data_cleaner import DataCleaner
            from ml.features.feature_engineer import FeatureEngineer

            # Create minimal test data
            df = pd.DataFrame({
                'Container_ID': ['C001'],
                'Declared_Value': [10000],
                'Declared_Weight': [100],
                'Measured_Weight': [105],
                'Origin_Country': ['US'],
                'Destination_Country': ['GB'],
                'Destination_Port': ['PORT1'],
                'HS_Code': ['2710'],
                'Dwell_Time_Hours': [24],
                'Shipping_Line': ['MAERSK'],
                'Trade_Regime': ['FREE'],
                'Declaration_Date': ['2024-01-01'],
                'Declaration_Time': ['10:00'],
                'Clearance_Status': ['Cleared'],
                'Importer_ID': ['IMP001'],
                'Exporter_ID': ['EXP001']
            })

            # Test cleaning
            cleaner = DataCleaner()
            df_clean, _ = cleaner.clean(df, strict=False)
            self.check("Data cleaning works", len(df_clean) > 0)

            # Test feature engineering
            engineer = FeatureEngineer()
            df_features = engineer.engineer_features(df_clean)
            self.check("Feature engineering works", 
                      len(engineer.get_available_features(df_features)) > 0)

        except Exception as e:
            self.check("Quick integration test", False, str(e))

    def generate_report(self):
        """Generate verification report."""
        total = self.checks_passed + self.checks_failed
        percentage = (self.checks_passed / total * 100) if total > 0 else 0

        logger.info("\n" + "="*50)
        logger.info(f"VERIFICATION REPORT")
        logger.info("="*50)
        logger.info(f"\nPassed:  {self.checks_passed}")
        logger.info(f"Failed:  {self.checks_failed}")
        logger.info(f"Total:   {total}")
        logger.info(f"Status:  {percentage:.1f}%")

        if self.checks_failed == 0:
            logger.info("\n✓ All checks passed! System is ready.")
            return True
        else:
            logger.warning(f"\n⚠ {self.checks_failed} issue(s) found. See above for details.")
            return False

    def run_all_checks(self):
        """Execute all verification checks."""
        self.verify_python_packages()
        self.verify_directory_structure()
        self.verify_required_files()
        self.verify_ml_modules()
        self.verify_api_module()
        self.verify_model()
        self.verify_test_suite()
        self.verify_docker()
        self.verify_sample_data()
        self.run_quick_test()
        
        return self.generate_report()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify SmartContainer Risk Engine setup')
    parser.add_argument('--dir', default='.', help='Project root directory')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    
    args = parser.parse_args()
    
    verifier = SystemVerifier(root_dir=args.dir)
    success = verifier.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
