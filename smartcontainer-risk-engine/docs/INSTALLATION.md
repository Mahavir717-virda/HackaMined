# Installation & Setup Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Training Your Model](#training-your-model)
5. [Running the System](#running-the-system)
6. [Verification](#verification)

---

## System Requirements

### Hardware
- **CPU:** 4+ cores recommended
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 5GB+ free space

### Software
- **Python:** 3.11 or higher
- **Node.js:** 18+ (for frontend)
- **Docker:** 20.10+ (optional, for containerized deployment)
- **Docker Compose:** 2.0+ (optional)

---

## Local Development Setup

### Step 1: Install Python Dependencies

```bash
# Navigate to project directory
cd smartcontainer-risk-engine

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env as needed (optional - defaults are production-ready)
# nano .env  # or use your preferred editor
```

### Step 3: Generate Training Data (Optional)

```bash
# Generate synthetic training data
python train.py --generate-data --samples 500

# This creates data/sample_data.csv with ~500 container records
```

### Step 4: Train the ML Model

```bash
# Train model from scratch
python train.py \
  --data ./data/sample_data.csv \
  --model-output ./models/risk_model.joblib

# Training takes ~5-10 minutes depending on data size
# Check training.log for detailed output
```

### Step 5: Start the API Server

```bash
# In one terminal, start the API
python -m backend.api.main

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

### Step 6: Test the API

```bash
# In another terminal, test health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true,"model_version":"1.0.0","timestamp":"2024-03-06T..."}
```

---

## Docker Deployment

### Option A: Using Docker Compose (Recommended)

```bash
# Navigate to project directory
cd smartcontainer-risk-engine

# Build all containers
docker-compose build

# Start all services
docker-compose up -d

# Wait for services to be healthy
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option B: Manual Docker Build

```bash
# Build backend image
docker build -f Dockerfile.backend -t smartcontainer-backend:latest .

# Build frontend image
docker build -f Dockerfile.frontend -t smartcontainer-frontend:latest .

# Run backend
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  smartcontainer-backend:latest

# Run frontend (in another terminal)
docker run -p 3000:3000 \
  -e REACT_APP_API_URL=http://localhost:8000 \
  smartcontainer-frontend:latest
```

---

## Training Your Model

### Using Provided Script

```bash
# With all options
python train.py \
  --generate-data \
  --samples 1000 \       # Number of synthetic samples
  --data ./data/train.csv \
  --model-output ./models/custom_model.joblib \
  --custom-lr             # Use custom logistic regression

# Using custom data
python train.py \
  --data ./data/your_data.csv \
  --model-output ./models/your_model.joblib
```

### Using Python API

```python
import sys
sys.path.insert(0, '/path/to/smartcontainer-risk-engine')

from train import generate_synthetic_data, train_pipeline

# Generate data
df = generate_synthetic_data(
    n_samples=1000,
    save_path='./data/custom_data.csv'
)

# Train model
model, stats = train_pipeline(
    data_path='./data/custom_data.csv',
    model_output_path='./models/custom_model.joblib',
    use_custom_lr=False
)

print(f"Model trained. AUC: {stats['auc']:.4f}")
```

---

## Running the System

### 1. API Only (Headless)

```bash
# Start API server
python -m backend.api.main

# API endpoints:
# - Health:         GET  http://localhost:8000/health
# - Upload:         POST http://localhost:8000/upload
# - Predict:        POST http://localhost:8000/predict
# - Summary:        GET  http://localhost:8000/summary
# - Single Predict: POST http://localhost:8000/predict-single
```

### 2. Full Stack (API + Frontend + Nginx)

```bash
# Using Docker Compose
docker-compose up -d

# Services will be available at:
# - Frontend:   http://localhost:3000
# - API:        http://localhost:8000
# - Nginx:      http://localhost (port 80)
```

### 3. Development with Hot Reload

```bash
# Terminal 1: API
python -m backend.api.main

# Terminal 2: Frontend (from DS-ML/custom-container)
cd DS-ML/custom-container
npm start

# Frontend available at http://localhost:3000
```

---

## Verification

### 1. Check Installation

```bash
# Verify Python packages
python -c "import fastapi, pandas, sklearn; print('✓ All packages installed')"

# Check model exists
ls -la models/risk_model.joblib

# Check data directory
ls -la data/
```

### 2. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=ml --cov-report=html
```

### 3. Quick Prediction Test

```python
import pandas as pd
import sys
sys.path.insert(0, '.')

from ml.preprocessing.data_cleaner import DataCleaner
from ml.features.feature_engineer import FeatureEngineer
from ml.core.ml_models import RiskDetectionModel, RiskScorer
from ml.core.explainability import RiskExplainer

# Create sample container
test_data = {
    'Container_ID': ['C999999'],
    'Declared_Value': [100000],
    'Declared_Weight': [500],
    'Measured_Weight': [520],
    'Origin_Country': ['CN'],
    'Destination_Country': ['US'],
    'Destination_Port': ['PORT_LA'],
    'HS_Code': ['2710'],
    'Dwell_Time_Hours': [48],
    'Shipping_Line': ['MAERSK'],
    'Trade_Regime': ['FREE'],
    'Declaration_Date': ['2024-03-06'],
    'Declaration_Time': ['14:30'],
    'Clearance_Status': ['Cleared']
}

df = pd.DataFrame(test_data)

# Clean and engineer features
cleaner = DataCleaner()
df_clean, _ = cleaner.clean(df, strict=False)

engineer = FeatureEngineer()
df_features = engineer.engineer_features(df_clean)

# Make prediction
model = RiskDetectionModel()
model.load('./models/risk_model.joblib')

features = engineer.get_available_features(df_features)
X = df_features[features].fillna(0)

clf_scores, anom_scores = model.predict(X)
risk_scores, risk_levels = RiskScorer.score_batch(clf_scores, anom_scores)

explainer = RiskExplainer()
explanation = explainer.generate_explanation(df_features.iloc[0], risk_levels[0], risk_scores[0])

print(f"Container: {df.iloc[0]['Container_ID']}")
print(f"Risk Level: {risk_levels[0]}")
print(f"Risk Score: {risk_scores[0]*100:.1f}%")
print(f"Explanation: {explanation}")
```

### 4. API Health Check

```bash
# Using curl
curl -s http://localhost:8000/health | python -m json.tool

# Expected output:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "model_version": "1.0.0",
#   "timestamp": "2024-03-06T..."
# }
```

---

## Troubleshooting

### Issue: "Model not found" Error

**Solution:**
```bash
# Retrain the model
python train.py --generate-data --samples 500

# Or use the provided sample model
cp ./DS-ML/model.joblib ./models/risk_model.joblib
```

### Issue: Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
API_PORT=8001 python -m backend.api.main
```

### Issue: Out of Memory During Training

```bash
# Train with fewer samples
python train.py --generate-data --samples 100

# Or increase swap space
# Linux:
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue: Docker Build Fails

```bash
# Clear Docker cache
docker-compose build --no-cache

# Increase Docker memory allocation
# Docker Desktop Settings → Resources → Memory: Increase to 8GB+
```

---

## Performance Tuning

### For Production

```bash
# Use Gunicorn instead of Uvicorn
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:8000 backend.api.main:app

# Or via Docker environment variable
WORKERS=4 docker-compose up -d
```

### Model Optimization

```python
# For faster inference but lower accuracy
model = RiskDetectionModel()
model.classifier = GradientBoostingClassifier(
    n_estimators=50,  # Reduced from 200
    max_depth=2,      # Reduced from 4
    subsample=0.5
)
```

---

## Next Steps

1. **Upload Your Data:** Place CSV files in `data/` directory
2. **Train Custom Model:** Run `python train.py --data ./data/your_file.csv`
3. **Access Dashboard:** Visit http://localhost:3000
4. **Monitor Predictions:** Check API logs and results

---

## Support

For issues see troubleshooting or check:
- `training.log` - ML pipeline logs
- `application.log` - API logs
- `docker-compose logs` - Container logs

---

**Last Updated:** 2024-03-06  
**Version:** 1.0.0
