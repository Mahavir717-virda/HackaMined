# API Documentation - SmartContainer Risk Engine

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, no authentication is required. Production deployments should implement JWT/OAuth2.

---

## Endpoints

### 1. Health Check
Check API status and model availability.

**Endpoint:**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2024-03-06T14:30:00.000Z"
}
```

**Status Codes:**
- `200 OK` - API is healthy
- `503 Service Unavailable` - Model not loaded

---

### 2. Upload Dataset
Upload a CSV file containing container shipment data.

**Endpoint:**
```http
POST /upload
Content-Type: multipart/form-data

file: <binary CSV file>
```

**Request:**
```bash
curl -X POST -F "file=@shipments.csv" http://localhost:8000/upload
```

**Response:**
```json
{
  "status": "success",
  "message": "File uploaded and validated successfully",
  "file_id": "a1b2c3d4",
  "rows_received": 500,
  "rows_valid": 495,
  "rows_failed": 5,
  "errors": []
}
```

**Required CSV Columns:**
```
Container_ID,Declaration_Date,Declaration_Time,Trade_Regime,
Origin_Country,Destination_Country,Destination_Port,HS_Code,
Importer_ID,Exporter_ID,Declared_Value,Declared_Weight,
Measured_Weight,Shipping_Line,Dwell_Time_Hours,Clearance_Status
```

**Status Codes:**
- `200 OK` - File uploaded successfully
- `400 Bad Request` - Invalid file or schema
- `413 Payload Too Large` - File exceeds size limit (100MB)

---

### 3. Generate Predictions
Generate risk predictions for uploaded dataset.

**Endpoint:**
```http
POST /predict?file_id=<file_id>&use_cached=true
```

**Parameters:**
- `file_id` (required): ID from upload response
- `use_cached` (optional): Use cached data (default: true)

**Request:**
```bash
curl -X POST "http://localhost:8000/predict?file_id=a1b2c3d4"
```

**Response:**
```json
{
  "status": "success",
  "file_id": "a1b2c3d4",
  "total_containers": 495,
  "predictions": [
    {
      "container_id": "C10001",
      "risk_score": 82.5,
      "risk_level": "Critical",
      "explanation_summary": "Large weight discrepancy and abnormal value density",
      "confidence": 0.95,
      "classifier_score": 0.85,
      "anomaly_score": 0.78
    },
    {
      "container_id": "C10002",
      "risk_score": 35.2,
      "risk_level": "Medium",
      "explanation_summary": "Minor weight variance within expected tolerances",
      "confidence": 0.72,
      "classifier_score": 0.32,
      "anomaly_score": 0.40
    }
  ],
  "summary": {
    "total_containers": 495,
    "critical": 45,
    "high": 95,
    "medium": 175,
    "low": 180,
    "anomalies": 52,
    "average_risk_score": 45.3,
    "processed_at": "2024-03-06T14:35:00.000Z"
  }
}
```

**Response Fields:**
- `container_id`: Unique container identifier
- `risk_score`: Prediction score 0-100
- `risk_level`: Classification (Critical/High/Medium/Low)
- `explanation_summary`: Human-readable explanation
- `confidence`: Model confidence 0-1
- `classifier_score`: ML classifier probability
- `anomaly_score`: Anomaly detection score

**Risk Level Mapping:**
| Risk Level | Score Range |
|-----------|------------|
| Critical | ≥ 75 |
| High | 50-75 |
| Medium | 25-50 |
| Low | < 25 |

**Status Codes:**
- `200 OK` - Predictions generated
- `404 Not Found` - File not found
- `503 Service Unavailable` - Model not loaded
- `500 Internal Server Error` - Processing error

---

### 4. Get Summary Statistics
Get aggregated summary statistics for a processed batch.

**Endpoint:**
```http
GET /summary?file_id=<file_id>
```

**Parameters:**
- `file_id` (required): ID from upload response

**Request:**
```bash
curl "http://localhost:8000/summary?file_id=a1b2c3d4"
```

**Response:**
```json
{
  "total_containers_processed": 495,
  "critical_count": 45,
  "high_count": 95,
  "medium_count": 175,
  "low_count": 180,
  "anomaly_count": 52,
  "risk_distribution": {
    "critical": 9.1,
    "high": 19.2,
    "medium": 35.4,
    "low": 36.4
  },
  "top_risk_factors": [
    "weight_discrepancy",
    "unusual_route",
    "high_value_density",
    "excessive_dwell",
    "off_hours_declaration"
  ],
  "average_risk_score": 45.3,
  "timestamp": "2024-03-06T14:35:00.000Z"
}
```

**Status Codes:**
- `200 OK` - Summary retrieved
- `404 Not Found` - File not found
- `500 Internal Server Error` - Retrieval error

---

### 5. Predict Single Container
Generate prediction for a single container without upload.

**Endpoint:**
```http
POST /predict-single
Content-Type: application/json

{
  "Container_ID": "C99999",
  "Declared_Value": 50000,
  "Declared_Weight": 500,
  "Measured_Weight": 510,
  "Origin_Country": "CN",
  "Destination_Country": "US",
  "Destination_Port": "PORT_LA",
  "HS_Code": "2710",
  "Dwell_Time_Hours": 48,
  "Shipping_Line": "MAERSK",
  "Trade_Regime": "FREE",
  "Declaration_Date": "2024-03-06",
  "Declaration_Time": "14:30",
  "Clearance_Status": "Cleared",
  "Importer_ID": "IMP001",
  "Exporter_ID": "EXP001"
}
```

**Request:**
```bash
curl -X POST http://localhost:8000/predict-single \
  -H "Content-Type: application/json" \
  -d '{
    "Container_ID": "C99999",
    "Declared_Value": 50000,
    "Declared_Weight": 500,
    "Measured_Weight": 510,
    "Origin_Country": "CN",
    "Destination_Country": "US",
    "Destination_Port": "PORT_LA",
    "HS_Code": "2710",
    "Dwell_Time_Hours": 48,
    "Shipping_Line": "MAERSK",
    "Trade_Regime": "FREE"
  }'
```

**Response:**
```json
{
  "container_id": "C99999",
  "risk_score": 62.3,
  "risk_level": "High",
  "explanation": "Measured weight is 2% higher than declared; Unusual trade route; Excessive dwell time",
  "classifier_score": 0.68,
  "anomaly_score": 0.52
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid container data
- `503 Service Unavailable` - Model not loaded
- `500 Internal Server Error` - Prediction error

---

## Error Responses

### Standard Error Format
All error responses follow this format:

```json
{
  "detail": "Error description",
  "status_code": 400
}
```

### Common Errors

**400 Bad Request**
```json
{
  "detail": "Invalid CSV schema. Missing required fields: Measured_Weight"
}
```

**404 Not Found**
```json
{
  "detail": "File abc123 not found. Upload first with /upload endpoint."
}
```

**503 Service Unavailable**
```json
{
  "detail": "Model not loaded. Service unavailable."
}
```

**500 Internal Server Error**
```json
{
  "detail": "An unexpected error occurred during processing."
}
```

---

## Rate Limiting (Future Enhancement)

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1614857400
```

---

## Pagination (For Large Result Sets)

```http
GET /predict?file_id=abc123&page=1&page_size=50
```

---

## Batch Operations (Future Enhancement)

```http
POST /batch-predict
Content-Type: application/json

{
  "file_ids": ["a1b2c3d4", "e5f6g7h8"],
  "export_format": "csv"
}
```

---

## Data Export

### CSV Export
```bash
curl "http://localhost:8000/predict?file_id=a1b2c3d4" \
  --header "Accept: text/csv" > predictions.csv
```

### JSON Export
```bash
curl "http://localhost:8000/predict?file_id=a1b2c3d4" \
  --header "Accept: application/json" | jq . > predictions.json
```

---

## Request Examples

### Python
```python
import requests
import json

# Upload file
with open('shipments.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f}
    )
    file_id = response.json()['file_id']

# Get predictions
response = requests.post(
    f'http://localhost:8000/predict?file_id={file_id}'
)
predictions = response.json()['predictions']

# Process results
for pred in predictions:
    print(f"{pred['container_id']}: {pred['risk_level']} ({pred['risk_score']}%)")
```

### JavaScript/Node.js
```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

// Upload file
const formData = new FormData();
formData.append('file', fs.createReadStream('shipments.csv'));

const uploadResponse = await axios.post(
  'http://localhost:8000/upload',
  formData
);
const fileId = uploadResponse.data.file_id;

// Get predictions
const predictResponse = await axios.post(
  `http://localhost:8000/predict?file_id=${fileId}`
);
const predictions = predictResponse.data.predictions;

// Process results
predictions.forEach(pred => {
  console.log(`${pred.container_id}: ${pred.risk_level} (${pred.risk_score}%)`);
});
```

### cURL
```bash
# Upload
UPLOAD_RESPONSE=$(curl -s -X POST -F "file=@shipments.csv" http://localhost:8000/upload)
FILE_ID=$(echo $UPLOAD_RESPONSE | grep -o '"file_id":"[^"]*' | grep -o '[^"]*$')

# Predict
curl -s -X POST "http://localhost:8000/predict?file_id=$FILE_ID" | jq .

# Single prediction
curl -s -X POST http://localhost:8000/predict-single \
  -H "Content-Type: application/json" \
  -d @container.json | jq .
```

---

## Performance Notes

- **Inference Time:** ~50-100ms per container
- **Batch Size:** Up to 10,000 containers per upload
- **Memory:** ~512MB baseline + ~1MB per 1000 predictions
- **Concurrent Requests:** Supports up to 100 concurrent requests (configurable)

---

## Webhooks (Future Enhancement)

```bash
POST /webhooks/register
Content-Type: application/json

{
  "url": "https://example.com/webhook",
  "event": "prediction_complete",
  "file_id": "a1b2c3d4"
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-03-06 | Initial release |
| 1.1.0 | TBD | Webhooks, Streaming |
| 2.0.0 | TBD | Database backend, Auth |

---

## Support

For API issues:
1. Check health endpoint: `GET /health`
2. Review request schema
3. Check logs: `docker-compose logs backend`
4. Create issue with request/response details

---

**Last Updated:** 2024-03-06  
**API Version:** 1.0.0  
**Status:** Production-Ready
