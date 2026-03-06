# Frontend Dashboard Integration Guide

## Overview

The SmartContainer Risk Engine includes an interactive React-based dashboard for analyzing container risk predictions, visualizing results, and managing batch processing jobs.

---

## Dashboard Features

### 1. **Upload Interface**
- Drag-and-drop CSV file upload
- File validation and preview
- Progress tracking
- Error reporting

### 2. **Results Table**
- Sortable risk predictions
- Container details view
- Expandable explanation summaries
- Export to CSV

### 3. **Risk Distribution Chart**
- Visual breakdown by risk level
- Interactive charts (pie, bar, line)
- Time-series analysis
- Trend indicators

### 4. **High-Risk Container List**
- Critical risks prioritized
- Quick view of key metrics
- Action buttons
- Batch operations

---

## Integration with Backend

### API Service (`containerRiskAPI.js`)

The frontend uses a centralized API service for all backend communication:

```javascript
import ContainerRiskAPI from './services/containerRiskAPI';

// Health check
await ContainerRiskAPI.health();

// Upload file
const response = await ContainerRiskAPI.uploadDataset(file);
const fileId = response.data.file_id;

// Get predictions
const predictions = await ContainerRiskAPI.predict(fileId);

// Get summary
const summary = await ContainerRiskAPI.getSummary(fileId);

// Predict single container
const result = await ContainerRiskAPI.predictSingle(containerData);

// Download predictions
await ContainerRiskAPI.downloadPredictions(predictions, 'results.csv');
```

---

## Component Structure

```
src/
├── pages/
│   ├── DashboardPage.jsx       # Main dashboard
│   ├── UploadPage.jsx          # File upload interface
│   ├── ReportsPage.jsx         # Analytics and reports
│   └── ContainerDetailPage.jsx # Individual container details
├── components/
│   ├── Navbar.jsx              # Navigation bar
│   ├── RiskChart.jsx           # Risk distribution chart
│   ├── PredictionTable.jsx     # Results table
│   └── shared.jsx              # Shared components
├── context/
│   └── DataContext.jsx         # Global state management
└── services/
    └── containerRiskAPI.js     # API integration
```

---

## Usage Flow

### Step 1: Upload Data
```typescript
// File format: CSV with required columns
Container_ID,Declaration_Date,Origin_Country,...,Measured_Weight,Dwell_Time_Hours

// Upload via UI or API
const formData = new FormData();
formData.append('file', csvFile);
const response = await fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
});
```

### Step 2: Generate Predictions
```typescript
// Once file uploaded, request predictions
const fileId = uploadResponse.file_id;
const predictions = await ContainerRiskAPI.predict(fileId);

// Returns:
{
  "status": "success",
  "file_id": "abc123",
  "total_containers": 100,
  "predictions": [...],
  "summary": {...}
}
```

### Step 3: View Results
- Display predictions in table
- Show risk distribution chart
- List critical containers
- Display explanations

### Step 4: Export Results
```typescript
// Download predictions as CSV
await ContainerRiskAPI.downloadPredictions(predictions, 'results.csv');
```

---

## Environment Configuration

```javascript
// .env file
REACT_APP_API_URL=http://localhost:8000
REACT_APP_LOG_LEVEL=debug
REACT_APP_ENABLE_ANALYTICS=true
```

---

## Error Handling

```javascript
try {
  const predictions = await ContainerRiskAPI.predict(fileId);
} catch (error) {
  if (error.response.status === 503) {
    // Model not loaded
    console.error('Model not available');
  } else if (error.response.status === 404) {
    // File not found
    console.error('File not found. Upload first.');
  } else {
    // Other errors
    console.error('Prediction failed:', error.message);
  }
}
```

---

## Dashboard Pages

### DashboardPage
Main analytics and summary view

```jsx
import React, { useState, useEffect } from 'react';
import ContainerRiskAPI from '../services/containerRiskAPI';

export default function DashboardPage() {
  const [predictions, setPredictions] = useState([]);
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    try {
      const fileId = localStorage.getItem('currentFileId');
      if (fileId) {
        const response = await ContainerRiskAPI.predict(fileId);
        setPredictions(response.data.predictions);
        setSummary(response.data.summary);
      }
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
  };

  return (
    <div className="dashboard">
      <h1>Container Risk Analysis</h1>
      <RiskChart summary={summary} />
      <PredictionTable predictions={predictions} />
    </div>
  );
}
```

### UploadPage
File upload and processing interface

```jsx
export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  const handleUpload = async () => {
    setUploading(true);
    try {
      const response = await ContainerRiskAPI.uploadDataset(file);
      setUploadResult(response.data);
      localStorage.setItem('currentFileId', response.data.file_id);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-container">
      <input 
        type="file" 
        accept=".csv"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handleUpload} disabled={!file || uploading}>
        {uploading ? 'Uploading...' : 'Upload & Analyze'}
      </button>
      {uploadResult && (
        <div>
          <p>File ID: {uploadResult.file_id}</p>
          <p>Valid rows: {uploadResult.rows_valid} / {uploadResult.rows_received}</p>
        </div>
      )}
    </div>
  );
}
```

---

## Visualization Examples

### Risk Distribution Pie Chart
```javascript
// Using Chart.js
const chartConfig = {
  type: 'doughnut',
  data: {
    labels: ['Critical', 'High', 'Medium', 'Low'],
    datasets: [{
      data: [
        summary.critical,
        summary.high,
        summary.medium,
        summary.low
      ],
      backgroundColor: [
        '#ef4444', // red
        '#f97316', // orange
        '#eab308', // yellow
        '#22c55e'  // green
      ]
    }]
  }
};
```

### Predictions Table
```javascript
// Sample columns
<table>
  <thead>
    <tr>
      <th>Container ID</th>
      <th>Risk Score</th>
      <th>Risk Level</th>
      <th>Explanation</th>
      <th>Actions</th>
    </tr>
  </thead>
  <tbody>
    {predictions.map(p => (
      <tr key={p.container_id}>
        <td>{p.container_id}</td>
        <td>{p.risk_score.toFixed(1)}%</td>
        <td>
          <span className={`badge badge-${p.risk_level.toLowerCase()}`}>
            {p.risk_level}
          </span>
        </td>
        <td>{p.explanation_summary}</td>
        <td>
          <button>View Details</button>
          <button>Export</button>
        </td>
      </tr>
    ))}
  </tbody>
</table>
```

---

## Advanced Features

### Real-Time Updates
```javascript
// Using WebSockets (future enhancement)
const ws = new WebSocket('ws://localhost:8000/ws/predictions');
ws.onmessage = (event) => {
  const newPrediction = JSON.parse(event.data);
  setPredictions(prev => [...prev, newPrediction]);
};
```

### Data Export
```javascript
// Export to multiple formats
async function exportResults(format) {
  const predictions = await ContainerRiskAPI.predict(fileId);
  
  if (format === 'csv') {
    await ContainerRiskAPI.downloadPredictions(predictions);
  } else if (format === 'json') {
    const json = JSON.stringify(predictions, null, 2);
    download(json, 'predictions.json');
  } else if (format === 'excel') {
    // Use XLSX library
    const ws = XLSX.utils.json_to_sheet(predictions);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Predictions');
    XLSX.writeFile(wb, 'predictions.xlsx');
  }
}
```

### Batch Operations
```javascript
// Select multiple containers and perform batch actions
const handleBatchAction = async (action, selectedIds) => {
  switch(action) {
    case 'export':
      const selected = predictions.filter(p => selectedIds.includes(p.container_id));
      await ContainerRiskAPI.downloadPredictions(selected);
      break;
    case 'detail_report':
      // Generate detailed report
      break;
    case 'escalate':
      // Mark for escalation/review
      break;
  }
};
```

---

## Testing

```javascript
// Test API integration
describe('ContainerRiskAPI', () => {
  it('should upload file successfully', async () => {
    const file = new File(['test'], 'test.csv');
    const response = await ContainerRiskAPI.uploadDataset(file);
    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('file_id');
  });

  it('should fetch predictions', async () => {
    const response = await ContainerRiskAPI.predict('test-id');
    expect(response.data).toHaveProperty('predictions');
    expect(Array.isArray(response.data.predictions)).toBe(true);
  });
});
```

---

## Performance Optimization

### Pagination
```javascript
// For large datasets
const [page, setPage] = useState(1);
const PAGE_SIZE = 50;

const visiblePredictions = predictions.slice(
  (page - 1) * PAGE_SIZE,
  page * PAGE_SIZE
);
```

### Caching
```javascript
// Cache API responses
const cache = new Map();

async function cachedPredict(fileId) {
  if (cache.has(fileId)) {
    return cache.get(fileId);
  }
  const response = await ContainerRiskAPI.predict(fileId);
  cache.set(fileId, response.data);
  return response.data;
}
```

---

## Accessibility

```jsx
// ARIA labels for accessibility
<button 
  aria-label="Upload CSV file for container analysis"
  onClick={handleUpload}
>
  Upload File
</button>

// Semantic HTML
<main role="main">
  <section aria-label="Risk summary statistics">
    {/* Summary content */}
  </section>
  <section aria-label="Container predictions table">
    {/* Table content */}
  </section>
</main>
```

---

## Deployment

### Frontend Build
```bash
cd DS-ML/custom-container
npm run build

# Output in dist/
```

### Environment Variables for Production
```bash
REACT_APP_API_URL=https://api.smartcontainer.io
REACT_APP_ENABLE_ANALYTICS=true
REACT_APP_LOG_LEVEL=error
```

---

## Troubleshooting

### CORS Issues
```javascript
// Ensure backend has CORS enabled
// In FastAPI:
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  // Restrict to frontend domain in production
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
```

### API Connection Failed
```javascript
// Check if API is running
fetch('http://localhost:8000/health')
  .then(r => console.log('API healthy'))
  .catch(() => console.error('API unavailable'))
```

---

## Best Practices

1. **Error Handling** - Always wrap API calls in try-catch
2. **Loading States** - Show spinners during API calls
3. **Data Validation** - Validate CSV before upload
4. **Caching** - Cache predictions to reduce API calls
5. **Pagination** - Paginate large result sets
6. **Accessibility** - Include ARIA labels and semantic HTML

---

**Last Updated:** 2024-03-06  
**Version:** 1.0.0
