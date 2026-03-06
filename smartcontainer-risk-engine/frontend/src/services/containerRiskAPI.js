/**
 * Frontend API Service
 * ====================
 * Axios service for frontend integration with backend API
 */

import axios from 'axios';

const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  import.meta.env.REACT_APP_API_URL ||
  'http://127.0.0.1:8000';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request/response interceptors for logging and error handling
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

apiClient.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status}`, response.data);
    return response;
  },
  (error) => {
    console.error('Response error:', error);
    return Promise.reject(error);
  }
);

// API Service methods
const ContainerRiskAPI = {
  RISK_LEVELS: ['Critical', 'High', 'Medium', 'Low'],

  // Health check
  health: () => apiClient.get('/health'),

  // Upload dataset
  uploadDataset: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  // Generate predictions
  predict: (fileId, useCached = true) => {
    return apiClient.post('/predict', null, {
      params: { file_id: fileId, use_cached: useCached },
    });
  },

  // Upload + predict helper
  runBatchPrediction: async (file) => {
    const uploadResponse = await ContainerRiskAPI.uploadDataset(file);
    const fileId = uploadResponse?.data?.file_id;
    if (!fileId) {
      throw new Error('Upload succeeded but file_id is missing.');
    }
    const predictResponse = await ContainerRiskAPI.predict(fileId, true);
    return {
      upload: uploadResponse,
      predict: predictResponse,
      fileId,
    };
  },

  // Get summary
  getSummary: (fileId) => {
    return apiClient.get('/summary', {
      params: { file_id: fileId },
    });
  },

  // Get predictions grouped by risk level (Critical/High/Medium/Low)
  getPredictionsByRisk: (fileId) => {
    return apiClient.get('/predictions-by-risk', {
      params: { file_id: fileId },
    });
  },

  // Fallback helper if backend returns a flat predictions list
  splitPredictionsByRisk: (predictions = []) => {
    const grouped = {
      Critical: [],
      High: [],
      Medium: [],
      Low: [],
    };
    predictions.forEach((prediction) => {
      const level = prediction?.risk_level;
      if (Object.prototype.hasOwnProperty.call(grouped, level)) {
        grouped[level].push(prediction);
      }
    });
    return grouped;
  },

  // Predict single container
  predictSingle: (containerData) => {
    return apiClient.post('/predict-single', containerData);
  },

  // Retrain model with new dataset
  retrain: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/retrain', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  // Get training status
  getTrainingStatus: (jobId) => {
    return apiClient.get(`/training-status/${jobId}`);
  },

  // Reload model after training
  reloadModel: () => {
    return apiClient.get('/reload-model');
  },

  // Download predictions as CSV
  downloadPredictions: async (predictions, filename = 'predictions.csv') => {
    if (!predictions || predictions.length === 0) {
      throw new Error('No predictions to download');
    }

    // Convert predictions to CSV
    const headers = ['Container_ID', 'Risk_Score', 'Risk_Level', 'Explanation'];
    const rows = predictions.map((p) => [
      p.container_id,
      p.risk_score.toFixed(2),
      p.risk_level,
      `"${p.explanation_summary}"`,
    ]);

    const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join(
      '\n'
    );

    // Trigger download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  },
};

export default ContainerRiskAPI;
