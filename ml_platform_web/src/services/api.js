import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000/api',
  timeout: 30000,
});

// Second axios instance for inference routes (no /api prefix)
const inferenceApi = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 30000,
});
inferenceApi.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('Inference API请求错误:', error);
    return Promise.reject(error);
  }
);

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API请求错误:', error);
    return Promise.reject(error);
  }
);

export const dataApi = {
  listDatasets(params) {
    return api.get('/data/list', { params });
  },
  uploadDataset(file, onUploadProgress) {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    });
  },
  previewDataset(datasetId) {
    return api.get(`/data/${datasetId}/preview`);
  },
  deleteDataset(datasetId) {
    return api.delete(`/data/${datasetId}`);
  },
};

export const trainingApi = {
  listModels() {
    return api.get('/training/models');
  },
  listTasks(params) {
    return api.get('/training/list', { params });
  },
  startTraining(payload) {
    return api.post('/training/start', payload);
  },
  getTrainingStatus(taskId) {
    return api.get(`/training/${taskId}/status`);
  },
  stopTraining(taskId) {
    return api.post(`/training/${taskId}/stop`);
  },
};

export const logsApi = {
  getLogs(taskId, params) {
    return api.get(`/logs/${taskId}`, { params });
  },
};

export const modelApi = {
  listModels(params) {
    return api.get('/models/list', { params });
  },
  getModelDetail(taskId) {
    return api.get(`/models/${taskId}/detail`);
  },
  compareModels(taskIds) {
    return api.get('/models/compare', {
      params: {
        task_ids: taskIds.join(','),
      },
    });
  },
  deleteModel(taskId) {
    return api.delete(`/models/${taskId}`);
  },
  predict(taskId, payload) {
    return api.post(`/models/${taskId}/predict`, payload);
  },
};

export const vizApi = {
  getConfusionMatrix(taskId, params) {
    return api.get(`/viz/${taskId}/confusion_matrix`, { params });
  },
  getRocCurve(taskId) {
    return api.get(`/viz/${taskId}/roc_curve`);
  },
  getFeatureImportance(taskId) {
    return api.get(`/viz/${taskId}/feature_importance`);
  },
  getLearningCurve(taskId) {
    return api.get(`/viz/${taskId}/learning_curve`);
  },
  getShapSummary(taskId, params) {
    return api.get(`/viz/${taskId}/shap_summary`, { params });
  },
  getResidualPlot(taskId) {
    return api.get(`/viz/${taskId}/residual_plot`);
  },
  getPredictedVsActual(taskId) {
    return api.get(`/viz/${taskId}/predicted_vs_actual`);
  },
};

export const dataEnhancedApi = {
  getCorrelation(datasetId, method = 'pearson') {
    return api.get(`/data/${datasetId}/correlation?method=${method}`);
  },
  getTargetDistribution(datasetId, targetColumn) {
    return api.get(`/data/${datasetId}/target_distribution?target_column=${encodeURIComponent(targetColumn)}`);
  },
};

// deploy/predict use separate axios instances because:
// - /api/deploy/... → api (has /api base)
// - /inference/... → inferenceApi (no /api prefix)
export const deployApi = {
  createDeployment(taskId, payload) {
    return api.post(`/deploy/${taskId}`, payload);
  },
  listDeployments(params = {}) {
    return api.get('/deploy/list', { params });
  },
  deleteDeployment(deploymentId) {
    return api.delete(`/deploy/${deploymentId}`);
  },
  updateStatus(deploymentId, status) {
    return api.patch(`/deploy/${deploymentId}/status?status=${status}`);
  },
  predict(deploymentId, payload) {
    return inferenceApi.post(`/inference/${deploymentId}/predict`, payload);
  },
  getResult(deploymentId, jobId) {
    return inferenceApi.get(`/inference/${deploymentId}/result/${jobId}`);
  },
};

export const dlApi = {
  listModels:    ()       => api.get('/dl/models'),
  startTraining: (data)   => api.post('/dl/train', data),
  listTasks:     (params) => api.get('/dl/list', { params }),
  getStatus:     (id)     => api.get(`/dl/${id}/status`),
  stopTask:      (id)     => api.post(`/dl/${id}/stop`),
};

export default api;
