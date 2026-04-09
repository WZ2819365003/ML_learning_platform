import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000/api',
  timeout: 30000,
});

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
};

export default api;
