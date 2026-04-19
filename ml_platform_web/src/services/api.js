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
  renameTask(taskId, name) {
    return api.patch(`/training/${taskId}/name`, { name });
  },
  updateMeta(taskId, meta) {
    return api.patch(`/training/${taskId}/meta`, meta);
  },
  deleteTask(taskId) {
    return api.delete(`/training/${taskId}`);
  },
};

export const logsApi = {
  getLogs(taskId, params) {
    return api.get(`/logs/${taskId}`, { params });
  },
  getMetrics(taskId) {
    return api.get(`/logs/${taskId}/metrics`);
  },
};

export const modelApi = {
  listAssets(params) {
    return api.get('/models/assets', { params });
  },
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
  listTags() {
    return api.get('/models/tags');
  },
  createTag(name, dimension, color) {
    return api.post('/models/tags/create', { name, dimension, color });
  },
  syncTags(tags) {
    return api.post('/models/tags/sync', { tags });
  },
  deleteTag(name) {
    return api.delete(`/models/tags/${encodeURIComponent(name)}`);
  },
  downloadModelUrl(taskId) {
    return `${api.defaults.baseURL}/models/${taskId}/download`;
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
  listUnifiedDeployments(params = {}) {
    return api.get('/deploy/assets', { params });
  },
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
  listModels:        ()             => api.get('/dl/models'),
  startTraining:     (data)         => api.post('/dl/train', data),
  listTasks:         (params)       => api.get('/dl/list', { params }),
  getStatus:         (id)           => api.get(`/dl/${id}/status`),
  stopTask:          (id)           => api.post(`/dl/${id}/stop`),
  renameTask:        (id, name)     => api.patch(`/dl/${id}/name`, { name }),
  updateMeta:        (id, meta)     => api.patch(`/dl/${id}/meta`, meta),
  deleteTask:        (id)           => api.delete(`/dl/${id}`),
  listTrainedModels: (params)       => api.get('/dl/trained-models', { params }),
  getLogs:           (id, params)   => api.get(`/dl/${id}/logs`, { params }),
  getEpochs:         (id, params)   => api.get(`/dl/${id}/epochs`, { params }),
  // Direct model prediction (no deployment needed)
  predictTask:       (taskId, data)   => api.post(`/dl/${taskId}/predict`, data),
  // Deployments
  createDeployment:  (dlTaskId, data) => api.post(`/dl/deployments/${dlTaskId}`, data),
  listDeployments:   ()             => api.get('/dl/deployments'),
  deleteDeployment:  (depId)        => api.delete(`/dl/deployments/${depId}`),
  toggleDeployment:  (depId, status) => api.patch(`/dl/deployments/${depId}/status`, null, { params: { status } }),
  predictDeployment: (depId, data)  => api.post(`/dl/deployments/${depId}/predict`, data),
};

export const timesfmApi = {
  modelStatus:     ()                 => api.get('/timesfm/model/status'),
  preloadModel:    (modelName)        => api.post('/timesfm/model/preload', null, { params: { model_name: modelName } }),
  startForecast:   (data)             => api.post('/timesfm/start', data),
  listForecasts:   (params)           => api.get('/timesfm/list', { params }),
  getForecast:     (id)               => api.get(`/timesfm/${id}`),
  deleteForecast:  (id)               => api.delete(`/timesfm/${id}`),
};

export const tsApi = {
  listTasks: (params = {}) => api.get('/ts/tasks', { params }),
  createTask: (data) => api.post('/ts/tasks', data),
  getTask: (taskId) => api.get(`/ts/tasks/${taskId}`),
  updateTaskMeta: (taskId, data) => api.patch(`/ts/tasks/${taskId}/meta`, data),
  deleteTask: (taskId) => api.delete(`/ts/tasks/${taskId}`),
  modelStatus: () => api.get('/ts/model/status'),
  preloadModel: (modelName) => api.post('/ts/model/preload', null, { params: { model_name: modelName } }),
  listDeployments: (params = {}) => api.get('/ts/deployments', { params }),
  createDeployment: (data) => api.post('/ts/deployments', data),
  getDeployment: (deploymentId) => api.get(`/ts/deployments/${deploymentId}`),
  updateDeploymentStatus: (deploymentId, status) =>
    api.patch(`/ts/deployments/${deploymentId}/status`, null, { params: { status } }),
  deleteDeployment: (deploymentId) => api.delete(`/ts/deployments/${deploymentId}`),
  predictDeployment: (deploymentId, data) => api.post(`/ts/deployments/${deploymentId}/predict`, data),
};

// ── V3 Platform APIs ────────────────────────────────────────────────────────

export const platformTasksApi = {
  list:   (params = {}) => api.get('/platform/tasks/', { params }),
  tree:   (params = {}) => api.get('/platform/tasks/tree', { params }),
  stats:  ()            => api.get('/platform/tasks/stats'),
  get:    (id)          => api.get(`/platform/tasks/${id}`),
  retry:  (id)          => api.post(`/platform/tasks/${id}/retry`),
  cancel: (id)          => api.post(`/platform/tasks/${id}/cancel`),
  delete: (id)          => api.delete(`/platform/tasks/${id}`),
};

export const platformExperimentsApi = {
  list: (params = {}) => api.get('/platform/experiments/', { params }),
  create: (data) => api.post('/platform/experiments/', data),
  get: (id) => api.get(`/platform/experiments/${id}`),
  delete: (id) => api.delete(`/platform/experiments/${id}`),
  listRuns: (experimentId, params = {}) => api.get(`/platform/experiments/${experimentId}/runs`, { params }),
  createRun: (experimentId, data) => api.post(`/platform/experiments/${experimentId}/runs`, data),
  getRun: (experimentId, runId) => api.get(`/platform/experiments/${experimentId}/runs/${runId}`),
  getLeaderboard: (experimentId) => api.get(`/platform/experiments/${experimentId}/leaderboard`),
  // Custom candidates (explicit list)
  submitAutoml: (experimentId, candidates) => api.post(`/platform/experiments/${experimentId}/automl`, { candidates }),
  // Registry-based one-click AutoML
  submitAutomlRegistry: (experimentId, config) => api.post(`/platform/experiments/${experimentId}/automl/registry`, config),
  // AutoML candidate list
  listAutomlCandidates: (taskType = 'classification') => api.get('/platform/experiments/automl/candidates', { params: { task_type: taskType } }),
  triggerExplain: (experimentId, runId) => api.post(`/platform/experiments/${experimentId}/runs/${runId}/explain`),
  getExplain: (experimentId, runId) => api.get(`/platform/experiments/${experimentId}/runs/${runId}/explain`),
};

// ── V3 Modeling Task Workbench ───────────────────────────────────────────────

export const modelingTaskApi = {
  list: (params = {}) => api.get('/v3/tasks/', { params }),
  create: (data) => api.post('/v3/tasks/', data),
  get: (taskId) => api.get(`/v3/tasks/${taskId}`),
  update: (taskId, data) => api.patch(`/v3/tasks/${taskId}`, data),
  delete: (taskId) => api.delete(`/v3/tasks/${taskId}`),
  leaderboard: (taskId, topK = 20) =>
    api.get(`/v3/tasks/${taskId}/leaderboard`, { params: { top_k: topK } }),
  createExperimentBatch: (taskId, data) =>
    api.post(`/v3/tasks/${taskId}/experiments`, data),
  tuningSpaces: (taskType) => api.get(`/v3/tasks/tuning-spaces/${taskType}`),
};

export const platformRunsApi = {
  inspect: (runId, params = {}) => api.get(`/platform/runs/${runId}/inspector`, { params }),
  shap: (runId) => api.get(`/platform/runs/${runId}/shap`),
};

// ── Training Plans (reusable templates) ──────────────────────────────────────

export const trainingPlansApi = {
  list: (params = {}) => api.get('/platform/training-plans', { params }),
  get: (id) => api.get(`/platform/training-plans/${id}`),
  create: (data) => api.post('/platform/training-plans', data),
  update: (id, data) => api.patch(`/platform/training-plans/${id}`, data),
  remove: (id) => api.delete(`/platform/training-plans/${id}`),
  markUsed: (id) => api.post(`/platform/training-plans/${id}/mark-used`),
};

// ── Dataset Versioning API ───────────────────────────────────────────────────

export const dataVersionsApi = {
  list: (datasetId) => api.get(`/data/${datasetId}/versions`),
  create: (datasetId, description = null) => api.post(`/data/${datasetId}/versions`, { description }),
  get: (datasetId, versionId) => api.get(`/data/${datasetId}/versions/${versionId}`),
};

export default api;
