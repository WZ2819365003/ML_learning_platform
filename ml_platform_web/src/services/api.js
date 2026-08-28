import axios from 'axios';

// ── Auth token helpers (single-admin bearer token, see backend app/core/auth) ──
const TOKEN_KEY = 'ml_platform_token';
export const getAuthToken = () => localStorage.getItem(TOKEN_KEY) || '';
export const setAuthToken = (token) => localStorage.setItem(TOKEN_KEY, token);
export const clearAuthToken = () => localStorage.removeItem(TOKEN_KEY);
/** Append ?token= for WebSocket URLs (browsers can't set WS headers). */
export const withWsToken = (url) => {
  const token = getAuthToken();
  if (!token) return url;
  return url + (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(token);
};

const redirectToLogin = () => {
  if (window.location.pathname !== '/login') {
    clearAuthToken();
    window.location.assign('/login');
  }
};

// Use relative path for API to work in both dev and Docker environments
const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

// Second axios instance for inference routes (no /api prefix)
const inferenceApi = axios.create({
  baseURL: '/',
  timeout: 30000,
});

const attachToken = (config) => {
  const token = getAuthToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
};
api.interceptors.request.use(attachToken);
inferenceApi.interceptors.request.use(attachToken);

inferenceApi.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error?.response?.status === 401) redirectToLogin();
    console.error('Inference API请求错误:', error);
    return Promise.reject(error);
  }
);

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    // 登录接口自身的 401 交给登录页展示错误，不做跳转循环
    if (error?.response?.status === 401 && !String(error?.config?.url).includes('/auth/login')) {
      redirectToLogin();
    }
    console.error('API请求错误:', error);
    return Promise.reject(error);
  }
);

export const authApi = {
  login(username, password) {
    return api.post('/auth/login', { username, password });
  },
  me() {
    return api.get('/auth/me');
  },
};

export const systemApi = {
  health() {
    return inferenceApi.get('/health');
  },
};

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
  // Data-pipeline-as-code: run Python that transforms the dataset → new dataset.
  runPipeline(datasetId, payload) {
    return api.post(`/data/${datasetId}/pipeline`, payload);
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
  getShapSummary(taskId, params, config = {}) {
    return api.get(`/viz/${taskId}/shap_summary`, { ...config, params });
  },
  getResidualPlot(taskId, params) {
    return api.get(`/viz/${taskId}/residual_plot`, { params });
  },
  getPredictedVsActual(taskId, params) {
    return api.get(`/viz/${taskId}/predicted_vs_actual`, { params });
  },
  getPerClass(taskId) {
    return api.get(`/viz/${taskId}/per_class`);
  },
  getPrCurve(taskId) {
    return api.get(`/viz/${taskId}/pr_curve`);
  },
  getCalibration(taskId, params) {
    return api.get(`/viz/${taskId}/calibration`, { params });
  },
  getThreshold(taskId, params) {
    return api.get(`/viz/${taskId}/threshold`, { params });
  },
  getDistribution(taskId, params) {
    return api.get(`/viz/${taskId}/distribution`, { params });
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
  // Batch prediction is file-in / file-out and runs asynchronously — the POST
  // returns a job id, not results. See batch_prediction_service.py.
  submitBatchPredict(deploymentId, file) {
    const form = new FormData();
    form.append('file', file);
    return inferenceApi.post(`/inference/${deploymentId}/batch-predict`, form);
  },
  getBatchPredict(deploymentId, jobId) {
    return inferenceApi.get(`/inference/${deploymentId}/batch-predict/${jobId}`);
  },
  batchPredictDownloadUrl(deploymentId, jobId) {
    // inferenceApi's baseURL is '/', so interpolating it would yield '//inference/…'.
    return `/inference/${deploymentId}/batch-predict/${jobId}/download`;
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

// Weighted multi-model deployments. Separate from deployApi because an
// ensemble has members and weights instead of a task_id, and its predict
// response reports which members actually contributed.
export const ensembleApi = {
  create: (payload) => api.post('/deploy/ensembles', payload),
  list: (params = {}) => api.get('/deploy/ensembles', { params }),
  delete: (id) => api.delete(`/deploy/ensembles/${id}`),
  predict: (id, payload) => inferenceApi.post(`/inference/ensembles/${id}/predict`, payload),
};

export const platformTasksApi = {
  list:   (params = {}) => api.get('/platform/tasks/', { params }),
  tree:   (params = {}) => api.get('/platform/tasks/tree', { params }),
  stats:  ()            => api.get('/platform/tasks/stats'),
  get:    (id)          => api.get(`/platform/tasks/${id}`),
  detail: (id, params = {}) => api.get(`/platform/tasks/${id}/detail`, { params }),
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
  // Registry-based one-click AutoML
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
  runs: (taskId, params = {}) =>
    api.get(`/v3/tasks/${taskId}/runs`, { params }),
  createExperimentBatch: (taskId, data) =>
    api.post(`/v3/tasks/${taskId}/experiments`, data),
  createExperimentBundle: (taskId, data) =>
    api.post(`/v3/tasks/${taskId}/experiments/bulk`, data),
  tuningSpaces: (taskType) => api.get(`/v3/tasks/tuning-spaces/${taskType}`),
  progressTree: (taskId) => api.get(`/v3/tasks/${taskId}/progress-tree`),
  // AutoML is a strategy of the normal batch pipeline — its runs land on the
  // leaderboard and qualify for final evaluation like any other.
  launchAutoml: (taskId, params = {}) =>
    api.post(`/v3/tasks/${taskId}/automl`, null, { params }),
  // Markdown, not JSON — responseType keeps axios from trying to parse it.
  report: (taskId) =>
    api.get(`/v3/tasks/${taskId}/report.md`, { responseType: 'text' }),
  aiReport: (taskId) =>
    api.post(`/v3/tasks/${taskId}/ai-report`, null, { timeout: 150000 }),
  aiReportArchives: (taskId) =>
    api.get(`/v3/tasks/${taskId}/ai-reports`),
  aiReportArchive: (taskId, reportId) =>
    api.get(`/v3/tasks/${taskId}/ai-reports/${reportId}`),
  strategyComparison: (taskId) =>
    api.get(`/v3/tasks/${taskId}/strategy-comparison`),
  finalize: (taskId) =>
    api.post(`/v3/tasks/${taskId}/final-evaluation`),
  // Deploy the model trained by a run (workflow 部署 step). Bridges to the
  // underlying ML/DL deployment via the run's domain task.
  deployRun: (taskId, runId, data) =>
    api.post(`/v3/tasks/${taskId}/runs/${runId}/deploy`, data),
  // Run user Python (code-config) → dispatch a batch through the normal pipeline.
  configExec: (taskId, data) =>
    api.post(`/v3/tasks/${taskId}/config-exec`, data),
};

// A run's trained model is downloadable via its domain_task_id (returned by
// modelingTaskApi.runs/leaderboard). ML models go through /api/models/{id}/download.
export const runModelDownloadUrl = (domainTaskId) =>
  `${api.defaults.baseURL}/models/${domainTaskId}/download`;

export const platformRunsApi = {
  inspect: (runId, params = {}) => api.get(`/platform/runs/${runId}/inspector`, { params }),
  shap: (runId) => api.get(`/platform/runs/${runId}/shap`),
};

// ── V3 cross-task Run list (powers 「Run 诊断中心」 page) ────────────────────
export const v3RunsApi = {
  // Flat list of every ExperimentRun joined with its parent ModelingTask
  // so the UI can filter/sort in a single table. See ml_platform/app/
  // api/routes/v3_runs.py for query params.
  list: (params = {}) => api.get('/v3/runs/', { params }),
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
