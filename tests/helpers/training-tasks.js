const fs = require('fs');
const path = require('path');
const { API_ROOT } = require('./e2e-env');

const DATASET_PATH = path.resolve(__dirname, '..', '..', 'examples', 'data', 'predictive_maintenance.csv');

const REGRESSION_MODELS = new Set([
  'linear_regression', 'ridge', 'lasso', 'elasticnet',
  'random_forest_regressor', 'xgboost_regressor', 'lightgbm_regressor',
  'svr', 'mlp_regressor', 'mlp_dl_regressor', 'lstm_regressor', 'tcn_regressor',
]);

function apiRoot() {
  return API_ROOT;
}

async function findSuccessfulTrainingTasks(request) {
  const response = await request.get(`${apiRoot()}/api/training/list?page=1&page_size=100&status=SUCCESS`);
  if (!response.ok()) return { classification: null, regression: null };
  const tasks = (await response.json()).items || [];
  const hasCv = (task) => Object.keys(task.result_metrics || {}).some((key) => key.startsWith('cv_avg_'));
  return {
    classification: tasks.find((task) => !REGRESSION_MODELS.has(task.model_type) && hasCv(task)) || null,
    regression: tasks.find((task) => REGRESSION_MODELS.has(task.model_type) && hasCv(task)) || null,
  };
}

async function waitForTrainingTask(request, taskId, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const response = await request.get(`${apiRoot()}/api/training/${taskId}/status`);
    if (response.ok()) {
      const task = await response.json();
      if (task.status === 'SUCCESS') return task;
      if (['FAILED', 'CANCELED'].includes(task.status)) {
        throw new Error(`Training task ${taskId} ended as ${task.status}: ${task.error_message || ''}`);
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(`Timed out waiting for training task ${taskId}`);
}

async function ensurePredictiveDataset(request) {
  const listResponse = await request.get(`${apiRoot()}/api/data/list?page=1&page_size=100`);
  if (!listResponse.ok()) throw new Error(`Dataset list failed with HTTP ${listResponse.status()}`);
  const datasets = (await listResponse.json()).items || [];
  const existing = datasets.find((item) => item.name?.includes('predictive_maintenance'));
  if (existing) return existing;

  const uploadResponse = await request.post(`${apiRoot()}/api/data/upload`, {
    multipart: { file: fs.createReadStream(DATASET_PATH) },
  });
  const payload = await uploadResponse.json();
  if (!uploadResponse.ok()) throw new Error(`Dataset upload failed: ${JSON.stringify(payload)}`);
  return payload;
}

async function startTrainingTask(request, payload) {
  const response = await request.post(`${apiRoot()}/api/training/start`, { data: payload });
  const task = await response.json();
  if (!response.ok()) throw new Error(`Training start failed: ${JSON.stringify(task)}`);
  await waitForTrainingTask(request, task.id);
}

async function ensureSuccessfulTrainingTasks(request) {
  let tasks = await findSuccessfulTrainingTasks(request);
  if (tasks.classification && tasks.regression) return tasks;

  const dataset = await ensurePredictiveDataset(request);
  if (!tasks.classification) {
    await startTrainingTask(request, {
      dataset_id: dataset.id,
      target_column: 'Target',
      model_type: 'logistic_regression',
      hyperparameters: { max_iter: 300 },
      test_size: 0.2,
      eval_metrics: ['accuracy', 'f1', 'roc_auc'],
      cross_validation: { enabled: true, folds: 3 },
    });
  }
  if (!tasks.regression) {
    await startTrainingTask(request, {
      dataset_id: dataset.id,
      target_column: 'Torque [Nm]',
      model_type: 'linear_regression',
      hyperparameters: {},
      test_size: 0.2,
      eval_metrics: ['rmse', 'mae', 'r2'],
      cross_validation: { enabled: true, folds: 3 },
    });
  }

  tasks = await findSuccessfulTrainingTasks(request);
  if (!tasks.classification || !tasks.regression) {
    throw new Error('Unable to seed successful classification and regression tasks with CV metrics');
  }
  return tasks;
}

module.exports = { apiRoot, ensureSuccessfulTrainingTasks, findSuccessfulTrainingTasks };
