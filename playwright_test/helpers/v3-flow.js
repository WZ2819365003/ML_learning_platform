// Shared V3 release-gate helpers — used by 08 (e2e gate) and 11 (pro-viz gate).
// Both specs need: fresh dataset upload + plan + task + launch + wait-for-run.

const fs = require('fs');
const path = require('path');
const { expect } = require('@playwright/test');
const { getJson, postJson, BASE_API } = require('./api');

const POLL_INTERVAL = 2000;
const DEFAULT_MAX_WAIT_MS = 120_000;
const DEFAULT_MODEL_TOKEN = 'logistic_regression';
const DEFAULT_TARGET_COLUMN = 'Target';

function runStatus(run) {
  return String(run?.status || run?.platform_task?.status || '').toUpperCase();
}

async function waitForRun(request, taskId, maxWaitMs = DEFAULT_MAX_WAIT_MS) {
  const t0 = Date.now();
  while (Date.now() - t0 < maxWaitMs) {
    const r = await getJson(request, `/v3/tasks/${taskId}/runs`);
    expect(r.ok, `poll runs failed: status=${r.status} body=${r.raw?.slice(0, 500)}`).toBeTruthy();
    const runs = r.body?.items || r.body?.runs || [];
    if (runs.length > 0) {
      const finished = runs.find((x) =>
        ['SUCCESS', 'FAILED', 'CANCELED'].includes(runStatus(x)),
      );
      if (finished) return finished;
    }
    await new Promise((res) => setTimeout(res, POLL_INTERVAL));
  }
  throw new Error(`Timed out after ${maxWaitMs}ms waiting for V3 task ${taskId} to finish`);
}

/**
 * Upload a fresh copy of examples/data/predictive_maintenance.csv with a unique
 * probe column appended.  Defeats backend content-dedup so the dataset row
 * created here always points to a freshly-stored file (Docker rebuilds wipe
 * /storage/uploads but leave dataset rows in MySQL → stale references).
 *
 * Target column is `Target` (binary classification label, 0/1).
 */
async function uploadFreshClassificationDataset(request, targetColumn = DEFAULT_TARGET_COLUMN) {
  const sourcePath = path.resolve(__dirname, '../../examples/data/predictive_maintenance.csv');
  expect(fs.existsSync(sourcePath), `Missing fixture dataset: ${sourcePath}`).toBeTruthy();

  const raw = fs.readFileSync(sourcePath, 'utf8').trimEnd();
  const lines = raw.split(/\r?\n/).filter(Boolean);
  expect(lines.length, 'Fixture dataset should contain header and rows').toBeGreaterThan(1);

  const probeColumn = `e2e_probe_${Date.now()}`;
  const csv = [
    `${lines[0]},${probeColumn}`,
    ...lines.slice(1).map((line) => `${line},1`),
  ].join('\n');

  const res = await request.post(`${BASE_API}/data/upload`, {
    multipart: {
      file: {
        name: `e2e-predictive-maintenance-${Date.now()}.csv`,
        mimeType: 'text/csv',
        buffer: Buffer.from(csv, 'utf8'),
      },
    },
  });
  const bodyText = await res.text();
  let body = null;
  try { body = bodyText ? JSON.parse(bodyText) : null; } catch { /* leave null */ }

  expect(res.ok(), `upload dataset failed: status=${res.status()} body=${bodyText.slice(0, 800)}`).toBeTruthy();
  expect(body?.id).toBeTruthy();
  expect(
    body?.columns_info?.[targetColumn],
    `Uploaded dataset must expose ${targetColumn}`,
  ).toBeTruthy();
  return body;
}

/**
 * Run the full V3 release-gate flow up to a SUCCESS run.  Returns
 * `{ dataset, plan, task, run }` for downstream assertions.  Caller is
 * responsible for `test.setTimeout(180_000)` since the run alone can take
 * up to 2 minutes on cold caches.
 */
async function runBaselineFlow(request, opts = {}) {
  const {
    modelToken = DEFAULT_MODEL_TOKEN,
    targetColumn = DEFAULT_TARGET_COLUMN,
    namePrefix = 'e2e',
    evalMetrics = ['accuracy', 'f1', 'roc_auc'],
    budget = { max_trials: 1, cv_folds: 3, test_size: 0.2 },
  } = opts;

  const dataset = await uploadFreshClassificationDataset(request, targetColumn);

  const plan = await postJson(request, '/platform/training-plans', {
    name: `${namePrefix}-plan-${Date.now()}`,
    description: 'playwright v3 release gate',
    task_type: 'classification',
    strategy_type: 'baseline',
    model_family: 'ml',
    selected_models: [modelToken],
    eval_metrics: evalMetrics,
    budget_config: budget,
  });
  expect(plan.ok, `create plan failed: ${plan.status} ${plan.raw?.slice(0, 800)}`).toBeTruthy();
  expect(plan.body?.id).toBeTruthy();

  const taskRes = await postJson(request, '/v3/tasks/', {
    name: `${namePrefix}-task-${Date.now()}`,
    task_type: 'classification',
    objective_metric: 'accuracy',
    objective_direction: 'max',
    training_plan_id: plan.body.id,
    dataset_id: dataset.id,
    target_column: targetColumn,
  });
  expect(taskRes.ok, `create task failed: ${taskRes.status} ${taskRes.raw?.slice(0, 800)}`).toBeTruthy();
  expect(taskRes.body?.id).toBeTruthy();
  const task = taskRes.body;

  const launch = await postJson(request, `/v3/tasks/${task.id}/experiments`, {
    name: `${namePrefix}-baseline-${Date.now()}`,
    strategy_type: 'baseline',
    selected_models: [modelToken],
    search_space: {},
    budget_config: budget,
    eval_metrics: evalMetrics,
    model_family: 'ml',
  });
  expect(launch.ok, `launch failed: ${launch.status} ${launch.raw?.slice(0, 1000)}`).toBeTruthy();

  const run = await waitForRun(request, task.id);
  expect(runStatus(run), `run not SUCCESS: ${JSON.stringify(run).slice(0, 1200)}`).toBe('SUCCESS');

  return { dataset, plan: plan.body, task, run };
}

module.exports = {
  POLL_INTERVAL,
  DEFAULT_MAX_WAIT_MS,
  DEFAULT_MODEL_TOKEN,
  DEFAULT_TARGET_COLUMN,
  runStatus,
  waitForRun,
  uploadFreshClassificationDataset,
  runBaselineFlow,
};
