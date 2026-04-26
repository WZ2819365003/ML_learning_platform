// 08 — V3 端到端发布门禁：
// 创建 TrainingPlan → 创建 ModelingTask → 启动 baseline 实验 → 等待 run
// → 校验 RunInspector / SHAP。该用例必须真跑完整链路，不能把失败降级为 annotation。
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');
const { getJson, postJson, BASE_API } = require('../helpers/api');

test.describe.configure({ mode: 'serial' });

const POLL_INTERVAL = 2000;
const MAX_WAIT_MS = 120_000;
const MODEL_TOKEN = 'logistic_regression';
const TARGET_COLUMN = 'Target';

function runStatus(run) {
  return String(run?.status || run?.platform_task?.status || '').toUpperCase();
}

async function waitForRun(request, taskId) {
  const t0 = Date.now();
  while (Date.now() - t0 < MAX_WAIT_MS) {
    const r = await getJson(request, `/v3/tasks/${taskId}/runs`);
    expect(r.ok, `poll runs failed: status=${r.status} body=${r.raw?.slice(0, 500)}`).toBeTruthy();
    const runs = r.body?.items || r.body?.runs || [];
    if (runs.length > 0) {
      const finished = runs.find((x) => ['SUCCESS', 'FAILED', 'CANCELED'].includes(runStatus(x)));
      if (finished) return finished;
    }
    await new Promise((res) => setTimeout(res, POLL_INTERVAL));
  }
  throw new Error(`Timed out after ${MAX_WAIT_MS}ms waiting for V3 task ${taskId} to finish`);
}

async function uploadFreshClassificationDataset(request) {
  const sourcePath = path.resolve(__dirname, '../../examples/data/predictive_maintenance.csv');
  expect(fs.existsSync(sourcePath), `Missing fixture dataset: ${sourcePath}`).toBeTruthy();

  const raw = fs.readFileSync(sourcePath, 'utf8').trimEnd();
  const lines = raw.split(/\r?\n/).filter(Boolean);
  expect(lines.length, 'Fixture dataset should contain header and rows').toBeGreaterThan(1);

  // Add a run-specific constant column so backend content-dedup cannot return a
  // stale DB row whose uploaded file disappeared during Docker rebuild.
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
  expect(body?.columns_info?.[TARGET_COLUMN], `Uploaded dataset must expose ${TARGET_COLUMN}`).toBeTruthy();
  return body;
}

test.describe('08 V3 端到端', () => {
  test('8.1 创建 plan + task + 启动 baseline 实验并校验 inspector / SHAP', async ({ request }) => {
    test.setTimeout(180_000);

    const ds = await uploadFreshClassificationDataset(request);

    const plan = await postJson(request, '/platform/training-plans', {
      name: `e2e-plan-${Date.now()}`,
      description: 'playwright v3 release gate',
      task_type: 'classification',
      strategy_type: 'baseline',
      model_family: 'ml',
      selected_models: [MODEL_TOKEN],
      eval_metrics: ['accuracy', 'f1', 'roc_auc'],
      budget_config: { max_trials: 1, cv_folds: 3, test_size: 0.2 },
    });
    expect(plan.ok, `create plan failed: status=${plan.status} body=${plan.raw?.slice(0, 800)}`).toBeTruthy();
    expect(plan.body?.id).toBeTruthy();

    const taskRes = await postJson(request, '/v3/tasks/', {
      name: `e2e-task-${Date.now()}`,
      task_type: 'classification',
      objective_metric: 'accuracy',
      objective_direction: 'max',
      training_plan_id: plan.body.id,
      dataset_id: ds.id,
      target_column: TARGET_COLUMN,
    });
    expect(taskRes.ok, `create task failed: status=${taskRes.status} body=${taskRes.raw?.slice(0, 800)}`).toBeTruthy();
    expect(taskRes.body?.id).toBeTruthy();
    const taskId = taskRes.body.id;

    const launch = await postJson(request, `/v3/tasks/${taskId}/experiments`, {
      name: `e2e-baseline-${Date.now()}`,
      strategy_type: 'baseline',
      selected_models: [MODEL_TOKEN],
      search_space: {},
      budget_config: { max_trials: 1, cv_folds: 3, test_size: 0.2 },
      eval_metrics: ['accuracy', 'f1', 'roc_auc'],
      model_family: 'ml',
    });
    expect(launch.ok, `launch experiment failed: status=${launch.status} body=${launch.raw?.slice(0, 1000)}`).toBeTruthy();
    expect(launch.body?.experiment?.id || launch.body?.id).toBeTruthy();

    const run = await waitForRun(request, taskId);
    expect(runStatus(run), `run failed: ${JSON.stringify(run).slice(0, 1200)}`).toBe('SUCCESS');
    expect(run.model_type || run.params?.model_type).toBe(MODEL_TOKEN);
    expect(typeof run.objective_value, `run has objective_value: ${JSON.stringify(run).slice(0, 800)}`).toBe('number');

    const runId = run.run_id || run.id;
    expect(runId).toBeTruthy();

    const inspector = await getJson(request, `/platform/runs/${runId}/inspector`);
    expect(inspector.ok, `inspector failed: status=${inspector.status} body=${inspector.raw?.slice(0, 800)}`).toBeTruthy();
    expect(inspector.body?.run?.id || inspector.body?.run?.run_id).toBeTruthy();
    expect(inspector.body?.experiment?.strategy_type).toBe('baseline');

    const shap = await getJson(request, `/platform/runs/${runId}/shap?compute=true`);
    expect(shap.ok, `shap failed: status=${shap.status} body=${shap.raw?.slice(0, 800)}`).toBeTruthy();
    expect(shap.body?.status, `shap payload: ${JSON.stringify(shap.body).slice(0, 800)}`).toBe('ready');
    expect(Number(shap.body?.feature_count || 0)).toBeGreaterThan(0);
  });
});
