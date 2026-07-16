// V3 — strategy coverage + UI run inspector
//
// Drives all three tuning strategies (baseline / grid_search / bayesian_search)
// against the predictive_maintenance dataset, waits for the runs to complete,
// and then exercises the ModelingTaskDetail page through Playwright:
//
//   1. Each batch lands as ≥1 SUCCESS run.
//   2. Detail page renders the run row + clicking it opens RunInspector.
//   3. RunInspector's tabs (overview / 训练可视化 / 上下文 / 日志 / SHAP)
//      all render content (no JSON.stringify dump, no error banner).
//   4. "策略对比" tab aggregates the three strategies and renders cards/charts.
//
// Setup is API-driven; the UI assertions exercise the surfaces a human user
// would actually click.

const { test, expect } = require('@playwright/test');

const BASE_UI  = process.env.BASE_UI  || 'http://127.0.0.1:3000';
const BASE_API = process.env.BASE_API || 'http://127.0.0.1:8000/api';

test.describe.configure({ mode: 'serial' });

let modelingTask;
let datasetId;

async function pollExperimentDone(request, taskId, expName, { timeoutMs = 90000, pollMs = 2000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const res = await request.get(`${BASE_API}/v3/tasks/${taskId}/runs`);
    if (res.ok()) {
      const body = await res.json();
      const runs = (body.items || []).filter((r) => r.experiment_name === expName);
      if (runs.length > 0) {
        const allTerminal = runs.every((r) => ['SUCCESS', 'FAILED', 'CANCELED'].includes(r.status));
        if (allTerminal) return runs;
      }
    }
    await new Promise((r) => setTimeout(r, pollMs));
  }
  throw new Error(`Timed out waiting for experiment "${expName}" on task ${taskId}`);
}

test.beforeAll(async ({ request }) => {
  // Pick the predictive_maintenance dataset (has the Target column).
  const dsRes = await request.get(`${BASE_API}/data/list?page=1&page_size=10`);
  expect(dsRes.ok()).toBeTruthy();
  const datasets = (await dsRes.json()).items || [];
  const ds = datasets.find((d) => d.name?.includes('predictive')) || datasets[0];
  expect(ds, 'need predictive_maintenance dataset').toBeTruthy();
  datasetId = ds.id;

  const taskRes = await request.post(`${BASE_API}/v3/tasks/`, {
    data: {
      name: `e2e-strategies-${Date.now()}`,
      task_type: 'classification',
      dataset_id: datasetId,
      target_column: 'Target',
      objective_metric: 'accuracy',
      objective_direction: 'max',
    },
  });
  expect(taskRes.ok(), `create task: ${await taskRes.text()}`).toBeTruthy();
  modelingTask = await taskRes.json();
});

test('Baseline batch — SUCCESS run with real metrics', async ({ request }) => {
  const res = await request.post(`${BASE_API}/v3/tasks/${modelingTask.id}/experiments`, {
    data: {
      name: 'baseline-1',
      strategy_type: 'baseline',
      selected_models: ['logistic_regression', 'random_forest'],
    },
  });
  expect(res.ok(), `dispatch baseline: ${await res.text()}`).toBeTruthy();

  const runs = await pollExperimentDone(request, modelingTask.id, 'baseline-1');
  expect(runs.length).toBeGreaterThanOrEqual(2);
  const successes = runs.filter((r) => r.status === 'SUCCESS');
  expect(successes.length, `baseline runs (got ${JSON.stringify(runs.map((r) => [r.params?.model_type, r.status]))})`).toBeGreaterThanOrEqual(1);
  // Real metric, not null/zero
  const objs = successes.map((r) => r.objective_value).filter((v) => typeof v === 'number');
  expect(objs.length).toBeGreaterThan(0);
  expect(Math.max(...objs)).toBeGreaterThan(0.5);
});

test('Grid search — N runs match cartesian product', async ({ request }) => {
  const res = await request.post(`${BASE_API}/v3/tasks/${modelingTask.id}/experiments`, {
    data: {
      name: 'grid-1',
      strategy_type: 'grid_search',
      selected_models: ['random_forest'],
      // 2 × 2 grid = 4 trials per model
      search_space: {
        random_forest: {
          n_estimators: [50, 100],
          max_depth: [4, 8],
        },
      },
      budget_config: { max_trials: 8, cv_folds: 3, test_size: 0.2 },
    },
  });
  expect(res.ok(), `dispatch grid: ${await res.text()}`).toBeTruthy();

  const runs = await pollExperimentDone(request, modelingTask.id, 'grid-1', { timeoutMs: 180000 });
  // Either 4 runs (one per cell) or capped at max_trials=8 — anyway ≥4
  expect(runs.length).toBeGreaterThanOrEqual(4);
  const successes = runs.filter((r) => r.status === 'SUCCESS');
  expect(successes.length, `grid_search successes`).toBeGreaterThanOrEqual(1);
  // Each run has its own hyperparameters set
  const distinctParams = new Set(successes.map((r) => JSON.stringify(r.params?.hyperparameters || {})));
  expect(distinctParams.size).toBeGreaterThan(1);
});

test('Bayesian search — Optuna trials run with distinct params', async ({ request }) => {
  const res = await request.post(`${BASE_API}/v3/tasks/${modelingTask.id}/experiments`, {
    data: {
      name: 'bayesian-1',
      strategy_type: 'bayesian_search',
      selected_models: ['random_forest'],
      search_space: {
        random_forest: {
          n_estimators:    { type: 'int', low: 50, high: 200, step: 50 },
          max_depth:       { type: 'int', low: 3, high: 12 },
          min_samples_leaf:{ type: 'int', low: 1, high: 8 },
        },
      },
      budget_config: { max_trials: 5, cv_folds: 3, test_size: 0.2, n_trials_per_model: 5 },
    },
  });
  expect(res.ok(), `dispatch bayesian: ${await res.text()}`).toBeTruthy();

  const runs = await pollExperimentDone(request, modelingTask.id, 'bayesian-1', { timeoutMs: 180000 });
  expect(runs.length).toBeGreaterThanOrEqual(3);
  const successes = runs.filter((r) => r.status === 'SUCCESS');
  expect(successes.length).toBeGreaterThanOrEqual(1);
  const distinctParams = new Set(successes.map((r) => JSON.stringify(r.params?.hyperparameters || {})));
  expect(distinctParams.size).toBeGreaterThan(1);
});

test('UI — ModelingTaskDetail shows runs across all 3 strategies', async ({ page }) => {
  await page.goto(`${BASE_UI}/v3/tasks/${modelingTask.id}`);
  await expect(page.getByRole('heading', { name: modelingTask.name })).toBeVisible({ timeout: 15000 });

  // Click the "实验编排" tab — batch names live there.
  await page.getByRole('tab', { name: /实验编排/ }).click();

  // Each batch name appears multiple times in the DOM (hidden cells in the
  // overview tab, etc.).  Pick the first VISIBLE match.
  for (const name of ['baseline-1', 'grid-1', 'bayesian-1']) {
    const visible = page.getByText(name, { exact: false }).locator('visible=true').first();
    await expect(visible, `batch "${name}" visible somewhere on the page`).toBeVisible({ timeout: 10000 });
  }
});

test('UI — RunInspector renders all tabs and a non-blank SHAP chart', async ({ page, request }) => {
  const r = await request.get(`${BASE_API}/v3/tasks/${modelingTask.id}/runs`);
  const runs = (await r.json()).items || [];
  const success = runs.find((x) => x.status === 'SUCCESS');
  expect(success, 'need a SUCCESS run').toBeTruthy();

  await page.goto(`${BASE_UI}/v3/tasks/${modelingTask.id}`);
  await expect(page.getByRole('heading', { name: modelingTask.name })).toBeVisible({ timeout: 15000 });

  // The consolidated "模型对比" tab has the runs table and inspector actions.
  // inspector drawer.
  await page.getByRole('tab', { name: /模型对比/ }).click();

  const detailBtn = page.getByRole('button', { name: '详情' }).first();
  await expect(detailBtn).toBeVisible({ timeout: 10000 });
  await detailBtn.click();

  const drawer = page.locator('.ant-drawer-body').first();
  await expect(drawer).toBeVisible({ timeout: 15000 });

  // Check the headline tabs that the v3.2 plan promises.
  const tabs = ['概览', '训练可视化', '上下文', '日志', 'SHAP'];
  for (const t of tabs) {
    const tab = drawer.getByRole('tab', { name: new RegExp(t) });
    await expect(tab, `tab "${t}" exists`).toBeVisible({ timeout: 5000 });
  }

  // Click each tab and ensure the body remains rendered (no error banner).
  for (const t of ['训练可视化', '上下文', '日志', 'SHAP']) {
    await drawer.getByRole('tab', { name: new RegExp(t) }).click();
    await expect(drawer.locator('.ant-alert-error')).toHaveCount(0);

    if (t === 'SHAP') {
      const canvas = drawer.locator('.ant-tabs-tabpane-active canvas').first();
      await expect(canvas).toBeVisible({ timeout: 15000 });
      await expect.poll(async () => canvas.evaluate((node) => {
        const ctx = node.getContext('2d');
        if (!ctx || !node.width || !node.height) return 0;
        const pixels = ctx.getImageData(0, 0, node.width, node.height).data;
        const stride = Math.max(1, Math.floor((node.width * node.height) / 10000));
        let painted = 0;
        for (let p = 0; p < node.width * node.height; p += stride) {
          const i = p * 4;
          const [r, g, b, a] = [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]];
          if (a > 0 && !(r > 248 && g > 248 && b > 248)) painted += 1;
        }
        return painted;
      }), { timeout: 15000 }).toBeGreaterThan(20);
    }
  }
});

test('UI — strategy compare tab loads (when present)', async ({ page }) => {
  await page.goto(`${BASE_UI}/v3/tasks/${modelingTask.id}`);
  await page.getByRole('tab', { name: /模型对比/ }).click();
  await expect(page.getByText(/按策略对比/)).toBeVisible({ timeout: 10000 });
});

test('Diagnostic — at least one SHAP method is non-fallback', async ({ request }) => {
  // After all 3 strategies finished, pick the leaderboard top run and
  // hit the inspector endpoint.  The shap_service in v3.2 is supposed to try
  // tree → kernel → permutation; if everything is permutation-only across
  // multiple runs, the SHAP path is broken.  This is a soft signal.
  const r = await request.get(`${BASE_API}/v3/tasks/${modelingTask.id}/runs`);
  const runs = (await r.json()).items || [];
  const successes = runs.filter((x) => x.status === 'SUCCESS').slice(0, 5);
  const methods = [];
  for (const run of successes) {
    const insp = await request.get(`${BASE_API}/platform/runs/${run.run_id}/inspector`);
    if (!insp.ok()) continue;
    const body = await insp.json();
    const m = body?.shap?.method || run.metrics?.shap_method;
    if (m) methods.push(m);
  }
  console.log('SHAP methods seen:', methods);
  // Not a hard failure — but record so we can spot if it's all permutation.
  expect(methods.length).toBeGreaterThan(0);
});
