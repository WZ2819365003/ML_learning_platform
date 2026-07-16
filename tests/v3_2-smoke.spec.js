// Current V3 release smoke tests. Fixtures are discovered at runtime so the
// suite works against both a clean database and a populated developer stack.

const { test, expect, request } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const WEB_BASE = process.env.BASE_UI || 'http://127.0.0.1:3000';
const API_BASE = process.env.BASE_API || 'http://127.0.0.1:8000';
const DATASET_PATH = path.resolve(__dirname, '..', 'examples', 'data', 'predictive_maintenance.csv');
let fixture;

async function findTaskWithSuccessfulRun(api) {
  const tasksResp = await api.get(`${API_BASE}/api/v3/tasks/?page=1&page_size=100`);
  expect(tasksResp.ok()).toBeTruthy();
  const tasks = (await tasksResp.json()).items || [];
  for (const task of tasks.filter((item) => (item.successful_run_count || 0) > 0)) {
    const runsResp = await api.get(`${API_BASE}/api/v3/tasks/${task.id}/runs?page=1&page_size=100`);
    if (!runsResp.ok()) continue;
    const run = ((await runsResp.json()).items || []).find((item) => item.status === 'SUCCESS');
    if (run) return { task, run };
  }
  return null;
}

async function ensureSuccessfulRun(api) {
  const existing = await findTaskWithSuccessfulRun(api);
  if (existing) return existing;

  const datasetsResponse = await api.get(`${API_BASE}/api/data/list?page=1&page_size=100`);
  expect(datasetsResponse.ok()).toBeTruthy();
  let dataset = ((await datasetsResponse.json()).items || [])
    .find((item) => item.name?.includes('predictive_maintenance'));
  if (!dataset) {
    const uploadResponse = await api.post(`${API_BASE}/api/data/upload`, {
      multipart: { file: fs.createReadStream(DATASET_PATH) },
    });
    const uploadPayload = await uploadResponse.json();
    expect(uploadResponse.ok(), JSON.stringify(uploadPayload)).toBeTruthy();
    dataset = uploadPayload;
  }

  const taskResponse = await api.post(`${API_BASE}/api/v3/tasks/`, { data: {
    name: `v3-smoke-${Date.now()}`,
    task_type: 'classification',
    dataset_id: dataset.id,
    target_column: 'Target',
    objective_metric: 'accuracy',
    objective_direction: 'max',
  } });
  const task = await taskResponse.json();
  expect(taskResponse.ok(), JSON.stringify(task)).toBeTruthy();

  const experimentResponse = await api.post(`${API_BASE}/api/v3/tasks/${task.id}/experiments`, { data: {
    name: 'v3-smoke-baseline',
    strategy_type: 'baseline',
    selected_models: ['random_forest'],
  } });
  expect(experimentResponse.ok(), await experimentResponse.text()).toBeTruthy();

  let successfulRun = null;
  await expect.poll(async () => {
    const runsResponse = await api.get(`${API_BASE}/api/v3/tasks/${task.id}/runs?page=1&page_size=100`);
    if (!runsResponse.ok()) return 'REQUEST_FAILED';
    const runs = (await runsResponse.json()).items || [];
    successfulRun = runs.find((item) => item.status === 'SUCCESS') || null;
    if (successfulRun) return 'SUCCESS';
    if (runs.length && runs.every((item) => ['FAILED', 'CANCELED'].includes(item.status))) return 'FAILED';
    return 'RUNNING';
  }, { timeout: 120_000, intervals: [1000, 2000, 5000] }).toBe('SUCCESS');
  return { task, run: successfulRun };
}

test.describe('V3 current release smoke', () => {
  test.setTimeout(150_000);

  test.beforeAll(async ({ request: api }) => {
    fixture = await ensureSuccessfulRun(api);
  });

  test('health endpoint reports the frontend-compatible 3.3 release', async () => {
    const api = await request.newContext();
    const resp = await api.get(`${API_BASE}/health`);
    expect(resp.ok()).toBeTruthy();
    const body = await resp.json();
    expect(body.version).toBe('3.3.0');
  });

  test('strategy-comparison endpoint returns the documented shape', async () => {
    const api = await request.newContext();

    const resp = await api.get(`${API_BASE}/api/v3/tasks/${fixture.task.id}/strategy-comparison`);
    expect(resp.ok()).toBeTruthy();
    const body = await resp.json();
    expect(body).toHaveProperty('task_id', fixture.task.id);
    expect(body).toHaveProperty('metric_name');
    expect(body).toHaveProperty('objective_direction');
    expect(Array.isArray(body.strategies)).toBeTruthy();
    expect(Array.isArray(body.raw_points)).toBeTruthy();
  });

  test('run inspector payload includes diagnosis state', async () => {
    const api = await request.newContext();

    const resp = await api.get(`${API_BASE}/api/platform/runs/${fixture.run.run_id}/inspector`);
    expect(resp.ok()).toBeTruthy();
    const body = await resp.json();
    expect(body).toHaveProperty('diagnosis');
    if (body.diagnosis) {
      expect(typeof body.diagnosis.narrative).toBe('string');
      expect(body.diagnosis).toHaveProperty('overfit');
      expect(body.diagnosis).toHaveProperty('peer_comparison');
    }
  });

  test('task detail consolidates strategy comparison under 模型对比', async ({ page }) => {
    await page.goto(`${WEB_BASE}/v3/tasks/${fixture.task.id}`);
    const comparisonTab = page.getByRole('tab', { name: '模型对比' });
    await expect(comparisonTab).toBeVisible({ timeout: 15000 });
    await comparisonTab.click();
    const strategySection = page.locator('.ant-card').filter({
      has: page.getByText('按策略对比（基线 / 网格 / 贝叶斯）', { exact: true }),
    }).first();
    await expect(strategySection).toBeVisible({ timeout: 15000 });
    await expect(strategySection.getByText('Baseline', { exact: true }).first()).toBeVisible();
    await expect(strategySection.getByText('Grid Search', { exact: true }).first()).toBeVisible();
    await expect(strategySection.getByText('Bayesian (TPE)', { exact: true }).first()).toBeVisible();
  });

  test('TrainingPlans drawer surfaces a 预估 tag', async ({ page }) => {
    await page.goto(`${WEB_BASE}/v3/training-plans`);
    await page.getByRole('button', { name: /新建方案/ }).first().click();
    await expect(page.getByText(/预估：/).first()).toBeVisible({ timeout: 8000 });
  });
});
