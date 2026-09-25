// V3 Visual Audit — capture screenshots of every surface so we can eyeball
// layout regressions. The suite seeds its own completed task so detail coverage
// never depends on a developer's local database or a hard-coded UUID.

const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');
const { WEB_BASE: BASE, API_ROOT: API } = require('./helpers/e2e-env');

const DATASET_PATH = path.resolve(__dirname, '..', 'examples', 'data', 'predictive_maintenance.csv');
const OUT = 'screenshots/v3-audit';
let taskId;

async function waitForSuccessfulRun(request) {
  await expect.poll(async () => {
    const response = await request.get(`${API}/api/v3/tasks/${taskId}/runs`);
    if (!response.ok()) return `HTTP_${response.status()}`;
    const runs = (await response.json()).items || [];
    if (!runs.length) return 'PENDING';
    if (runs.some((run) => run.status === 'SUCCESS')) return 'SUCCESS';
    if (runs.every((run) => ['FAILED', 'CANCELED'].includes(run.status))) return 'FAILED';
    return 'RUNNING';
  }, { timeout: 120_000, intervals: [1000, 2000, 5000] }).toBe('SUCCESS');
}

test.describe('V3 Visual Audit', () => {
  test.describe.configure({ mode: 'serial' });
  test.use({ viewport: { width: 1440, height: 900 } });
  test.setTimeout(150_000);

  test.beforeAll(async ({ request }) => {
    const datasetsResponse = await request.get(`${API}/api/data/list?page=1&page_size=100`);
    expect(datasetsResponse.ok()).toBeTruthy();
    let dataset = ((await datasetsResponse.json()).items || [])
      .find((item) => item.name?.includes('predictive_maintenance'));

    if (!dataset) {
      const uploadResponse = await request.post(`${API}/api/data/upload`, {
        multipart: { file: fs.createReadStream(DATASET_PATH) },
      });
      const uploadPayload = await uploadResponse.json();
      expect(uploadResponse.ok(), JSON.stringify(uploadPayload)).toBeTruthy();
      dataset = uploadPayload;
    }

    const taskResponse = await request.post(`${API}/api/v3/tasks/`, { data: {
      name: `visual-audit-${Date.now()}`,
      task_type: 'classification',
      dataset_id: dataset.id,
      target_column: 'Target',
      objective_metric: 'accuracy',
      objective_direction: 'max',
    } });
    const taskPayload = await taskResponse.json();
    expect(taskResponse.ok(), JSON.stringify(taskPayload)).toBeTruthy();
    taskId = taskPayload.id;

    const experimentResponse = await request.post(`${API}/api/v3/tasks/${taskId}/experiments`, { data: {
      name: 'visual-audit-baseline',
      strategy_type: 'baseline',
      selected_models: ['random_forest'],
    } });
    expect(experimentResponse.ok(), await experimentResponse.text()).toBeTruthy();
    await waitForSuccessfulRun(request);
  });

  test('workbench list', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);
    await expect(page.getByText('建模任务工作台')).toBeVisible();
    await page.waitForLoadState('networkidle');
    const workflowButton = page.getByRole('button', { name: '工作流' }).first();
    await expect(workflowButton).toBeVisible();
    const workflowStyles = await workflowButton.evaluate((node) => ({
      background: getComputedStyle(node).backgroundColor,
      color: getComputedStyle(node).color,
    }));
    expect(workflowStyles.background).toBe('rgba(0, 0, 0, 0)');
    expect(['rgb(37, 99, 235)', 'rgb(59, 130, 246)']).toContain(workflowStyles.color);
    await page.screenshot({ path: `${OUT}/01-list.png`, fullPage: true });
  });

  test('create task workflow', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);
    await page.getByRole('button', { name: /新建建模任务/ }).click();
    await expect(page).toHaveURL(/\/v3\/tasks\/new\/workflow/);
    await expect(page.getByRole('heading', { name: '新建建模任务' })).toBeVisible();
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${OUT}/02-create-modal.png`, fullPage: true });
  });

  test('task detail — overview', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks/${taskId}`);
    await expect(page.getByText('任务信息')).toBeVisible({ timeout: 10000 });
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${OUT}/03-detail-overview.png`, fullPage: true });
  });

  test('task detail — experiments', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks/${taskId}`);
    await page.getByRole('tab', { name: /实验编排/ }).click();
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${OUT}/04-detail-experiments.png`, fullPage: true });
  });

  test('task detail — runs leaderboard', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks/${taskId}`);
    await page.getByRole('tab', { name: /模型对比/ }).click();
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${OUT}/05-detail-runs.png`, fullPage: true });
  });

  test('task detail — consolidated comparison', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks/${taskId}`);
    await page.getByRole('tab', { name: /模型对比/ }).click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${OUT}/06-detail-explain.png`, fullPage: true });
  });

  test('batch modal — bayesian', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks/${taskId}`);
    await page.getByRole('tab', { name: /实验编排/ }).click();
    await page.getByRole('button', { name: /启动新批次/ }).first().click();
    const dialog = page.getByRole('dialog').filter({ hasText: '启动新的实验批次' });
    await expect(dialog).toBeVisible();
    await dialog.locator('label.ant-radio-button-wrapper').filter({ hasText: '贝叶斯' }).click();
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${OUT}/07-batch-bayesian.png`, fullPage: true });
  });

  test('run inspector drawer', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks/${taskId}`);
    await page.getByRole('tab', { name: /模型对比/ }).click();
    await page.waitForLoadState('networkidle');
    const detailBtn = page.getByRole('button', { name: /详情/ }).first();
    await expect(detailBtn).toBeVisible();
    await detailBtn.click();
    await expect(page.getByText('Run 诊断')).toBeVisible({ timeout: 5000 });
    const metricLabel = page.getByText('selection_cv_mean_accuracy', { exact: true });
    await expect(metricLabel).toBeVisible();
    expect(await metricLabel.evaluate((node) => getComputedStyle(node).textOverflow)).toBe('ellipsis');
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${OUT}/08-run-inspector.png`, fullPage: true });
  });
});
