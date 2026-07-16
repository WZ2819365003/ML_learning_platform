// V3 Modeling Workbench — smoke tests
//
// These tests exercise the new task-centric workflow end-to-end at the UI layer.
// They assume the dev stack is running (frontend on :3000, backend on :8000)
// and that at least one dataset has been seeded (example CSVs ship in-tree).
//
// Scope:
//   1. Workbench list page renders and pagination is visible
//   2. Create-task workflow renders its required fields
//   3. Detail page exposes the three consolidated tabs
//   4. "启动新批次" modal switches strategy and reveals per-model tabs
//
// We deliberately stop short of actually running training — that path is
// covered by backend pytest (tests/v3/test_tuning_service.py).

const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const BASE = process.env.BASE_UI || 'http://127.0.0.1:3000';
const API = process.env.BASE_API || 'http://127.0.0.1:8000';
const DATASET_PATH = path.resolve(__dirname, '..', 'examples', 'data', 'predictive_maintenance.csv');

async function ensureTaskWithSuccessfulRun(request) {
  const tasksResponse = await request.get(`${API}/api/v3/tasks/?page=1&page_size=100`);
  expect(tasksResponse.ok()).toBeTruthy();
  const existing = ((await tasksResponse.json()).items || [])
    .find((task) => (task.successful_run_count || 0) > 0);
  if (existing) return existing;

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
    name: `workbench-inspector-${Date.now()}`,
    task_type: 'classification',
    dataset_id: dataset.id,
    target_column: 'Target',
    objective_metric: 'accuracy',
    objective_direction: 'max',
  } });
  const task = await taskResponse.json();
  expect(taskResponse.ok(), JSON.stringify(task)).toBeTruthy();

  const experimentResponse = await request.post(`${API}/api/v3/tasks/${task.id}/experiments`, { data: {
    name: 'workbench-inspector-baseline',
    strategy_type: 'baseline',
    selected_models: ['random_forest'],
  } });
  expect(experimentResponse.ok(), await experimentResponse.text()).toBeTruthy();

  await expect.poll(async () => {
    const runsResponse = await request.get(`${API}/api/v3/tasks/${task.id}/runs`);
    if (!runsResponse.ok()) return 'REQUEST_FAILED';
    const runs = (await runsResponse.json()).items || [];
    if (runs.some((run) => run.status === 'SUCCESS')) return 'SUCCESS';
    if (runs.length && runs.every((run) => ['FAILED', 'CANCELED'].includes(run.status))) return 'FAILED';
    return 'RUNNING';
  }, { timeout: 120_000, intervals: [1000, 2000, 5000] }).toBe('SUCCESS');
  return task;
}

test.describe('V3 Modeling Workbench', () => {
  test('workbench list page is reachable and has pagination', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);

    // Title visible
    await expect(page.getByText('建模任务工作台')).toBeVisible();
    // V3 badge in title bar
    await expect(page.getByText('V3').first()).toBeVisible();

    // Stats chips (use .first() — labels may also appear inside table rows)
    await expect(page.getByText('总任务')).toBeVisible();
    await expect(page.getByText('运行中').first()).toBeVisible();
    await expect(page.getByText('已完成', { exact: true }).first()).toBeVisible();

    // "New task" button
    await expect(page.getByRole('button', { name: /新建建模任务/ })).toBeVisible();

    // Table header present
    await expect(page.getByRole('columnheader', { name: '任务' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: '优化目标' })).toBeVisible();
  });

  test('create-task workflow validates and shows form', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);

    await page.getByRole('button', { name: /新建建模任务/ }).click();
    await expect(page).toHaveURL(/\/v3\/tasks\/new\/workflow/);
    await expect(page.getByRole('heading', { name: '新建建模任务' })).toBeVisible();
    await expect(page.getByLabel('任务名称')).toBeVisible();
    await expect(page.getByLabel('任务类型')).toBeVisible();
    await expect(page.getByLabel('优化目标')).toBeVisible();
    await expect(page.getByRole('button', { name: /创建并继续/ })).toBeVisible();
  });

  test('sidebar surfaces task list under 建模', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);

    // V3 menu group expanded by default
    const link = page.getByRole('link', { name: '任务列表' }).first();
    await expect(link).toBeVisible();

    // Clicking it takes us to /v3/tasks
    await link.click();
    await expect(page).toHaveURL(/\/v3\/tasks/);
    await expect(page.getByText('建模任务工作台')).toBeVisible();
  });

  test('mobile sidebar overlays content without horizontal overflow', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(`${BASE}/dashboard`);

    await page.locator('.anticon-menu-unfold').click();
    await expect(page.getByRole('link', { name: '任务列表' }).first()).toBeVisible();

    const viewport = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
    }));
    expect(viewport.scrollWidth).toBe(viewport.clientWidth);

    await page.getByRole('link', { name: '模型管理' }).click();
    await expect(page).toHaveURL(/\/models/);
    await expect(page.locator('.anticon-menu-unfold')).toBeVisible();
  });

  test('task detail page renders 3 consolidated tabs when a task exists', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);

    // If there is no task row yet, create one via the API to avoid coupling to seed state
    const taskLinks = page.locator('tbody a[href^="/v3/tasks/"]');
    const hasTask = await taskLinks.count() > 0;

    if (!hasTask) {
      // Create via backend API directly — keeps the test self-sufficient
      const created = await page.request.post(`${API}/api/v3/tasks/`, {
        data: {
          name: `smoke-${Date.now()}`,
          task_type: 'classification',
          objective_metric: 'accuracy',
          objective_direction: 'max',
        },
      });
      expect(created.ok()).toBeTruthy();
      await page.reload();
    }

    const taskLink = page.locator('tbody a[href^="/v3/tasks/"]').first();
    await expect(taskLink).toBeVisible({ timeout: 10000 });
    await taskLink.click();

    // Detail URL
    await expect(page).toHaveURL(/\/v3\/tasks\/[\w-]+/);

    // Comparison and explanation are consolidated into one shared surface.
    await expect(page.getByRole('tab', { name: /任务概览/ })).toBeVisible();
    await expect(page.getByRole('tab', { name: /实验编排/ })).toBeVisible();
    await expect(page.getByRole('tab', { name: /模型对比/ })).toBeVisible();

    // Overview tab content (task info card)
    await expect(page.getByText('任务信息')).toBeVisible();

    // Switch to experiments tab
    await page.getByRole('tab', { name: /实验编排/ }).click();
    await expect(page.getByRole('button', { name: /启动新批次/ }).first()).toBeVisible();

    // Open the batch modal and verify strategy selector
    await page.getByRole('button', { name: /启动新批次/ }).first().click();
    const batchDialog = page.getByRole('dialog').filter({ hasText: '启动新的实验批次' });
    await expect(batchDialog).toBeVisible();
    // Ant Radio.Button hides the underlying <input>; use the visible label
    await expect(batchDialog.locator('label.ant-radio-button-wrapper').filter({ hasText: '基线' })).toBeVisible();
    await expect(batchDialog.locator('label.ant-radio-button-wrapper').filter({ hasText: '网格' })).toBeVisible();
    await expect(batchDialog.locator('label.ant-radio-button-wrapper').filter({ hasText: '贝叶斯' })).toBeVisible();

    // Switching to 网格 should change the info banner content
    await batchDialog.locator('label.ant-radio-button-wrapper').filter({ hasText: '网格' }).click();
    await expect(batchDialog.getByText(/网格搜索/)).toBeVisible();

    // Close modal via keyboard
    await page.keyboard.press('Escape');
    await expect(batchDialog).toBeHidden({ timeout: 5000 });
  });

  test('run inspector drawer opens for a successful baseline run', async ({ page }) => {
    test.setTimeout(150_000);
    const task = await ensureTaskWithSuccessfulRun(page.request);
    await page.goto(`${BASE}/v3/tasks/${task.id}`);
    await page.getByRole('tab', { name: /模型对比/ }).click();

    const detailButton = page.getByRole('button', { name: /详情/ }).first();
    await expect(detailButton).toBeVisible({ timeout: 15_000 });
    await detailButton.click();
    await expect(page.getByText('Run 诊断')).toBeVisible({ timeout: 5000 });
  });
});
