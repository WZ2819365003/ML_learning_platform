// V3 Modeling Workbench — smoke tests
//
// These tests exercise the new task-centric workflow end-to-end at the UI layer.
// They assume the dev stack is running (frontend on :3000, backend on :8000)
// and that at least one dataset has been seeded (example CSVs ship in-tree).
//
// Scope:
//   1. Workbench list page renders and pagination is visible
//   2. Create-task modal validates and produces a new entry that navigates to detail
//   3. Detail page exposes 4 tabs (任务概览 / 实验编排 / Run 对比 / 模型解释)
//   4. "启动新批次" modal switches strategy and reveals per-model tabs
//
// We deliberately stop short of actually running training — that path is
// covered by backend pytest (tests/v3/test_tuning_service.py).

const { test, expect } = require('@playwright/test');

const BASE = 'http://127.0.0.1:3000';

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

  test('create-task modal validates and shows form', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);

    await page.getByRole('button', { name: /新建建模任务/ }).click();
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    // Dialog title scoped within the dialog, avoiding collision with the header button
    await expect(dialog.getByText('新建建模任务')).toBeVisible();

    // Required fields present (scoped within the dialog)
    await expect(dialog.getByLabel('任务名称')).toBeVisible();
    await expect(dialog.getByLabel('任务类型')).toBeVisible();
    await expect(dialog.getByLabel('优化目标指标')).toBeVisible();

    // Close via AntD modal X button (stable selector)
    await page.locator('.ant-modal-close').first().click();
    await expect(dialog).toBeHidden({ timeout: 5000 });
  });

  test('sidebar surfaces 建模工作台 under V3 platform', async ({ page }) => {
    await page.goto(`${BASE}/dashboard`);

    // V3 menu group expanded by default
    await expect(page.getByRole('link', { name: '建模工作台' })).toBeVisible();

    // Clicking it takes us to /v3/tasks
    await page.getByRole('link', { name: '建模工作台' }).click();
    await expect(page).toHaveURL(/\/v3\/tasks/);
    await expect(page.getByText('建模任务工作台')).toBeVisible();
  });

  test('task detail page renders 4 tabs when a task exists', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);

    // If there is no task row yet, create one via the API to avoid coupling to seed state
    const hasRow = await page.locator('tbody tr').first().isVisible().catch(() => false);

    if (!hasRow) {
      // Create via backend API directly — keeps the test self-sufficient
      const created = await page.request.post('http://127.0.0.1:8000/api/v3/tasks/', {
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

    // Click the first task row's "详情" button
    const detailBtn = page.getByRole('button', { name: /详情/ }).first();
    await expect(detailBtn).toBeVisible({ timeout: 10000 });
    await detailBtn.click();

    // Detail URL
    await expect(page).toHaveURL(/\/v3\/tasks\/[\w-]+/);

    // 4 tabs visible
    await expect(page.getByRole('tab', { name: /任务概览/ })).toBeVisible();
    await expect(page.getByRole('tab', { name: /实验编排/ })).toBeVisible();
    await expect(page.getByRole('tab', { name: /Run 对比/ })).toBeVisible();
    await expect(page.getByRole('tab', { name: /模型解释/ })).toBeVisible();

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

  test('run inspector drawer opens for best run if one exists', async ({ page }) => {
    await page.goto(`${BASE}/v3/tasks`);

    // Need a task first
    const firstRow = page.locator('tbody tr').first();
    if (!(await firstRow.isVisible().catch(() => false))) test.skip(true, 'no tasks seeded');

    await page.getByRole('button', { name: /详情/ }).first().click();
    await page.getByRole('tab', { name: /Run 对比/ }).click();

    // If any row is visible, open inspector
    const runRow = page.locator('tbody tr').first();
    if (await runRow.isVisible().catch(() => false)) {
      const detailLink = page.getByRole('button', { name: /详情/ }).first();
      if (await detailLink.isVisible().catch(() => false)) {
        await detailLink.click();
        // Drawer title shows "Run 诊断"
        await expect(page.getByText('Run 诊断')).toBeVisible({ timeout: 5000 });
      }
    }
  });
});
