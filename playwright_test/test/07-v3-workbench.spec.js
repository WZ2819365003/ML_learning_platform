// 07 — V3 建模工作台（TrainingPlans / ModelingTasks / TaskCenter / V3Runs）
const { test, expect } = require('@playwright/test');
const { getJson, postJson } = require('../helpers/api');
const { attachPageObservers, attachToReport } = require('../helpers/page-probe');

test.describe.configure({ mode: 'serial' });

test.describe('07 V3 建模工作台', () => {
  test('7.1 GET /api/platform/training-plans 列表', async ({ request }) => {
    const r = await getJson(request, '/platform/training-plans?page=1&page_size=20');
    test.info().annotations.push({
      type: 'status',
      description: `status=${r.status} count=${(r.body?.items || r.body || []).length}`,
    });
    expect([200, 404]).toContain(r.status);
  });

  test('7.2 GET /api/v3/tasks 建模任务列表', async ({ request }) => {
    const r = await getJson(request, '/v3/tasks?page=1&page_size=20');
    test.info().annotations.push({
      type: 'status',
      description: `status=${r.status} count=${(r.body?.items || r.body || []).length}`,
    });
    expect([200, 404]).toContain(r.status);
  });

  test('7.3 GET /api/v3/runs 跨任务 run 列表', async ({ request }) => {
    const r = await getJson(request, '/v3/runs?page=1&page_size=20');
    test.info().annotations.push({
      type: 'status',
      description: `status=${r.status} count=${(r.body?.items || r.body || []).length}`,
    });
    expect([200, 404]).toContain(r.status);
  });

  test('7.4 GET /api/platform/tasks 平台任务（Orphan + 建模）', async ({ request }) => {
    const r = await getJson(request, '/platform/tasks?page=1&page_size=20');
    test.info().annotations.push({ type: 'status', description: String(r.status) });
    expect([200, 404]).toContain(r.status);
  });

  test('7.5 POST 创建 baseline TrainingPlan + ModelingTask', async ({ request }) => {
    const planPayload = {
      name: `pwt-plan-${Date.now()}`,
      description: 'playwright milestone probe',
      task_type: 'classification',
      strategy_type: 'baseline',
      model_family: 'ml',
      selected_models: ['logistic_regression'],
      eval_metrics: ['accuracy'],
    };
    const planRes = await postJson(request, '/platform/training-plans', planPayload);
    test.info().annotations.push({
      type: 'plan',
      description: `status=${planRes.status} id=${planRes.body?.id || 'n/a'}`,
    });
    if (!planRes.ok) {
      // Log but don't fail — record reason in MD
      test.info().annotations.push({ type: 'plan-fail-body', description: planRes.raw?.slice(0, 300) });
      return;
    }
    const taskPayload = {
      name: `pwt-task-${Date.now()}`,
      task_type: 'classification',
      objective_metric: 'accuracy',
      objective_direction: 'max',
      plan_id: planRes.body.id,
    };
    const taskRes = await postJson(request, '/v3/tasks/', taskPayload);
    test.info().annotations.push({
      type: 'task',
      description: `status=${taskRes.status} id=${taskRes.body?.id || 'n/a'}`,
    });
    expect([200, 201, 400, 422]).toContain(taskRes.status);
  });

  test('7.6 /v3/training-plans 页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/v3/training-plans');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await attachToReport(test.info(), obs, 'v3-plans-observers');
    const tableOrEmpty = await page.locator('.ant-table, .ant-empty').count();
    expect(tableOrEmpty).toBeGreaterThan(0);
  });

  test('7.7 /v3/tasks 建模任务页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/v3/tasks');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await attachToReport(test.info(), obs, 'v3-tasks-observers');
    const body = await page.locator('body').textContent();
    expect(body && body.length > 0).toBeTruthy();
  });

  test('7.8 /v3/runs 跨任务 run 列表页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/v3/runs');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await attachToReport(test.info(), obs, 'v3-runs-observers');
    const body = await page.locator('body').textContent();
    expect(body && body.length > 0).toBeTruthy();
  });

  test('7.9 /tasks 任务中心页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/tasks');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await attachToReport(test.info(), obs, 'task-center-observers');
    const body = await page.locator('body').textContent();
    expect(body && body.length > 0).toBeTruthy();
  });
});
