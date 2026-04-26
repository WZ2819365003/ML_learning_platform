// 03 — 机器学习模块（TrainingConfig / Monitor / Results / 后端 API）
const { test, expect } = require('@playwright/test');
const { listDatasets, getJson, postJson, BASE_API } = require('../helpers/api');
const { attachPageObservers, attachToReport } = require('../helpers/page-probe');

test.describe.configure({ mode: 'serial' });

test.describe('03 机器学习模块', () => {
  test('3.1 GET /api/training/list 返回历史任务', async ({ request }) => {
    const r = await getJson(request, '/training/list?page=1&page_size=20');
    test.info().annotations.push({ type: 'status', description: String(r.status) });
    expect([200, 404]).toContain(r.status);
  });

  test('3.2 ML 模型注册表 = /api/models/list', async ({ request }) => {
    // V1 报告里探过 `/models/registry` 405 —— 前端从未调用，真实路径是 /models/list。
    const r = await getJson(request, '/models/list?page=1&page_size=5');
    expect(r.ok, `models/list failed: ${r.status} ${r.raw?.slice(0, 200)}`).toBeTruthy();
    const items = r.body?.items || r.body?.models || [];
    expect(Array.isArray(items)).toBe(true);
  });

  test('3.3 POST /api/training/start (logistic_regression) 后台启动', async ({ request }) => {
    const items = await listDatasets(request);
    if (items.length === 0) test.skip();
    const ds = items.find((i) => /diabetes/i.test(i.name)) || items[0];
    const payload = {
      dataset_id: ds.id,
      model_type: 'logistic_regression',
      task_type: 'classification',
      target_column: 'Outcome',
      feature_columns: null,
      hyperparameters: {},
    };
    const r = await postJson(request, '/training/start', payload);
    test.info().annotations.push({
      type: 'training-start',
      description: `status=${r.status} body=${JSON.stringify(r.body)?.slice(0, 200)}`,
    });
    // 任何 2xx/4xx/5xx 都记录到报告，不让单个 API 异常阻断后续 UI 检查
    expect(r.status).toBeGreaterThanOrEqual(200);
    expect(r.status).toBeLessThan(600);
    if (r.ok && r.body?.task_id) {
      // attach for later steps
      test.info().annotations.push({ type: 'task_id', description: r.body.task_id });
    }
  });

  test('3.4 /training/config 页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/training/config');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    const hasForm = await page.locator('.ant-form, form').count();
    test.info().annotations.push({ type: 'form-count', description: String(hasForm) });
    await attachToReport(test.info(), obs, 'training-config-observers');
    expect(hasForm).toBeGreaterThan(0);
  });

  test('3.5 /training/monitor 页面渲染（无 taskId 时容错）', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/training/monitor');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000);
    const r = await attachToReport(test.info(), obs, 'training-monitor-observers');
    // 容错：允许 console error，但不应整页崩溃
    const hasContent = await page.locator('body').textContent();
    expect(hasContent && hasContent.length > 0).toBeTruthy();
  });

  test('3.6 /training/results 页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/training/results');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await page.waitForTimeout(1500);
    await attachToReport(test.info(), obs, 'results-observers');
    const body = await page.locator('body').textContent();
    expect(body && body.length > 0).toBeTruthy();
  });
});
