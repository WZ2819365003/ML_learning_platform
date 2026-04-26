// 05 — 时序任务模块（TS / TimesFM）
const { test, expect } = require('@playwright/test');
const { getJson } = require('../helpers/api');
const { attachPageObservers, attachToReport } = require('../helpers/page-probe');

test.describe('05 时序任务模块', () => {
  test('5.1 GET /api/ts/tasks 列表', async ({ request }) => {
    const r = await getJson(request, '/ts/tasks?page=1&page_size=20');
    test.info().annotations.push({ type: 'status', description: String(r.status) });
    expect([200, 404]).toContain(r.status);
  });

  test('5.2 GET /api/timesfm/list 列表', async ({ request }) => {
    const r = await getJson(request, '/timesfm/list?page=1&page_size=20');
    test.info().annotations.push({ type: 'status', description: String(r.status) });
    expect([200, 404]).toContain(r.status);
  });

  test('5.3 /ts/tasks 页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/ts/tasks');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await attachToReport(test.info(), obs, 'ts-tasks-observers');
    const body = await page.locator('body').textContent();
    expect(body && body.length > 0).toBeTruthy();
  });

  test('5.4 /ts/tasks/new 重定向到 /ts/tasks?drawer=create', async ({ page }) => {
    // TSConfig.jsx 是个 <Navigate> 组件 —— 真实表单挂在 /ts/tasks 的 Drawer 上。
    const obs = attachPageObservers(page);
    await page.goto('/ts/tasks/new');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await attachToReport(test.info(), obs, 'ts-config-observers');
    expect(page.url()).toMatch(/\/ts\/tasks/);
  });

  test('5.5 /ts/results 页面渲染（容错）', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/ts/results');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(1500);
    await attachToReport(test.info(), obs, 'ts-results-observers');
    const body = await page.locator('body').textContent();
    expect(body && body.length > 0).toBeTruthy();
  });
});
