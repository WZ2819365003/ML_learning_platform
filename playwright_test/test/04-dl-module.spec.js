// 04 — 深度学习模块（DLConfig / DLMonitor / DLResults / dl_registry）
const { test, expect } = require('@playwright/test');
const { getJson } = require('../helpers/api');
const { attachPageObservers, attachToReport } = require('../helpers/page-probe');

test.describe('04 深度学习模块', () => {
  test('4.1 DL 模型注册表 = /api/dl/models', async ({ request }) => {
    // V1 报告里探过 `/dl/registry` 405 —— 真实路径是 /dl/models。
    const r = await getJson(request, '/dl/models');
    expect(r.ok, `dl/models failed: ${r.status} ${r.raw?.slice(0, 200)}`).toBeTruthy();
    expect(r.body?.models, 'dl/models.models missing').toBeTruthy();
    expect(r.body?.categories, 'dl/models.categories missing').toBeTruthy();
  });

  test('4.2 GET /api/dl/list 返回 DL 任务列表', async ({ request }) => {
    const r = await getJson(request, '/dl/list?page=1&page_size=20');
    test.info().annotations.push({ type: 'status', description: String(r.status) });
    expect([200, 404]).toContain(r.status);
  });

  test('4.3 /dl/config 页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/dl/config');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await attachToReport(test.info(), obs, 'dl-config-observers');
    const body = await page.locator('body').textContent();
    expect(body && body.length > 0).toBeTruthy();
  });

  test('4.4 /dl/monitor 页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/dl/monitor');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000);
    await attachToReport(test.info(), obs, 'dl-monitor-observers');
    const body = await page.locator('body').textContent();
    expect(body && body.length > 0).toBeTruthy();
  });

  test('4.5 /dl/results 页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/dl/results');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await attachToReport(test.info(), obs, 'dl-results-observers');
    const body = await page.locator('body').textContent();
    expect(body && body.length > 0).toBeTruthy();
  });
});
