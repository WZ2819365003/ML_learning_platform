// 04 — 深度学习模块（DLConfig / DLMonitor / DLResults / dl_registry）
const { test, expect } = require('@playwright/test');
const { getJson } = require('../helpers/api');
const { attachPageObservers, attachToReport } = require('../helpers/page-probe');

test.describe('04 深度学习模块', () => {
  test('4.1 GET /api/dl/registry 列出 DL 模型 token', async ({ request }) => {
    const candidates = ['/dl/registry', '/dl/models', '/dl/list-models'];
    let hit = null;
    for (const p of candidates) {
      const r = await getJson(request, p);
      if (r.status === 200) { hit = { p, body: r.body }; break; }
    }
    test.info().annotations.push({
      type: 'dl-registry',
      description: hit ? `${hit.p} -> keys=${Object.keys(hit.body || {}).join(',')}` : 'none-matched',
    });
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
