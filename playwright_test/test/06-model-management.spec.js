// 06 — 模型管理 & 部署
const { test, expect } = require('@playwright/test');
const { getJson } = require('../helpers/api');
const { attachPageObservers, attachToReport } = require('../helpers/page-probe');

test.describe('06 模型管理与部署', () => {
  test('6.1 GET /api/models/list', async ({ request }) => {
    const r = await getJson(request, '/models/list?page=1&page_size=20');
    test.info().annotations.push({
      type: 'status',
      description: `status=${r.status} count=${(r.body?.items || r.body?.models || []).length}`,
    });
    expect([200, 404]).toContain(r.status);
  });

  test('6.2 GET /api/deploy/list 部署列表', async ({ request }) => {
    const r = await getJson(request, '/deploy/list?page=1&page_size=20');
    test.info().annotations.push({ type: 'status', description: String(r.status) });
    expect([200, 404]).toContain(r.status);
  });

  test('6.3 /models 模型管理页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/models');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await attachToReport(test.info(), obs, 'models-observers');
    const tableOrEmpty = await page.locator('.ant-table, .ant-empty').count();
    expect(tableOrEmpty).toBeGreaterThan(0);
  });

  test('6.4 /deploy 模型部署页面渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/deploy');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await attachToReport(test.info(), obs, 'deploy-observers');
    const body = await page.locator('body').textContent();
    expect(body && body.length > 0).toBeTruthy();
  });

  test('6.5 GET /api/models/tags 标签库', async ({ request }) => {
    const r = await getJson(request, '/models/tags');
    test.info().annotations.push({
      type: 'tags',
      description: r.status === 200 ? `count=${(r.body || []).length}` : `status=${r.status}`,
    });
    expect([200, 404]).toContain(r.status);
  });
});
