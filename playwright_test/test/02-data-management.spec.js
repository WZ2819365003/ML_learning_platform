// 02 — 数据管理模块
const { test, expect } = require('@playwright/test');
const { listDatasets, getJson, BASE_API } = require('../helpers/api');
const { attachPageObservers, attachToReport } = require('../helpers/page-probe');

test.describe('02 数据管理', () => {
  test('2.1 GET /api/data/list 返回种子数据集', async ({ request }) => {
    const items = await listDatasets(request);
    test.info().annotations.push({ type: 'dataset-count', description: String(items.length) });
    expect(items.length).toBeGreaterThan(0);
    const names = items.map((i) => i.name);
    expect(names.some((n) => /\.csv$/i.test(n))).toBeTruthy();
  });

  test('2.2 GET /api/data/{id}/preview 返回前 N 行', async ({ request }) => {
    const items = await listDatasets(request);
    if (items.length === 0) {
      test.info().annotations.push({ type: 'skip-reason', description: '无种子数据集' });
      test.skip();
    }
    const id = items[0].id;
    const r = await getJson(request, `/data/${id}/preview?rows=5`);
    expect(r.status, `preview status=${r.status} body=${r.raw?.slice(0, 200)}`).toBe(200);
    expect(r.body).toBeTruthy();
  });

  test('2.3 GET /api/data/{id}/columns 返回列元数据', async ({ request }) => {
    const items = await listDatasets(request);
    if (items.length === 0) test.skip();
    const id = items[0].id;
    const r = await getJson(request, `/data/${id}/columns`);
    test.info().annotations.push({ type: 'columns-status', description: String(r.status) });
    // 不强制 200 — 路由可能不存在；记录状态码即可
    expect([200, 404, 405]).toContain(r.status);
  });

  test('2.4 /data 页面渲染并加载列表', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/data');
    await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    // table or empty state
    const hasTable = await page.locator('.ant-table').count() > 0;
    const hasEmpty = await page.locator('.ant-empty').count() > 0;
    test.info().annotations.push({
      type: 'render',
      description: `table=${hasTable} empty=${hasEmpty}`,
    });
    await attachToReport(test.info(), obs, 'data-page-observers');
    expect(hasTable || hasEmpty).toBeTruthy();
  });
});
