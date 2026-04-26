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

  test('2.3 数据集列元数据通过 columns_info 暴露（非独立路由）', async ({ request }) => {
    // 历史 V1 报告里探过 `/data/{id}/columns` 路由，确认前端不调用、
    // 后端不存在；列元数据直接挂在 /data/list 的每条记录的 columns_info 字段。
    const items = await listDatasets(request);
    if (items.length === 0) test.skip();
    const ds = items[0];
    expect(ds?.columns_info, `dataset.columns_info missing on ${ds?.id}`).toBeTruthy();
    expect(typeof ds.columns_info).toBe('object');
    expect(Object.keys(ds.columns_info).length).toBeGreaterThan(0);
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
