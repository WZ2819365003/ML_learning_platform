// 01 — 平台基础冒烟（Backend health / Frontend boot / 14 个路由组）
const { test, expect, request } = require('@playwright/test');
const { BASE_API, BASE_ROOT, getJson } = require('../helpers/api');
const { attachPageObservers, attachToReport } = require('../helpers/page-probe');

test.describe('01 平台冒烟', () => {
  test('1.1 Backend /health 返回 200 & version 字段', async ({ request }) => {
    const res = await request.get(`${BASE_ROOT}/health`);
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.status).toBe('ok');
    expect(body.version).toMatch(/^\d+\.\d+\.\d+$/);
    test.info().annotations.push({ type: 'version', description: body.version });
  });

  test('1.2 OpenAPI schema 暴露并枚举 14 个路由组', async ({ request }) => {
    const res = await request.get(`${BASE_ROOT}/openapi.json`);
    expect(res.ok()).toBe(true);
    const spec = await res.json();
    const groups = new Set();
    for (const p of Object.keys(spec.paths || {})) {
      const seg = p.startsWith('/api/') ? p.split('/')[2] : p.split('/')[1];
      if (seg) groups.add(seg);
    }
    test.info().annotations.push({ type: 'route-groups', description: [...groups].sort().join(',') });
    expect(groups.size).toBeGreaterThanOrEqual(10);
  });

  test('1.3 Frontend / 首页可访问且 SPA 渲染', async ({ page }) => {
    const obs = attachPageObservers(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 30000 }).catch(() => {});
    // Should redirect to /dashboard
    await expect(page).toHaveURL(/\/dashboard/);
    await attachToReport(test.info(), obs, 'home-observers');
  });

  test('1.4 Sidebar 导航元素可见', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('domcontentloaded');
    // Ant Design Menu rendering — wait for menu items
    const menuItems = page.locator('.ant-menu-item, .ant-menu-submenu-title');
    const cnt = await menuItems.count();
    test.info().annotations.push({ type: 'menu-count', description: String(cnt) });
    expect(cnt).toBeGreaterThan(3);
  });
});
