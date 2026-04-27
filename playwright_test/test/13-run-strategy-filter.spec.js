// 13 — UI smoke for the Run 对比 strategy filter + 实验编排 deep link.
// Drives /v3/tasks/<id> in the rebuilt SPA to confirm:
//   (a) the strategy dropdown shows up on the Run 对比 tab
//   (b) clicking "查看 Run →" on an experiment row jumps to Run 对比 and
//       pre-selects that batch's strategy in the dropdown
const { test, expect } = require('@playwright/test');
const { getJson } = require('../helpers/api');

// Drive the SPA through nginx (port 80), not the static :3000 server —
// otherwise /api/* requests miss the proxy and the page renders empty.
test.use({ baseURL: process.env.BASE_NGINX || 'http://127.0.0.1' });

test('13.1 Run 对比 has strategy dropdown + experiments→runs deep link works', async ({ page, request }) => {
  test.setTimeout(60_000);

  // Pick a modeling task that has at least one experiment so 实验编排 row exists.
  const list = await getJson(request, '/v3/tasks?page=1&page_size=20');
  const items = list.body?.items || [];
  const target = items.find(t => (t.experiment_count || 0) > 0) || items[0];
  expect(target, 'no V3 task available — run scripts/seed_v3_demo.py first').toBeTruthy();

  await page.goto(`/v3/tasks/${target.id}`);
  await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
  await page.locator('h2', { hasText: target.name }).waitFor({ timeout: 10000 });

  // Switch to Run 对比 tab and confirm dropdown rendered with at least 2 options
  // (always-present "全部策略" + at least one real strategy).
  await page.locator('.ant-tabs-tab', { hasText: 'Run 对比' }).click();
  await page.waitForTimeout(300);
  const filterLabel = page.locator('text=策略：').first();
  await filterLabel.waitFor({ timeout: 5000 });

  // Open the strategy dropdown and check options.
  const strategySelect = page.locator('.ant-tabs-tabpane-active .ant-select-selector').first();
  await strategySelect.click();
  await page.waitForTimeout(200);
  const options = page.locator('.ant-select-dropdown:visible .ant-select-item-option');
  const optCount = await options.count();
  test.info().annotations.push({ type: 'option-count', description: String(optCount) });
  expect(optCount, 'strategy dropdown should expose ≥ 2 options (全部 + ≥1 real)').toBeGreaterThanOrEqual(2);
  // Close the dropdown to avoid blocking the next click
  await page.keyboard.press('Escape');
  await page.waitForTimeout(150);

  // Now jump back to 实验编排 and click "查看 Run →" on the first row.
  await page.locator('.ant-tabs-tab', { hasText: '实验编排' }).click();
  await page.waitForTimeout(300);
  const firstStrategyTagEl = page.locator('.ant-tabs-tabpane-active .ant-table-row').first()
    .locator('.ant-tag').nth(0);
  const firstStrategyText = (await firstStrategyTagEl.textContent())?.trim() || '';
  test.info().annotations.push({ type: 'first-row-strategy', description: firstStrategyText });

  await page.locator('.ant-tabs-tabpane-active button', { hasText: '查看 Run →' }).first().click();
  await page.waitForTimeout(400);

  // After the deep-link, the active dropdown's selection text should match the
  // experiment row's strategy (e.g. "bayesian_search (2)").
  const selectedText = await page
    .locator('.ant-tabs-tabpane-active .ant-select-selection-item').first().textContent();
  test.info().annotations.push({ type: 'selected-after-deeplink', description: selectedText || '' });
  expect(selectedText || '').toContain(firstStrategyText);
});
