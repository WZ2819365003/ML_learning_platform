// 12 — 通过前端 UI 插入 4 条机器学习训练任务（一次性 seeding 用）。
//
// 与发布门禁不同，这里是"操作型"脚本：用户清空 legacy training_tasks 后用它
// 在 /training/config 表单上模拟人工新建 4 个不同 (数据集 × 模型) 组合的任务，
// 让 /training/monitor 立刻有可观察的演示数据。
//
// 复跑：npx playwright test --config=playwright.config.js test/12-frontend-seed-training.spec.js
// 注意：每次跑都会再插 4 条 —— 想清空就到 MySQL 跑 `DELETE FROM training_tasks;`
const { test, expect } = require('@playwright/test');
const { getJson } = require('../helpers/api');

test.describe.configure({ mode: 'serial' });

// 这条 spec 真正驱动 SPA — 必须走 nginx (port 80)，不能直连 frontend 容器
// (port 3000)。frontend 镜像里只有静态 `serve`，没有 /api 反代，直连会让
// SPA 的 axios 收到 index.html 当 JSON 解析，导致 datasets 始终空。
test.use({ baseURL: process.env.BASE_NGINX || 'http://127.0.0.1' });

// 4 个组合：覆盖二分类 / 多模型，不重复 (dataset, model)。
// modelLabel 必须与 /api/training/models 返回的 display_name 完全一致 —— 中文。
const SCENARIOS = [
  { dataset: 'diabetes.csv',               target: 'Outcome', modelLabel: '逻辑回归',  modelToken: 'logistic_regression' },
  { dataset: 'diabetes.csv',               target: 'Outcome', modelLabel: '随机森林',  modelToken: 'random_forest' },
  { dataset: 'predictive_maintenance.csv', target: 'Target',  modelLabel: 'XGBoost',  modelToken: 'xgboost' },
  { dataset: 'predictive_maintenance.csv', target: 'Target',  modelLabel: 'LightGBM', modelToken: 'lightgbm' },
];

/**
 * 打开一个 antd Select 后再用 keyboard 选项是不可靠的（虚拟列表 + 滚动）。
 * 这里用 visible-text 命中下拉项：先 click selector 触发下拉，再 click 文本节点。
 * - openLocator: 触发下拉的元素（通常是 .ant-select-selector 或它的容器）
 * - optionText:  下拉项的可见文本子串（hasText 默认为子串匹配）
 */
async function pickAntSelect(page, openLocator, optionText) {
  await openLocator.click();
  await page.waitForTimeout(150); // 等待下拉动画
  // antd 下拉渲染在 portal，class 链 .ant-select-dropdown > rc-virtual-list >
  // .ant-select-item.ant-select-item-option。直接锁 .ant-select-item:has-text 最稳。
  const option = page
    .locator('.ant-select-dropdown:visible .ant-select-item-option', { hasText: optionText })
    .first();
  await option.waitFor({ state: 'visible', timeout: 5000 });
  await option.click();
}

test('12.1 在 /training/config 提交 4 个不同 model × dataset 组合', async ({ page, request }) => {
  test.setTimeout(120_000);

  // 自洁：清掉之前其他 spec (08/11/15) 上传的 e2e-predictive-maintenance-*.csv
  // 残留。dataset Select 没有 showSearch，结果按最近上传排序；残留多了之后
  // 真正想点的种子数据集 (diabetes.csv) 落在下拉折叠区点不到。这一步保证
  // spec 12 在脏环境里也能跑过。
  const allDatasets = await getJson(request, '/data/list?page=1&page_size=50');
  for (const d of allDatasets.body?.items || []) {
    if (typeof d.name === 'string' && d.name.startsWith('e2e-predictive-maintenance-')) {
      await request.delete(`/api/data/${d.id}`).catch(() => {});
    }
  }

  // 抓初始 count，不假设是 0（后端可能已经有别的 task）
  const before = await getJson(request, '/training/list?page=1&page_size=50');
  const beforeCount = (before.body?.items || []).length;

  for (const [i, sc] of SCENARIOS.entries()) {
    // /training/config 在 mount 时并行 fetch /api/data/list + /api/training/models。
    // 等两个响应都回来再点 dataset Select，避免下拉显示 "No data"。
    const [dataResp, _] = await Promise.all([
      page.waitForResponse(r => r.url().includes('/api/data/list') && r.status() === 200, { timeout: 15000 }),
      page.waitForResponse(r => r.url().includes('/api/training/models') && r.status() === 200, { timeout: 15000 }),
      page.goto('/training/config'),
    ]);
    expect(dataResp.ok()).toBeTruthy();
    await page.locator('h2:has-text("训练配置")').waitFor({ timeout: 10000 });
    await page.waitForTimeout(200); // React state propagation after fetch

    // 数据集
    await pickAntSelect(
      page,
      page.locator('[data-testid="dataset-select"] .ant-select-selector').first(),
      sc.dataset,
    );

    // 目标列（依赖 dataset 加载，给点反应时间）
    await page.waitForTimeout(400);
    await pickAntSelect(
      page,
      page.locator('[data-testid="target-select"] .ant-select-selector').first(),
      sc.target,
    );

    // 模型选择器（在 .model-selector 内部）
    await pickAntSelect(
      page,
      page.locator('.model-selector .ant-select-selector').first(),
      sc.modelLabel,
    );

    // 提交
    const submit = page.locator('button:has-text("启动训练")').first();
    await submit.waitFor({ state: 'visible', timeout: 5000 });
    // 提交后会 navigate 到 /training/monitor?taskId=<id>，等 URL 变化
    await Promise.all([
      page.waitForURL(/\/training\/monitor/, { timeout: 15000 }),
      submit.click(),
    ]);

    test.info().annotations.push({
      type: `seed-${i + 1}`,
      description: `${sc.dataset} × ${sc.modelLabel} -> ${page.url()}`,
    });
  }

  // 服务端核对：列表 +4
  // 训练是后台启动，POST 已经返回 task 行，列表立刻可见
  const after = await getJson(request, '/training/list?page=1&page_size=50');
  const afterItems = after.body?.items || [];
  expect(afterItems.length, `expected ${beforeCount + 4} tasks, got ${afterItems.length}`).toBeGreaterThanOrEqual(beforeCount + 4);

  // 校验最近 4 条 task 的 model_type 跟我们的脚本一致（按时间倒序）
  const recent = afterItems.slice(0, 4).map(t => t.model_type).sort();
  const expected = SCENARIOS.map(s => s.modelToken).sort();
  expect(recent).toEqual(expected);
});
