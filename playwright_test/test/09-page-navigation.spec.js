// 09 — 全站页面导航 smoke（每个路由都进一遍，记录控制台/网络异常）
const { test, expect } = require('@playwright/test');
const { attachPageObservers, attachToReport } = require('../helpers/page-probe');

const ROUTES = [
  ['Dashboard', '/dashboard'],
  ['Data', '/data'],
  ['TrainingConfig', '/training/config'],
  ['TrainingMonitor', '/training/monitor'],
  ['TrainingResults', '/training/results'],
  ['ModelManagement', '/models'],
  ['ModelDeploy', '/deploy'],
  ['Settings', '/settings'],
  ['DLConfig', '/dl/config'],
  ['DLMonitor', '/dl/monitor'],
  ['DLResults', '/dl/results'],
  ['TSTasks', '/ts/tasks'],
  ['TSConfig', '/ts/tasks/new'],
  ['TSMonitor', '/ts/monitor'],
  ['TSResults', '/ts/results'],
  ['TaskCenter', '/tasks'],
  ['Experiments', '/experiments'],
  ['V3Tasks', '/v3/tasks'],
  ['V3Plans', '/v3/training-plans'],
  ['V3Runs', '/v3/runs'],
];

test.describe('09 全站导航 smoke', () => {
  for (const [name, path] of ROUTES) {
    test(`9.x ${name} (${path})`, async ({ page }) => {
      const obs = attachPageObservers(page);
      const t0 = Date.now();
      await page.goto(path);
      await page.waitForLoadState('domcontentloaded');
      await page.waitForTimeout(1500);
      const elapsed = Date.now() - t0;

      const summary = await attachToReport(test.info(), obs, `${name}-observers`);
      test.info().annotations.push({
        type: 'load',
        description: `t=${elapsed}ms console-err=${summary.summary.consoleErrors} 4xx5xx=${summary.summary.badResponses} pageerror=${summary.summary.pageErrors}`,
      });

      const body = await page.locator('body').textContent();
      // 仅断言"页面有内容"，所有错误都记录为 annotation
      expect(body && body.length > 50).toBeTruthy();
    });
  }
});
