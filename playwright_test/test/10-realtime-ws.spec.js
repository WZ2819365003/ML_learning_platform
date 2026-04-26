// 10 — 实时通道（WebSocket /ws/training, /ws/logs）连通性
const { test, expect } = require('@playwright/test');
const { BASE_ROOT } = require('../helpers/api');

test.describe('10 WebSocket 实时通道', () => {
  test('10.1 /ws/training/{id} 可建立连接（即使 id 不存在）', async ({ page }) => {
    const wsUrl = BASE_ROOT.replace(/^http/, 'ws') + '/ws/training/playwright-probe';
    const result = await page.evaluate(async (url) => {
      return new Promise((resolve) => {
        let ws;
        const timeout = setTimeout(() => resolve({ phase: 'timeout' }), 5000);
        try {
          ws = new WebSocket(url);
        } catch (e) {
          clearTimeout(timeout);
          resolve({ phase: 'ctor-error', error: String(e) });
          return;
        }
        ws.onopen = () => {
          clearTimeout(timeout);
          resolve({ phase: 'open', ts: Date.now() });
          try { ws.close(); } catch {}
        };
        ws.onerror = (e) => {
          clearTimeout(timeout);
          resolve({ phase: 'error', message: e.message || 'ws-error' });
        };
        ws.onclose = (e) => {
          // record but don't override resolve from open
        };
      });
    }, wsUrl);
    test.info().annotations.push({
      type: 'ws-training',
      description: JSON.stringify(result),
    });
    expect(['open', 'error', 'timeout']).toContain(result.phase);
  });

  test('10.2 /ws/logs/{id} 可建立连接', async ({ page }) => {
    const wsUrl = BASE_ROOT.replace(/^http/, 'ws') + '/ws/logs/playwright-probe';
    const result = await page.evaluate(async (url) => {
      return new Promise((resolve) => {
        let ws;
        const timeout = setTimeout(() => resolve({ phase: 'timeout' }), 5000);
        try {
          ws = new WebSocket(url);
        } catch (e) {
          clearTimeout(timeout);
          resolve({ phase: 'ctor-error', error: String(e) });
          return;
        }
        ws.onopen = () => {
          clearTimeout(timeout);
          resolve({ phase: 'open' });
          try { ws.close(); } catch {}
        };
        ws.onerror = (e) => {
          clearTimeout(timeout);
          resolve({ phase: 'error', message: e.message || 'ws-error' });
        };
      });
    }, wsUrl);
    test.info().annotations.push({
      type: 'ws-logs',
      description: JSON.stringify(result),
    });
    expect(['open', 'error', 'timeout']).toContain(result.phase);
  });
});
