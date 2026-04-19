// @ts-check
import { defineConfig, devices } from '@playwright/test'

/**
 * Playwright configuration for V3 Modeling Workbench smoke tests.
 *
 * The dev server is expected to already be running at http://localhost:3000
 * with the FastAPI backend at http://localhost:8000 (Vite proxies /api).
 * We don't spawn the server here so the tests can reuse a warm cache.
 */
export default defineConfig({
  testDir: './tests-e2e',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,   // The backend has shared state (DB) — run serially.
  workers: 1,
  retries: 0,
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    locale: 'zh-CN',
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
})
