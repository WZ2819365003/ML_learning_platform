// @ts-check
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './test',
  fullyParallel: false,
  retries: 0,
  workers: 1,
  // maxFailures=0 means Playwright keeps executing the remaining probes after
  // a failure. Strict gate specs still fail the run normally.
  maxFailures: 0,
  reporter: [
    ['list'],
    ['json', { outputFile: 'artifacts/results.json' }],
    ['html', { outputFolder: 'artifacts/html', open: 'never' }],
  ],
  outputDir: 'artifacts/test-results',
  timeout: 60_000,
  use: {
    baseURL: process.env.BASE_UI || 'http://127.0.0.1:3000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    extraHTTPHeaders: {
      'Accept': 'application/json',
    },
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
});
