// @ts-check
const { defineConfig, devices } = require('@playwright/test');
const path = require('path');

// Ports are overridable so a locally running docker stack (which also binds
// 8000/3000) doesn't get reused as the system under test — `reuseExistingServer`
// would otherwise silently point the suite at whatever build is already there.
const API_PORT = process.env.E2E_API_PORT || '8000';
const WEB_PORT = process.env.E2E_WEB_PORT || '3000';

const e2eDatabaseUrl = process.env.E2E_DATABASE_URL
  || `sqlite+aiosqlite:///${path.resolve(__dirname, 'ml_platform/storage/e2e-playwright.db')}`;

/**
 * @see https://playwright.dev/docs/test-configuration
 */
module.exports = defineConfig({
  testDir: './tests',
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 1 : undefined,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: 'html',
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: `http://127.0.0.1:${WEB_PORT}`,

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },

    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },

    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },

    /* Test against mobile viewports. */
    // {
    //   name: 'Mobile Chrome',
    //   use: { ...devices['Pixel 5'] },
    // },
    // {
    //   name: 'Mobile Safari',
    //   use: { ...devices['iPhone 12'] },
    // },

    /* Test against branded browsers. */
    // {
    //   name: 'Microsoft Edge',
    //   use: { ...devices['Desktop Edge'], channel: 'msedge' },
    // },
    // {
    //   name: 'Google Chrome',
    //   use: { ...devices['Desktop Chrome'], channel: 'chrome' },
    // },
  ],

  /* Run your local dev server before starting the tests */
  webServer: [
    {
      command: `python -c "import uvicorn; uvicorn.run('app.main:app', host='127.0.0.1', port=${API_PORT}, loop='asyncio', log_level='warning')"`,
      url: `http://127.0.0.1:${API_PORT}/health`,
      cwd: './ml_platform',
      env: {
        ...process.env,
        DATABASE_URL: e2eDatabaseUrl,
        S3_ENABLED: 'false',
      },
      timeout: 120000,
      reuseExistingServer: !process.env.CI,
    },
    {
      command: `npm run dev -- --host 127.0.0.1 --port ${WEB_PORT}`,
      url: `http://127.0.0.1:${WEB_PORT}`,
      cwd: './ml_platform_web',
      env: {
        ...process.env,
        VITE_API_TARGET: `http://127.0.0.1:${API_PORT}`,
      },
      timeout: 120000,
      reuseExistingServer: !process.env.CI,
    },
  ],
});
