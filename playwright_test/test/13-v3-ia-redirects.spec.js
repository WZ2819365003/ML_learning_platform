import { test, expect } from '@playwright/test'

// IA cleanup: the "V3 平台" group is dissolved into 建模; legacy list pages
// redirect. Uses the suite's baseURL (BASE_UI env or 127.0.0.1:3000) — no
// hardcoded port.
test('legacy V3 routes redirect into 建模', async ({ page }) => {
  await page.goto('/experiments')
  await expect(page).toHaveURL(/\/v3\/tasks$/)
  await page.goto('/tasks')
  await expect(page).toHaveURL(/\/v3\/runs$/)
})

test('sidebar has no V3 平台 group; 建模 has 运行诊断', async ({ page }) => {
  await page.goto('/v3/tasks')
  const sider = page.locator('.ant-layout-sider')
  // V3 menu title carried a BETA badge ("V3 平台BETA") — use contains match.
  await expect(sider.getByText('V3 平台')).toHaveCount(0)
  await expect(sider.getByText('实验管理')).toHaveCount(0)
  await expect(sider.getByText('任务中心')).toHaveCount(0)
  await expect(sider.getByText('运行诊断')).toBeVisible()
})
