// @ts-check
import { test, expect } from '@playwright/test'

test.describe('V3 cleanup regressions', () => {
  test('new workflow never loads an undefined task', async ({ page }) => {
    const undefinedRequests = []
    page.on('request', request => {
      if (request.url().includes('/api/v3/tasks/undefined')) {
        undefinedRequests.push(request.url())
      }
    })

    await page.goto('/v3/tasks/new/workflow')
    await expect(page.getByText('新建建模任务', { exact: true })).toBeVisible()
    await page.waitForLoadState('networkidle')
    expect(undefinedRequests).toEqual([])
  })

  test('mobile layout leaves usable space for the main content', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto('/v3/tasks')

    const readLayout = () => page.evaluate(() => {
      const sider = document.querySelector('.ant-layout-sider')
      const main = document.querySelector('.ant-layout-content')
      return {
        siderWidth: sider?.getBoundingClientRect().width ?? 0,
        mainWidth: main?.getBoundingClientRect().width ?? 0,
        documentWidth: document.documentElement.scrollWidth,
      }
    })

    await expect.poll(async () => (await readLayout()).siderWidth).toBeLessThanOrEqual(72)
    const layout = await readLayout()

    expect(layout.mainWidth).toBeGreaterThanOrEqual(250)
    expect(layout.documentWidth).toBeLessThanOrEqual(390)
  })
})
