// @ts-check
import { test, expect } from '@playwright/test'

/**
 * The legacy task-center deep link redirects to Run diagnostics, where the
 * orphan-task panel remains available as a secondary tab.
 */
test.describe('V3 task center — hierarchy + orphan banner', () => {
  test('tabs switch, banner + disabled batch retry render correctly', async ({ page }) => {
    await page.goto('/tasks')
    await expect(page).toHaveURL(/\/v3\/runs/)
    await expect(page.getByRole('heading', { name: /运行诊断/ })).toBeVisible()

    const runsTab = page.getByRole('tab', { name: /全部 Run/ })
    await expect(runsTab).toHaveAttribute('aria-selected', 'true')

    // Switch to orphan tab
    await page.getByRole('tab', { name: /孤立任务/ }).click()

    // Banner explains scope
    await expect(page.getByText(/未关联到建模任务/)).toBeVisible()

    // Batch retry button is rendered but disabled with no selection
    const retryBtn = page.getByRole('button', { name: /批量重试/ })
    await expect(retryBtn).toBeVisible()
    await expect(retryBtn).toBeDisabled()

    // Switch back — the run table and filters remain mounted
    await runsTab.click()
    await expect(runsTab).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByPlaceholder(/搜索任务名/)).toBeVisible()
  })

  test('global stats cards render without errors', async ({ page }) => {
    await page.goto('/tasks')
    // Summary chips are derived from the current Run result set.
    for (const label of ['运行中', '成功', '失败']) {
      await expect(page.getByText(new RegExp(`^${label} \\d+$`)).first()).toBeVisible()
    }
  })
})
