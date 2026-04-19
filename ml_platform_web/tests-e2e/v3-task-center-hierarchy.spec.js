// @ts-check
import { test, expect } from '@playwright/test'

/**
 * Stage-4 acceptance: the Task Center has a hierarchical "建模任务视图" tab
 * and an "孤立任务" tab whose batch-retry button is disabled until a
 * retriable row is selected.  The orphan banner should explain the scope.
 */
test.describe('V3 task center — hierarchy + orphan banner', () => {
  test('tabs switch, banner + disabled batch retry render correctly', async ({ page }) => {
    await page.goto('/tasks')
    await expect(page.getByRole('heading', { name: /任务中心/ })).toBeVisible()

    // Hierarchy tab exists and is selected by default
    const hierarchyTab = page.getByRole('tab', { name: /建模任务视图/ })
    await expect(hierarchyTab).toBeVisible()

    // Switch to orphan tab
    await page.getByRole('tab', { name: /孤立任务/ }).click()

    // Banner explains scope
    await expect(page.getByText(/未关联到建模任务/)).toBeVisible()

    // Batch retry button is rendered but disabled with no selection
    const retryBtn = page.getByRole('button', { name: /批量重试/ })
    await expect(retryBtn).toBeVisible()
    await expect(retryBtn).toBeDisabled()

    // Switch back — hierarchy tab still clickable and tabpanel still mounts
    await hierarchyTab.click()
    await expect(hierarchyTab).toHaveAttribute('aria-selected', 'true')
  })

  test('global stats cards render without errors', async ({ page }) => {
    await page.goto('/tasks')
    // Each of the four stat labels present
    for (const label of ['运行中', '已排队', '成功', '失败']) {
      await expect(page.getByText(label, { exact: true }).first()).toBeVisible()
    }
  })
})
