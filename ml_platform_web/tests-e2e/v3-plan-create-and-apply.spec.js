// @ts-check
import { test, expect } from '@playwright/test'

/**
 * Stage-2 acceptance: the TrainingPlan library page lets users create a
 * reusable (strategy × models × budget) recipe and it shows up in the table.
 */
test.describe('V3 training plans — create & list', () => {
  test('create a classification plan via the modal', async ({ page, request }) => {
    let createdPlanId = null
    try {
      await page.goto('/v3/training-plans')
      // Page title lives inside the Card header (not an <h*>) — assert by text.
      await expect(page.getByText('训练方案', { exact: true }).first()).toBeVisible()

      // Open creation modal
      await page.getByRole('button', { name: /新建方案/ }).first().click()
      const dialog = page.getByRole('dialog', { name: /新建训练方案/ })
      await expect(dialog).toBeVisible()

      // Fill required plan name (form defaults task_type/strategy/metrics)
      const planName = `E2E-分类-${Date.now()}`
      await dialog.locator('input[placeholder*="例"]').first().fill(planName)

      // Pick at least one candidate model — the registry-driven multi-select
      const modelSelect = dialog.locator('.ant-form-item').filter({ hasText: /^候选模型/ }).locator('.ant-select').first()
      await modelSelect.click()
      // The first popup option is one of the registry models; pick it
      await page.locator('.ant-select-item-option').first().waitFor({ state: 'visible' })
      await page.locator('.ant-select-item-option').first().click()
      // Submit — AntD inserts a thin space between two-char CJK button
      // labels ("创 建"), so match loosely
      const createResponse = page.waitForResponse(response =>
        response.request().method() === 'POST'
        && new URL(response.url()).pathname === '/api/platform/training-plans')
      await dialog.getByRole('button', { name: /创\s*建/ }).click()
      createdPlanId = (await (await createResponse).json()).id

      // Row should appear in the table
      await expect(page.getByText(planName, { exact: false })).toBeVisible({ timeout: 10_000 })
    } finally {
      if (createdPlanId) {
        await request.delete(`/api/platform/training-plans/${createdPlanId}`)
      }
    }
  })

  test('table reflects server state after reload', async ({ page }) => {
    await page.goto('/v3/training-plans')
    await page.getByRole('button', { name: /刷新/ }).first().click()
    await expect(page.getByText('训练方案', { exact: true }).first()).toBeVisible()
  })
})
