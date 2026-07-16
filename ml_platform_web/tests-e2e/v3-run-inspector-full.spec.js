// @ts-check
import { test, expect } from '@playwright/test'

/**
 * Creating a modeling task starts in the unified workflow and advances to
 * model configuration without going through the retired creation modal.
 */
test.describe('V3 modeling task — create & inspect', () => {
  test('create a classification task on the diabetes dataset', async ({ page, request }) => {
    let createdTaskId = null
    try {
      // --- Fixture: find a valid dataset id via the proxied API
      const ds = await request.get('/api/data/list').then(r => r.json())
      const diabetes = Array.isArray(ds)
        ? ds.find(d => /diabetes/i.test(d?.name || ''))
        : (ds?.items || []).find(d => /diabetes/i.test(d?.name || ''))
      expect(diabetes, 'diabetes.csv fixture must exist').toBeTruthy()

      await page.goto('/v3/tasks')

      // Open the unified creation workflow
      await page.getByRole('button', { name: /新建建模任务/ }).first().click()
      await expect(page).toHaveURL(/\/v3\/tasks\/new\/workflow/)
      await expect(page.getByText('新建建模任务', { exact: true })).toBeVisible()

      // Fill name (required)
      const name = `E2E-分类-${Date.now()}`
      await page.getByRole('textbox', { name: '任务名称' }).fill(name)

      // Select dataset
      const dsSelect = page.locator('.ant-form-item').filter({ hasText: /^数据集/ }).locator('.ant-select').first()
      await dsSelect.click()
      await page.locator('.ant-select-item-option').filter({ hasText: /diabetes/i }).first().click()

      // Pick target column: Outcome.  The Select is virtualized, so type via
      // keyboard after focusing the combobox to filter the options list.
      const targetCombo = page.locator('.ant-form-item').filter({ hasText: /^目标列/ }).getByRole('combobox').first()
      await expect(targetCombo).toBeEnabled({ timeout: 10_000 })
      await targetCombo.click()
      await targetCombo.pressSequentially('Outcome', { delay: 30 })
      // Wait for the filtered option and click it
      const outcomeOption = page.locator('.ant-select-item-option').filter({ hasText: 'Outcome' }).first()
      await outcomeOption.waitFor({ state: 'visible', timeout: 5000 })
      await outcomeOption.click()

      const createResponse = page.waitForResponse(response =>
        response.request().method() === 'POST'
        && new URL(response.url()).pathname === '/api/v3/tasks/')
      await page.getByRole('button', { name: /创建并继续/ }).click()
      createdTaskId = (await (await createResponse).json()).id
      await expect(page).toHaveURL(/\/v3\/tasks\/[0-9a-f-]+\/workflow/i, { timeout: 10_000 })
      for (const tabName of ['机器学习', '深度学习', '混合策略', '调参策略']) {
        await expect(page.getByRole('tab', { name: tabName })).toBeVisible()
      }
    } finally {
      if (createdTaskId) {
        await request.delete(`/api/v3/tasks/${createdTaskId}`)
      }
    }
  })
})
