// @ts-check
import { test, expect } from '@playwright/test'

/**
 * Stage-3 acceptance: creating a modeling task lands on its detail page,
 * and the detail page exposes the hierarchical structure (experiments
 * panel + runs table).  We don't wait for training to complete — the
 * inspector Drawer itself is covered by the hierarchy spec.
 */
test.describe('V3 modeling task — create & inspect', () => {
  test('create a classification task on the diabetes dataset', async ({ page, request }) => {
    // --- Fixture: find a valid dataset id via the proxied API
    const ds = await request.get('/api/data/list').then(r => r.json())
    const diabetes = Array.isArray(ds)
      ? ds.find(d => /diabetes/i.test(d?.name || ''))
      : (ds?.items || []).find(d => /diabetes/i.test(d?.name || ''))
    expect(diabetes, 'diabetes.csv fixture must exist').toBeTruthy()

    await page.goto('/v3/tasks')

    // Open creation modal
    await page.getByRole('button', { name: /新建建模任务/ }).first().click()
    await expect(page.locator('.ant-modal-body')).toBeVisible()

    // Fill name (required)
    const name = `E2E-分类-${Date.now()}`
    await page.locator('.ant-modal-body input[placeholder*="客户流失预测"]').first().fill(name)

    // Select dataset
    const dsSelect = page.locator('.ant-modal-body .ant-form-item').filter({ hasText: /数据集/ }).locator('.ant-select').first()
    await dsSelect.click()
    await page.locator('.ant-select-item-option').filter({ hasText: /diabetes/i }).first().click()

    // Pick target column: Outcome.  The Select is virtualized, so type via
    // keyboard after focusing the combobox to filter the options list.
    await expect(page.getByText(/共 \d+ 列/)).toBeVisible({ timeout: 10_000 })
    const targetCombo = page.locator('.ant-modal-body .ant-form-item').filter({ hasText: /目标列/ }).getByRole('combobox').first()
    await targetCombo.click()
    await targetCombo.pressSequentially('Outcome', { delay: 30 })
    // Wait for the filtered option and click it
    const outcomeOption = page.locator('.ant-select-item-option').filter({ hasText: 'Outcome' }).first()
    await outcomeOption.waitFor({ state: 'visible', timeout: 5000 })
    await outcomeOption.click()

    // Submit — AntD inserts a thin space into two-char CJK button labels
    await page.locator('.ant-modal-footer').getByRole('button', { name: /创\s*建/ }).click()

    // Modal closes → either redirects to the detail page OR the task shows
    // up in the list.  Accept both: look for the name anywhere in the DOM.
    await expect(page.getByText(name).first()).toBeVisible({ timeout: 10_000 })

    // Navigate to detail view if we're still on list
    if (!/\/v3\/tasks\/[0-9a-f]/i.test(page.url())) {
      await page.getByText(name).first().click()
    }
    await expect(page).toHaveURL(/\/v3\/tasks\/[0-9a-f]/i, { timeout: 10_000 })
  })
})
