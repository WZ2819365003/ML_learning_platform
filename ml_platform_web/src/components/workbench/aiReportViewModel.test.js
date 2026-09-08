import { describe, expect, it } from 'vitest'

import { buildMetaItems, buildReportViewModel } from './aiReportViewModel'
import { legacyReport, overviewReport } from './__fixtures__/reportCharts.fixtures'

describe('buildReportViewModel', () => {
  it('takes the cover from headline and meta of the contract payload', () => {
    const vm = buildReportViewModel(overviewReport, '忽略的任务名')

    expect(vm.title).toBe('测试1-电力负荷预测 · 建模报告')
    expect(vm.taskName).toBe('测试1-电力负荷预测')
    expect(vm.headline).toBe('xgboost 胜出，误差不到平均负荷的 1%；但这批模型是随机切分训练的，分数偏乐观。')
    expect(vm.metaItems).toEqual(['电力负荷预测数据.csv', '目标列 load', '7 个 Run / 5 种模型'])
    expect(vm.generatedAt).toBe('2026-09-04T10:12:00Z')
    expect(vm.archiveLabel).toBe('fb837adc')
  })

  it('has no score, metric cards or figure counts', () => {
    const vm = buildReportViewModel(overviewReport)
    expect(vm.score).toBeUndefined()
    expect(vm.metrics).toBeUndefined()
    expect(vm.chartCount).toBeUndefined()
    expect(vm.tableCount).toBeUndefined()
    expect(vm.summary).toBeUndefined()
  })

  it('falls back to the first paragraph and omits the meta row for an old archive', () => {
    const vm = buildReportViewModel(legacyReport, '旧任务')
    expect(vm.title).toBe('AI 建模报告')
    expect(vm.headline).toBe('总分：60/100。 当前不存在可直接认定的全局最优模型。')
    expect(vm.metaItems).toBeNull()
    expect(vm.archiveLabel).toBe('legacy-a')
  })

  it('builds a title from meta or the task name when the markdown has no h1', () => {
    expect(buildReportViewModel({ markdown: '正文', meta: { task_name: '任务 A' } }).title).toBe('任务 A · 建模报告')
    expect(buildReportViewModel({ markdown: '正文' }, '任务 B').title).toBe('任务 B · 建模报告')
    expect(buildReportViewModel({}).headline).toBe('')
  })
})

describe('buildMetaItems', () => {
  it('skips facts the payload does not have rather than printing undefined', () => {
    expect(buildMetaItems({ dataset_name: 'a.csv' })).toEqual(['a.csv'])
    expect(buildMetaItems({ run_count: 3 })).toEqual(['3 个 Run'])
    expect(buildMetaItems({ run_count: 0, model_count: 0 })).toEqual([])
    expect(buildMetaItems(null)).toBeNull()
  })
})
