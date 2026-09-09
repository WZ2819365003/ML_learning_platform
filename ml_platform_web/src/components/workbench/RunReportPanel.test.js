import { describe, expect, it } from 'vitest'

import { OVERVIEW, buildTreeItems } from './RunReportPanel'

describe('buildTreeItems', () => {
  it('puts 总报告 first, as the root the models hang off', () => {
    const items = buildTreeItems([{ run_id: 'a', model_type: 'xgboost' }])
    expect(items[0].id).toBe(OVERVIEW)
    expect(items[0].label).toBe('总报告')
    expect(items[1].label).toBe('xgboost')
  })

  it('keeps the backend ordering of the models', () => {
    // The leaderboard is already ranked; re-sorting here would disagree with
    // the ranking the overall report argues from.
    const items = buildTreeItems([
      { run_id: 'a', model_type: 'first' },
      { run_id: 'b', model_type: 'second' },
    ])
    expect(items.slice(1).map(i => i.label)).toEqual(['first', 'second'])
  })

  it('distinguishes repeated model types by Trial and shows validation metadata', () => {
    const items = buildTreeItems([
      { run_id: 'a', model_type: 'xgboost', trial_no: 1, validation_scheme: '交叉验证' },
      { run_id: 'b', model_type: 'xgboost', trial_no: 2, validation_scheme: '交叉验证' },
    ])
    expect(items.slice(1).map(i => i.label)).toEqual([
      'xgboost · Trial 1', 'xgboost · Trial 2',
    ])
    expect(items[1].meta).toBe('交叉验证')
  })

  it('falls back to a short Run id for repeated legacy reports', () => {
    const items = buildTreeItems([
      { run_id: 'abcdefgh-1', model_type: 'lstm' },
      { run_id: 'ijklmnop-2', model_type: 'lstm' },
    ])
    expect(items.slice(1).map(i => i.label)).toEqual([
      'lstm · abcdefgh', 'lstm · ijklmnop',
    ])
  })

  it('marks exactly the best run', () => {
    const items = buildTreeItems(
      [{ run_id: 'a', model_type: 'x' }, { run_id: 'b', model_type: 'y' }], 'b',
    )
    expect(items.filter(i => i.best).map(i => i.id)).toEqual(['b'])
  })

  it('marks nothing when there is no best run rather than matching undefined', () => {
    // Both run_id and bestRunId can be absent; === would then mark every model.
    const items = buildTreeItems([{ model_type: 'x' }, { model_type: 'y' }], null)
    expect(items.some(i => i.best)).toBe(false)
  })

  it('flags a failed sub-report so the nav shows it before you click', () => {
    const items = buildTreeItems([{ run_id: 'a', model_type: 'x', error: '上游超时' }])
    expect(items[1].failed).toBe(true)
  })

  it('returns just the root when no model reported', () => {
    expect(buildTreeItems()).toEqual([{ id: OVERVIEW, label: '总报告' }])
  })
})
