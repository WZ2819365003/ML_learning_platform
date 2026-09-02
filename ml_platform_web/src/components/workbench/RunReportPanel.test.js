import { describe, expect, it } from 'vitest'

import { OVERVIEW, buildTreeItems, gridColumns } from './RunReportPanel'

// The chart-marker splitter these tests used to cover is gone: figures are no
// longer placed by a second model call that emitted {{chart:id}} into the
// prose. The backend now fixes each section's charts, so what is left to get
// wrong here is the nav tree and the grid shape.

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

describe('gridColumns', () => {
  it('gives a lone chart the full row', () => {
    expect(gridColumns(1)).toBe('1fr')
  })

  it('lays two or more out 2-up so the section stays rectangular', () => {
    for (const n of [2, 3, 4]) {
      expect(gridColumns(n), String(n)).toBe('repeat(2, minmax(0, 1fr))')
    }
  })
})
