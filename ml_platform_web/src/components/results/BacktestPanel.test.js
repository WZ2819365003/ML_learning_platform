import { describe, expect, it } from 'vitest'

import { backtestStats, buildScatterOption } from './BacktestPanel'
import { getResultViewEntries } from './resultViewRegistry'

describe('backtestStats', () => {
  it('computes rmse, mae and mape over the pairs', () => {
    const s = backtestStats([100, 200], [110, 180])
    expect(s.count).toBe(2)
    expect(s.mae).toBeCloseTo(15)
    expect(s.rmse).toBeCloseTo(Math.sqrt((100 + 400) / 2))
    expect(s.mape).toBeCloseTo(((10 / 100) + (20 / 200)) / 2 * 100)
  })

  it('skips zero actuals rather than returning Infinity for mape', () => {
    // A single zero-valued row would otherwise make the whole metric useless.
    const s = backtestStats([0, 100], [5, 110])
    expect(Number.isFinite(s.mape)).toBe(true)
    expect(s.mape).toBeCloseTo(10)
  })

  it('reports null mape when every actual is zero', () => {
    expect(backtestStats([0, 0], [1, 2]).mape).toBeNull()
  })

  it('returns null for empty input instead of dividing by zero', () => {
    expect(backtestStats([], [])).toBeNull()
  })

  it('never reads past the shorter array', () => {
    expect(backtestStats([1, 2, 3], [1]).count).toBe(1)
  })
})

describe('buildScatterOption', () => {
  it('plots pairs against a y=x diagonal spanning the data', () => {
    const o = buildScatterOption([1, 5], [2, 4])
    expect(o.series[0].data).toEqual([[1, 2], [5, 4]])
    expect(o.series[1].data).toEqual([[1, 1], [5, 5]])
  })
})

describe('result view registry ordering', () => {
  it('orders a successful ML regression run: logs, viz, backtest, explain', () => {
    const keys = getResultViewEntries({
      family: 'ml', taskType: 'regression', status: 'SUCCESS',
    }).map(e => e.key)
    expect(keys).toEqual(['logs', 'visualization', 'backtest', 'explain'])
  })

  it('gives DL a download tab that ML does not get', () => {
    const dl = getResultViewEntries({ family: 'dl', taskType: 'regression', status: 'SUCCESS' })
      .map(e => e.key)
    const ml = getResultViewEntries({ family: 'ml', taskType: 'regression', status: 'SUCCESS' })
      .map(e => e.key)
    expect(dl).toContain('download')
    expect(ml).not.toContain('download')
  })

  it('omits backtest for classification', () => {
    // Lining predictions up against the truth is a numeric comparison; a
    // classifier gets its confusion matrix under 训练可视化 instead.
    const keys = getResultViewEntries({
      family: 'ml', taskType: 'classification', status: 'SUCCESS',
    }).map(e => e.key)
    expect(keys).not.toContain('backtest')
  })

  it('shows only logs while a run is still going', () => {
    const keys = getResultViewEntries({
      family: 'dl', taskType: 'regression', status: 'RUNNING',
    }).map(e => e.key)
    expect(keys).toEqual(['logs'])
  })
})
