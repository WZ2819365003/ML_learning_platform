import { describe, expect, it } from 'vitest'

import { getResultViewEntries } from './resultViewRegistry'

describe('result view registry', () => {
  // ML and DL share one skeleton. Download is not a tab: 模型管理's drawer
  // has the button for both families, so a tab was a second route to it.
  it('uses the shared successful-result tab structure for ml', () => {
    expect(getResultViewEntries({
      family: 'ml', taskType: 'regression', status: 'SUCCESS',
    }).map(entry => entry.key)).toEqual(['logs', 'visualization', 'backtest', 'explain'])
  })

  it('gives dl the same tabs as ml', () => {
    expect(getResultViewEntries({
      family: 'dl', taskType: 'regression', status: 'SUCCESS',
    }).map(entry => entry.key)).toEqual(['logs', 'visualization', 'backtest', 'explain'])
  })

  it('keeps logs as the only tab before a task succeeds', () => {
    expect(getResultViewEntries({
      family: 'dl', taskType: 'classification', status: 'RUNNING',
    }).map(entry => entry.key)).toEqual(['logs'])
  })
})
