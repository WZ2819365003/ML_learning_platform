import { describe, expect, it } from 'vitest'

import { getResultViewEntries } from './resultViewRegistry'

describe('result view registry', () => {
  // ML and DL share the skeleton; DL adds 模型下载 because a checkpoint is the
  // thing you take away to serve elsewhere and the result view had no way to
  // reach it, while ML models are already downloadable from 模型管理.
  it('uses the shared successful-result tab structure for ml', () => {
    expect(getResultViewEntries({
      family: 'ml', taskType: 'regression', status: 'SUCCESS',
    }).map(entry => entry.key)).toEqual(['logs', 'visualization', 'backtest', 'explain'])
  })

  it('adds a download tab for dl', () => {
    expect(getResultViewEntries({
      family: 'dl', taskType: 'regression', status: 'SUCCESS',
    }).map(entry => entry.key)).toEqual([
      'logs', 'visualization', 'backtest', 'download', 'explain',
    ])
  })

  it('keeps logs as the only tab before a task succeeds', () => {
    expect(getResultViewEntries({
      family: 'dl', taskType: 'classification', status: 'RUNNING',
    }).map(entry => entry.key)).toEqual(['logs'])
  })
})
