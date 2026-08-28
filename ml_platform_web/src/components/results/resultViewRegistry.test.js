import { describe, expect, it } from 'vitest'

import { getResultViewEntries } from './resultViewRegistry'

describe('result view registry', () => {
  it.each(['ml', 'dl'])('uses one successful-result tab structure for %s', (family) => {
    expect(getResultViewEntries({
      family, taskType: 'regression', status: 'SUCCESS',
    }).map(entry => entry.key)).toEqual(['logs', 'visualization', 'explain'])
  })

  it('keeps logs as the only tab before a task succeeds', () => {
    expect(getResultViewEntries({
      family: 'dl', taskType: 'classification', status: 'RUNNING',
    }).map(entry => entry.key)).toEqual(['logs'])
  })
})
