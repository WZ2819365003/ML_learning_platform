import { describe, expect, it } from 'vitest'

import {
  buildResultsUrl,
  normalizeResultFamily,
  redirectLegacyDlSearch,
} from './resultRoutes'

describe('unified result routes', () => {
  it('keeps ML as the compact default route', () => {
    expect(buildResultsUrl()).toBe('/training/results')
    expect(buildResultsUrl({ taskId: 'ml-1' })).toBe('/training/results?taskId=ml-1')
  })

  it('makes the DL renderer explicit on the same route', () => {
    expect(buildResultsUrl({ family: 'dl', taskId: 'dl-1' }))
      .toBe('/training/results?family=dl&taskId=dl-1')
    expect(normalizeResultFamily('DL')).toBe('dl')
  })

  it('preserves old deep links while adding the DL family marker', () => {
    expect(redirectLegacyDlSearch('?taskId=legacy-1'))
      .toBe('/training/results?taskId=legacy-1&family=dl')
  })
})
