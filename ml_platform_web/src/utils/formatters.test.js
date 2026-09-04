import { describe, expect, it } from 'vitest'

import { formatMetricByKey, parseServerDate, percentageMetricValue } from './formatters'

// The API serializes DB datetime columns without an offset ("2026-08-25
// 12:52:01") while they hold UTC. `new Date()` reads a timezone-less string as
// *local* time, so in UTC+8 a log written seconds ago rendered as "8小时前" and
// a live log panel looked completely stale.
describe('parseServerDate', () => {
  it('reads a timezone-less DB timestamp as UTC, not local time', () => {
    expect(parseServerDate('2026-08-25 12:52:01').toISOString())
      .toBe('2026-08-25T12:52:01.000Z')
  })

  it('handles the ISO-with-T form the API also emits', () => {
    expect(parseServerDate('2026-08-25T12:52:01').toISOString())
      .toBe('2026-08-25T12:52:01.000Z')
  })

  it('leaves an explicit offset alone', () => {
    // WebSocket payloads carry "+00:00" already; re-tagging would double-shift.
    expect(parseServerDate('2026-08-25T13:24:14+00:00').toISOString())
      .toBe('2026-08-25T13:24:14.000Z')
    expect(parseServerDate('2026-08-25T21:24:14+08:00').toISOString())
      .toBe('2026-08-25T13:24:14.000Z')
  })

  it('keeps sub-second precision', () => {
    expect(parseServerDate('2026-08-25T13:24:14.785Z').toISOString())
      .toBe('2026-08-25T13:24:14.785Z')
  })

  it('leaves a date-only string to the spec, which already means UTC', () => {
    expect(parseServerDate('2026-08-25').toISOString())
      .toBe('2026-08-25T00:00:00.000Z')
  })

  it('returns null for empty and unparseable input instead of Invalid Date', () => {
    for (const bad of [null, undefined, '', 'not a date']) {
      expect(parseServerDate(bad), String(bad)).toBeNull()
    }
  })

  it('passes a Date through unchanged', () => {
    const d = new Date('2026-08-25T13:24:14Z')
    expect(parseServerDate(d)).toBe(d)
  })
})

describe('percentage metric formatting', () => {
  it('renders ratio-form MAPE as a percentage', () => {
    expect(percentageMetricValue('val_mape', 0.0118)).toBeCloseTo(1.18)
    expect(formatMetricByKey('val_mape', 0.0118)).toBe('1.18%')
  })

  it('does not scale MAPE that is already expressed as percent', () => {
    expect(formatMetricByKey('mape', 8.4)).toBe('8.40%')
  })

  it('uses the same rule for accuracy-like metrics', () => {
    expect(formatMetricByKey('val_acc', 0.979)).toBe('97.90%')
  })

  it('keeps error metrics in their original unit', () => {
    expect(percentageMetricValue('val_rmse', 0.75)).toBeNull()
    expect(formatMetricByKey('val_rmse', 0.75)).toBe('0.7500')
  })
})
