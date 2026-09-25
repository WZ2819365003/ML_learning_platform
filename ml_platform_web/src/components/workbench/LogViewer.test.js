import { describe, expect, it } from 'vitest'

import { formatRelative } from './LogViewer'

describe('formatRelative', () => {
  const now = new Date('2026-09-04T10:00:00Z').getTime()

  it('uses Chinese labels from seconds through days', () => {
    expect(formatRelative('2026-09-04T09:59:59.500Z', now)).toBe('刚刚')
    expect(formatRelative('2026-09-04T09:59:48Z', now)).toBe('12 秒前')
    expect(formatRelative('2026-09-04T09:53:00Z', now)).toBe('7 分钟前')
    expect(formatRelative('2026-09-04T07:00:00Z', now)).toBe('3 小时前')
    expect(formatRelative('2026-09-02T10:00:00Z', now)).toBe('2 天前')
  })

  it('returns an empty label for missing or invalid timestamps', () => {
    expect(formatRelative('', now)).toBe('')
    expect(formatRelative('not-a-date', now)).toBe('')
  })
})
