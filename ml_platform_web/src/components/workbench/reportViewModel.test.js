import { describe, expect, it } from 'vitest'

import { pickLatestArchive, resolveReportSource } from './reportViewModel'

describe('reportViewModel', () => {
  it('prefers a freshly generated AI report over archived and legacy reports', () => {
    const generated = { archive_id: 'new-report', markdown: '# 新 AI 报告' }
    const archived = { archive_id: 'old-report', markdown: '# 旧 AI 报告' }

    expect(resolveReportSource({
      generatedAiReport: generated,
      archivedAiReport: archived,
      legacyMarkdown: '# 旧静态报告',
    })).toEqual({
      kind: 'ai',
      report: generated,
      markdown: '# 新 AI 报告',
      sourceLabel: 'AI 报告',
    })
  })

  it('falls back to the latest archive, then legacy markdown', () => {
    const archived = { archive_id: 'archived-report', markdown: '# 归档 AI 报告' }

    expect(resolveReportSource({
      archivedAiReport: archived,
      legacyMarkdown: '# 旧静态报告',
    })).toMatchObject({
      kind: 'ai',
      report: archived,
      markdown: '# 归档 AI 报告',
    })

    expect(resolveReportSource({
      legacyMarkdown: '# 旧静态报告',
    })).toEqual({
      kind: 'legacy',
      report: null,
      markdown: '# 旧静态报告',
      sourceLabel: '基础报告',
    })
  })

  it('sorts report archives by archive time before opening the latest one', () => {
    expect(pickLatestArchive([
      { id: 'older', archived_at: '2026-07-26T10:00:00Z' },
      { id: 'newer', archived_at: '2026-07-26T11:00:00Z' },
    ])?.id).toBe('newer')
  })
})
