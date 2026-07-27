export function pickLatestArchive(items = []) {
  return [...items]
    .filter((item) => item?.id)
    .sort((a, b) => archiveTime(b) - archiveTime(a))[0] || null
}

export function resolveReportSource({
  generatedAiReport = null,
  archivedAiReport = null,
  legacyMarkdown = '',
} = {}) {
  const report = generatedAiReport || archivedAiReport
  if (report) {
    return {
      kind: 'ai',
      report,
      markdown: report.markdown || '',
      sourceLabel: 'AI 报告',
    }
  }
  return {
    kind: 'legacy',
    report: null,
    markdown: legacyMarkdown || '',
    sourceLabel: '基础报告',
  }
}

function archiveTime(item) {
  const value = item?.archived_at || item?.generated_at || ''
  const time = Date.parse(value)
  return Number.isFinite(time) ? time : 0
}
