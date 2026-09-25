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

const CHART_MARKER = /\{\{\s*chart\s*:\s*([a-z0-9_]+)\s*\}\}/gi

/**
 * The Markdown export: overview with its figures noted in place, the appendix
 * tables, then every Run report. A chart marker becomes a one-line note with
 * the figure's title and reading, since a .md file cannot carry the canvas.
 */
export function buildCompleteReportMarkdown(report = null, fallbackMarkdown = '') {
  if (!report) return String(fallbackMarkdown || '').trim() + '\n'

  const charts = Array.isArray(report.charts) ? report.charts : []
  const placed = new Set()
  const parts = [inlineChartNotes(report.markdown || fallbackMarkdown || '', charts, placed)]

  const tables = Array.isArray(report.appendix_tables)
    ? report.appendix_tables
    : (Array.isArray(report.tables) ? report.tables : [])
  if (tables.length) {
    parts.push('# 附录')
    tables.forEach((table) => {
      const rendered = tableToMarkdown(table)
      if (rendered) parts.push(rendered)
    })
  }

  // Figures the prose never placed (legacy archives) still get listed.
  const unplaced = charts.filter((chart) => !placed.has(String(chart.id || '').toLowerCase()))
  if (unplaced.length) {
    parts.push([
      '# 图表索引',
      '',
      ...unplaced.map((chart) => `- **${chart.title || chart.id || '图表'}**：${chartNote(chart)}`),
    ].join('\n'))
  }

  const runReports = Array.isArray(report.run_reports) ? report.run_reports : []
  if (runReports.length) {
    parts.push('# Run 分报告')
    runReports.forEach((run) => {
      const markdown = inlineChartNotes(run.markdown || '', run.charts || [], new Set())
      if (markdown) parts.push(markdown)
    })
  }

  return parts.filter(Boolean).join('\n\n---\n\n') + '\n'
}

function inlineChartNotes(markdown, charts, placed) {
  const byId = Object.fromEntries((charts || []).map((chart) => [String(chart.id || '').toLowerCase(), chart]))
  return String(markdown || '').replace(CHART_MARKER, (_match, id) => {
    const key = String(id).toLowerCase()
    const chart = byId[key]
    placed.add(key)
    return `> 图表：${chart?.title || id}。${chartNote(chart)}`
  }).trim()
}

function chartNote(chart) {
  return chart?.caption || chart?.description || '交互图请在在线归档中查看。'
}

function tableToMarkdown(table) {
  const columns = Array.isArray(table?.columns) ? table.columns : []
  const rows = Array.isArray(table?.rows) ? table.rows : []
  if (!columns.length || !rows.length) return ''
  const header = `| ${columns.map(column => escapeCell(column.title || column.key)).join(' | ')} |`
  const divider = `|${columns.map(() => '---').join('|')}|`
  const body = rows.map(row => (
    `| ${columns.map(column => escapeCell(row?.[column.key])).join(' | ')} |`
  ))
  return [`## ${table.title || table.id || '数据表'}`, '', header, divider, ...body].join('\n')
}

function escapeCell(value) {
  if (value === undefined || value === null || value === '') return '—'
  const text = typeof value === 'object' ? JSON.stringify(value) : String(value)
  return text.replace(/\|/g, '\\|').replace(/\r?\n/g, '<br>')
}

function archiveTime(item) {
  const value = item?.archived_at || item?.generated_at || ''
  const time = Date.parse(value)
  return Number.isFinite(time) ? time : 0
}
