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

export function buildCompleteReportMarkdown(report = null, fallbackMarkdown = '') {
  if (!report) return String(fallbackMarkdown || '').trim() + '\n'

  const parts = [String(report.markdown || fallbackMarkdown || '').trim()]
  const tables = Array.isArray(report.tables) ? report.tables : []
  if (tables.length) {
    parts.push('# 结构化数据附表')
    tables.forEach((table) => {
      const rendered = tableToMarkdown(table)
      if (rendered) parts.push(rendered)
    })
  }

  const charts = Array.isArray(report.charts) ? report.charts : []
  if (charts.length) {
    parts.push([
      '# 图表索引',
      '',
      ...charts.map((chart) => (
        `- **${chart.title || chart.id || '图表'}**：${chart.description || '交互图请在在线归档中查看。'}`
      )),
    ].join('\n'))
  }

  const runReports = Array.isArray(report.run_reports) ? report.run_reports : []
  if (runReports.length) {
    parts.push('# Run 分报告')
    runReports.forEach((run) => {
      const runCharts = Object.fromEntries((run.charts || []).map(chart => [chart.id, chart]))
      const markdown = String(run.markdown || '').replace(
        /\{\{\s*chart\s*:\s*([a-z0-9_]+)\s*\}\}/gi,
        (_match, id) => {
          const chart = runCharts[String(id).toLowerCase()]
          const title = chart?.title || id
          const description = chart?.description || '请在在线归档中查看交互图。'
          return `> 图表：${title}。${description}`
        },
      ).trim()
      if (markdown) parts.push(markdown)
    })
  }

  return parts.filter(Boolean).join('\n\n---\n\n') + '\n'
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
