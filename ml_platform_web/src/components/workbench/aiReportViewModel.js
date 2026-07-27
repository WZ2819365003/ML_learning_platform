import { parseMarkdown } from '../../utils/markdown'

export function buildReportViewModel(report = {}, taskName = '建模任务') {
  const markdown = visibleReportMarkdown(report)
  const markdownBlocks = parseMarkdown(markdown)

  return {
    title: extractTitle(markdownBlocks),
    taskName,
    score: extractScore(report, `${markdown}\n${report.markdown || ''}`),
    toc: extractToc(markdownBlocks),
    summary: extractSummary(markdownBlocks),
    metrics: normalizeMetrics(report.headline_metrics || []),
    chartCount: Array.isArray(report.charts) ? report.charts.length : 0,
    tableCount: Array.isArray(report.tables) ? report.tables.length : 0,
    archiveLabel: report.archive_id ? String(report.archive_id).slice(0, 8) : null,
  }
}

function visibleReportMarkdown(report) {
  const blockMarkdown = (report.report_blocks || [])
    .filter((block) => block?.type === 'markdown' && block.markdown)
    .map((block) => block.markdown)
    .join('\n\n')
    .trim()
  return blockMarkdown || report.markdown || ''
}

function extractTitle(blocks) {
  const heading = blocks.find((block) => block.type === 'heading' && block.level === 1)
  return heading?.text || 'AI 建模报告'
}

function extractScore(report, markdown) {
  const metric = (report.headline_metrics || []).find((item) => item.key === 'ai_score')
  if (metric?.value) return String(metric.value)
  const match = String(markdown || '').match(/(?:总分|综合得分)[：:]\s*([0-9]{1,3}\s*\/\s*100)/)
  return match ? match[1].replace(/\s+/g, '') : '—'
}

function extractToc(blocks) {
  return blocks
    .filter((block) => block.type === 'heading' && block.level >= 2 && block.level <= 3)
    .slice(0, 10)
    .map((block) => ({
      level: block.level,
      title: block.text,
    }))
}

function extractSummary(blocks) {
  const paragraph = blocks.find((block) => block.type === 'paragraph')
  return paragraph?.text?.replace(/\*\*/g, '') || ''
}

function normalizeMetrics(items) {
  return items
    .filter((item) => item && item.key !== 'ai_score')
    .slice(0, 4)
    .map((item) => ({
      key: item.key || item.label,
      label: item.label,
      value: item.value,
      detail: item.detail,
      tone: item.tone || 'default',
    }))
}
