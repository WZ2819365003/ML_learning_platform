import { parseMarkdown } from '../../utils/markdown'

/**
 * What the cover shows: the title, one line of facts, and the headline.
 *
 * The payload carries `headline` and `meta`; older archives have neither, so
 * the headline falls back to the first paragraph of the markdown and the meta
 * row is simply omitted rather than invented.
 */
export function buildReportViewModel(report = {}, taskName = '建模任务') {
  const blocks = parseMarkdown(report.markdown || '')
  const meta = report.meta && typeof report.meta === 'object' ? report.meta : null
  return {
    title: extractTitle(blocks, meta, taskName),
    taskName: meta?.task_name || taskName,
    headline: extractHeadline(report, blocks),
    metaItems: buildMetaItems(meta),
    generatedAt: report.generated_at || report.archived_at || null,
    archiveLabel: report.archive_id ? String(report.archive_id).slice(0, 8) : null,
  }
}

function extractTitle(blocks, meta, taskName) {
  const heading = blocks.find((block) => block.type === 'heading' && block.level === 1)
  if (heading?.text) return heading.text
  return `${meta?.task_name || taskName} · 建模报告`
}

function extractHeadline(report, blocks) {
  if (typeof report.headline === 'string' && report.headline.trim()) return report.headline.trim()
  const paragraph = blocks.find((block) => block.type === 'paragraph')
  return paragraph?.text?.replace(/\*\*/g, '') || ''
}

/** The facts of the meta row, in reading order; null when the payload has no meta. */
export function buildMetaItems(meta) {
  if (!meta) return null
  const items = []
  if (meta.dataset_name) items.push(String(meta.dataset_name))
  if (meta.target_column) items.push(`目标列 ${meta.target_column}`)
  const runs = Number(meta.run_count)
  const models = Number(meta.model_count)
  if (Number.isFinite(runs) && runs > 0) {
    items.push(Number.isFinite(models) && models > 0
      ? `${runs} 个 Run / ${models} 种模型`
      : `${runs} 个 Run`)
  }
  return items
}
