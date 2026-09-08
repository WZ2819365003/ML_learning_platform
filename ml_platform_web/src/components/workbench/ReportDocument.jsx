/**
 * ReportDocument — one renderer for the overall report and every sub-report.
 *
 * The body is markdown with `{{chart:<id>}}` markers on their own line; each
 * marker becomes the figure with that id, drawn through renderReportChart so
 * the look is defined once. The prose has no tables: what used to be a table
 * lives in the figure tooltips, and only the two wide appendix tables remain,
 * folded shut under the body.
 *
 * Safety: markdown reaches the DOM through the token renderer (never as
 * innerHTML) and the tooltip strings are escaped in reportCharts.js, so a
 * dataset named `<img onerror=…>` stays text.
 */
import React, { useMemo } from 'react'
import { Collapse, Table, Typography } from 'antd'
import { CustomChart } from 'echarts/charts'
import { MarkAreaComponent } from 'echarts/components'

import echarts from '../../utils/echarts'
import EChart from '../EChart'
import MarkdownReport from './MarkdownReport'
import { chartHeight, renderReportChart } from './reportCharts'

// Error bars are a custom series and the early-stop shade a markArea; neither
// is in the app-wide bundle, so the report registers them for itself.
echarts.use([CustomChart, MarkAreaComponent])

const { Text } = Typography

const CHART_MARKER = /\{\{\s*chart\s*:\s*([a-z0-9_]+)\s*\}\}/gi

/**
 * Split a report on its chart markers.
 *
 * The placement pass puts markers on their own line between paragraphs, so the
 * text either side is complete markdown; rendering each span separately keeps
 * the chart inline without needing a markdown extension.
 */
export function splitReportOnCharts(markdown = '') {
  const segments = []
  let cursor = 0
  CHART_MARKER.lastIndex = 0
  let match = CHART_MARKER.exec(markdown)
  while (match) {
    const before = markdown.slice(cursor, match.index).trim()
    if (before) segments.push({ kind: 'markdown', value: before })
    segments.push({ kind: 'chart', value: match[1].toLowerCase() })
    cursor = match.index + match[0].length
    match = CHART_MARKER.exec(markdown)
  }
  const tail = markdown.slice(cursor).trim()
  if (tail) segments.push({ kind: 'markdown', value: tail })
  return segments
}

/**
 * Charts the prose never placed. New reports place every figure with a
 * marker; old archives have no markers at all, and their charts would
 * otherwise vanish, so they follow the prose in payload order.
 */
export function unplacedCharts(charts = [], segments = []) {
  const placed = new Set(segments.filter((s) => s.kind === 'chart').map((s) => s.value))
  return (charts || []).filter((chart) => !placed.has(String(chart?.id ?? '').toLowerCase()))
}

/** Drop a leading `# title` line — the cover already shows the title. */
export function stripLeadingTitle(markdown = '') {
  return String(markdown ?? '').replace(/^\s*#\s+[^\n]*\n+/, '')
}

/** One figure: title, the chart, and the reading of it underneath. */
export function ReportChart({ spec }) {
  const option = useMemo(() => renderReportChart(spec), [spec])
  // A marker whose chart cannot be drawn renders as nothing: an empty frame
  // reads as a broken chart rather than as an absent one.
  if (!option) return null
  const caption = spec.caption || spec.description
  return (
    <figure className="ai-report-figure" data-chart-id={spec.id}>
      {spec.title && <div className="ai-report-figure-title">{spec.title}</div>}
      <EChart option={option} style={{ height: chartHeight(spec) }} />
      {caption && <figcaption className="ai-report-figure-caption">{caption}</figcaption>}
    </figure>
  )
}

function cellText(value) {
  if (value === null || value === undefined || value === '') return '—'
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}

function AppendixTable({ table }) {
  const columns = (table.columns || []).map((column) => ({
    title: column.title || column.key,
    dataIndex: column.key,
    key: column.key,
    render: (value) => <span className="ai-report-appendix-cell">{cellText(value)}</span>,
  }))
  const dataSource = (table.rows || []).map((row, index) => ({ key: index, ...row }))
  return (
    <section className="ai-report-appendix-table">
      <h4>{table.title || table.id}</h4>
      {columns.length && dataSource.length ? (
        <Table
          size="small"
          columns={columns}
          dataSource={dataSource}
          pagination={false}
          scroll={{ x: 'max-content' }}
        />
      ) : (
        <Text type="secondary">暂无数据</Text>
      )}
    </section>
  )
}

/** The two wide tables, folded shut by default. */
export function ReportAppendix({ tables = [], open = false }) {
  if (!tables.length) return null
  const label = `附录 · ${tables.map((t) => t.title || t.id).join(' / ')}`
  const items = [{
    key: 'appendix',
    label,
    children: tables.map((table) => <AppendixTable key={table.id || table.title} table={table} />),
  }]
  return (
    <Collapse
      className="ai-report-appendix"
      size="small"
      items={items}
      defaultActiveKey={open ? ['appendix'] : []}
    />
  )
}

export default function ReportDocument({
  markdown = '',
  charts = [],
  appendixTables = [],
  stripTitle = false,
  appendixOpen = false,
  className = '',
}) {
  const chartsById = useMemo(
    () => Object.fromEntries((charts || []).map((chart) => [String(chart.id).toLowerCase(), chart])),
    [charts],
  )
  const body = stripTitle ? stripLeadingTitle(markdown) : String(markdown ?? '')
  const segments = useMemo(() => splitReportOnCharts(body), [body])
  const trailing = useMemo(() => unplacedCharts(charts, segments), [charts, segments])

  return (
    <div className={`ai-report-document-body ${className}`.trim()}>
      {segments.length === 0 && (
        <div className="ai-report-prose"><MarkdownReport markdown="" /></div>
      )}
      {segments.map((segment, index) => {
        if (segment.kind === 'markdown') {
          return (
            <div key={index} className="ai-report-prose">
              <MarkdownReport markdown={segment.value} />
            </div>
          )
        }
        const spec = chartsById[segment.value]
        return spec ? <ReportChart key={index} spec={spec} /> : null
      })}
      {trailing.map((spec, index) => <ReportChart key={`trailing-${spec.id || index}`} spec={spec} />)}
      <ReportAppendix tables={appendixTables || []} open={appendixOpen} />
    </div>
  )
}
