/**
 * RunReportPanel — the whole AI report, navigated from a side rail.
 *
 * 总报告 sits at the top of the rail with each model nested under it, which is
 * the shape of the document: the sub-reports belong to the overall report
 * rather than sitting beside it. A side rail rather than a top strip because a
 * grid search can produce eight models and a vertical list takes eight names
 * without wrapping, while the body keeps a single uninterrupted column.
 *
 * Charts appear wherever the placement pass put them — the model reads the
 * finished prose and the already-rendered figures and decides which paragraph
 * each belongs after, or that a figure does not belong at all.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Empty, Space, Tag, Typography } from 'antd'
import { TrophyOutlined } from '@ant-design/icons'

import EChart from '../EChart'
import MarkdownReport from './MarkdownReport'

const { Text } = Typography

export const OVERVIEW = '__overview__'

const CHART_MARKER = /\{\{\s*chart\s*:\s*([a-z0-9_]+)\s*\}\}/gi

/**
 * Split a sub-report on its chart markers.
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

/** The nav's items: 总报告 first, then one per model. */
export function buildTreeItems(runReports = [], bestRunId = null, overviewLabel = '总报告') {
  return [
    { id: OVERVIEW, label: overviewLabel },
    ...runReports.map(r => ({
      id: r.run_id,
      label: r.model_type,
      best: Boolean(bestRunId) && r.run_id === bestRunId,
      failed: Boolean(r.error),
    })),
  ]
}

export function RunReportBody({ report }) {
  const chartsById = useMemo(
    () => Object.fromEntries((report?.charts || []).map(c => [c.id, c])),
    [report],
  )
  const segments = useMemo(
    () => splitReportOnCharts(report?.markdown || ''), [report],
  )

  if (report?.error) {
    return (
      <Alert type="warning" showIcon
        message="该模型的分报告生成失败"
        description={<>其余模型和总报告不受影响。失败原因：{report.error}</>} />
    )
  }
  if (!report?.markdown) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="该模型暂无分报告" />
  }

  return (
    <div className="report-body">
      {segments.map((seg, i) => {
        if (seg.kind === 'markdown') {
          return <MarkdownReport key={i} markdown={seg.value} />
        }
        const chart = chartsById[seg.value]
        // A marker whose chart was not built renders as nothing: an empty frame
        // reads as a broken chart rather than as an absent one.
        if (!chart?.option) return null
        return (
          <Card key={i} size="small" variant="outlined" title={chart.title}
            style={{ margin: '14px 0' }} styles={{ body: { padding: 12 } }}>
            <EChart option={chart.option} style={{ height: 300 }} />
            {chart.description && (
              <Text type="secondary" style={{ fontSize: 12 }}>{chart.description}</Text>
            )}
          </Card>
        )
      })}
    </div>
  )
}

export default function RunReportPanel({
  runReports = [], bestRunId = null, overviewLabel = '总报告', overview = null,
}) {
  const [activeId, setActiveId] = useState(OVERVIEW)

  const items = useMemo(
    () => buildTreeItems(runReports, bestRunId, overviewLabel),
    [runReports, bestRunId, overviewLabel],
  )

  useEffect(() => {
    // Keep the selection only while it still exists — a regenerated report can
    // carry a different set of runs.
    setActiveId(prev => (items.some(i => i.id === prev) ? prev : OVERVIEW))
  }, [items])

  const active = runReports.find(r => r.run_id === activeId) || null
  const [root, ...children] = items

  return (
    <Card size="small" styles={{ body: { padding: 0 } }}
      title={<Space><span>AI 报告</span>
        <Tag color="blue" style={{ margin: 0 }}>{runReports.length} 个模型</Tag></Space>}>
      <div className="report-shell">
        <nav className="report-nav">
          <button type="button"
            className={`report-nav-item is-root${activeId === root.id ? ' is-active' : ''}`}
            onClick={() => setActiveId(root.id)}>
            {root.label}
          </button>
          <div className="report-nav-children">
            {children.map(item => (
              <button key={item.id} type="button"
                className={`report-nav-item${activeId === item.id ? ' is-active' : ''}`}
                onClick={() => setActiveId(item.id)}>
                <span className="report-nav-label">{item.label}</span>
                {item.best && <TrophyOutlined style={{ color: '#f59e0b' }} />}
                {item.failed && <Text type="danger" style={{ fontSize: 11 }}>失败</Text>}
              </button>
            ))}
          </div>
        </nav>
        <div className="report-content">
          {activeId === OVERVIEW ? overview : <RunReportBody report={active} />}
        </div>
      </div>
    </Card>
  )
}
