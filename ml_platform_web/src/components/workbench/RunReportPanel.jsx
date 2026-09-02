/**
 * RunReportPanel — the whole AI report, as one document with a tree of parts.
 *
 * The nav is a horizontal tree across the top: 总报告 is the root, each model
 * hangs off it. That shape says what the earlier left-rail master/detail did
 * not — the sub-reports belong to the overall report, they are not a separate
 * list beside it — and it leaves the full width to the content, which matters
 * because the content is mostly charts.
 *
 * Sections and figures are fixed by the backend, not by the model: every
 * sub-report is 训练过程 then 训练结果, and each chart renders in its section's
 * grid rather than wherever a second model call decided to wedge a marker.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Empty, Space, Tag, Typography } from 'antd'
import { TrophyOutlined } from '@ant-design/icons'

import EChart from '../EChart'
import MarkdownReport from './MarkdownReport'

const { Text } = Typography

export const OVERVIEW = '__overview__'

/**
 * The nav's items: 总报告 first, then one per model.
 *
 * Exported because the ordering and the "best" marker are the only real logic
 * in this file; the rest is layout.
 */
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

/**
 * One chart spans the row; two or more sit 2-up.
 *
 * A lone half-width tile leaves the section ragged, which is the shape this
 * layout exists to avoid.
 */
export function gridColumns(count) {
  return count === 1 ? '1fr' : 'repeat(2, minmax(0, 1fr))'
}

function ChartGrid({ charts = [] }) {
  if (charts.length === 0) return null
  const columns = gridColumns(charts.length)
  return (
    <div style={{ display: 'grid', gridTemplateColumns: columns, gap: 12, margin: '12px 0' }}>
      {charts.map((chart, i) => (
        <Card key={chart.id || i} size="small" variant="outlined" title={chart.title}
          styles={{ body: { padding: 12 } }}>
          <EChart option={chart.option} style={{ height: 260 }} />
          {chart.description && (
            <Text type="secondary" style={{ fontSize: 12 }}>{chart.description}</Text>
          )}
        </Card>
      ))}
    </div>
  )
}

export function RunReportBody({ report }) {
  if (report?.error) {
    return (
      <Alert type="warning" showIcon
        message="该模型的分报告生成失败"
        description={<>其余模型和总报告不受影响。失败原因：{report.error}</>} />
    )
  }
  const sections = report?.sections || []
  if (sections.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="该模型暂无分报告" />
  }
  return (
    <div className="report-body">
      {sections.map(section => (
        <section key={section.key} style={{ marginBottom: 20 }}>
          <div className="run-report-section-title">{section.title}</div>
          {section.markdown && <MarkdownReport markdown={section.markdown} />}
          <ChartGrid charts={section.charts} />
        </section>
      ))}
    </div>
  )
}

/** 总报告 as the root, every model hanging off it. */
function ReportTree({ items, activeId, onSelect }) {
  const root = items[0]
  const children = items.slice(1)
  return (
    <div className="report-tree">
      <button type="button"
        className={`report-tree-node is-root${activeId === root.id ? ' is-active' : ''}`}
        onClick={() => onSelect(root.id)}>
        {root.label}
      </button>
      {children.length > 0 && (
        <div className="report-tree-children">
          {children.map(item => (
            <button key={item.id} type="button"
              className={`report-tree-node${activeId === item.id ? ' is-active' : ''}`}
              onClick={() => onSelect(item.id)}>
              <Space size={4}>
                {item.best && <TrophyOutlined style={{ color: '#f59e0b' }} />}
                <span>{item.label}</span>
              </Space>
              {item.failed && <Text type="danger" style={{ fontSize: 11 }}> 失败</Text>}
            </button>
          ))}
        </div>
      )}
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

  return (
    <Card size="small" styles={{ body: { padding: 0 } }}
      title={<Space><span>AI 报告</span>
        <Tag color="blue" style={{ margin: 0 }}>{runReports.length} 个模型</Tag></Space>}>
      <div className="report-tree-wrap">
        <ReportTree items={items} activeId={activeId} onSelect={setActiveId} />
      </div>
      <div className="report-tree-content">
        {activeId === OVERVIEW ? overview : <RunReportBody report={active} />}
      </div>
    </Card>
  )
}
