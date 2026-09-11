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
 * each belongs after, or that a figure does not belong at all. Both the
 * overview and every sub-report render through ReportDocument, so the two
 * halves of the report share one look.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Empty, Space, Tag, Typography } from '../../ui'
import { TrophyOutlined } from '@ant-design/icons'

import ReportDocument from './ReportDocument'

const { Text } = Typography

export const OVERVIEW = '__overview__'

/** The nav's items: 总报告 first, then one per model. */
export function buildTreeItems(runReports = [], bestRunId = null, overviewLabel = '总报告') {
  const modelCounts = runReports.reduce((counts, report) => {
    const model = report.model_type || '未命名模型'
    counts[model] = (counts[model] || 0) + 1
    return counts
  }, {})
  return [
    { id: OVERVIEW, label: overviewLabel },
    ...runReports.map(r => ({
      id: r.run_id,
      label: runReportLabel(r, modelCounts),
      meta: r.validation_scheme || null,
      best: Boolean(bestRunId) && r.run_id === bestRunId,
      failed: Boolean(r.error),
    })),
  ]
}

function runReportLabel(report, modelCounts) {
  const model = report.model_type || '未命名模型'
  if ((modelCounts[model] || 0) <= 1) return model
  if (report.trial_no !== undefined && report.trial_no !== null) {
    return `${model} · Trial ${report.trial_no}`
  }
  const shortId = String(report.run_id || '').slice(0, 8)
  return shortId ? `${model} · ${shortId}` : model
}

export function RunReportBody({ report, appendixOpen = false }) {
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
    <div className="report-body ai-report-run">
      <ReportDocument
        markdown={report.markdown}
        charts={report.charts || []}
        appendixTables={report.appendix_tables || []}
        appendixOpen={appendixOpen}
      />
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
  const modelCount = new Set(runReports.map(report => report.model_type).filter(Boolean)).size

  return (
    <Card size="small" styles={{ body: { padding: 0 } }}
      title={<Space><span>AI 报告</span>
        <Tag color="blue" style={{ margin: 0 }}>
          {runReports.length} 个 Run / {modelCount} 种模型
        </Tag></Space>}>
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
                <span className="report-nav-label">
                  <span>{item.label}</span>
                  {item.meta && <small>{item.meta}</small>}
                </span>
                {item.best && <TrophyOutlined style={{ color: '#ed7b2f' }} />}
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
