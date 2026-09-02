/**
 * RunReportPanel — the per-model half of the AI report, master/detail.
 *
 * The overall report answers "can we use this"; a sub-report answers "how did
 * this particular model train and how do I read its score". They are different
 * documents for different moments, which is why they are no longer one.
 *
 * Master/detail rather than tabs or an accordion: a grid search can produce
 * dozens of runs, tabs stop fitting past about six, and an accordion makes you
 * scroll a whole report to reach the next model. Here the model list stays put
 * and switching costs one click.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Empty, Space, Tag, Typography } from 'antd'
import { TrophyOutlined } from '@ant-design/icons'

import EChart from '../EChart'
import MarkdownReport from './MarkdownReport'

const { Text } = Typography

const CHART_MARKER = /\{\{\s*chart\s*:\s*([a-z0-9_]+)\s*\}\}/gi

/**
 * Split a sub-report on its chart markers.
 *
 * The generator places markers on their own line between paragraphs, so the
 * text either side is complete markdown; rendering each span separately keeps
 * the chart inline without a markdown extension.
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

function RunReportBody({ report }) {
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
        // A marker whose chart was not built renders as nothing: an empty
        // frame would read as a broken chart rather than as an absent one.
        if (!chart?.option) return null
        return (
          <Card key={i} size="small" variant="outlined" style={{ margin: '12px 0' }}
            title={chart.title}>
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

export default function RunReportPanel({ runReports = [], bestRunId = null }) {
  const [activeId, setActiveId] = useState(null)

  useEffect(() => {
    if (runReports.length === 0) { setActiveId(null); return }
    setActiveId(prev => (
      runReports.some(r => r.run_id === prev) ? prev : runReports[0].run_id
    ))
  }, [runReports])

  const active = runReports.find(r => r.run_id === activeId) || null

  if (runReports.length === 0) {
    return (
      <Card size="small" title="模型分报告">
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="本次没有生成分报告（需要至少一个训练成功的 Run）" />
      </Card>
    )
  }

  return (
    <Card size="small" title={<Space><span>模型分报告</span>
      <Tag color="blue" style={{ margin: 0 }}>{runReports.length} 个模型</Tag></Space>}
      styles={{ body: { padding: 0 } }}>
      <div style={{ display: 'flex', minHeight: 420 }}>
        <div style={{
          width: 210, flexShrink: 0, borderRight: '1px solid #f0f0f0',
          padding: '8px 0', maxHeight: 620, overflowY: 'auto',
        }}>
          {runReports.map(r => {
            const isActive = r.run_id === activeId
            return (
              <div
                key={r.run_id}
                onClick={() => setActiveId(r.run_id)}
                style={{
                  padding: '8px 14px', cursor: 'pointer',
                  borderLeft: `3px solid ${isActive ? '#2563eb' : 'transparent'}`,
                  background: isActive ? 'rgba(37,99,235,0.06)' : 'transparent',
                }}
              >
                <Space size={6}>
                  {r.run_id === bestRunId && <TrophyOutlined style={{ color: '#f59e0b' }} />}
                  <Text strong={isActive} style={{ fontSize: 13 }}>{r.model_type}</Text>
                </Space>
                {r.error && (
                  <div><Text type="danger" style={{ fontSize: 11 }}>生成失败</Text></div>
                )}
              </div>
            )
          })}
        </div>

        <div style={{ flex: 1, minWidth: 0, padding: 16, maxHeight: 620, overflowY: 'auto' }}>
          {active
            ? <RunReportBody report={active} />
            : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="选择左侧模型查看分报告" />}
        </div>
      </div>
    </Card>
  )
}
