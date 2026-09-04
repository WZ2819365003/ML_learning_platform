import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert, Button, Modal, Space, Spin, Table,
  Tag, Typography, message,
} from 'antd'
import {
  BulbOutlined,
  DownloadOutlined,
  FileTextOutlined,
  LineChartOutlined,
  ReloadOutlined,
  TableOutlined,
  TrophyOutlined,
} from '@ant-design/icons'

import { modelingTaskApi } from '../../services/api'
import EChart from '../EChart'
import MarkdownReport from './MarkdownReport'
import { buildReportViewModel } from './aiReportViewModel'
import { formatDateTime } from '../../utils/formatters'

const { Paragraph, Text } = Typography

function extractError(err) {
  let body = err?.response?.data
  if (typeof body === 'string') {
    try { body = JSON.parse(body) } catch { /* use string as-is */ }
  }
  return (typeof body === 'string' ? body : body?.detail) || err?.message || 'AI 报告生成失败'
}

const toneColor = {
  success: '#059669',
  warning: '#d97706',
  processing: '#2563eb',
  error: '#dc2626',
  default: '#0f172a',
}

const wrappingColumns = new Set([
  'column',
  'note',
  'training_setup',
  'key_params',
  'process_summary',
  'effect_summary',
  'effect_note',
  'selected_models',
  'search_profile',
  'validation_setting',
  'best_selection_metric',
  'best_final_metric',
  'selection_metric',
  'stability_risk',
  'meaning',
  'evidence',
  'action',
  'expected_benefit',
])

const columnWidths = {
  column: 260,
  rank: 70,
  model_type: 150,
  strategy_type: 150,
  trial_no: 80,
  note: 260,
  training_setup: 240,
  key_params: 260,
  process_summary: 280,
  effect_summary: 280,
  effect_note: 280,
  selected_models: 220,
  search_profile: 220,
  validation_setting: 240,
  best_selection_metric: 180,
  best_final_metric: 180,
  selection_metric: 260,
  test_accuracy: 140,
  test_f1: 120,
  test_roc_auc: 150,
  test_rmse: 140,
  test_mae: 140,
  stability_risk: 300,
  meaning: 340,
  evidence: 300,
  action: 340,
  expected_benefit: 300,
  run_id: 120,
}

function ChartBlock({ chart, caption }) {
  if (!chart) return null
  const option = withReportChartDefaults(chart)
  return (
    <section className="ai-report-figure ai-report-chart">
      <div className="ai-report-figure-head">
        <div>
          <div className="ai-report-eyebrow"><LineChartOutlined /> ECharts 图</div>
          <h3>{chart.title}</h3>
        </div>
      </div>
      {caption && <Paragraph className="ai-report-caption">{caption}</Paragraph>}
      {chart.description && <Text className="ai-report-support">{chart.description}</Text>}
      <div className="ai-report-chart-canvas">
        <EChart option={option} style={{ height: chart.height || 300 }} />
      </div>
    </section>
  )
}

export function withReportChartDefaults(chart) {
  const option = chart.option || {}
  const grid = option.grid && !Array.isArray(option.grid) ? option.grid : {}
  const next = {
    ...option,
    grid: {
      ...grid,
      containLabel: true,
      right: maxChartSpace(grid.right, 48),
    },
  }
  if (chart.id === 'training_curves') {
    next.grid = {
      ...next.grid,
      right: maxChartSpace(grid.right, 64),
      bottom: maxChartSpace(grid.bottom, 108),
    }
    next.xAxis = mapAxis(option.xAxis, (axis) => ({
      ...axis,
      nameLocation: 'middle',
      nameGap: maxChartSpace(axis?.nameGap, 64),
      axisLabel: {
        ...(axis?.axisLabel || {}),
        hideOverlap: true,
        margin: maxChartSpace(axis?.axisLabel?.margin, 14),
      },
    }))
    next.yAxis = mapAxis(option.yAxis, (axis, index) => ({
      ...axis,
      scale: true,
      ...adaptiveYAxisRange(option.series, index),
    }))
  }
  return next
}

function mapAxis(axis, mapper) {
  if (Array.isArray(axis)) return axis.map((item, index) => mapper(item || {}, index))
  return mapper(axis || {}, 0)
}

function maxChartSpace(value, minimum) {
  return typeof value === 'number' ? Math.max(value, minimum) : minimum
}

function adaptiveYAxisRange(series, axisIndex) {
  const values = numericSeriesValues(series, axisIndex)
  if (values.length < 2) return {}
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min
  const padding = span === 0
    ? Math.max(Math.abs(max) * 0.02, 0.01)
    : Math.max(span * 0.18, 0.002)
  const unitMetric = values.every((value) => value >= 0 && value <= 1)
  const lower = unitMetric ? Math.max(0, min - padding) : min - padding
  const upper = unitMetric ? Math.min(1, max + padding) : max + padding
  return {
    min: roundAxisValue(lower),
    max: roundAxisValue(upper),
  }
}

function numericSeriesValues(series, axisIndex) {
  return (series || [])
    .filter((item) => Number(item?.yAxisIndex || 0) === axisIndex)
    .flatMap((item) => item?.data || [])
    .map((point) => {
      if (Array.isArray(point)) return Number(point[point.length - 1])
      if (point && typeof point === 'object') return Number(point.value)
      return Number(point)
    })
    .filter((value) => Number.isFinite(value))
}

function roundAxisValue(value) {
  return Number(value.toFixed(6))
}

function TableBlock({ table, caption }) {
  if (!table) return null
  const columns = (table.columns || []).map((column) => ({
    title: column.title,
    dataIndex: column.key,
    key: column.key,
    ellipsis: !wrappingColumns.has(column.key),
    width: columnWidths[column.key],
    render: (value) => {
      const text = value == null || value === '' ? '—' : String(value)
      if (column.key === 'run_id') {
        return <Text code style={{ fontSize: 11 }}>{text.slice(0, 12)}</Text>
      }
      if (wrappingColumns.has(column.key)) {
        return <Paragraph style={{ margin: 0, whiteSpace: 'normal' }}>{text}</Paragraph>
      }
      return <span>{text}</span>
    },
  }))
  const dataSource = (table.rows || []).map((row, index) => ({ key: index, ...row }))
  const scrollX = Math.max(
    columns.reduce((sum, column) => sum + (Number(column.width) || 128), 0),
    760,
  )
  return (
    <section className="ai-report-figure ai-report-table-block">
      <div className="ai-report-figure-head">
        <div>
          <div className="ai-report-eyebrow"><TableOutlined /> 结构化表</div>
          <h3>{table.title}</h3>
        </div>
      </div>
      {caption && <Paragraph className="ai-report-caption">{caption}</Paragraph>}
      <Table
        size="small"
        columns={columns}
        dataSource={dataSource}
        pagination={false}
        scroll={{ x: scrollX }}
        tableLayout="fixed"
      />
    </section>
  )
}

function EvidenceList({ items = [] }) {
  if (!items.length) return null
  return (
    <Alert
      type="info"
      showIcon
      message="报告依据"
      description={
        <Space direction="vertical" size={2}>
          {items.map((item, index) => <Text key={index} style={{ fontSize: 12 }}>{item}</Text>)}
        </Space>
      }
    />
  )
}

function ReportBlock({ block, report }) {
  const chartsById = Object.fromEntries((report?.charts || []).map((chart) => [chart.id, chart]))
  const tablesById = Object.fromEntries((report?.tables || []).map((table) => [table.id, table]))

  if (block.type === 'markdown') {
    const markdown = block.id === 'conclusion'
      ? stripReportTitle(block.markdown)
      : block.markdown
    return (
      <div className={`ai-report-prose ai-report-section ai-report-section-${block.id || 'markdown'}`}>
        <MarkdownReport markdown={markdown} />
      </div>
    )
  }
  if (block.type === 'metric_strip') {
    return null
  }
  if (block.type === 'chart') {
    return <ChartBlock chart={chartsById[block.chart_id]} caption={block.caption} />
  }
  if (block.type === 'table') {
    return <TableBlock table={tablesById[block.table_id]} caption={block.caption} />
  }
  if (block.type === 'evidence') {
    return <EvidenceList items={report?.evidence || []} />
  }
  return null
}

function ReportFlow({ report, markdown }) {
  const blocks = report?.report_blocks || []
  if (!blocks.length) {
    return (
      <>
        {(report?.charts || []).map((chart) => <ChartBlock key={chart.id} chart={chart} />)}
        {(report?.tables || []).map((table) => <TableBlock key={table.id} table={table} />)}
        <EvidenceList items={report?.evidence || []} />
        <div className="ai-report-prose">
          <MarkdownReport markdown={stripReportTitle(markdown)} />
        </div>
      </>
    )
  }
  return (
    <div className="ai-report-flow">
      {blocks.map((block) => (
        <ReportBlock key={block.id || `${block.type}-${block.chart_id || block.table_id}`} block={block} report={report} />
      ))}
    </div>
  )
}

function stripReportTitle(markdown = '') {
  return String(markdown).replace(/^#\s+AI 建模报告\s*\n+/, '')
}

function ReportCover({ viewModel, report }) {
  const generatedAt = report?.generated_at || report?.archived_at
  return (
    <section className="ai-report-cover">
      <div className="ai-report-cover-main">
        <div className="ai-report-eyebrow"><FileTextOutlined /> 建模研究报告</div>
        <h1>{viewModel.title}</h1>
        <p>{viewModel.summary || '本报告汇总任务目标、训练过程、结构化图表和模型评价结果。'}</p>
        <div className="ai-report-meta-row">
          <Tag color="blue">归档 {viewModel.archiveLabel || '—'}</Tag>
          {generatedAt && (
            <span>{formatDateTime(generatedAt)}</span>
          )}
          <span>{report?.model || 'doubao'}</span>
        </div>
      </div>
      <div className="ai-report-score-card">
        <div className="ai-report-score-label"><TrophyOutlined /> 评估就绪度</div>
        <div className="ai-report-score-value">{viewModel.score}</div>
        <div className="ai-report-score-note">{viewModel.taskName}</div>
      </div>
    </section>
  )
}

function ReportMetricStrip({ viewModel }) {
  const cards = [
    ...viewModel.metrics,
    { key: 'charts', label: '图表', value: `${viewModel.chartCount}`, detail: '训练过程数据', tone: 'processing' },
    { key: 'tables', label: '表格', value: `${viewModel.tableCount}`, detail: '数据/参数/评价', tone: 'default' },
  ].slice(0, 6)
  if (!cards.length) return null
  return (
    <div className="ai-report-metric-strip">
      {cards.map((item) => (
        <div className="ai-report-metric-card" key={item.key || item.label}>
          <span>{item.label}</span>
          <strong style={{ color: toneColor[item.tone] || toneColor.default }}>{item.value}</strong>
          {item.detail && <em>{item.detail}</em>}
        </div>
      ))}
    </div>
  )
}

export function AiReportReader({ report, taskName = '建模任务', className = '' }) {
  const markdown = report?.markdown || ''
  const viewModel = useMemo(
    () => buildReportViewModel(report || {}, taskName || '建模任务'),
    [report, taskName],
  )
  return (
    <div className={`report-body ai-report-body ${className}`.trim()}>
      <ReportCover viewModel={viewModel} report={report} />
      <ReportMetricStrip viewModel={viewModel} />
      {/* No table of contents. It duplicated the side rail beside it, and the
          two together left the body 818px for tables that need 1000–1160 —
          every structured table had to be dragged sideways to be read. The
          report is three sections; the headings are enough. */}
      <main className="ai-report-document">
        <ReportFlow report={report} markdown={markdown} />
      </main>
    </div>
  )
}

export default function AiReportModal({
  open,
  taskId,
  taskName,
  onClose,
  initialReport = null,
  onGenerated,
}) {
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    if (!taskId) return
    setLoading(true)
    setError(null)
    try {
      const payload = await modelingTaskApi.aiReport(taskId)
      setReport(payload)
      onGenerated?.(payload)
    } catch (err) {
      setReport(null)
      setError(extractError(err))
    } finally {
      setLoading(false)
    }
  }, [taskId, onGenerated])

  useEffect(() => {
    if (!open) return
    if (initialReport) {
      setReport(initialReport)
      setError(null)
      setLoading(false)
      return
    }
    void load()
  }, [open, initialReport, load])

  const markdown = report?.markdown || ''
  const download = () => {
    if (!markdown) return
    const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ai-report-${String(taskId).slice(0, 8)}.md`
    a.click()
    URL.revokeObjectURL(url)
    message.success('AI 报告已下载')
  }

  const footer = [
    <Button key="download" icon={<DownloadOutlined />} disabled={!markdown} onClick={download}>
      下载 Markdown
    </Button>,
    <Button key="reload" icon={<ReloadOutlined />} loading={loading} onClick={() => void load()}>
      重新生成并归档
    </Button>,
    <Button key="close" type="primary" onClick={onClose}>
      关闭
    </Button>,
  ]

  return (
    <Modal
      open={open}
      title={<Space><BulbOutlined />AI 建模报告</Space>}
      width="min(1180px, calc(100vw - 32px))"
      footer={footer}
      destroyOnHidden
      onCancel={onClose}
      className="ai-report-modal"
      style={{ top: 16 }}
      styles={{ body: { padding: 0, maxHeight: 'calc(100vh - 184px)', overflow: 'auto' } }}
    >
      <div className="ai-report-reader">
        {loading && !markdown ? (
          <Spin tip="生成 AI 报告中…">
            <div style={{ minHeight: 160 }} />
          </Spin>
        ) : error ? (
          <Alert
            type="error"
            showIcon
            message="AI 报告生成失败"
            description={error}
          />
        ) : (
          <AiReportReader report={report} taskName={taskName} />
        )}
      </div>
    </Modal>
  )
}
