import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Alert, Button, Modal, Space, Spin, message } from 'antd'
import {
  BulbOutlined,
  DownloadOutlined,
  FileTextOutlined,
  ReloadOutlined,
} from '@ant-design/icons'

import { modelingTaskApi } from '../../services/api'
import ReportDocument from './ReportDocument'
import { buildReportViewModel } from './aiReportViewModel'
import { formatDateTime } from '../../utils/formatters'

function extractError(err) {
  let body = err?.response?.data
  if (typeof body === 'string') {
    try { body = JSON.parse(body) } catch { /* use string as-is */ }
  }
  return (typeof body === 'string' ? body : body?.detail) || err?.message || 'AI 报告生成失败'
}

/**
 * The cover: one line of facts and one sentence of judgement.
 *
 * No score card and no chart/table counts — the number that used to dominate
 * the cover was a three-item checklist, and counting figures tells the reader
 * nothing about the model. The headline is the finding.
 */
function ReportCover({ viewModel }) {
  return (
    <header className="ai-report-cover">
      <div className="ai-report-eyebrow"><FileTextOutlined /> 建模报告</div>
      <h1>{viewModel.title}</h1>
      {viewModel.metaItems && (
        <div className="ai-report-meta-row">
          {viewModel.metaItems.map((item) => <span key={item}>{item}</span>)}
          {viewModel.generatedAt && <span>{formatDateTime(viewModel.generatedAt)}</span>}
        </div>
      )}
      {viewModel.headline && <p className="ai-report-headline">{viewModel.headline}</p>}
    </header>
  )
}

export function AiReportReader({
  report, taskName = '建模任务', className = '', appendixOpen = false,
}) {
  const viewModel = useMemo(
    () => buildReportViewModel(report || {}, taskName || '建模任务'),
    [report, taskName],
  )
  return (
    <div className={`report-body ai-report-body ${className}`.trim()}>
      <ReportCover viewModel={viewModel} />
      <main className="ai-report-document">
        <ReportDocument
          markdown={report?.markdown || ''}
          charts={report?.charts || []}
          // Old archives carry `tables`; folding them into the appendix keeps
          // them reachable without putting a table back in the prose.
          appendixTables={report?.appendix_tables || report?.tables || []}
          stripTitle
          appendixOpen={appendixOpen}
        />
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
