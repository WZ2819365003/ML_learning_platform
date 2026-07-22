/**
 * ReportView — renders the backend's Markdown report (M3-1).
 *
 * The backend deliberately emits text only; charts live here. See
 * app/services/report_service.py for why.
 *
 * Safety note: every value below reaches the DOM as a React child, never as
 * innerHTML. The report interpolates user-controlled strings (dataset names,
 * feature names), so this is what keeps a dataset named `<img onerror=…>` from
 * becoming stored XSS. Do not "improve" this with dangerouslySetInnerHTML.
 */
import React, { useEffect, useState, useCallback } from 'react'
import {
  Alert, Button, Card, Empty, Space, Spin, Table, Typography, message,
} from 'antd'
import { DownloadOutlined, PrinterOutlined, ReloadOutlined } from '@ant-design/icons'

import { modelingTaskApi } from '../../services/api'
import { parseMarkdown, parseInline } from '../../utils/markdown'

const { Title, Paragraph, Text } = Typography

/** Render inline markup as React nodes — escaping comes free. */
function Inline({ text }) {
  return (
    <>
      {parseInline(text).map((seg, i) => {
        if (seg.kind === 'bold') {
          return (
            <strong key={i}>
              {seg.children.map((child, j) =>
                child.kind === 'code'
                  ? <Text key={j} code>{child.value}</Text>
                  : <React.Fragment key={j}>{child.value}</React.Fragment>,
              )}
            </strong>
          )
        }
        if (seg.kind === 'code') return <Text key={i} code>{seg.value}</Text>
        return <React.Fragment key={i}>{seg.value}</React.Fragment>
      })}
    </>
  )
}

function MarkdownTable({ headers, rows }) {
  const columns = headers.map((h, i) => ({
    title: h,
    dataIndex: String(i),
    key: String(i),
    render: (value) => <Inline text={value ?? ''} />,
  }))
  const dataSource = rows.map((row, r) => {
    const record = { key: r }
    row.forEach((cell, c) => { record[String(c)] = cell })
    return record
  })
  return (
    <Table
      size="small"
      bordered
      pagination={false}
      columns={columns}
      dataSource={dataSource}
      style={{ marginBottom: 16 }}
    />
  )
}

/** A blockquote carrying ⚠️ is a correctness warning, not decoration. */
function Blockquote({ text }) {
  const isWarning = text.includes('⚠')
  return (
    <Alert
      type={isWarning ? 'warning' : 'info'}
      showIcon={isWarning}
      style={{ marginBottom: 16 }}
      message={<Inline text={text} />}
    />
  )
}

function Block({ block }) {
  switch (block.type) {
    case 'heading': {
      const level = Math.min(block.level, 5)
      return (
        <Title level={level} style={{ marginTop: level <= 2 ? 28 : 20 }}>
          <Inline text={block.text} />
        </Title>
      )
    }
    case 'table':
      return <MarkdownTable headers={block.headers} rows={block.rows} />
    case 'blockquote':
      return <Blockquote text={block.text} />
    default:
      return (
        <Paragraph>
          <Inline text={block.text} />
        </Paragraph>
      )
  }
}

export default function ReportView({ taskId }) {
  const [markdown, setMarkdown] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      // api.js unwraps response.data in a response interceptor, so this is
      // already the Markdown string — not an axios response.
      const md = await modelingTaskApi.report(taskId)
      setMarkdown(typeof md === 'string' ? md : String(md ?? ''))
    } catch (err) {
      // 409 means "not finalized yet" and carries actionable guidance — surface
      // it verbatim instead of a generic failure the user cannot act on.
      // responseType:'text' applies to error bodies too, so the JSON envelope
      // arrives as a string and has to be parsed before `.detail` exists.
      let body = err?.response?.data
      if (typeof body === 'string') {
        try { body = JSON.parse(body) } catch { /* not JSON — use as-is */ }
      }
      const detail = typeof body === 'string' ? body : body?.detail
      setError(detail || err?.message || '报告加载失败')
    } finally {
      setLoading(false)
    }
  }, [taskId])

  useEffect(() => { load() }, [load])

  const download = () => {
    const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `report-${String(taskId).slice(0, 8)}.md`
    a.click()
    URL.revokeObjectURL(url)
    message.success('报告已下载')
  }

  if (loading) {
    // `tip` only renders in nest/fullscreen mode, so wrap content rather
    // than using a bare <Spin tip=…/> whose label silently never appears.
    return (
      <div style={{ padding: 48 }}>
        <Spin tip="生成报告中…">
          <div style={{ minHeight: 80 }} />
        </Spin>
      </div>
    )
  }

  if (error) {
    return (
      <Alert
        type="info"
        showIcon
        message="暂时无法生成报告"
        description={<span>{error}</span>}
        action={<Button size="small" icon={<ReloadOutlined />} onClick={load}>重试</Button>}
      />
    )
  }

  const blocks = parseMarkdown(markdown)
  if (!blocks.length) return <Empty description="报告为空" />

  return (
    <div>
      <Space style={{ marginBottom: 16 }} className="report-actions">
        <Button icon={<DownloadOutlined />} onClick={download}>下载 Markdown</Button>
        <Button icon={<PrinterOutlined />} onClick={() => window.print()}>打印 / 导出 PDF</Button>
        <Button icon={<ReloadOutlined />} onClick={load}>刷新</Button>
      </Space>

      <Card variant="outlined" className="report-body">
        {blocks.map((block, i) => <Block key={i} block={block} />)}
      </Card>
    </div>
  )
}
