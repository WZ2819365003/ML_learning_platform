/**
 * Safe Markdown renderer for generated modeling reports.
 *
 * The parser returns tokens and React renders them as children, so report
 * content is escaped by construction. Do not replace this with raw HTML.
 */
import React from 'react'
import { Alert, Empty, Table, Typography } from 'antd'

import { parseMarkdown, parseInline } from '../../utils/markdown'

const { Title, Paragraph, Text } = Typography

export function Inline({ text }) {
  return (
    <>
      {parseInline(text).map((seg, i) => {
        if (seg.kind === 'bold') {
          return (
            <strong key={i} className="ai-report-lead">
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

export default function MarkdownReport({ markdown, emptyDescription = '报告为空' }) {
  const blocks = parseMarkdown(markdown)
  if (!blocks.length) return <Empty description={emptyDescription} />
  return (
    <>
      {blocks.map((block, i) => <Block key={i} block={block} />)}
    </>
  )
}
