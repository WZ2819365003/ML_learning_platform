/**
 * ModelDetail — the standalone page for one trained model / run.
 *
 * Replaces the detail halves of the legacy `/training/monitor?taskId=` and
 * `/training/results?taskId=` pages. The tab body is RunDetailBody, the same
 * component the run drawer renders, so the page and the drawer cannot drift.
 *
 * Back navigation is caller-supplied, because the same page is reached from
 * two places that must return to different ones:
 *
 *   workflow  → back to the task's workflow, on the 训练过程 step
 *   models    → back to 模型管理
 *
 * The origin travels in router state (invisible in the URL). A deep link or a
 * refresh loses that state, so `?from=` is read as a fallback and 模型管理 is
 * the final default — never a dead end.
 *
 * Layout is deliberately wide-and-short rather than one long column: a fixed
 * identity card beside a KPI grid, then everything else behind tabs.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  Button, Card, Col, Descriptions, Empty, Row, Space, Spin, Tag, Tooltip, Typography, message,
} from 'antd'
import { ArrowLeftOutlined, DownloadOutlined } from '@ant-design/icons'

import { RunDetailBody, STATUS_TAG } from '../components/workbench/RunDetailBody'
import { modelApi, runModelDownloadUrl } from '../services/api'
import { formatBytes, formatDateTime } from '../utils/formatters'

const { Title, Text } = Typography

/** Where "返回" goes, and what it is called. Exported for tests. */
export function resolveBackTarget({ state, search }) {
  const from = state?.from || search.get('from')
  const taskId = state?.taskId || search.get('taskId')
  if (from === 'workflow' && taskId) {
    // ?step=2 rather than router state: the workflow already reads that param,
    // and it survives a refresh on the page we navigate back to.
    return { label: '返回训练过程', to: `/v3/tasks/${taskId}/workflow?step=2` }
  }
  if (from === 'workflow') {
    return { label: '返回工作流', to: '/v3/tasks' }
  }
  return { label: '返回模型管理', to: '/models' }
}

/** Square-ish KPI tiles — four across, so the row stays a row. */
function MetricTiles({ metrics }) {
  const entries = Object.entries(metrics || {})
    .filter(([, v]) => typeof v === 'number')
    .slice(0, 8)
  if (entries.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无指标" />
  }
  return (
    <Row gutter={[12, 12]}>
      {entries.map(([k, v]) => (
        <Col key={k} xs={12} sm={8} md={6}>
          <div style={{
            border: '1px solid #e2e8f0', borderRadius: 8, padding: '12px 14px',
            background: '#f8fafc', minHeight: 76,
            display: 'flex', flexDirection: 'column', justifyContent: 'center',
          }}>
            <Tooltip title={k}>
              <div style={{
                fontSize: 11, color: '#64748b',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>{k}</div>
            </Tooltip>
            <div style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', fontFamily: 'monospace' }}>
              {v.toFixed(4)}
            </div>
          </div>
        </Col>
      ))}
    </Row>
  )
}

export default function ModelDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const [search] = useSearchParams()

  const [run, setRun] = useState(null)
  const [ttask, setTtask] = useState(null)

  const back = useMemo(
    () => resolveBackTarget({ state: location.state, search }),
    [location.state, search],
  )

  // RunDetailBody already fetches the whole inspector payload; reuse what it
  // loaded for the header instead of issuing a second request for the same rows.
  const handleData = (payload) => {
    setRun(payload?.run || null)
    setTtask(payload?.training_task || null)
  }

  useEffect(() => { setRun(null); setTtask(null) }, [id])

  const download = () => {
    const domainId = ttask?.id
    if (!domainId) { message.warning('该 Run 没有可下载的模型文件'); return }
    const a = document.createElement('a')
    a.href = runModelDownloadUrl(domainId)
    a.download = ''
    a.click()
  }

  return (
    <div style={{ padding: 16 }}>
      {/* Header — back, identity, actions. One line, always visible. */}
      <Card
        variant="borderless"
        styles={{ body: { padding: '12px 20px' } }}
        style={{ marginBottom: 12, boxShadow: '0 1px 2px rgba(15,23,42,0.04)' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate(back.to, back.state ? { state: back.state } : undefined)}
          >
            {back.label}
          </Button>
          <Title level={4} style={{ margin: 0 }}>
            {ttask?.model_type || run?.params?.model_type || '模型详情'}
          </Title>
          {ttask?.family && (
            <Tag color={ttask.family === 'dl' ? 'purple' : 'blue'}>
              {ttask.family === 'dl' ? 'DL' : 'ML'}
            </Tag>
          )}
          {run && STATUS_TAG[run.status]}
          {run?.id && <Text type="secondary" style={{ fontSize: 12 }}>#{String(run.id).slice(0, 8)}</Text>}
          <div style={{ flex: 1 }} />
          <Button icon={<DownloadOutlined />} onClick={download} disabled={!ttask?.id}>
            下载模型
          </Button>
        </div>
      </Card>

      {/* Identity beside KPIs — two columns, so neither becomes a long list. */}
      <Row gutter={[12, 12]} style={{ marginBottom: 12 }}>
        <Col xs={24} lg={8}>
          <Card size="small" title="基本信息" variant="outlined" styles={{ body: { padding: 12 } }}>
            <Descriptions column={1} size="small" colon={false}>
              <Descriptions.Item label="数据集">{ttask?.dataset?.name ?? '—'}</Descriptions.Item>
              <Descriptions.Item label="目标列">{ttask?.target_column ?? '—'}</Descriptions.Item>
              <Descriptions.Item label="模型">{ttask?.model_type ?? '—'}</Descriptions.Item>
              <Descriptions.Item label="完成时间">
                {ttask?.finished_at ? formatDateTime(ttask.finished_at) : '—'}
              </Descriptions.Item>
              <Descriptions.Item label="模型大小">
                {ttask?.model_size ? formatBytes(ttask.model_size) : '—'}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
        <Col xs={24} lg={16}>
          <Card size="small" title="关键指标" variant="outlined" styles={{ body: { padding: 12 } }}>
            <MetricTiles metrics={run?.metrics} />
          </Card>
        </Col>
      </Row>

      {/* Everything else lives behind tabs so the page never grows a third screen. */}
      <Card size="small" variant="outlined" styles={{ body: { padding: '8px 16px 16px' } }}>
        {id ? (
          <RunDetailBody runId={id} active defaultTab="overview" onData={handleData} />
        ) : (
          <Spin />
        )}
      </Card>
    </div>
  )
}
