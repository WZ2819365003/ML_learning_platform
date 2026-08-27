/**
 * TrainingLogModal — the live training log for one task, in a modal.
 *
 * Wraps LogViewer so every place that wants "show me this model's log" gets
 * the same panel — the orchestration tree, 模型管理, and anything later —
 * instead of each growing its own copy.
 *
 * Seeding differs by what the caller has to offer:
 *   runId given        → the run-inspector endpoint, which owns the messy
 *                        id-resolution (V3 native logs, then the legacy table,
 *                        then the .log file on disk)
 *   only domainTaskId  → GET /logs/{id}, which is keyed by that id directly
 *
 * The live tail is always keyed by the *domain* task id: that is the event-bus
 * channel the trainer publishes to.
 */
import React, { useEffect, useState } from 'react'
import { Empty, Modal, Space, Spin, Tag, Typography } from 'antd'
import { FileTextOutlined } from '@ant-design/icons'

import LogViewer from './LogViewer'
import { logsApi, platformRunsApi } from '../../services/api'

const { Text } = Typography

export default function TrainingLogModal({
  open,
  onClose,
  domainTaskId,
  runId = null,
  title,
  statusTag = null,
  isLive = false,
}) {
  const [historical, setHistorical] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!open || (!runId && !domainTaskId)) { setHistorical([]); return undefined }
    let cancelled = false
    setLoading(true)

    const seed = runId
      ? platformRunsApi.inspect(runId, { log_limit: 500, include_siblings: false })
          .then(resp => resp?.logs || [])
      : logsApi.getLogs(domainTaskId, { page_size: 500 })
          .then(resp => resp?.entries || [])

    seed
      .then(entries => { if (!cancelled) setHistorical(entries) })
      .catch(() => { if (!cancelled) setHistorical([]) })
      .finally(() => { if (!cancelled) setLoading(false) })

    return () => { cancelled = true }
  }, [open, runId, domainTaskId])

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={960}
      destroyOnClose
      title={
        <Space>
          <FileTextOutlined />
          <span>训练日志</span>
          {title && <Text strong>{title}</Text>}
          {statusTag}
        </Space>
      }
    >
      <Spin spinning={loading}>
        {domainTaskId ? (
          <LogViewer historical={historical} domainTaskId={domainTaskId} isLive={isLive} />
        ) : (
          <Empty description="该模型没有关联的训练任务，暂无日志" />
        )}
      </Spin>
    </Modal>
  )
}
