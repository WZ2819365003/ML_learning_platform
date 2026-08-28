import React, { useEffect, useState } from 'react'
import { Alert, Spin } from 'antd'

import LogViewer from '../workbench/LogViewer'
import { dlApi, logsApi } from '../../services/api'

export default function ResultLogPanel({ family, taskId, status }) {
  const [historical, setHistorical] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)
    const request = family === 'dl'
      ? dlApi.getLogs(taskId, { page: 1, page_size: 1000 })
      : logsApi.getLogs(taskId, { page: 1, page_size: 500 })

    request
      .then(payload => {
        if (!active) return
        setHistorical((payload?.entries ?? []).map(entry => ({
          ...entry,
          timestamp: entry.timestamp ?? entry.created_at,
        })))
      })
      .catch(err => {
        if (!active) return
        setHistorical([])
        setError(err?.response?.data?.detail || err?.message || '训练日志加载失败')
      })
      .finally(() => { if (active) setLoading(false) })

    return () => { active = false }
  }, [family, taskId])

  return (
    <Spin spinning={loading}>
      {error && (
        <Alert
          type="warning"
          showIcon
          message="历史日志暂时不可用"
          description={error}
          style={{ marginBottom: 12 }}
        />
      )}
      <LogViewer
        historical={historical}
        domainTaskId={taskId}
        isLive={['PENDING', 'RUNNING'].includes(status)}
        streamEnabled={['PENDING', 'RUNNING'].includes(status)}
      />
    </Spin>
  )
}
