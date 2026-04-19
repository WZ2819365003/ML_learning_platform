/**
 * /experiments/:experimentId is legacy — the dedicated ExperimentDetail page
 * has been retired.  A PlatformExperiment now always lives under a
 * ModelingTask, so we look up the modeling_task_id and redirect into the
 * workbench at /v3/tasks/:modeling_task_id (which has a "Run 对比" tab that
 * supersedes the old experiment compare UI).
 *
 * Falls back to /experiments if resolution fails so the user isn't stranded.
 */
import React, { useEffect, useState } from 'react'
import { useParams, Navigate } from 'react-router-dom'
import { Alert, Skeleton, Space, Typography } from 'antd'
import { platformExperimentsApi } from '../services/api'

const { Text } = Typography

export default function ExperimentRedirect() {
  const { experimentId } = useParams()
  const [target, setTarget] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!experimentId) return
    let cancelled = false
    platformExperimentsApi.get(experimentId)
      .then((exp) => {
        if (cancelled) return
        if (exp?.modeling_task_id) {
          setTarget(`/v3/tasks/${exp.modeling_task_id}`)
        } else {
          setError('该实验未关联建模任务 — 无法跳转到工作台')
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err?.response?.data?.detail || err.message || '加载实验失败')
        }
      })
    return () => { cancelled = true }
  }, [experimentId])

  if (target) return <Navigate to={target} replace />

  if (error) {
    return (
      <div style={{ padding: 40, maxWidth: 720, margin: '0 auto' }}>
        <Alert
          type="warning"
          showIcon
          message="实验详情页已合并至建模任务工作台"
          description={
            <Space direction="vertical" size={4}>
              <Text>{error}</Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                请从 <a href="/v3/tasks">建模任务列表</a> 进入对应任务查看实验详情。
              </Text>
            </Space>
          }
        />
      </div>
    )
  }

  return (
    <div style={{ padding: 40 }}>
      <Skeleton active paragraph={{ rows: 3 }} />
      <Text type="secondary" style={{ fontSize: 12 }}>正在跳转到建模任务工作台…</Text>
    </div>
  )
}
