/**
 * BatchPredictPanel — upload a CSV, watch the job, download the result (M3-2).
 *
 * The backend runs this asynchronously, so the panel polls rather than waiting
 * on the request. Polling stops on a terminal status and on unmount; a forgotten
 * interval here would keep hitting the API for as long as the tab is open.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  Alert, Button, Descriptions, Progress, Space, Tag, Upload, message,
} from 'antd'
import {
  CloudUploadOutlined, DownloadOutlined, ReloadOutlined,
} from '@ant-design/icons'

import { deployApi } from '../../services/api'

const POLL_MS = 1500
const TERMINAL = ['completed', 'failed']

const STATUS_META = {
  pending:   { color: 'default',    label: '排队中' },
  running:   { color: 'processing', label: '预测中' },
  completed: { color: 'success',    label: '已完成' },
  failed:    { color: 'error',      label: '失败' },
}

export default function BatchPredictPanel({ deploymentId }) {
  const [job, setJob] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const timer = useRef(null)

  const stopPolling = useCallback(() => {
    if (timer.current) {
      clearInterval(timer.current)
      timer.current = null
    }
  }, [])

  // Stop on unmount too — otherwise closing the drawer leaves the interval
  // running against a deployment nobody is looking at.
  useEffect(() => stopPolling, [stopPolling])

  const poll = useCallback(async (jobId) => {
    try {
      const data = await deployApi.getBatchPredict(deploymentId, jobId)
      setJob(data)
      if (TERMINAL.includes(data?.status)) stopPolling()
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || '状态查询失败')
      stopPolling()
    }
  }, [deploymentId, stopPolling])

  const startPolling = useCallback((jobId) => {
    stopPolling()
    timer.current = setInterval(() => void poll(jobId), POLL_MS)
  }, [poll, stopPolling])

  const handleUpload = async (file) => {
    setSubmitting(true)
    setError(null)
    try {
      const data = await deployApi.submitBatchPredict(deploymentId, file)
      setJob(data)
      message.success('已提交，正在后台预测')
      startPolling(data.job_id)
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || '提交失败')
    } finally {
      setSubmitting(false)
    }
    return false // never let antd upload the file itself
  }

  const meta = STATUS_META[job?.status] || { color: 'default', label: job?.status || '—' }
  const done = job?.status === 'completed'
  const total = job?.input_rows || 0
  const processed = job?.processed_rows || 0
  const percent = total > 0 ? Math.round((processed / total) * 100) : (done ? 100 : 0)

  return (
    <div>
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="批量预测"
        description="上传 CSV，后台异步预测。结果会在原表基础上追加 prediction 列，完成后可下载。"
      />

      <Space style={{ marginBottom: 16 }}>
        <Upload accept=".csv" showUploadList={false} beforeUpload={handleUpload}>
          <Button icon={<CloudUploadOutlined />} loading={submitting}>上传 CSV</Button>
        </Upload>
        {job && (
          <Button
            icon={<ReloadOutlined />}
            onClick={() => void poll(job.job_id)}
          >
            刷新状态
          </Button>
        )}
        {done && (
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            href={deployApi.batchPredictDownloadUrl(deploymentId, job.job_id)}
           target="_blank"
          >
            下载结果 CSV
          </Button>
        )}
      </Space>

      {error && (
        <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} />
      )}

      {job && (
        <>
          <Progress
            percent={percent}
            status={job.status === 'failed' ? 'exception' : (done ? 'success' : 'active')}
            style={{ marginBottom: 12 }}
          />
          <Descriptions bordered size="small" column={2}>
            <Descriptions.Item label="任务 ID">{job.job_id}</Descriptions.Item>
            <Descriptions.Item label="状态">
              <Tag color={meta.color}>{meta.label}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="已处理行数">{processed}</Descriptions.Item>
            <Descriptions.Item label="总行数">{total || '统计中'}</Descriptions.Item>
            {job.error_message && (
              <Descriptions.Item label="错误" span={2}>{job.error_message}</Descriptions.Item>
            )}
          </Descriptions>
        </>
      )}
    </div>
  )
}
