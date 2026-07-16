import React, { useMemo, useState, useEffect } from 'react'
import {
  Card, Select, Input, Button, Space, Tag, Typography, Alert, Descriptions,
  message, Empty, Divider, Tooltip,
} from 'antd'
import {
  CloudUploadOutlined, DownloadOutlined, ThunderboltOutlined, TrophyOutlined,
} from '@ant-design/icons'
import { deployApi, runModelDownloadUrl } from '../../services/api'
import { useDeployRun } from '../../hooks/useDeployRun'

const { Text } = Typography

/**
 * Workflow 部署 step. Given the task's successful runs (with domain_task_id +
 * family from the runs/leaderboard payload), lets the user deploy one run's
 * model, then run a quick prediction and download the artifact.
 *
 * props: task, runs (array), bestRunId
 */
export default function DeployStep({ task, runs = [], bestRunId }) {
  const successRuns = useMemo(
    () => runs.filter(r => String(r.status).toUpperCase() === 'SUCCESS' && r.domain_task_id),
    [runs]
  )
  const [runId, setRunId] = useState(null)
  const [name, setName] = useState('')
  const { deploying, deployment, deploy, reset } = useDeployRun(task)
  const [predictInput, setPredictInput] = useState('[\n  {}\n]')
  const [predicting, setPredicting] = useState(false)
  const [predictResult, setPredictResult] = useState(null)

  // Default selection = best run (or first success run)
  useEffect(() => {
    if (!runId && successRuns.length) {
      setRunId(bestRunId && successRuns.some(r => r.run_id === bestRunId)
        ? bestRunId : successRuns[0].run_id)
    }
  }, [successRuns, bestRunId, runId])

  useEffect(() => {
    if (task?.name) setName(`${task.name}-部署`)
  }, [task?.name])

  const selectedRun = successRuns.find(r => r.run_id === runId)

  const handleDeploy = async () => {
    if (!runId) { message.warning('请先选择一个成功的 Run'); return }
    if (!name.trim()) { message.warning('请填写部署名称'); return }
    setPredictResult(null)
    await deploy(runId, { name })
  }

  const handlePredict = async () => {
    if (!deployment?.deployment_id) return
    let rows
    try {
      rows = JSON.parse(predictInput)
      if (!Array.isArray(rows)) throw new Error('need array')
    } catch {
      message.error('预测输入需为 JSON 数组，例如 [{"feature_a": 1.0}]')
      return
    }
    setPredicting(true)
    try {
      const resp = await deployApi.predict(deployment.deployment_id, {
        rows, include_probabilities: true,
      })
      setPredictResult(resp)
    } catch (err) {
      message.error(err?.response?.data?.detail || '预测失败')
    } finally {
      setPredicting(false)
    }
  }

  if (!successRuns.length) {
    return (
      <Card size="small" bodyStyle={{ padding: 32 }}>
        <Empty description={
          <Text type="secondary">还没有训练成功的 Run，完成「训练」步骤后即可部署。</Text>
        } />
      </Card>
    )
  }

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card size="small" title={<span><CloudUploadOutlined /> 选择要上线的模型</span>}
        bodyStyle={{ padding: 16 }}>
        <Space direction="vertical" size={12} style={{ width: '100%' }}>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>成功的 Run（默认选中最佳）</Text>
            <Select
              style={{ width: '100%', marginTop: 4 }}
              value={runId}
              onChange={(v) => { setRunId(v); reset(); setPredictResult(null) }}
              options={successRuns.map(r => ({
                value: r.run_id,
                label: (
                  <Space size={6}>
                    {r.run_id === bestRunId && <TrophyOutlined style={{ color: '#f59e0b' }} />}
                    <span>{r.params?.model_type || r.family || 'model'}</span>
                    <Tag color="blue" style={{ fontSize: 10, margin: 0 }}>{r.strategy_type}</Tag>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {task?.objective_metric}={typeof r.objective_value === 'number' ? r.objective_value.toFixed(4) : '-'}
                    </Text>
                    <Tag style={{ fontSize: 10, margin: 0 }}>{r.family || 'ml'}</Tag>
                  </Space>
                ),
              }))}
            />
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>部署名称</Text>
            <Input style={{ marginTop: 4 }} value={name} onChange={e => setName(e.target.value)}
              placeholder="例：iris-分类-prod" />
          </div>
          <Space>
            <Button type="primary" icon={<CloudUploadOutlined />} loading={deploying}
              onClick={handleDeploy}>
              部署上线
            </Button>
            {selectedRun?.domain_task_id && (
              <Tooltip title="下载该 Run 训练出的模型文件 (.joblib)">
                <Button icon={<DownloadOutlined />}
                  href={runModelDownloadUrl(selectedRun.domain_task_id)} target="_blank">
                  下载模型
                </Button>
              </Tooltip>
            )}
          </Space>
        </Space>
      </Card>

      {deployment && (
        <Card size="small" title={<span><ThunderboltOutlined style={{ color: '#10b981' }} /> 已上线 · 推理接口</span>}
          bodyStyle={{ padding: 16 }}>
          <Descriptions column={1} size="small" bordered labelStyle={{ width: 120, background: '#f8fafc' }}>
            <Descriptions.Item label="部署 ID"><code>{deployment.deployment_id}</code></Descriptions.Item>
            <Descriptions.Item label="状态"><Tag color="green">{deployment.status || 'active'}</Tag></Descriptions.Item>
            {deployment.endpoints?.predict && (
              <Descriptions.Item label="预测端点">
                <code style={{ fontSize: 12 }}>{deployment.endpoints.predict}</code>
              </Descriptions.Item>
            )}
          </Descriptions>

          <Divider style={{ margin: '14px 0 10px' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>快速预测（JSON 行数组）</Text>
          </Divider>
          <Input.TextArea rows={4} value={predictInput} onChange={e => setPredictInput(e.target.value)}
            style={{ fontFamily: 'monospace', fontSize: 12 }}
            placeholder='[{"feature_a": 1.2, "feature_b": 3.4}]' />
          <Button style={{ marginTop: 8 }} loading={predicting} onClick={handlePredict}>
            调用预测
          </Button>
          {predictResult && (
            <Alert style={{ marginTop: 10 }} type="success" showIcon
              message="预测结果"
              description={<pre style={{ margin: 0, fontSize: 12, whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(predictResult, null, 2)}
              </pre>} />
          )}
        </Card>
      )}
    </Space>
  )
}
