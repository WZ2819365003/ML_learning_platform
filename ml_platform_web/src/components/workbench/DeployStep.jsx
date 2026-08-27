import React, { useMemo, useState, useEffect, useCallback } from 'react'
import {
  Card, Select, Input, Button, Space, Tag, Typography, Alert, Descriptions,
  message, Empty, Divider, Tooltip, Tabs, Table, InputNumber, Progress, Row, Col,
} from 'antd'
import {
  CloudUploadOutlined, DownloadOutlined, ThunderboltOutlined, TrophyOutlined,
  CopyOutlined, ApiOutlined, BlockOutlined, ReloadOutlined,
} from '@ant-design/icons'
import { dataApi, deployApi, runModelDownloadUrl } from '../../services/api'
import { useDeployRun } from '../../hooks/useDeployRun'
import {
  buildCurl, buildRequestExample, buildResponseExample, deploymentNotes,
  normaliseWeights, predictUrl, suggestWeights,
} from './deploySchema'

const { Text } = Typography

/** Copy helper that degrades to a message rather than throwing. */
async function copyText(value, label = '内容') {
  try {
    await navigator.clipboard.writeText(value)
    message.success(`${label}已复制`)
  } catch {
    message.warning('复制失败，请手动选中复制')
  }
}

/** A labelled, copyable JSON block. */
function JsonBlock({ title, value, extra, height = 170 }) {
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
        <Text strong style={{ fontSize: 12 }}>{title}</Text>
        <div style={{ flex: 1 }} />
        {extra}
        <Button size="small" type="text" icon={<CopyOutlined />}
          onClick={() => copyText(text, title)}>复制</Button>
      </div>
      <pre style={{
        margin: 0, padding: 10, borderRadius: 6, background: '#0f172a', color: '#e2e8f0',
        fontSize: 11.5, lineHeight: 1.55, height, overflow: 'auto',
        fontFamily: 'ui-monospace, Menlo, Monaco, monospace',
      }}>{text}</pre>
    </div>
  )
}

/** Label for a run in a picker or a table. */
function RunLabel({ run, bestRunId, objectiveMetric }) {
  return (
    <Space size={6}>
      {run.run_id === bestRunId && <TrophyOutlined style={{ color: '#f59e0b' }} />}
      <span>{run.params?.model_type || run.family || 'model'}</span>
      <Tag color="blue" style={{ fontSize: 10, margin: 0 }}>{run.strategy_type}</Tag>
      <Text type="secondary" style={{ fontSize: 11 }}>
        {objectiveMetric}={typeof run.objective_value === 'number' ? run.objective_value.toFixed(4) : '-'}
      </Text>
      <Tag style={{ fontSize: 10, margin: 0 }}>{run.family || 'ml'}</Tag>
    </Space>
  )
}

// ---------------------------------------------------------------------------
// Tab 1 — single model
// ---------------------------------------------------------------------------

function SingleDeployTab({ task, successRuns, bestRunId, schema }) {
  const [runId, setRunId] = useState(null)
  const [name, setName] = useState('')
  const { deploying, deployment, deploy, reset } = useDeployRun(task)
  const [predictInput, setPredictInput] = useState('')
  const [predicting, setPredicting] = useState(false)
  const [predictResult, setPredictResult] = useState(null)

  useEffect(() => {
    if (!runId && successRuns.length) {
      setRunId(bestRunId && successRuns.some(r => r.run_id === bestRunId)
        ? bestRunId : successRuns[0].run_id)
    }
  }, [successRuns, bestRunId, runId])

  useEffect(() => { if (task?.name) setName(`${task.name}-部署`) }, [task?.name])

  // Seed the try-it box with the same example the contract shows, so the first
  // call a user makes is one that actually works.
  useEffect(() => {
    if (schema.requestExample) {
      setPredictInput(JSON.stringify(schema.requestExample.rows, null, 2))
    }
  }, [schema.requestExample])

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

  const curl = useMemo(() => buildCurl({
    deploymentId: deployment?.deployment_id,
    requestExample: schema.requestExample,
  }), [deployment?.deployment_id, schema.requestExample])

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Card size="small" title={<span><CloudUploadOutlined /> 选择要上线的模型</span>}
        styles={{ body: { padding: 16 } }}>
        <Space direction="vertical" size={12} style={{ width: '100%' }}>
          <Row gutter={[12, 8]}>
            <Col xs={24} lg={14}>
              <Text type="secondary" style={{ fontSize: 12 }}>成功的 Run（默认选中最佳）</Text>
              <Select
                style={{ width: '100%', marginTop: 4 }}
                value={runId}
                onChange={(v) => { setRunId(v); reset(); setPredictResult(null) }}
                options={successRuns.map(r => ({
                  value: r.run_id,
                  label: <RunLabel run={r} bestRunId={bestRunId} objectiveMetric={task?.objective_metric} />,
                }))}
              />
            </Col>
            <Col xs={24} lg={10}>
              <Text type="secondary" style={{ fontSize: 12 }}>部署名称</Text>
              <Input style={{ marginTop: 4 }} value={name} onChange={e => setName(e.target.value)}
                placeholder="例：iris-分类-prod" />
            </Col>
          </Row>
          <Space>
            <Button type="primary" icon={<CloudUploadOutlined />} loading={deploying}
              onClick={handleDeploy}>部署上线</Button>
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

      {/* The contract is shown before deploying too — the shape does not depend
          on the deployment, only the URL does, and seeing it up front is what
          lets someone plan the integration. */}
      <Card size="small" title={<span><ApiOutlined /> 接口契约</span>}
        styles={{ body: { padding: 16 } }}
        extra={!deployment && <Text type="secondary" style={{ fontSize: 11 }}>部署后 URL 才会确定</Text>}>
        {schema.loading ? (
          <Text type="secondary">正在读取数据集字段…</Text>
        ) : (
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <Descriptions column={1} size="small" bordered
              labelStyle={{ width: 110, background: '#f8fafc' }}>
              <Descriptions.Item label="请求方法"><code>POST</code></Descriptions.Item>
              <Descriptions.Item label="端点">
                <code style={{ fontSize: 12 }}>{predictUrl(deployment?.deployment_id)}</code>
              </Descriptions.Item>
              <Descriptions.Item label="特征列数">
                {schema.featureCount} 列（目标列 <code>{task?.target_column}</code> 不要传）
              </Descriptions.Item>
            </Descriptions>

            {/* Request and response side by side — they are read together,
                and stacking them was most of this page's height. */}
            <Row gutter={[12, 12]}>
              <Col xs={24} lg={12}>
                <JsonBlock title="入参 JSON" value={schema.requestExample} />
              </Col>
              <Col xs={24} lg={12}>
                <JsonBlock title="出参 JSON" value={schema.responseExample} />
              </Col>
              <Col xs={24} lg={12}>
                <JsonBlock title="curl 示例" value={curl} height={150} />
              </Col>
              <Col xs={24} lg={12}>
                <Text strong style={{ fontSize: 12 }}>部署说明</Text>
                <ul style={{
                  margin: '6px 0 0', paddingLeft: 18, fontSize: 11.5, color: '#475569',
                  lineHeight: 1.75, height: 150, overflowY: 'auto',
                }}>
                  {schema.notes.map((n, i) => <li key={i}>{n}</li>)}
                </ul>
              </Col>
            </Row>
          </Space>
        )}
      </Card>

      {deployment && (
        <Card size="small" title={<span><ThunderboltOutlined style={{ color: '#10b981' }} /> 已上线 · 试调用</span>}
          styles={{ body: { padding: 16 } }}>
          <Descriptions column={1} size="small" bordered labelStyle={{ width: 110, background: '#f8fafc' }}>
            <Descriptions.Item label="部署 ID"><code>{deployment.deployment_id}</code></Descriptions.Item>
            <Descriptions.Item label="状态"><Tag color="green">{deployment.status || 'active'}</Tag></Descriptions.Item>
          </Descriptions>

          <Divider style={{ margin: '14px 0 10px' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>快速预测（JSON 行数组）</Text>
          </Divider>
          <Input.TextArea rows={5} value={predictInput} onChange={e => setPredictInput(e.target.value)}
            style={{ fontFamily: 'monospace', fontSize: 12 }} />
          <Button style={{ marginTop: 8 }} loading={predicting} onClick={handlePredict}>调用预测</Button>
          {predictResult && (
            <Alert style={{ marginTop: 10 }} type="success" showIcon message="预测结果"
              description={<pre style={{ margin: 0, fontSize: 12, whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(predictResult, null, 2)}
              </pre>} />
          )}
        </Card>
      )}
    </Space>
  )
}

// ---------------------------------------------------------------------------
// Tab 2 — weighted ensemble
// ---------------------------------------------------------------------------

function MultiDeployTab({ task, successRuns, bestRunId }) {
  const direction = task?.objective_direction === 'min' ? 'min' : 'max'
  const [selectedIds, setSelectedIds] = useState([])
  const [weights, setWeights] = useState({})
  const [name, setName] = useState('')

  useEffect(() => { if (task?.name) setName(`${task.name}-融合部署`) }, [task?.name])

  const selected = useMemo(
    () => successRuns.filter(r => selectedIds.includes(r.run_id)),
    [successRuns, selectedIds],
  )

  const applySuggested = useCallback((members) => {
    setWeights(suggestWeights(members, direction))
  }, [direction])

  // Re-suggest whenever the membership changes; manual edits survive until then.
  useEffect(() => { applySuggested(selected) }, [selectedIds]) // eslint-disable-line react-hooks/exhaustive-deps

  const normalised = useMemo(() => normaliseWeights(weights), [weights])
  const weightSum = Object.values(weights).reduce((a, b) => a + (b || 0), 0)

  const ensemblePreview = useMemo(() => ({
    name,
    strategy: 'weighted_average',
    members: selected.map(r => ({
      run_id: r.run_id,
      model_type: r.params?.model_type || r.family,
      family: r.family || 'ml',
      weight: Number((normalised[r.run_id] ?? 0).toFixed(4)),
    })),
  }), [name, selected, normalised])

  const columns = [
    {
      title: '模型', key: 'model',
      render: (_, r) => <RunLabel run={r} bestRunId={bestRunId} objectiveMetric={task?.objective_metric} />,
    },
    {
      title: '权重', key: 'weight', width: 130,
      render: (_, r) => (
        <InputNumber size="small" min={0} step={0.05} style={{ width: 100 }}
          value={weights[r.run_id]}
          onChange={(v) => setWeights(w => ({ ...w, [r.run_id]: v ?? 0 }))} />
      ),
    },
    {
      title: '归一化后', key: 'normalised', width: 150,
      render: (_, r) => {
        const pct = Math.round((normalised[r.run_id] ?? 0) * 100)
        return (
          <Space size={6}>
            <Progress percent={pct} size="small" style={{ width: 70 }} showInfo={false} />
            <Text style={{ fontFamily: 'monospace', fontSize: 12 }}>{pct}%</Text>
          </Space>
        )
      },
    },
  ]

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Alert
        type="info" showIcon
        message="这个页面还没接后端"
        description={
          <span>
            界面可以试用，但「创建融合部署」尚未接通 —— 后端还没有存放成员和权重的表，
            也还没有把推理扇出到多个模型。先看排版和交互是否合用。
          </span>
        }
      />

      <Card size="small" title={<span><BlockOutlined /> 选择参与融合的模型（至少 2 个）</span>}
        styles={{ body: { padding: 16 } }}>
        <Space direction="vertical" size={12} style={{ width: '100%' }}>
          <Select
            mode="multiple" style={{ width: '100%' }} value={selectedIds} onChange={setSelectedIds}
            placeholder="从训练成功的 Run 里挑选"
            options={successRuns.map(r => ({
              value: r.run_id,
              label: <RunLabel run={r} bestRunId={bestRunId} objectiveMetric={task?.objective_metric} />,
            }))}
          />

          {selected.length < 2 ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={<Text type="secondary">再选一个模型即可配置权重</Text>} />
          ) : (
            <>
              <Table size="small" rowKey="run_id" columns={columns} dataSource={selected}
                pagination={false} />
              <Space>
                <Button size="small" icon={<ReloadOutlined />} onClick={() => applySuggested(selected)}>
                  按成绩重算权重
                </Button>
                <Button size="small" onClick={() => setWeights(
                  Object.fromEntries(selected.map(r => [r.run_id, 1 / selected.length])))}>
                  等权
                </Button>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  当前合计 {weightSum.toFixed(3)}，提交时按比例归一化
                </Text>
              </Space>

              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>部署名称</Text>
                <Input style={{ marginTop: 4 }} value={name} onChange={e => setName(e.target.value)} />
              </div>

              <JsonBlock title="融合配置（预览）" value={ensemblePreview} />

              <Alert
                type="warning" showIcon
                message="权重是起点，不是结论"
                description={
                  <span>
                    这里的权重按各模型的<b>选择分</b>推出来，而选择分是在选择集上得到的 ——
                    用它定权重等于又看了一次同一份数据。等权平均是个很难打败的基线，
                    融合是否真的更好，要拿封存测试集单独评一次才算数。
                  </span>
                }
              />

              <Space>
                <Tooltip title="后端尚未实现：需要 ensemble_deployments / ensemble_members 两张表，以及扇出推理">
                  <Button type="primary" icon={<CloudUploadOutlined />} disabled>
                    创建融合部署
                  </Button>
                </Tooltip>
                <Button icon={<CopyOutlined />}
                  onClick={() => copyText(JSON.stringify(ensemblePreview, null, 2), '融合配置')}>
                  复制配置
                </Button>
              </Space>
            </>
          )}
        </Space>
      </Card>
    </Space>
  )
}

// ---------------------------------------------------------------------------

/**
 * Workflow 部署 step — 单模型部署 / 多模型部署.
 *
 * props: task, runs (array), bestRunId
 */
export default function DeployStep({ task, runs = [], bestRunId, fillHeight = false }) {
  const successRuns = useMemo(
    () => runs.filter(r => String(r.status).toUpperCase() === 'SUCCESS' && r.domain_task_id),
    [runs]
  )
  const [tab, setTab] = useState('single')
  const [schema, setSchema] = useState({
    loading: true, requestExample: null, responseExample: null, notes: [], featureCount: 0,
  })

  // The request/response contract comes from the dataset's own preview — real
  // column names and a real row, rather than invented placeholders.
  useEffect(() => {
    if (!task?.dataset_id) { setSchema(s => ({ ...s, loading: false })); return undefined }
    let cancelled = false
    setSchema(s => ({ ...s, loading: true }))
    dataApi.previewDataset(task.dataset_id)
      .then((preview) => {
        if (cancelled) return
        const columnsInfo = preview?.columns_info || {}
        const columnNames = Object.keys(columnsInfo)
        const requestExample = buildRequestExample({
          sampleRow: preview?.rows?.[0] || {},
          columnsInfo,
          columnNames,
          targetColumn: task.target_column,
        })
        const hasTextFeatures = Object.entries(columnsInfo).some(
          ([n, info]) => n !== task.target_column && String(info?.dtype || '').startsWith('object'))
        setSchema({
          loading: false,
          requestExample,
          responseExample: buildResponseExample({ taskType: task.task_type }),
          notes: deploymentNotes({ hasTextFeatures }),
          featureCount: Object.keys(requestExample.rows[0] || {}).length,
        })
      })
      .catch(() => {
        if (cancelled) return
        setSchema({
          loading: false,
          requestExample: { rows: [{}], include_probabilities: true },
          responseExample: buildResponseExample({ taskType: task?.task_type }),
          notes: deploymentNotes({}),
          featureCount: 0,
        })
      })
    return () => { cancelled = true }
  }, [task?.dataset_id, task?.target_column, task?.task_type])

  if (!successRuns.length) {
    return (
      <Card size="small" styles={{ body: { padding: 32 } }}>
        <Empty description={
          <Text type="secondary">还没有训练成功的 Run，完成「训练」步骤后即可部署。</Text>
        } />
      </Card>
    )
  }

  const paneStyle = fillHeight
    ? { height: '100%', overflowY: 'auto', paddingRight: 4 }
    : undefined

  return (
    <>
      {fillHeight && (
        <style>{`
          .deploy-fill { display: flex; flex-direction: column; }
          .deploy-fill > .ant-tabs-content-holder,
          .deploy-fill .ant-tabs-content,
          .deploy-fill .ant-tabs-tabpane {
            flex: 1;
            min-height: 0;
            height: 100%;
          }
        `}</style>
      )}
    <Tabs
      activeKey={tab}
      onChange={setTab}
      // In a fixed frame the tab body is the scroller, so the pane fills the
      // frame and nothing below it gets pushed off the page.
      style={fillHeight ? { height: '100%' } : undefined}
      className={fillHeight ? 'deploy-fill' : undefined}
      items={[
        {
          key: 'single',
          label: <span><CloudUploadOutlined /> 单模型部署</span>,
          children: (
            <div style={paneStyle}>
              <SingleDeployTab task={task} successRuns={successRuns}
                bestRunId={bestRunId} schema={schema} />
            </div>
          ),
        },
        {
          key: 'multi',
          label: <span><BlockOutlined /> 多模型部署</span>,
          children: (
            <div style={paneStyle}>
              <MultiDeployTab task={task} successRuns={successRuns} bestRunId={bestRunId} />
            </div>
          ),
        },
      ]}
    />
    </>
  )
}
