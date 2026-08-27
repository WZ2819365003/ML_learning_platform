import React, { useMemo, useState, useEffect, useCallback } from 'react'
import {
  Card, Select, Input, Button, Space, Tag, Typography, Alert, Descriptions,
  message, Empty, Divider, Tooltip, Tabs, Table, InputNumber, Progress, Row, Col,
  Collapse,
} from 'antd'
import {
  CloudUploadOutlined, DownloadOutlined, ThunderboltOutlined, TrophyOutlined,
  CopyOutlined, ApiOutlined, BlockOutlined, ReloadOutlined,
} from '@ant-design/icons'
import { dataApi, deployApi, ensembleApi, runModelDownloadUrl } from '../../services/api'
import { useDeployRun } from '../../hooks/useDeployRun'
import {
  buildCurl, buildRequestExample, buildResponseExample, deploymentNotes,
  normaliseWeights, predictUrl, suggestWeights,
} from './deploySchema'

const { Text } = Typography

/**
 * Copy to clipboard, with a fallback for pages served over plain HTTP.
 *
 * `navigator.clipboard` only exists in a secure context — HTTPS or localhost.
 * This app is reached over http:// on a bare IP, so on the deployed site the
 * API is simply `undefined` and every copy button silently did nothing but
 * show "请手动复制". The execCommand path is deprecated but still works
 * everywhere, and it is the only option left on an insecure origin.
 */
async function copyText(value, label = '内容') {
  const text = String(value ?? '')

  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      message.success(`${label}已复制`)
      return true
    } catch {
      // Permission denied or a non-focused document — fall through.
    }
  }

  try {
    const ta = document.createElement('textarea')
    ta.value = text
    // Keep it off-screen but still selectable; display:none would not copy.
    ta.setAttribute('readonly', '')
    ta.style.position = 'fixed'
    ta.style.top = '-1000px'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    ta.setSelectionRange(0, text.length)
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    if (ok) {
      message.success(`${label}已复制`)
      return true
    }
  } catch {
    // fall through to the failure message
  }

  message.warning('复制失败，请手动选中复制')
  return false
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
  const [description, setDescription] = useState('')
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
    await deploy(runId, { name, description })
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
      <Collapse
        size="small"
        defaultActiveKey={['pick']}
        items={[{
          key: 'pick',
          label: (
            <Space size={8}>
              <CloudUploadOutlined />
              <Text strong style={{ fontSize: 13 }}>选择要上线的模型</Text>
              {selectedRun && (
                <Text type="secondary" style={{ fontSize: 11 }}>
                  {selectedRun.params?.model_type || selectedRun.family} · {name}
                </Text>
              )}
            </Space>
          ),
          children: (
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
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>
              模型说明（可选，会写入部署记录）
            </Text>
            <Input.TextArea
              style={{ marginTop: 4 }} rows={2} value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="例：用于日前负荷预测，输入为前一日 48 点负荷与气象特征；上线人 张三" />
          </div>
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
          ),
        }]}
      />

      {/* Collapsed by default. The contract matters when you sit down to write
          a client, not while you are picking a model — folded away it costs
          one line instead of half the page. It stays available before
          deploying because only the URL depends on the deployment. */}
      <Collapse
        size="small"
        items={[{
          key: 'contract',
          label: (
            <Space size={8}>
              <ApiOutlined />
              <Text strong style={{ fontSize: 13 }}>接口契约与调用说明</Text>
              <Text type="secondary" style={{ fontSize: 11 }}>
                入参 / 出参 / curl / 部署说明
              </Text>
              {!deployment && (
                <Text type="secondary" style={{ fontSize: 11 }}>· 部署后 URL 才会确定</Text>
              )}
            </Space>
          ),
          children: schema.loading ? (
            <Text type="secondary">正在读取数据集字段…</Text>
          ) : (
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Descriptions column={{ xs: 1, lg: 3 }} size="small" bordered
                labelStyle={{ background: '#f8fafc' }}>
                <Descriptions.Item label="请求方法"><code>POST</code></Descriptions.Item>
                <Descriptions.Item label="端点" span={2}>
                  <code style={{ fontSize: 12 }}>{predictUrl(deployment?.deployment_id)}</code>
                </Descriptions.Item>
                <Descriptions.Item label="特征列数" span={3}>
                  {schema.featureCount} 列（目标列 <code>{task?.target_column}</code> 不要传）
                </Descriptions.Item>
              </Descriptions>

              {/* Request and response are read together, so they share a row. */}
              <Row gutter={[12, 12]}>
                <Col xs={24} lg={12}>
                  <JsonBlock title="入参 JSON" value={schema.requestExample} />
                </Col>
                <Col xs={24} lg={12}>
                  <JsonBlock title="出参 JSON" value={schema.responseExample} />
                </Col>
              </Row>

              {/* curl gets the full width — it is one long line per flag and
                  wrapping it into half a column made it unreadable. */}
              <JsonBlock title="curl 示例" value={curl} height={130} />

              <div>
                <Text strong style={{ fontSize: 12 }}>部署说明</Text>
                <ul style={{
                  margin: '6px 0 0', paddingLeft: 18, fontSize: 12, color: '#475569',
                  lineHeight: 1.85,
                }}>
                  {schema.notes.map((n, i) => <li key={i}>{n}</li>)}
                </ul>
              </div>
            </Space>
          ),
        }]}
      />

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

function MultiDeployTab({ task, successRuns, bestRunId, schema }) {
  const direction = task?.objective_direction === 'min' ? 'min' : 'max'
  const [selectedIds, setSelectedIds] = useState([])
  const [weights, setWeights] = useState({})
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [creating, setCreating] = useState(false)
  const [ensemble, setEnsemble] = useState(null)
  const [predictInput, setPredictInput] = useState('[\n  {}\n]')
  // Same body as a single-model call: the ensemble sends this row to every
  // member and combines their answers, so the request contract is unchanged.

  const [predicting, setPredicting] = useState(false)
  const [predictResult, setPredictResult] = useState(null)

  useEffect(() => { if (task?.name) setName(`${task.name}-融合部署`) }, [task?.name])

  useEffect(() => {
    if (schema?.requestExample) {
      setPredictInput(JSON.stringify(schema.requestExample.rows, null, 2))
    }
  }, [schema?.requestExample])

  const selected = useMemo(
    () => successRuns.filter(r => selectedIds.includes(r.run_id)),
    [successRuns, selectedIds],
  )
  const enoughMembers = selected.length >= 2

  const applySuggested = useCallback((members) => {
    setWeights(suggestWeights(members, direction))
  }, [direction])

  // Re-suggest whenever the membership changes; manual edits survive until then.
  useEffect(() => { applySuggested(selected) }, [selectedIds]) // eslint-disable-line react-hooks/exhaustive-deps

  const normalised = useMemo(() => normaliseWeights(weights), [weights])
  const weightSum = Object.values(weights).reduce((a, b) => a + (b || 0), 0)

  const ensemblePreview = useMemo(() => ({
    name,
    description,
    strategy: 'weighted_average',
    members: selected.map(r => ({
      run_id: r.run_id,
      model_type: r.params?.model_type || r.family,
      family: r.family || 'ml',
      weight: Number((normalised[r.run_id] ?? 0).toFixed(4)),
    })),
  }), [name, description, selected, normalised])

  const handleCreate = async () => {
    if (!name.trim()) { message.warning('请填写部署名称'); return }
    setCreating(true)
    setPredictResult(null)
    try {
      const resp = await ensembleApi.create({
        modeling_task_id: task.id,
        name: name.trim(),
        description: description.trim() || undefined,
        members: selected.map(r => ({
          domain_task_id: r.domain_task_id,
          family: r.family || 'ml',
          weight: normalised[r.run_id] ?? 0,
          run_id: r.run_id,
          model_type: r.params?.model_type || r.family,
        })),
      })
      setEnsemble(resp)
      message.success('融合部署已创建')
    } catch (err) {
      message.error(err?.response?.data?.detail || '创建融合部署失败')
    } finally {
      setCreating(false)
    }
  }

  const handlePredict = async () => {
    if (!ensemble?.id) return
    let parsed
    try {
      parsed = JSON.parse(predictInput)
      if (!Array.isArray(parsed)) throw new Error('need array')
    } catch {
      message.error('预测输入需为 JSON 数组')
      return
    }
    setPredicting(true)
    try {
      setPredictResult(await ensembleApi.predict(ensemble.id, { rows: parsed }))
    } catch (err) {
      message.error(err?.response?.data?.detail || '融合预测失败')
    } finally {
      setPredicting(false)
    }
  }

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
      {/* Mirrors 单模型部署: the picker panel always holds the picker, the
          name and the note, so the form is complete before anything is chosen
          rather than appearing once a second model is picked. */}
      <Collapse
        size="small"
        defaultActiveKey={['members']}
        items={[{
          key: 'members',
          label: (
            <Space size={8}>
              <BlockOutlined />
              <Text strong style={{ fontSize: 13 }}>选择参与融合的模型（至少 2 个）</Text>
              {selected.length > 0 && (
                <Text type="secondary" style={{ fontSize: 11 }}>已选 {selected.length} 个 · {name}</Text>
              )}
            </Space>
          ),
          children: (
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Select
                mode="multiple" style={{ width: '100%' }} value={selectedIds} onChange={setSelectedIds}
                placeholder="从训练成功的 Run 里挑选"
                options={successRuns.map(r => ({
                  value: r.run_id,
                  label: <RunLabel run={r} bestRunId={bestRunId} objectiveMetric={task?.objective_metric} />,
                }))}
              />

              {/* Weights belong with the thing they configure, directly above
                  the button that submits them — a separate panel made the user
                  set them somewhere else and then come back here to deploy. */}
              {enoughMembers && (
                <>
                  <Table size="small" rowKey="run_id" columns={columns} dataSource={selected}
                    pagination={false} />
                  <Space wrap>
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
                </>
              )}

              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>部署名称</Text>
                <Input style={{ marginTop: 4 }} value={name} onChange={e => setName(e.target.value)} />
              </div>
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  模型说明（可选，会写入部署记录）
                </Text>
                <Input.TextArea style={{ marginTop: 4 }} rows={2} value={description}
                  onChange={e => setDescription(e.target.value)}
                  placeholder="例：xgboost 与 lstm 加权融合，权重按 rmse 反比；上线人 张三" />
              </div>

              <Space>
                <Tooltip title={enoughMembers ? '' : '至少选择 2 个模型'}>
                  <span style={{ display: 'inline-flex' }}>
                    <Button type="primary" icon={<CloudUploadOutlined />}
                      loading={creating} disabled={!enoughMembers}
                      onClick={handleCreate}>
                      创建融合部署
                    </Button>
                  </span>
                </Tooltip>
                <Button icon={<CopyOutlined />} disabled={!enoughMembers}
                  onClick={() => copyText(JSON.stringify(ensemblePreview, null, 2), '融合配置')}>
                  复制配置
                </Button>
              </Space>
            </Space>
          ),
        }]}
      />

      {ensemble && (
        <Card size="small" styles={{ body: { padding: 16 } }}
          title={<span><ThunderboltOutlined style={{ color: '#10b981' }} /> 已上线 · 融合推理</span>}>
          <Descriptions column={1} size="small" bordered labelStyle={{ width: 110, background: '#f8fafc' }}>
            <Descriptions.Item label="部署 ID"><code>{ensemble.id}</code></Descriptions.Item>
            <Descriptions.Item label="端点">
              <code style={{ fontSize: 12 }}>
                {predictUrl(null).replace('/inference/{deployment_id}/predict',
                  `/inference/ensembles/${ensemble.id}/predict`)}
              </code>
            </Descriptions.Item>
            <Descriptions.Item label="成员">
              <Space size={4} wrap>
                {(ensemble.members || []).map(m => (
                  <Tag key={m.id} style={{ margin: 0 }}>
                    {m.model_type} · {(m.weight * 100).toFixed(0)}%
                  </Tag>
                ))}
              </Space>
            </Descriptions.Item>
          </Descriptions>

          <Divider style={{ margin: '14px 0 10px' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>快速预测（JSON 行数组）</Text>
          </Divider>
          <Input.TextArea rows={4} value={predictInput} onChange={e => setPredictInput(e.target.value)}
            style={{ fontFamily: 'monospace', fontSize: 12 }} />
          <Button style={{ marginTop: 8 }} loading={predicting} onClick={handlePredict}>调用融合预测</Button>

          {predictResult && (
            <>
              {/* A member can fail at call time; the blend then runs on the
                  survivors with renormalised weights. That is a different model
                  from the one configured, so it is reported rather than hidden. */}
              {predictResult.members_failed?.length > 0 && (
                <Alert style={{ marginTop: 10 }} type="warning" showIcon
                  message={`${predictResult.members_failed.length} 个成员本次未参与，权重已在剩余成员间重新归一化`}
                  description={
                    <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12 }}>
                      {predictResult.members_failed.map((f, i) => (
                        <li key={i}>{f.model_type || f.domain_task_id}：{f.error}</li>
                      ))}
                    </ul>
                  } />
              )}
              <Alert style={{ marginTop: 10 }} type="success" showIcon message="融合预测结果"
                description={<pre style={{ margin: 0, fontSize: 12, whiteSpace: 'pre-wrap' }}>
                  {JSON.stringify(predictResult, null, 2)}
                </pre>} />
            </>
          )}
        </Card>
      )}
    </Space>
  )
}

// ---------------------------------------------------------------------------

/**
 * Workflow 部署 step — 单模型部署 / 多模型部署.
 *
 * props: task, runs (array), bestRunId
 */
// One height for both 部署 tabs, so switching between them does not resize the
// step. Taller than the folded content needs: 多模型部署 with members picked
// carries a weights table the single tab has no equivalent of.
const DEPLOY_TAB_BODY_HEIGHT = 700

export default function DeployStep({ task, runs = [], bestRunId }) {
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

  return (
    <Tabs
      activeKey={tab}
      onChange={setTab}
      items={[
        {
          key: 'single',
          label: <span><CloudUploadOutlined /> 单模型部署</span>,
          children: (
            <div style={{ height: DEPLOY_TAB_BODY_HEIGHT, overflowY: 'auto', paddingRight: 4 }}>
              <SingleDeployTab task={task} successRuns={successRuns}
                bestRunId={bestRunId} schema={schema} />
            </div>
          ),
        },
        {
          key: 'multi',
          label: <span><BlockOutlined /> 多模型部署</span>,
          children: (
            <div style={{ height: DEPLOY_TAB_BODY_HEIGHT, overflowY: 'auto', paddingRight: 4 }}>
              <MultiDeployTab task={task} successRuns={successRuns}
                bestRunId={bestRunId} schema={schema} />
            </div>
          ),
        },
      ]}
    />
  )
}
