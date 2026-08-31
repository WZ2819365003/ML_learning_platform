import React, { useEffect, useState } from 'react'
import { Alert, Button, Card, Empty, Spin, Typography } from 'antd'
import { BulbOutlined } from '@ant-design/icons'

import EChart from '../EChart'
import ShapView from '../viz/ShapView'
import { vizApi } from '../../services/api'

const { Paragraph, Text } = Typography

/** Native feature importance — the model's own ranking, when it has one. */
function buildImportanceOption(payload) {
  const items = (payload?.features || payload?.feature_names || []).slice(0, 15)
  const values = (payload?.importances || payload?.values || []).slice(0, 15)
  if (items.length === 0 || values.length === 0) return null
  // Horizontal bars ascending, so the most important sits at the top.
  const pairs = items.map((f, i) => [String(f), Number(values[i]) || 0])
    .sort((a, b) => a[1] - b[1])
  return {
    grid: { left: 140, right: 24, top: 10, bottom: 30 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: pairs.map(p => p[0]), axisLabel: { fontSize: 11 } },
    series: [{
      type: 'bar', data: pairs.map(p => p[1]),
      itemStyle: { color: '#8b5cf6', borderRadius: [0, 4, 4, 0] },
    }],
  }
}

export default function ModelExplainPanel({ taskId, modelType }) {
  const [payload, setPayload] = useState(null)
  const [importance, setImportance] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    setPayload(null)
    setError(null)
    setLoading(false)
    if (!taskId) { setImportance(null); return undefined }
    let cancelled = false
    // Cheap and always available for models that expose it — no reason to make
    // the user press a button for this one. SHAP stays on demand.
    vizApi.getFeatureImportance(taskId)
      .then(resp => { if (!cancelled) setImportance(resp) })
      .catch(() => { if (!cancelled) setImportance(null) })
    return () => { cancelled = true }
  }, [taskId])

  async function loadShap() {
    setLoading(true)
    setError(null)
    try {
      setPayload(await vizApi.getShapSummary(
        taskId,
        { max_samples: 5 },
        { timeout: 180000 },
      ))
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || '模型解释计算失败')
    } finally {
      setLoading(false)
    }
  }

  const importanceOption = buildImportanceOption(importance)

  const nativeImportance = importanceOption ? (
    <Card size="small" variant="outlined" title="原生特征重要度 Top-15"
      style={{ marginTop: 12 }}>
      <EChart option={importanceOption} style={{ height: 340 }} />
      <Text type="secondary" style={{ fontSize: 12 }}>
        模型自带的重要度，反映特征被用于分裂的程度；SHAP 则给出每个特征对单条预测的贡献方向与大小。
      </Text>
    </Card>
  ) : null

  if (payload) {
    return (
      <>
        <ShapView payload={payload} />
        {nativeImportance}
      </>
    )
  }

  return (
    <Spin spinning={loading} tip="正在计算 SHAP 解释">
      {error ? (
        <Alert
          type="error"
          showIcon
          message="模型解释计算失败"
          description={error}
          action={<Button size="small" onClick={loadShap}>重试</Button>}
        />
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={(
            <div>
              <Text>按需计算 {modelType || '当前模型'} 的 SHAP 解释</Text>
              <Paragraph type="secondary" style={{ margin: '6px 0 0' }}>
                计算完成后展示特征贡献排序和方向；这里不再创建没有数据的原生特征重要性卡片。
              </Paragraph>
            </div>
          )}
          style={{ padding: '72px 20px' }}
        >
          <Button type="primary" icon={<BulbOutlined />} onClick={loadShap}>
            计算模型解释
          </Button>
        </Empty>
      )}
      {nativeImportance}
    </Spin>
  )
}
