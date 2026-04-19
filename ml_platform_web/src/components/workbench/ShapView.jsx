/**
 * ShapView — professional SHAP visualisation for a single Run.
 *
 * Three views toggled by a segmented control:
 *   1. 重要度条形 (bar)   — mean(|SHAP|) per feature, sorted descending.
 *   2. Beeswarm              — each dot is one sample; color = feature value, x = SHAP.
 *   3. 依赖散点 (dependence)  — feature value vs SHAP value for a selected feature.
 *
 * Data flows from GET /platform/runs/{run_id}/shap:
 *   { status, feature_importances, feature_names, samples: { feature_values, shap_values } }
 *
 * Graceful degradation:
 *   - No payload yet → "pending" state with hint message.
 *   - Aggregated only (inline) → bar-only view, other tabs disabled.
 *   - Full MinIO payload → all three views active.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert, Button, Empty, message, Segmented, Select, Skeleton, Space, Tag, Typography,
} from 'antd'
import { ThunderboltOutlined, ReloadOutlined } from '@ant-design/icons'
import { platformRunsApi, platformExperimentsApi } from '../../services/api'
import EChart from '../EChart'

const { Text } = Typography

// ── Helpers ──────────────────────────────────────────────────────────────

function toTopN(importances, n = 15) {
  const entries = Object.entries(importances || {})
    .map(([f, v]) => [f, Number(v) || 0])
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
  return entries.slice(0, n)
}

function minMax(arr) {
  let lo = Infinity, hi = -Infinity
  for (const v of arr) {
    if (typeof v === 'number' && !Number.isNaN(v)) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  if (!Number.isFinite(lo)) return [0, 1]
  if (lo === hi) return [lo - 1, hi + 1]
  return [lo, hi]
}

// ── Chart options ────────────────────────────────────────────────────────

function buildBarOption(importances) {
  const top = toTopN(importances, 15).reverse() // reverse so largest at top
  const features = top.map(([f]) => f)
  const values = top.map(([, v]) => Number(v))
  return {
    grid: { left: 160, right: 40, top: 16, bottom: 32 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
      valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(6) : v) },
    xAxis: { type: 'value', name: 'mean(|SHAP|)', nameGap: 22,
      nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#475569' } },
    yAxis: { type: 'category', data: features, axisLabel: { color: '#0f172a', fontSize: 12 } },
    series: [{
      type: 'bar',
      data: values,
      itemStyle: {
        color: (p) => {
          // Gradient from muted to vivid blue based on rank
          const pct = p.dataIndex / Math.max(1, features.length - 1)
          const mix = Math.round(60 + 140 * pct)
          return `rgb(37, 99, 235, ${0.4 + 0.6 * pct})`.replace(
            'rgb(', 'rgba(',
          )
        },
      },
      barWidth: 14,
      label: { show: true, position: 'right', color: '#2563eb',
        formatter: (p) => p.value.toFixed(4), fontSize: 11 },
    }],
  }
}

function buildBeeswarmOption(samples, featureNames, importances) {
  // Pick top 10 features by importance for readability.
  const ranked = toTopN(importances, 10)
  const nameToIdx = new Map(featureNames.map((f, i) => [f, i]))

  const sv = samples.shap_values // [n_samples][n_features]
  const fv = samples.feature_values // [n_samples][n_features]

  // Per-feature jitter bucket so markers don't overlap.
  const series = ranked.map(([feature], rowIdx) => {
    const fi = nameToIdx.get(feature)
    if (fi == null) return null
    const col = sv.map((row) => row[fi])
    const values = fv.map((row) => row[fi])
    const [vmin, vmax] = minMax(values.filter((v) => v != null))

    // Deterministic jitter: hash of sample index → small Y offset.
    const data = col.map((shap, si) => {
      const val = values[si]
      const jitter = ((si * 2654435761) % 100) / 100 - 0.5 // [-0.5, 0.5]
      return {
        value: [shap, rowIdx + jitter * 0.7],
        featureValue: val,
        feature,
        _colorVal: val == null ? 0.5 : (val - vmin) / Math.max(1e-9, vmax - vmin),
      }
    })

    return {
      type: 'scatter',
      name: feature,
      data,
      symbolSize: 5,
      itemStyle: {
        color: (p) => {
          const t = p.data._colorVal
          // Blue → red diverging scale (matches SHAP's default)
          const r = Math.round(59 + (239 - 59) * t)
          const g = Math.round(130 + (68 - 130) * t)
          const b = Math.round(246 + (68 - 246) * t)
          return `rgba(${r},${g},${b},0.65)`
        },
      },
      emphasis: { itemStyle: { borderColor: '#0f172a', borderWidth: 1 } },
    }
  }).filter(Boolean)

  return {
    grid: { left: 140, right: 80, top: 20, bottom: 50 },
    tooltip: {
      trigger: 'item',
      formatter: (p) => {
        const d = p.data
        const fv = d.featureValue == null ? '—' : Number(d.featureValue).toFixed(4)
        return `<b>${d.feature}</b><br/>SHAP = ${Number(d.value[0]).toFixed(4)}<br/>特征值 = ${fv}`
      },
    },
    xAxis: {
      type: 'value', name: 'SHAP value', nameGap: 28,
      axisLine: { show: true, lineStyle: { color: '#94a3b8' } },
      splitLine: { lineStyle: { color: '#e2e8f0' } },
      axisLabel: { color: '#475569' },
    },
    yAxis: {
      type: 'category',
      data: ranked.map(([f]) => f),
      axisLabel: { color: '#0f172a', fontSize: 12 },
    },
    // Colour legend (feature-value low → high)
    visualMap: {
      show: true,
      type: 'continuous',
      min: 0,
      max: 1,
      text: ['high', 'low'],
      textStyle: { color: '#64748b' },
      itemWidth: 10,
      itemHeight: 160,
      right: 10,
      top: 40,
      inRange: { color: ['#3b82f6', '#e0e7ff', '#ef4444'] },
      calculable: false,
      // We drive colour ourselves via itemStyle; visualMap here is legend-only.
      dimension: 2,
      seriesIndex: [],
    },
  }
}

function buildDependenceOption(samples, featureNames, importances, selected) {
  const nameToIdx = new Map(featureNames.map((f, i) => [f, i]))
  const fi = nameToIdx.get(selected)
  if (fi == null) return { series: [] }

  const sv = samples.shap_values
  const fv = samples.feature_values
  const xs = fv.map((row) => row[fi])
  const ys = sv.map((row) => row[fi])

  // Second-order: colour dots by the next-most-correlated feature for richer insight.
  // For MVP we just colour by the sample's own feature magnitude.
  const [vmin, vmax] = minMax(xs.filter((v) => v != null))
  const data = xs.map((x, i) => ({
    value: [x, ys[i]],
    _colorVal: x == null ? 0.5 : (x - vmin) / Math.max(1e-9, vmax - vmin),
  }))

  return {
    grid: { left: 70, right: 30, top: 30, bottom: 50 },
    tooltip: {
      trigger: 'item',
      formatter: (p) =>
        `特征值 = ${p.value[0] == null ? '—' : Number(p.value[0]).toFixed(4)}<br/>SHAP = ${Number(p.value[1]).toFixed(4)}`,
    },
    xAxis: {
      type: 'value', name: selected, nameGap: 28,
      axisLabel: { color: '#475569' },
      splitLine: { lineStyle: { color: '#e2e8f0' } },
    },
    yAxis: {
      type: 'value', name: 'SHAP value',
      axisLabel: { color: '#475569' },
      splitLine: { lineStyle: { color: '#e2e8f0' } },
    },
    series: [{
      type: 'scatter',
      data,
      symbolSize: 7,
      itemStyle: {
        color: (p) => {
          const t = p.data._colorVal
          const r = Math.round(59 + (239 - 59) * t)
          const g = Math.round(130 + (68 - 130) * t)
          const b = Math.round(246 + (68 - 246) * t)
          return `rgba(${r},${g},${b},0.7)`
        },
      },
    }],
  }
}

// ── Component ────────────────────────────────────────────────────────────

export default function ShapView({ runId, initialSummary, experimentId, runStatus }) {
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [view, setView] = useState('bar') // 'bar' | 'swarm' | 'dep'
  const [depFeature, setDepFeature] = useState(null)
  const [triggering, setTriggering] = useState(false)
  const pollRef = useRef(null)

  const fetchPayload = React.useCallback(() => {
    if (!runId) return Promise.resolve(null)
    setLoading(true)
    setError(null)
    return platformRunsApi.shap(runId)
      .then((resp) => { setPayload(resp); return resp })
      .catch((err) => { setError(err?.response?.data?.detail || err.message); return null })
      .finally(() => setLoading(false))
  }, [runId])

  useEffect(() => {
    void fetchPayload()
    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
    }
  }, [fetchPayload])

  const handleTrigger = async () => {
    if (!experimentId || !runId) {
      message.warning('缺少 experimentId — 无法触发 SHAP 计算')
      return
    }
    if (runStatus && runStatus !== 'SUCCESS') {
      message.warning('只能为成功完成的 Run 计算 SHAP')
      return
    }
    setTriggering(true)
    try {
      const resp = await platformExperimentsApi.triggerExplain(experimentId, runId)
      if (resp?.already_computed) {
        message.info('SHAP 已计算过，正在刷新')
        await fetchPayload()
        return
      }
      message.success('已提交 SHAP 任务 — 结果就绪后自动刷新')
      // Poll every 3s up to ~2 min
      let ticks = 0
      pollRef.current = setInterval(async () => {
        ticks += 1
        const p = await fetchPayload()
        if (p && p.status !== 'pending') {
          clearInterval(pollRef.current); pollRef.current = null
        } else if (ticks >= 40) {
          clearInterval(pollRef.current); pollRef.current = null
          message.warning('SHAP 计算仍在进行中，请稍后手动刷新')
        }
      }, 3000)
    } catch (err) {
      message.error(err?.response?.data?.detail || '触发 SHAP 失败')
    } finally {
      setTriggering(false)
    }
  }

  const importances = payload?.feature_importances || null
  const featureNames = payload?.feature_names || (importances ? Object.keys(importances) : [])
  const samples = payload?.samples || null
  const hasSamples = !!(samples && samples.shap_values && samples.feature_values)

  // Preselect top-1 feature for dependence view once payload arrives.
  useEffect(() => {
    if (!depFeature && importances) {
      const top = toTopN(importances, 1)
      if (top.length) setDepFeature(top[0][0])
    }
  }, [importances, depFeature])

  const barOption = useMemo(
    () => (importances ? buildBarOption(importances) : null),
    [importances],
  )
  const swarmOption = useMemo(
    () => (hasSamples ? buildBeeswarmOption(samples, featureNames, importances) : null),
    [samples, featureNames, importances, hasSamples],
  )
  const depOption = useMemo(
    () => (hasSamples && depFeature
      ? buildDependenceOption(samples, featureNames, importances, depFeature)
      : null),
    [samples, featureNames, importances, depFeature, hasSamples],
  )

  if (loading) return <Skeleton active paragraph={{ rows: 6 }} />

  // No payload at all — fall back to inline summary from inspector aggregate.
  if (error) {
    return <Alert type="error" showIcon message="加载 SHAP 失败" description={error} />
  }

  const isPending = !payload || payload.status === 'pending'
  if (isPending) {
    const canTrigger = !!experimentId && (!runStatus || runStatus === 'SUCCESS')
    return (
      <Empty
        description={
          <div style={{ textAlign: 'center' }}>
            <div style={{ marginBottom: 6 }}>该 Run 暂无 SHAP 解释</div>
            <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 12 }}>
              实验完成后会自动为 Top-3 Run 计算 SHAP；其他 Run 可手动触发
            </Text>
            <Space>
              <Button
                type="primary"
                icon={<ThunderboltOutlined />}
                loading={triggering}
                disabled={!canTrigger}
                onClick={handleTrigger}
              >
                立即触发 SHAP
              </Button>
              <Button icon={<ReloadOutlined />} onClick={fetchPayload} loading={loading}>
                刷新
              </Button>
            </Space>
            {!canTrigger && runStatus && runStatus !== 'SUCCESS' && (
              <div style={{ marginTop: 8, fontSize: 11, color: '#f59e0b' }}>
                仅成功完成的 Run 支持计算 SHAP（当前状态 {runStatus}）
              </div>
            )}
          </div>
        }
      />
    )
  }

  const method = payload.explanation_method || 'shap'
  const sampleSize = payload.sample_size
  const featureCount = payload.feature_count ?? featureNames.length

  return (
    <div>
      <Space wrap style={{ marginBottom: 12 }}>
        <Tag color={method === 'shap' ? 'blue' : 'orange'}>
          method: <code>{method}</code>
        </Tag>
        <Tag>{featureCount} 特征</Tag>
        {sampleSize != null && <Tag>{sampleSize} 样本</Tag>}
        {payload.source && <Tag color={payload.source === 'minio' ? 'green' : 'default'}>
          source: {payload.source}
        </Tag>}
        {!hasSamples && (
          <Tag color="warning">仅聚合重要度（无 per-sample，Beeswarm/依赖不可用）</Tag>
        )}
      </Space>

      <Segmented
        size="small"
        value={view}
        onChange={setView}
        options={[
          { label: '重要度', value: 'bar' },
          { label: 'Beeswarm', value: 'swarm', disabled: !hasSamples },
          { label: '依赖散点', value: 'dep', disabled: !hasSamples },
        ]}
        style={{ marginBottom: 12 }}
      />

      {view === 'bar' && barOption && (
        <EChart option={barOption} style={{ height: 420 }} />
      )}

      {view === 'swarm' && swarmOption && (
        <EChart option={swarmOption} style={{ height: 440 }} />
      )}

      {view === 'dep' && (
        <>
          <Space style={{ marginBottom: 8 }}>
            <Text type="secondary" style={{ fontSize: 12 }}>选择特征:</Text>
            <Select
              size="small"
              value={depFeature}
              onChange={setDepFeature}
              style={{ minWidth: 200 }}
              options={toTopN(importances, 30).map(([f]) => ({ label: f, value: f }))}
            />
          </Space>
          {depOption && <EChart option={depOption} style={{ height: 400 }} />}
        </>
      )}
    </div>
  )
}
