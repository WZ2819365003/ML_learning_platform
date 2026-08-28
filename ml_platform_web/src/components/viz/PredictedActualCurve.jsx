/**
 * PredictedActualCurve — actual and predicted as two lines over the sample index.
 *
 * Distinct from the existing scatter view, which plots predicted against actual
 * with a diagonal: that answers "how tight is the fit overall", while this one
 * answers "where does it go wrong" — which peaks it misses, whether it lags,
 * whether error grows over the horizon. For a load-forecasting series that
 * second question is usually the one being asked.
 *
 * Feeds off the same `/viz/{id}/predicted_vs_actual` payload as the scatter, so
 * no extra request is needed when both are shown.
 */
import React, { useMemo, useState } from 'react'
import { Empty, Radio, Space, Typography } from 'antd'

import EChart from '../EChart'

const { Text } = Typography

// Plotting tens of thousands of points makes the chart unreadable and slow.
// The window is a view, not a summary — the numbers are never resampled.
const WINDOW_OPTIONS = [200, 500, 1000]
const EMPTY_SERIES = []

export function buildPredictedActualOption(actual = [], predicted = [], windowSize = 200, sampleOffset = 0) {
  const n = Math.min(actual.length, predicted.length)
  const size = Math.min(windowSize, n)
  const from = Math.max(0, n - size)          // most recent slice
  const idx = Array.from({ length: size }, (_, i) => sampleOffset + from + i + 1)

  return {
    grid: { left: 56, right: 20, top: 34, bottom: 44 },
    tooltip: {
      trigger: 'axis',
      valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(4) : v),
    },
    legend: { data: ['实际值', '预测值'], top: 0, itemWidth: 18 },
    xAxis: { type: 'category', data: idx, name: '样本序号', nameLocation: 'middle', nameGap: 26 },
    yAxis: { type: 'value', scale: true },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 6 }],
    series: [
      {
        name: '实际值', type: 'line', showSymbol: false, smooth: false,
        data: actual.slice(from, from + size),
        lineStyle: { width: 1.6, color: '#0f172a' },
      },
      {
        name: '预测值', type: 'line', showSymbol: false, smooth: false,
        data: predicted.slice(from, from + size),
        lineStyle: { width: 1.6, color: '#2563eb' },
      },
    ],
  }
}

export default function PredictedActualCurve({ payload, height = 320 }) {
  const [windowSize, setWindowSize] = useState(WINDOW_OPTIONS[0])

  const actual = Array.isArray(payload?.actual) ? payload.actual : EMPTY_SERIES
  const predicted = Array.isArray(payload?.predicted) ? payload.predicted : EMPTY_SERIES

  const option = useMemo(
    () => buildPredictedActualOption(actual, predicted, windowSize, payload?.sample_offset ?? 0),
    [actual, predicted, windowSize, payload?.sample_offset],
  )

  if (actual.length === 0 || predicted.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无预测/实际对比数据" />
  }

  const loaded = Math.min(actual.length, predicted.length)
  const total = Number(payload?.total_count) || loaded
  const shown = Math.min(windowSize, loaded)
  return (
    <div>
      <Space size={8} style={{ marginBottom: 6 }} wrap>
        <Text type="secondary" style={{ fontSize: 12 }}>显示最近</Text>
        <Radio.Group size="small" value={windowSize} onChange={e => setWindowSize(e.target.value)}
          options={WINDOW_OPTIONS.filter(w => w <= Math.max(loaded, WINDOW_OPTIONS[0]))
            .map(w => ({ label: w, value: w }))}
          optionType="button" buttonStyle="solid" />
        <Text type="secondary" style={{ fontSize: 12 }}>
          条（当前显示 {shown} 条）/ 已加载最近 {loaded} 条 / 测试集共 {total} 条，可框选缩放
        </Text>
      </Space>
      <EChart option={option} style={{ height }} />
    </div>
  )
}
