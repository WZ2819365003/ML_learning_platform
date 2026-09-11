import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Segmented, Space, Typography } from '../../ui'
import echarts from '../../utils/echarts'
import { buildOption, buildSeries } from './forecastChartOption'

const { Text } = Typography

const MIN_HEIGHT = 280
const DEFAULT_HEIGHT = 420

export default function ForecastChart({ result }) {
  const containerRef = useRef(null)
  const horizon = result?.point_forecast?.length ?? 0
  const totalHistory = result?.historical?.length ?? 0

  const windowOptions = useMemo(() => {
    const focused = Math.max(horizon * 6, 48)
    const candidates = [
      { label: '聚焦预测段', value: focused },
      { label: '近 500 点', value: 500 },
      { label: '近 5000 点', value: 5000 },
    ].filter((item, index, list) => item.value < totalHistory
      && list.findIndex(other => other.value === item.value) === index)
    return [...candidates, { label: `全部 ${totalHistory.toLocaleString('zh-CN')} 点`, value: Infinity }]
  }, [horizon, totalHistory])

  const [windowSize, setWindowSize] = useState(() => Math.max(horizon * 6, 48))
  const [height, setHeight] = useState(DEFAULT_HEIGHT)

  useEffect(() => {
    setWindowSize(Math.max(horizon * 6, 48))
  }, [horizon])

  useEffect(() => {
    if (!containerRef.current || !result) return undefined

    const chart = echarts.init(containerRef.current)
    chart.setOption(buildOption(buildSeries(result, windowSize)), true)

    const handleResize = () => chart.resize()
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      chart.dispose()
    }
  }, [result, windowSize])

  // 容器自身可拖拽改高，utils/echarts 的 ResizeObserver 会跟着重绘。
  const onHandleDrag = (event) => {
    event.preventDefault()
    const startY = event.clientY
    const startHeight = containerRef.current?.clientHeight ?? height
    const onMove = (move) => {
      setHeight(Math.max(MIN_HEIGHT, Math.round(startHeight + move.clientY - startY)))
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  return (
    <div className="forecast-chart">
      <Space className="forecast-chart-toolbar" size={12} wrap>
        <Text type="secondary">历史窗口</Text>
        <Segmented
          size="small"
          value={windowSize}
          onChange={setWindowSize}
          options={windowOptions}
        />
      </Space>
      <div ref={containerRef} style={{ width: '100%', height }} />
      <div
        className="forecast-chart-handle"
        role="separator"
        aria-label="拖拽调整图表高度"
        onMouseDown={onHandleDrag}
      >
        <span />
        <Text type="secondary" className="forecast-chart-hint">
          上下拖拽调整高度 · 按住 Shift 滚轮可缩放横轴
        </Text>
      </div>
    </div>
  )
}
