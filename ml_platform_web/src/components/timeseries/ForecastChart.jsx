import React, { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

function buildLabelSeries(result) {
  const historical = result?.historical ?? []
  const forecast = result?.point_forecast ?? []
  const historicalAxis = result?.time_axis?.historical

  const historyLabels = historicalAxis?.length
    ? historicalAxis.slice(-historical.length)
    : historical.map((_, index) => `t-${historical.length - index}`)

  const forecastLabels = forecast.map((_, index) => `t+${index + 1}`)

  return {
    historical,
    forecast,
    q10: result?.q10 ?? [],
    q90: result?.q90 ?? [],
    labels: [...historyLabels, ...forecastLabels],
  }
}

export default function ForecastChart({ result }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || !result) {
      return undefined
    }

    const chart = echarts.init(containerRef.current)
    const series = buildLabelSeries(result)

    chart.setOption(
      {
        color: ['#0f4c81', '#d97706', '#d97706', '#d97706'],
        tooltip: {
          trigger: 'axis',
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          borderWidth: 0,
          textStyle: { color: '#f8fafc' },
        },
        legend: {
          bottom: 0,
          itemGap: 18,
          data: ['历史值', '预测值', '置信上界', '置信下界'],
        },
        grid: {
          left: 52,
          right: 24,
          top: 28,
          bottom: 64,
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: series.labels,
        },
        yAxis: {
          type: 'value',
          splitLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.18)' } },
        },
        series: [
          {
            name: '历史值',
            type: 'line',
            smooth: true,
            symbol: 'none',
            lineStyle: { width: 3 },
            data: [...series.historical, ...series.forecast.map(() => null)],
          },
          {
            name: '预测值',
            type: 'line',
            smooth: true,
            symbol: 'none',
            lineStyle: { width: 3, type: 'dashed' },
            data: [...series.historical.map(() => null), ...series.forecast],
          },
          {
            name: '置信上界',
            type: 'line',
            smooth: true,
            symbol: 'none',
            lineStyle: { opacity: 0 },
            data: [...series.historical.map(() => null), ...series.q90],
            stack: 'confidence-band',
            areaStyle: {
              color: 'rgba(217, 119, 6, 0.14)',
            },
          },
          {
            name: '置信下界',
            type: 'line',
            smooth: true,
            symbol: 'none',
            lineStyle: { opacity: 0 },
            data: [...series.historical.map(() => null), ...series.q10],
            stack: 'confidence-band',
            areaStyle: {
              color: '#ffffff',
            },
          },
        ],
      },
      true,
    )

    const handleResize = () => chart.resize()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.dispose()
    }
  }, [result])

  return <div ref={containerRef} style={{ width: '100%', height: 320 }} />
}
