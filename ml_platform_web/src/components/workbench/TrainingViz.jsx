/**
 * TrainingViz — detailed training-visualisation dashboard for a Run.
 *
 * Rendered inside RunInspector as the "训练可视化" tab.  Drives all the
 * viz_service endpoints off `training_task.id` (not run.id — the backend
 * helpers re-load the model + dataset on the fly).
 *
 * Layout (ordered by diagnostic importance):
 *
 *   ┌─────────────┬─────────────┐
 *   │ 核心性能图  │ Learning    │   ← classification: 混淆矩阵
 *   │ (CM / 残差) │ Curve       │     regression: 残差散点
 *   ├─────────────┼─────────────┤
 *   │ ROC /       │ 特征重要度  │
 *   │ 预测 vs 真实│ Top-10      │
 *   └─────────────┴─────────────┘
 *
 *  - Each chart is in its own Card with a uniform 16-px gutter and a
 *    360-px canvas so the drawer stays readable.
 *  - Missing data renders an <Empty /> placeholder (endpoints return 4xx
 *    for non-applicable model types — we swallow errors silently).
 *  - A single 刷新 button re-runs every query in parallel.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Card, Row, Col, Empty, Spin, Button, Space, Tag, Tooltip, Alert, Typography,
} from 'antd'
import {
  ReloadOutlined,
  HeatMapOutlined,
  LineChartOutlined,
  BarChartOutlined,
  DotChartOutlined,
  RadarChartOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons'
import EChart from '../EChart'
import { vizApi } from '../../services/api'

const { Text } = Typography

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function inferTaskType(modelType) {
  if (!modelType) return 'classification'
  const m = String(modelType).toLowerCase()
  if (m.includes('regress') || m.endsWith('regressor')) return 'regression'
  return 'classification'
}

function safeFetch(promise) {
  return promise.then((v) => v).catch(() => null)
}

// ─────────────────────────────────────────────────────────────────────────────
// ECharts option builders
// ─────────────────────────────────────────────────────────────────────────────

function buildConfusionMatrixOption(cm) {
  if (!cm?.matrix || !cm?.labels) return null
  const max = Math.max(...cm.matrix.flat())
  return {
    tooltip: {
      position: 'top',
      formatter: ({ value }) =>
        `真实=${cm.labels[value[1]]} · 预测=${cm.labels[value[0]]}<br/>计数=<b>${value[2]}</b>`,
    },
    grid: { height: '68%', top: 36, left: 72, right: 28, bottom: 56 },
    xAxis: {
      type: 'category', data: cm.labels, name: '预测类别',
      nameGap: 28, nameLocation: 'middle',
    },
    yAxis: { type: 'category', data: cm.labels, name: '真实类别', nameGap: 40 },
    visualMap: {
      min: 0, max: max || 1, calculable: true,
      orient: 'horizontal', left: 'center', bottom: 4,
      inRange: { color: ['#e0f2fe', '#2563eb', '#1e3a8a'] },
    },
    series: [{
      type: 'heatmap',
      data: cm.matrix.flatMap((row, ri) => row.map((v, ci) => [ci, ri, v])),
      label: { show: true, color: '#0f172a' },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' } },
    }],
  }
}

function buildRocCurveOption(roc) {
  if (!roc) return null
  const baseline = {
    name: '随机基线', type: 'line',
    data: [[0, 0], [1, 1]],
    lineStyle: { type: 'dashed', color: '#94a3b8' }, symbol: 'none',
    tooltip: { show: false },
  }
  const curves = roc.multiclass
    ? (roc.curves || []).map((c) => ({
        name: `${c.class} (AUC ${Number(c.auc).toFixed(3)})`,
        type: 'line', smooth: true, showSymbol: false,
        data: c.fpr.map((v, i) => [v, c.tpr[i]]),
      }))
    : [{
        name: `ROC (AUC ${Number(roc.auc).toFixed(3)})`,
        type: 'line', smooth: true, showSymbol: false,
        areaStyle: { color: 'rgba(37,99,235,0.12)' },
        lineStyle: { color: '#2563eb', width: 2 },
        data: (roc.fpr || []).map((v, i) => [v, roc.tpr[i]]),
      }]
  return {
    tooltip: { trigger: 'axis' },
    legend: { bottom: 4, textStyle: { fontSize: 11 } },
    grid: { top: 24, left: 56, right: 24, bottom: 56 },
    xAxis: { type: 'value', min: 0, max: 1, name: 'FPR', nameGap: 24, nameLocation: 'middle' },
    yAxis: { type: 'value', min: 0, max: 1, name: 'TPR' },
    series: [...curves, baseline],
  }
}

function buildFeatureImportanceOption(fi) {
  if (!fi?.features?.length) return null
  const N = Math.min(10, fi.features.length)
  const features = [...fi.features].slice(0, N).reverse()
  const importance = [...fi.importance].slice(0, N).reverse()
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { top: 16, left: 140, right: 24, bottom: 24 },
    xAxis: { type: 'value', name: '重要性' },
    yAxis: {
      type: 'category', data: features,
      axisLabel: { fontSize: 11 },
    },
    series: [{
      type: 'bar',
      data: importance,
      itemStyle: {
        color: {
          type: 'linear', x: 0, y: 0, x2: 1, y2: 0,
          colorStops: [{ offset: 0, color: '#0f766e' }, { offset: 1, color: '#38bdf8' }],
        },
        borderRadius: [0, 4, 4, 0],
      },
      label: { show: true, position: 'right', fontSize: 10, formatter: (p) => Number(p.value).toFixed(3) },
    }],
  }
}

function buildLearningCurveOption(lc) {
  if (!lc?.steps?.length) return null
  const knownKeys = ['accuracy', 'f1', 'precision', 'recall', 'r2', 'rmse', 'mae']
  const availableKeys = knownKeys.filter((k) =>
    lc.steps.some((s) => typeof s.metrics?.[k] === 'number')
  )
  if (availableKeys.length === 0) return null
  return {
    tooltip: { trigger: 'axis' },
    legend: { bottom: 4, textStyle: { fontSize: 11 } },
    grid: { top: 24, left: 56, right: 24, bottom: 56 },
    xAxis: {
      type: 'category',
      data: lc.steps.map((s) => `Fold ${s.step}`),
    },
    yAxis: { type: 'value', scale: true },
    series: availableKeys.map((k, i) => ({
      name: k,
      type: 'line', smooth: true,
      data: lc.steps.map((s) => s.metrics?.[k] ?? null),
      lineStyle: { width: 2 },
      itemStyle: { color: ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][i % 5] },
    })),
  }
}

function buildResidualOption(res) {
  if (!res?.predicted?.length) return null
  const scatter = res.predicted.map((p, i) => [p, res.residuals[i]])
  return {
    tooltip: {
      trigger: 'item',
      formatter: (p) => `预测=<b>${p.value[0]}</b><br/>残差=<b>${p.value[1]}</b>`,
    },
    grid: { top: 24, left: 56, right: 24, bottom: 56 },
    xAxis: { type: 'value', name: '预测值', nameGap: 24, nameLocation: 'middle' },
    yAxis: {
      type: 'value', name: '残差',
      axisLine: { lineStyle: { color: '#94a3b8' } },
    },
    series: [
      {
        type: 'scatter', symbolSize: 6,
        data: scatter,
        itemStyle: { color: 'rgba(37,99,235,0.55)' },
      },
      {
        // y=0 reference line
        type: 'line', markLine: {
          silent: true, symbol: 'none',
          lineStyle: { color: '#ef4444', type: 'dashed' },
          data: [{ yAxis: 0 }],
        },
      },
    ],
    graphic: [{
      type: 'text', right: 24, top: 12,
      style: {
        text: `均值=${res.mean_residual} · 标准差=${res.std_residual}`,
        fontSize: 11, fill: '#64748b',
      },
    }],
  }
}

function buildPredVsActualOption(pva) {
  if (!pva?.actual?.length) return null
  const scatter = pva.actual.map((a, i) => [a, pva.predicted[i]])
  const all = [...pva.actual, ...pva.predicted]
  const lo = Math.min(...all)
  const hi = Math.max(...all)
  return {
    tooltip: {
      trigger: 'item',
      formatter: (p) => `真实=<b>${p.value[0]}</b><br/>预测=<b>${p.value[1]}</b>`,
    },
    grid: { top: 24, left: 56, right: 24, bottom: 56 },
    xAxis: { type: 'value', name: '真实值', nameGap: 24, nameLocation: 'middle' },
    yAxis: { type: 'value', name: '预测值' },
    series: [
      {
        type: 'scatter', symbolSize: 6,
        data: scatter,
        itemStyle: { color: 'rgba(16,185,129,0.6)' },
      },
      {
        type: 'line', showSymbol: false,
        data: [[lo, lo], [hi, hi]],
        lineStyle: { color: '#ef4444', type: 'dashed' },
        tooltip: { show: false },
      },
    ],
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart Card wrapper
// ─────────────────────────────────────────────────────────────────────────────

function VizCard({ icon, title, hint, option, height = 320, empty = '暂无数据' }) {
  return (
    <Card
      size="small"
      variant="borderless"
      styles={{ body: { padding: 12 } }}
      title={
        <Space size={6}>
          {icon}
          <Text strong style={{ fontSize: 13 }}>{title}</Text>
          {hint && (
            <Tooltip title={hint}>
              <InfoCircleOutlined style={{ color: '#94a3b8', fontSize: 12 }} />
            </Tooltip>
          )}
        </Space>
      }
      style={{ background: '#ffffff', boxShadow: '0 1px 3px rgba(15,23,42,0.06)' }}
    >
      {option ? (
        <EChart option={option} style={{ height }} />
      ) : (
        <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={empty} />
        </div>
      )}
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export default function TrainingViz({ trainingTaskId, modelType, taskStatus }) {
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState({
    cm: null, roc: null, fi: null, lc: null, res: null, pva: null,
  })
  const [error, setError] = useState(null)

  const taskType = inferTaskType(modelType)

  const reload = useCallback(async () => {
    if (!trainingTaskId) return
    setLoading(true)
    setError(null)
    try {
      if (taskType === 'classification') {
        const [cm, roc, fi, lc] = await Promise.all([
          safeFetch(vizApi.getConfusionMatrix(trainingTaskId)),
          safeFetch(vizApi.getRocCurve(trainingTaskId)),
          safeFetch(vizApi.getFeatureImportance(trainingTaskId)),
          safeFetch(vizApi.getLearningCurve(trainingTaskId)),
        ])
        setData({ cm, roc, fi, lc, res: null, pva: null })
      } else {
        const [res, pva, fi, lc] = await Promise.all([
          safeFetch(vizApi.getResidualPlot(trainingTaskId)),
          safeFetch(vizApi.getPredictedVsActual(trainingTaskId)),
          safeFetch(vizApi.getFeatureImportance(trainingTaskId)),
          safeFetch(vizApi.getLearningCurve(trainingTaskId)),
        ])
        setData({ cm: null, roc: null, fi, lc, res, pva })
      }
    } catch (e) {
      setError(e?.message || '加载训练可视化失败')
    } finally {
      setLoading(false)
    }
  }, [trainingTaskId, taskType])

  useEffect(() => { void reload() }, [reload])

  const options = useMemo(() => ({
    cm: buildConfusionMatrixOption(data.cm),
    roc: buildRocCurveOption(data.roc),
    fi: buildFeatureImportanceOption(data.fi),
    lc: buildLearningCurveOption(data.lc),
    res: buildResidualOption(data.res),
    pva: buildPredVsActualOption(data.pva),
  }), [data])

  if (!trainingTaskId) {
    return <Empty description="该 Run 未绑定训练任务，无法生成图表" />
  }

  const notReady = taskStatus && !['SUCCESS', 'FAILED'].includes(taskStatus)

  return (
    <Spin spinning={loading}>
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Space size={8}>
            <Tag color={taskType === 'classification' ? 'blue' : 'purple'}>
              {taskType === 'classification' ? '分类' : '回归'}
            </Tag>
            <Text type="secondary" style={{ fontSize: 12 }}>
              模型 · {modelType || '未知'} · TrainingTask #{String(trainingTaskId).slice(0, 8)}
            </Text>
          </Space>
          <Button size="small" icon={<ReloadOutlined />} onClick={reload} loading={loading}>
            刷新
          </Button>
        </Space>

        {notReady && (
          <Alert
            type="info" showIcon
            message="训练尚在进行中"
            description="部分图表要等训练完成后才能渲染，你可以继续观察日志 Tab，训练结束后回到这里点击刷新。"
          />
        )}
        {error && <Alert type="error" showIcon message={error} />}

        {/* Row 1 — 最核心的性能诊断 */}
        <Row gutter={[12, 12]}>
          {taskType === 'classification' ? (
            <>
              <Col span={24} xl={12}>
                <VizCard
                  icon={<HeatMapOutlined style={{ color: '#2563eb' }} />}
                  title="混淆矩阵"
                  hint="真实类别 × 预测类别的计数。对角线越深越好；非对角线透露主要误分类方向。"
                  option={options.cm}
                  empty="暂无混淆矩阵数据（可能训练未完成或非分类模型）"
                />
              </Col>
              <Col span={24} xl={12}>
                <VizCard
                  icon={<LineChartOutlined style={{ color: '#10b981' }} />}
                  title="学习曲线 (交叉验证)"
                  hint="每折的评估指标走势。各折差异大 = 方差高，指标随折上升 = 数据顺序敏感。"
                  option={options.lc}
                  empty="暂无学习曲线（任务未开启 CV 或指标未记录）"
                />
              </Col>
            </>
          ) : (
            <>
              <Col span={24} xl={12}>
                <VizCard
                  icon={<DotChartOutlined style={{ color: '#2563eb' }} />}
                  title="残差图"
                  hint="残差 = 真实值 − 预测值。理想情况下围绕 0 线均匀散布，出现明显模式（漏斗 / 曲线）说明模型欠拟合。"
                  option={options.res}
                  empty="暂无残差数据"
                />
              </Col>
              <Col span={24} xl={12}>
                <VizCard
                  icon={<LineChartOutlined style={{ color: '#10b981' }} />}
                  title="学习曲线 (交叉验证)"
                  hint="每折的 R² / RMSE / MAE 走势。"
                  option={options.lc}
                  empty="暂无学习曲线"
                />
              </Col>
            </>
          )}
        </Row>

        {/* Row 2 — 细节图 */}
        <Row gutter={[12, 12]}>
          {taskType === 'classification' ? (
            <Col span={24} xl={12}>
              <VizCard
                icon={<RadarChartOutlined style={{ color: '#f59e0b' }} />}
                title="ROC 曲线"
                hint="横轴 FPR 纵轴 TPR。曲线越贴近左上角越好，AUC ≥ 0.8 为可用模型。"
                option={options.roc}
                empty="暂无 ROC 数据（模型不支持 predict_proba 或非二分类）"
              />
            </Col>
          ) : (
            <Col span={24} xl={12}>
              <VizCard
                icon={<DotChartOutlined style={{ color: '#f59e0b' }} />}
                title="预测 vs 真实"
                hint="散点越贴近 y=x 虚线越好。偏离可以显示模型的系统性高估或低估。"
                option={options.pva}
                empty="暂无预测数据"
              />
            </Col>
          )}

          <Col span={24} xl={12}>
            <VizCard
              icon={<BarChartOutlined style={{ color: '#8b5cf6' }} />}
              title="特征重要度 Top-10"
              hint="模型自带的 feature_importances_ / coef_。越靠上贡献越大，为特征选择提供参考。"
              option={options.fi}
              empty="该模型不暴露特征重要度（如 kNN、SVM 默认内核）"
            />
          </Col>
        </Row>
      </Space>
    </Spin>
  )
}
