/**
 * EvaluationDetail — the deeper evaluation charts for one run.
 *
 * These five views used to exist only on the legacy `/training/results` page.
 * They are the ones the run drawer never had, so folding that page away
 * without bringing them across would have been a straight capability loss:
 *
 *   每类指标 · PR 曲线 · 校准曲线 · 阈值调优 · 预测分布
 *
 * Layout: a two-column grid of equal-height cards rather than a single tall
 * column, so the panel stays roughly square and does not stretch the page.
 *
 * Fetching is lazy — the parent only mounts this once its tab is opened, and
 * the endpoints are chosen by task kind so regression runs never call the
 * classification-only ones (they would cleanly 400, but the round-trip and the
 * error noise are pointless).
 */
import React, { useEffect, useState } from 'react'
import { Alert, Card, Col, Empty, Row, Spin } from 'antd'

import { vizApi } from '../../services/api'
import PerClassMetricsTable from '../viz/PerClassMetricsTable'
import PRCurveChart from '../viz/PRCurveChart'
import CalibrationCurveChart from '../viz/CalibrationCurveChart'
import ThresholdTuningTable from '../viz/ThresholdTuningTable'
import PredictionDistributionChart from '../viz/PredictionDistributionChart'

// One height for every card here, so the grid reads as a grid instead of a
// ragged column. Charts get the full box; tables scroll inside it.
const CARD_BODY_HEIGHT = 340

function Panel({ title, children, span = 12 }) {
  return (
    <Col xs={24} xl={span}>
      <Card
        size="small"
        title={title}
        variant="outlined"
        styles={{ body: { height: CARD_BODY_HEIGHT, overflow: 'auto', padding: 12 } }}
      >
        {children}
      </Card>
    </Col>
  )
}

/** Render a payload, or a placeholder explaining why there is nothing. */
function orEmpty(payload, node, emptyText) {
  if (!payload) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyText} />
  return node
}

export default function EvaluationDetail({ trainingTaskId, taskKind }) {
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState({})
  const [error, setError] = useState(null)

  const isClassification = taskKind !== 'regression'

  useEffect(() => {
    if (!trainingTaskId) return undefined
    let cancelled = false
    setLoading(true)
    setError(null)

    const requests = [['distribution', () => vizApi.getDistribution(trainingTaskId)]]
    if (isClassification) {
      requests.push(
        ['perClass', () => vizApi.getPerClass(trainingTaskId)],
        ['prCurve', () => vizApi.getPrCurve(trainingTaskId)],
        ['calibration', () => vizApi.getCalibration(trainingTaskId)],
        ['threshold', () => vizApi.getThreshold(trainingTaskId)],
      )
    }

    // allSettled, not all: one unavailable chart must not blank the others.
    Promise.allSettled(requests.map(([, fn]) => fn()))
      .then((results) => {
        if (cancelled) return
        const next = {}
        results.forEach((r, i) => {
          if (r.status === 'fulfilled') next[requests[i][0]] = r.value
        })
        setData(next)
        if (results.every((r) => r.status === 'rejected')) {
          setError('评估图表暂不可用 — 该 Run 可能还没有产出可视化数据')
        }
      })
      .finally(() => { if (!cancelled) setLoading(false) })

    return () => { cancelled = true }
  }, [trainingTaskId, isClassification])

  if (!trainingTaskId) {
    return <Empty description="该 Run 没有关联的训练任务，无法加载评估图表" />
  }

  return (
    <Spin spinning={loading}>
      {error && <Alert type="info" showIcon message={error} style={{ marginBottom: 12 }} />}
      <Row gutter={[16, 16]}>
        {isClassification ? (
          <>
            <Panel title="每类指标">
              {orEmpty(data.perClass,
                <PerClassMetricsTable payload={data.perClass} />,
                '暂无每类指标')}
            </Panel>
            <Panel title="PR 曲线">
              {orEmpty(data.prCurve,
                <PRCurveChart payload={data.prCurve} height={CARD_BODY_HEIGHT - 40} />,
                '暂无 PR 曲线')}
            </Panel>
            <Panel title="校准曲线">
              {orEmpty(data.calibration,
                <CalibrationCurveChart payload={data.calibration} height={CARD_BODY_HEIGHT - 40} />,
                '暂无校准曲线')}
            </Panel>
            <Panel title="阈值调优">
              {orEmpty(data.threshold,
                <ThresholdTuningTable payload={data.threshold} />,
                '暂无阈值数据')}
            </Panel>
          </>
        ) : null}
        <Panel title="预测分布" span={isClassification ? 24 : 12}>
          {orEmpty(data.distribution,
            <PredictionDistributionChart payload={data.distribution} height={CARD_BODY_HEIGHT - 40} />,
            '暂无预测分布')}
        </Panel>
        {!isClassification && (
          <Panel title="说明">
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description="每类指标 / PR 曲线 / 校准曲线 / 阈值调优 仅适用于分类任务"
            />
          </Panel>
        )}
      </Row>
    </Spin>
  )
}
