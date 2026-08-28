/**
 * Shared visualization request registry.
 *
 * Chart eligibility is described by task type, runtime family, and the UI
 * surface/tab that consumes it.  Model names deliberately do not appear here:
 * capabilities such as predict_proba or feature importance are declared as
 * requirements and a missing capability remains an isolated chart response.
 */
import { dlApi, vizApi } from '../../services/api'

export const VIZ_REGISTRY = [
  {
    key: 'confusionMatrix', title: '混淆矩阵', taskTypes: ['classification'],
    families: ['ml'], resultsTabs: ['performance'], workbench: true,
    fetch: (taskId) => vizApi.getConfusionMatrix(taskId),
  },
  {
    key: 'rocCurve', title: 'ROC 曲线', taskTypes: ['classification'],
    families: ['ml'], resultsTabs: ['performance'], workbench: true,
    requires: 'predict_proba', fetch: (taskId) => vizApi.getRocCurve(taskId),
  },
  {
    key: 'perClass', title: '逐类指标', taskTypes: ['classification'],
    families: ['ml'], resultsTabs: ['performance'], workbench: false,
    fetch: (taskId) => vizApi.getPerClass(taskId),
  },
  {
    key: 'prCurve', title: 'Precision-Recall 曲线', taskTypes: ['classification'],
    families: ['ml'], resultsTabs: ['performance'], workbench: false,
    requires: 'predict_proba', fetch: (taskId) => vizApi.getPrCurve(taskId),
  },
  {
    key: 'learningCurve', title: '学习曲线', taskTypes: ['classification', 'regression'],
    families: ['ml'], resultsTabs: ['training'], workbench: true,
    fetch: (taskId) => vizApi.getLearningCurve(taskId),
  },
  {
    key: 'dlEpochs', title: 'Epoch 训练历史', taskTypes: ['classification', 'regression'],
    families: ['dl'], resultsTabs: ['training'], workbench: false,
    fetch: (taskId) => dlApi.getEpochs(taskId, { page_size: 200 }),
  },
  {
    key: 'predictedVsActual', title: '预测 vs 实际', taskTypes: ['regression'],
    families: ['ml'], resultsTabs: ['comparison'], workbench: true,
    fetch: (taskId) => vizApi.getPredictedVsActual(taskId, { max_samples: 1000 }),
  },
  {
    key: 'featureImportance', title: '特征重要度', taskTypes: ['classification', 'regression'],
    families: ['ml'], resultsTabs: ['explain'], workbench: true,
    requires: 'feature_importance', fetch: (taskId) => vizApi.getFeatureImportance(taskId),
  },
  {
    key: 'shap', title: 'SHAP 解释', taskTypes: ['classification', 'regression'],
    families: ['ml', 'dl'], resultsTabs: ['explain'], workbench: false,
    requires: 'explainable', loadPolicy: 'manual',
    // Kernel SHAP for models such as SVR can take minutes.  Keep this behind
    // an explicit user action so neither page boot nor tab switching starts an
    // expensive job that continues after the browser request has timed out.
    fetch: (taskId) => vizApi.getShapSummary(
      taskId,
      { max_samples: 5 },
      { timeout: 180000 },
    ),
  },
  {
    key: 'threshold', title: '阈值敏感度', taskTypes: ['classification'],
    families: ['ml'], resultsTabs: ['threshold'], workbench: false,
    requires: 'predict_proba', fetch: (taskId) => vizApi.getThreshold(taskId),
  },
  {
    key: 'calibration', title: '校准曲线', taskTypes: ['classification'],
    families: ['ml'], resultsTabs: ['threshold'], workbench: false,
    requires: 'predict_proba', fetch: (taskId) => vizApi.getCalibration(taskId),
  },
  {
    key: 'distribution', title: '预测分布', taskTypes: ['classification'],
    families: ['ml'], resultsTabs: ['threshold'], workbench: false,
    requires: 'predict_proba',
    fetch: (taskId) => vizApi.getDistribution(taskId, { max_samples: 5000 }),
  },
]

export function getVizEntries({ taskType, family = 'ml', surface, tab = null }) {
  return VIZ_REGISTRY.filter((entry) => {
    if (!entry.taskTypes.includes(taskType)) return false
    if (!entry.families.includes(family)) return false
    if (surface === 'workbench') return entry.workbench
    if (surface === 'results') return entry.resultsTabs.includes(tab)
    return false
  })
}

export function getVizEntry(key) {
  return VIZ_REGISTRY.find((entry) => entry.key === key) ?? null
}

function histogram(values, binCount = 30) {
  const finite = values.filter(Number.isFinite)
  if (!finite.length) return { bin_edges: [], counts: [] }
  const min = Math.min(...finite)
  const max = Math.max(...finite)
  if (min === max) {
    return { bin_edges: [min - 0.5, max + 0.5], counts: [finite.length] }
  }
  const width = (max - min) / binCount
  const bin_edges = Array.from({ length: binCount + 1 }, (_, index) => min + index * width)
  const counts = Array.from({ length: binCount }, () => 0)
  finite.forEach((value) => {
    const index = Math.min(binCount - 1, Math.max(0, Math.floor((value - min) / width)))
    counts[index] += 1
  })
  return { bin_edges, counts }
}

export function deriveRegressionViz(predictedVsActual) {
  const actual = Array.isArray(predictedVsActual?.actual) ? predictedVsActual.actual : []
  const predicted = Array.isArray(predictedVsActual?.predicted) ? predictedVsActual.predicted : []
  const n = Math.min(actual.length, predicted.length)
  if (!n) return { residualPlot: null, distribution: null }

  const residuals = Array.from({ length: n }, (_, i) => Number(actual[i]) - Number(predicted[i]))
  const mean = residuals.reduce((sum, value) => sum + value, 0) / n
  const variance = residuals.reduce((sum, value) => sum + (value - mean) ** 2, 0) / n
  const { bin_edges, counts } = histogram(residuals)
  const meta = {
    sample_count: predictedVsActual.sample_count ?? n,
    total_count: predictedVsActual.total_count ?? n,
    sample_offset: predictedVsActual.sample_offset ?? 0,
    truncated: predictedVsActual.truncated ?? false,
  }

  return {
    residualPlot: {
      predicted: predicted.slice(0, n),
      residuals,
      mean_residual: mean,
      std_residual: Math.sqrt(variance),
      ...meta,
    },
    distribution: {
      kind: 'regression_residuals',
      bin_edges,
      counts,
      mean,
      std: Math.sqrt(variance),
      min: Math.min(...residuals),
      max: Math.max(...residuals),
      ...meta,
    },
  }
}
