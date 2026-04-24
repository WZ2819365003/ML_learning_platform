/**
 * Shared API wrapper for viz components.
 *
 * Components receive a `scope` prop so the same component can render
 * against either the legacy /api/viz/:id/* routes or V3 per-run routes
 * without duplicating fetch logic.
 */
import { vizApi } from '../../services/api'

export const VizScope = {
  legacy: 'legacy',
  run: 'run',
}

export function fetchPerClass(taskId) {
  return vizApi.getPerClass(taskId)
}

export function fetchPrCurve(taskId) {
  return vizApi.getPrCurve(taskId)
}

export function fetchCalibration(taskId, nBins = 10) {
  return vizApi.getCalibration(taskId, { n_bins: nBins })
}

export function fetchThreshold(taskId, step = 0.05) {
  return vizApi.getThreshold(taskId, { step })
}

export function fetchDistribution(taskId, bins = 30) {
  return vizApi.getDistribution(taskId, { bins })
}

export function fetchShap(taskId, maxSamples = 100) {
  return vizApi.getShapSummary(taskId, { max_samples: maxSamples })
}
