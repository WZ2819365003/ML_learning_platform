function errorDetail(error) {
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string') return detail
  return error?.message || ''
}

export function classifyVizUnavailable(key, error) {
  if (error?.response?.status !== 400) return null
  const detail = errorDetail(error).toLowerCase()

  if (key === 'featureImportance' && detail.includes('feature importance')) {
    return '当前模型不提供原生特征重要性。SVR 等模型可使用上方 SHAP 解释，但不会生成 feature_importances_。'
  }
  if (detail.includes('predict_proba')) {
    return '当前模型不提供概率输出，因此不适用这项分析。'
  }
  if (detail.includes('不支持') || detail.includes('unavailable')) {
    return errorDetail(error)
  }
  return null
}
