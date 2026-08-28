export const RESULTS_PATH = '/training/results'

export function normalizeResultFamily(value) {
  return String(value || '').toLowerCase() === 'dl' ? 'dl' : 'ml'
}

export function buildResultsUrl({ family = 'ml', taskId = null } = {}) {
  const params = new URLSearchParams()
  if (normalizeResultFamily(family) === 'dl') params.set('family', 'dl')
  if (taskId) params.set('taskId', taskId)
  const query = params.toString()
  return query ? `${RESULTS_PATH}?${query}` : RESULTS_PATH
}

export function redirectLegacyDlSearch(search = '') {
  const params = new URLSearchParams(search)
  params.set('family', 'dl')
  return `${RESULTS_PATH}?${params.toString()}`
}
