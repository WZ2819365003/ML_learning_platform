import React from 'react'
import { Navigate, useLocation } from 'react-router-dom'

import UnifiedResultDetail from '../components/results/UnifiedResultDetail'
import { normalizeResultFamily, redirectLegacyDlSearch } from '../utils/resultRoutes'

export default function UnifiedResults() {
  const location = useLocation()

  // Historical links remain valid but converge immediately on the one public
  // result route.  The family query only selects a renderer; it is not a
  // second page hierarchy.
  if (location.pathname === '/dl/results') {
    return <Navigate to={redirectLegacyDlSearch(location.search)} replace />
  }

  const params = new URLSearchParams(location.search)
  const family = normalizeResultFamily(params.get('family'))
  const taskId = params.get('taskId')

  if (!taskId) return <Navigate to="/models" replace />
  return <UnifiedResultDetail family={family} taskId={taskId} />
}
