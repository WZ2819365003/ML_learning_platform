import React, { lazy } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { Segmented, Space, Typography } from 'antd'
import { ApartmentOutlined, DeploymentUnitOutlined } from '@ant-design/icons'

import {
  buildResultsUrl,
  normalizeResultFamily,
  redirectLegacyDlSearch,
} from '../utils/resultRoutes'

const MLResults = lazy(() => import('./Results'))
const DLResults = lazy(() => import('./DLResults'))
const { Text } = Typography

export default function UnifiedResults() {
  const location = useLocation()
  const navigate = useNavigate()

  // Historical links remain valid but converge immediately on the one public
  // result route.  The family query only selects a renderer; it is not a
  // second page hierarchy.
  if (location.pathname === '/dl/results') {
    return <Navigate to={redirectLegacyDlSearch(location.search)} replace />
  }

  const params = new URLSearchParams(location.search)
  const family = normalizeResultFamily(params.get('family'))
  const taskId = params.get('taskId')

  return (
    <div>
      {!taskId && (
        <Space
          size={12}
          wrap
          style={{
            width: '100%',
            justifyContent: 'space-between',
            marginBottom: 16,
            padding: '12px 16px',
            background: 'rgba(255,255,255,0.72)',
            border: '1px solid rgba(148,163,184,0.22)',
            borderRadius: 14,
          }}
        >
          <div>
            <Text strong>结果工作台</Text>
            <Text type="secondary" style={{ marginLeft: 10 }}>
              同一入口查看不同训练引擎的结果
            </Text>
          </div>
          <Segmented
            value={family}
            options={[
              { label: '机器学习', value: 'ml', icon: <ApartmentOutlined /> },
              { label: '深度学习', value: 'dl', icon: <DeploymentUnitOutlined /> },
            ]}
            onChange={nextFamily => navigate(buildResultsUrl({ family: nextFamily }))}
          />
        </Space>
      )}
      {family === 'dl' ? <DLResults /> : <MLResults />}
    </div>
  )
}
