/**
 * RunInspector — the Drawer shell around RunDetailBody.
 *
 * Kept for the "quick peek without leaving the list" flows (V3Runs,
 * ModelComparison, ModelingTaskDetail). The tab content itself lives in
 * RunDetailBody, which the standalone model detail page renders too — so the
 * two presentations can never drift apart.
 */
import React, { useEffect, useState } from 'react'
import { Drawer, Space, Typography } from 'antd'
import { LineChartOutlined } from '@ant-design/icons'

import { RunDetailBody, STATUS_TAG } from './RunDetailBody'

const { Text } = Typography

export default function RunInspector({ open, runId, onClose, defaultTab = 'overview' }) {
  const [run, setRun] = useState(null)

  // Drop the previous run's title as soon as the drawer closes, so reopening
  // on a different run never flashes the old id.
  useEffect(() => { if (!open) setRun(null) }, [open])

  return (
    <Drawer
      title={
        <Space>
          <LineChartOutlined style={{ color: '#2563eb' }} />
          <span>Run 诊断</span>
          {run && <Text type="secondary" style={{ fontSize: 12 }}>#{run.id?.slice(0, 8)}</Text>}
          {run && STATUS_TAG[run.status]}
        </Space>
      }
      placement="right"
      width={780}
      open={open}
      onClose={onClose}
      styles={{ body: { padding: '16px 20px' } }}
    >
      <RunDetailBody
        runId={runId}
        active={open}
        defaultTab={defaultTab}
        onData={(d) => setRun(d?.run || null)}
      />
    </Drawer>
  )
}
