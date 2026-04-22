import React, { useState } from 'react'
import { Alert, Button, Card, Descriptions, Modal, Space, Tag, Typography } from 'antd'
import { FileTextOutlined, WarningOutlined, SnippetsOutlined } from '@ant-design/icons'

const { Text } = Typography

/**
 * TrainingPlanSnapshotView — read-only summary of the TrainingPlan frozen at
 * ModelingTask create-time.
 *
 * Props:
 *   snapshot       — object from ModelingTask.training_plan_snapshot
 *   status         — object from ModelingTask.training_plan_status
 *                     { plan_exists, snapshot_version, current_version, stale, current_name }
 *
 * If snapshot is null/undefined, renders an inline hint instead of a panel
 * (so the parent can embed this unconditionally).
 */
export default function TrainingPlanSnapshotView({ snapshot, status }) {
  const [jsonOpen, setJsonOpen] = useState(false)

  if (!snapshot) {
    return (
      <Card size="small" style={{ borderStyle: 'dashed' }}>
        <Space align="start">
          <FileTextOutlined style={{ color: '#94a3b8' }} />
          <div>
            <Text type="secondary">本任务未绑定训练方案</Text>
            <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>
              使用的是创建时的即时配置，无可追溯的方案引用。
            </div>
          </div>
        </Space>
      </Card>
    )
  }

  const payload = snapshot.payload || {}
  const dagShape = snapshot.dag_shape || {}
  const familyColor =
    payload.model_family === 'dl' ? 'purple'
    : payload.model_family === 'mixed' ? 'geekblue' : 'blue'

  return (
    <Card
      size="small"
      title={
        <Space>
          <SnippetsOutlined />
          <span>训练方案快照</span>
          <Tag color="default">v{snapshot.plan_version}</Tag>
        </Space>
      }
      extra={
        <Button size="small" icon={<FileTextOutlined />} onClick={() => setJsonOpen(true)}>
          查看 JSON
        </Button>
      }
    >
      {status?.stale && (
        <Alert
          type="warning"
          showIcon
          icon={<WarningOutlined />}
          style={{ marginBottom: 12 }}
          message={
            <span>
              方案原版已更新至 <b>v{status.current_version}</b>
              ，但本任务仍使用绑定时的快照 <b>v{status.snapshot_version}</b>
              （保证可重现）。
            </span>
          }
        />
      )}
      {status && !status.plan_exists && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message="原方案已被删除 — 本任务仍可按快照执行。"
        />
      )}

      <Descriptions size="small" column={2} bordered>
        <Descriptions.Item label="方案名" span={2}>
          <Text strong>{snapshot.plan_name}</Text>
          {status?.current_name && status.current_name !== snapshot.plan_name && (
            <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
              （当前名: {status.current_name}）
            </Text>
          )}
        </Descriptions.Item>
        <Descriptions.Item label="模型族">
          <Tag color={familyColor}>{(payload.model_family || 'ml').toUpperCase()}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="策略">
          <Tag>{payload.strategy_type || 'baseline'}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="模型数量">
          {dagShape.num_models ?? (payload.selected_models || []).length}
        </Descriptions.Item>
        <Descriptions.Item label="目标指标">
          {payload.default_objective_metric || '—'}
          {payload.default_objective_direction && (
            <Text type="secondary" style={{ marginLeft: 4 }}>
              ({payload.default_objective_direction})
            </Text>
          )}
        </Descriptions.Item>
        <Descriptions.Item label="选中模型" span={2}>
          <Space size={4} wrap>
            {(payload.selected_models || []).map((m) => (
              <Tag key={m} style={{ margin: 0 }}>{m}</Tag>
            ))}
          </Space>
        </Descriptions.Item>
        <Descriptions.Item label="绑定时间" span={2}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {snapshot.captured_at}
          </Text>
        </Descriptions.Item>
      </Descriptions>

      <Modal
        open={jsonOpen}
        title={`快照 JSON — v${snapshot.plan_version}`}
        onCancel={() => setJsonOpen(false)}
        footer={null}
        width={720}
      >
        <pre style={{
          background: '#0f172a', color: '#e2e8f0',
          padding: 16, borderRadius: 8, fontSize: 12,
          maxHeight: 480, overflow: 'auto',
        }}>
          {JSON.stringify(snapshot, null, 2)}
        </pre>
      </Modal>
    </Card>
  )
}
