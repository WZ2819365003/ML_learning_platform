/**
 * ModelDownloadPanel — the trained artifact, and what to do with it.
 *
 * The unified download endpoint serves both families and picks the extension
 * from the row it finds: .pt for a DL checkpoint, .joblib for a classic-ML
 * bundle. DL is where this tab earns its place — a checkpoint is the thing you
 * take away to serve elsewhere, and there was no way to get it from the result
 * view at all.
 */
import React from 'react'
import { Alert, Button, Card, Descriptions, Space, Typography } from 'antd'
import { DownloadOutlined } from '@ant-design/icons'

import { runModelDownloadUrl } from '../../services/api'
import { formatBytes, formatDateTime } from '../../utils/formatters'

const { Text, Paragraph } = Typography

export default function ModelDownloadPanel({ family, taskId, result }) {
  const isDl = family === 'dl'
  const available = Boolean(taskId && result?.status === 'SUCCESS')

  if (!available) {
    return (
      <Alert type="info" showIcon message="模型文件尚不可用"
        description="训练成功后才会产出可下载的模型文件。" />
    )
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Card size="small" variant="outlined">
        <Descriptions column={{ xs: 1, md: 2 }} size="small">
          <Descriptions.Item label="模型">{result?.modelType || '—'}</Descriptions.Item>
          <Descriptions.Item label="文件格式">
            <code>{isDl ? '.pt（PyTorch 检查点）' : '.joblib（scikit-learn 产物）'}</code>
          </Descriptions.Item>
          <Descriptions.Item label="文件大小">
            {result?.modelSize ? formatBytes(result.modelSize) : '—'}
          </Descriptions.Item>
          <Descriptions.Item label="完成时间">
            {result?.finishedAt ? formatDateTime(result.finishedAt) : '—'}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Button type="primary" icon={<DownloadOutlined />}
        href={runModelDownloadUrl(taskId)} target="_blank">
        下载模型文件
      </Button>

      <Card size="small" variant="outlined" title="使用说明">
        {isDl ? (
          <Paragraph type="secondary" style={{ marginBottom: 0, fontSize: 12 }}>
            检查点包含模型权重与结构参数。同目录下的 <code>.scaler.joblib</code> 与
            <code>.preprocessor.joblib</code> 保存了特征标准化与编码状态 ——
            <Text strong>脱离它们单独加载权重会得到错误的预测</Text>，
            因为推理输入必须经过与训练时完全相同的变换。需要在平台外部署时，
            建议连同这两个文件一并导出。
          </Paragraph>
        ) : (
          <Paragraph type="secondary" style={{ marginBottom: 0, fontSize: 12 }}>
            产物是一个自包含的 joblib 包，已内置特征预处理，
            用 <code>joblib.load()</code> 载入后可直接对原始特征行调用 <code>predict()</code>。
          </Paragraph>
        )}
      </Card>
    </Space>
  )
}
