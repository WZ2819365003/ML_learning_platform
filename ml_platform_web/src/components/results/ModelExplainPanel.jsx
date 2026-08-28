import React, { useEffect, useState } from 'react'
import { Alert, Button, Empty, Spin, Typography } from 'antd'
import { BulbOutlined } from '@ant-design/icons'

import ShapView from '../viz/ShapView'
import { vizApi } from '../../services/api'

const { Paragraph, Text } = Typography

export default function ModelExplainPanel({ taskId, modelType }) {
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    setPayload(null)
    setError(null)
    setLoading(false)
  }, [taskId])

  async function loadShap() {
    setLoading(true)
    setError(null)
    try {
      setPayload(await vizApi.getShapSummary(
        taskId,
        { max_samples: 5 },
        { timeout: 180000 },
      ))
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || '模型解释计算失败')
    } finally {
      setLoading(false)
    }
  }

  if (payload) return <ShapView payload={payload} />

  return (
    <Spin spinning={loading} tip="正在计算 SHAP 解释">
      {error ? (
        <Alert
          type="error"
          showIcon
          message="模型解释计算失败"
          description={error}
          action={<Button size="small" onClick={loadShap}>重试</Button>}
        />
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={(
            <div>
              <Text>按需计算 {modelType || '当前模型'} 的 SHAP 解释</Text>
              <Paragraph type="secondary" style={{ margin: '6px 0 0' }}>
                计算完成后展示特征贡献排序和方向；这里不再创建没有数据的原生特征重要性卡片。
              </Paragraph>
            </div>
          )}
          style={{ padding: '72px 20px' }}
        >
          <Button type="primary" icon={<BulbOutlined />} onClick={loadShap}>
            计算模型解释
          </Button>
        </Empty>
      )}
    </Spin>
  )
}
