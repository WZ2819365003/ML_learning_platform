import React, { useEffect, useState } from 'react'
import { Modal, Input, Alert, Typography, message } from 'antd'
import { CodeOutlined } from '@ant-design/icons'
import { dataApi } from '../../services/api'

const { Text, Paragraph } = Typography

const TEMPLATE = `# df 是源数据集（pandas.DataFrame），可用 pd / np。
# 清洗 / 特征工程后重新赋值 df 或定义 result；
# 平台会把结果存为一个新数据集，供本任务训练使用。

df = df.dropna()

# 示例：新增特征
# df["petal_area"] = df["petal_length"] * df["petal_width"]

result = df
`

/**
 * 数据 Pipeline（代码）— run Python that transforms the source dataset's df and
 * saves the output as a new dataset. On success calls onCreated(newDataset).
 */
export default function DataPipelineModal({ open, datasetId, onClose, onCreated }) {
  const [code, setCode] = useState(TEMPLATE)
  const [saveAs, setSaveAs] = useState('')
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => { if (open) { setCode(TEMPLATE); setSaveAs(''); setError(null) } }, [open])

  const run = async () => {
    if (!datasetId) { message.warning('请先选择源数据集'); return }
    setRunning(true)
    setError(null)
    try {
      const ds = await dataApi.runPipeline(datasetId, { code, save_as: saveAs || null })
      message.success(`已生成新数据集：${ds.name}`)
      onCreated?.(ds)
    } catch (err) {
      setError(err?.response?.data?.detail || '执行失败')
    } finally {
      setRunning(false)
    }
  }

  return (
    <Modal
      title={<span><CodeOutlined style={{ marginRight: 8, color: '#2563eb' }} />数据 Pipeline（Python）</span>}
      open={open} onCancel={onClose} onOk={run} confirmLoading={running}
      okText="运行并生成数据集" cancelText="取消" width={720} destroyOnClose
    >
      <Paragraph type="secondary" style={{ fontSize: 12, marginBottom: 8 }}>
        源数据集以 <Text code>df</Text> 传入（pandas，可用 <Text code>pd</Text> / <Text code>np</Text>）。
        处理后重新赋值 <Text code>df</Text> 或定义 <Text code>result</Text>；输出存为新数据集。
        受限环境执行（禁 import / 文件 / 进程）。
      </Paragraph>
      <Input placeholder="新数据集名称（可选，默认 <源名>_pipeline）" value={saveAs}
        onChange={e => setSaveAs(e.target.value)} style={{ marginBottom: 8 }} />
      <Input.TextArea value={code} onChange={e => setCode(e.target.value)} rows={14} spellCheck={false}
        style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: 12.5, lineHeight: 1.5 }} />
      {error && <Alert type="error" showIcon style={{ marginTop: 10 }} message="执行失败"
        description={<pre style={{ margin: 0, fontSize: 12, whiteSpace: 'pre-wrap' }}>{error}</pre>} />}
    </Modal>
  )
}
