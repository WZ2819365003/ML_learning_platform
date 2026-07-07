import React, { useEffect, useState } from 'react'
import { Modal, Input, Alert, Typography, message } from 'antd'
import { CodeOutlined } from '@ant-design/icons'
import { modelingTaskApi } from '../../services/api'

const { Text, Paragraph } = Typography

/**
 * 代码配置 — write Python that assigns a `config` dict; the backend executor
 * runs it (restricted builtins) and dispatches the resulting experiment batch
 * through the normal pipeline. No client-side parsing.
 */
export default function CodeConfigModal({ open, task, defaultCode = '', onClose, onSubmitted }) {
  const [code, setCode] = useState(defaultCode)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => { if (open) { setCode(defaultCode); setError(null) } }, [open, defaultCode])

  const run = async () => {
    setRunning(true)
    setError(null)
    try {
      await modelingTaskApi.configExec(task.id, { code })
      message.success('代码执行成功，已提交训练')
      onSubmitted?.()
    } catch (err) {
      setError(err?.response?.data?.detail || '执行失败')
    } finally {
      setRunning(false)
    }
  }

  return (
    <Modal
      title={<span><CodeOutlined style={{ marginRight: 8, color: '#2563eb' }} />代码配置（Python）</span>}
      open={open}
      onCancel={onClose}
      onOk={run}
      confirmLoading={running}
      okText="运行并训练"
      cancelText="取消"
      width={720}
      destroyOnClose
    >
      <Paragraph type="secondary" style={{ fontSize: 12, marginBottom: 8 }}>
        用 Python 定义一个名为 <Text code>config</Text> 的 dict（至少含 <Text code>selected_models</Text>）。
        代码在受限环境中执行（禁用 import / 文件 / 进程），输出直接交给现有训练管线。
      </Paragraph>
      <Input.TextArea
        value={code}
        onChange={e => setCode(e.target.value)}
        rows={16}
        spellCheck={false}
        style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: 12.5, lineHeight: 1.5 }}
      />
      {error && <Alert type="error" showIcon style={{ marginTop: 10 }} message="执行失败"
        description={<pre style={{ margin: 0, fontSize: 12, whiteSpace: 'pre-wrap' }}>{error}</pre>} />}
    </Modal>
  )
}
