import React from 'react'
import { Modal } from 'antd'
import { ThunderboltOutlined } from '@ant-design/icons'
import ExperimentBatchForm from './ExperimentBatchForm'

/**
 * Thin Modal shell around ExperimentBatchForm (used by the task detail page's
 * 「启动新批次」). The form owns all state/logic and its own submit button;
 * this wrapper only provides the dialog chrome. `active={open}` drives the
 * form's reset/data-loading; on submit success the form calls onSubmitted,
 * which we chain to onClose.
 */
export default function ExperimentBatchModal({ open, task, onClose, onSubmitted }) {
  return (
    <Modal
      title={<span><ThunderboltOutlined style={{ marginRight: 8, color: '#2563eb' }} />启动新的实验批次</span>}
      open={open}
      onCancel={onClose}
      footer={null}
      width={780}
      destroyOnHidden
    >
      <ExperimentBatchForm
        task={task}
        active={open}
        resetKey={open ? 1 : 0}
        onSubmitted={() => { onSubmitted?.(); onClose?.() }}
      />
    </Modal>
  )
}
