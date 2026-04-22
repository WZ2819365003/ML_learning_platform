import React, { useEffect, useMemo, useState } from 'react'
import { Select, Space, Tag, Typography, Tooltip } from 'antd'
import { ExperimentOutlined, ThunderboltOutlined, AppstoreOutlined } from '@ant-design/icons'
import { trainingPlansApi } from '../../services/api'

const { Text } = Typography

/**
 * TrainingPlanPicker — compact plan selector for the ModelingTask create form.
 *
 * Fetches plans filtered by task_type, sorts by (use_count desc, last_used_at desc)
 * so frequently-used recipes surface first. Renders a rich option with:
 *   - plan name + description tooltip
 *   - strategy_type tag
 *   - model_family chip (ml / dl / mixed)
 *   - selected model count
 *
 * Purely presentational: the parent controls the value via Form.Item or useState.
 */
export default function TrainingPlanPicker({
  value,
  onChange,
  taskType,
  disabled,
  placeholder = '选择训练方案（可选，不选则使用即时配置）',
  style,
  allowClear = true,
}) {
  const [plans, setPlans] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      setLoading(true)
      try {
        const params = { page_size: 100 }
        if (taskType) params.task_type = taskType
        const resp = await trainingPlansApi.list(params)
        if (cancelled) return
        const items = resp?.items || []
        // Re-rank client-side: prefer most-used, then most-recently-used.
        items.sort((a, b) => {
          const ua = a.use_count || 0
          const ub = b.use_count || 0
          if (ua !== ub) return ub - ua
          const la = a.last_used_at || ''
          const lb = b.last_used_at || ''
          return lb.localeCompare(la)
        })
        setPlans(items)
      } catch (err) {
        console.warn('加载训练方案失败', err)
        setPlans([])
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [taskType])

  const options = useMemo(() => plans.map((p) => {
    const family = p.model_family || 'ml'
    const strategy = p.strategy_type || 'baseline'
    const modelCount = (p.selected_models || []).length
    const familyColor = family === 'dl' ? 'purple' : family === 'mixed' ? 'geekblue' : 'blue'
    const familyIcon = family === 'dl'
      ? <ThunderboltOutlined />
      : family === 'mixed' ? <AppstoreOutlined /> : <ExperimentOutlined />
    return {
      value: p.id,
      // Plain label used by Select when search is performed
      label: p.name,
      // Rich render for the dropdown
      data: p,
      rich: (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'space-between' }}>
          <div style={{ minWidth: 0, flex: 1 }}>
            <Tooltip title={p.description || ''} mouseEnterDelay={0.4}>
              <Text ellipsis style={{ fontWeight: 600 }}>{p.name}</Text>
            </Tooltip>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
              {modelCount} 个模型 · v{p.version || 1}
              {p.use_count > 0 && <> · 已使用 {p.use_count} 次</>}
            </div>
          </div>
          <Space size={4}>
            <Tag icon={familyIcon} color={familyColor} style={{ margin: 0 }}>
              {family.toUpperCase()}
            </Tag>
            <Tag color="default" style={{ margin: 0 }}>{strategy}</Tag>
          </Space>
        </div>
      ),
    }
  }), [plans])

  return (
    <Select
      showSearch
      allowClear={allowClear}
      disabled={disabled}
      loading={loading}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      style={style}
      optionFilterProp="label"
      filterOption={(input, option) => {
        const hay = `${option?.label || ''} ${option?.data?.description || ''}`.toLowerCase()
        return hay.includes(input.toLowerCase())
      }}
      options={options}
      optionRender={(opt) => opt.data?.rich || opt.label}
      notFoundContent={loading ? '加载中…' : (taskType ? `无 ${taskType} 方案` : '暂无方案')}
    />
  )
}
