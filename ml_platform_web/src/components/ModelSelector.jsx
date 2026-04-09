/**
 * ModelSelector — category tabs + filtered model dropdown with task-type badges.
 *
 * Props:
 *   models          - array of model metadata from GET /api/training/models
 *   categories      - array of category metadata from the same response
 *   value           - currently selected model id (controlled)
 *   onChange        - (modelId) => void
 *   taskFilter      - 'classification' | 'regression' | null (null = show all)
 */
import React, { useMemo, useState } from 'react';
import { Select, Tabs, Tag, Tooltip } from 'antd';

const { Option } = Select;

const TASK_TAG = {
  classification: { color: 'blue',  text: '分类' },
  regression:     { color: 'green', text: '回归' },
};

export default function ModelSelector({ models = [], categories = [], value, onChange, taskFilter = null }) {
  const [activeCategory, setActiveCategory] = useState('all');

  // Filter by both category tab and optional task filter from outside
  const visibleModels = useMemo(() => {
    return models.filter(m => {
      const catOk = activeCategory === 'all' || m.category === activeCategory;
      const taskOk = !taskFilter || m.task_types.includes(taskFilter);
      return catOk && taskOk;
    });
  }, [models, activeCategory, taskFilter]);

  // When category changes, clear selection if current model not in new list
  const handleCategoryChange = (cat) => {
    setActiveCategory(cat);
    if (value) {
      const stillVisible = models.find(
        m => m.id === value && (cat === 'all' || m.category === cat)
      );
      if (!stillVisible) onChange(undefined);
    }
  };

  const tabItems = [
    { key: 'all', label: '全部' },
    ...categories.map(c => ({ key: c.id, label: c.display_name })),
  ];

  return (
    <div className="model-selector">
      <Tabs
        size="small"
        activeKey={activeCategory}
        onChange={handleCategoryChange}
        items={tabItems}
        style={{ marginBottom: 8 }}
      />
      <Select
        value={value}
        onChange={onChange}
        placeholder="请选择模型"
        style={{ width: '100%' }}
        showSearch
        optionFilterProp="label"
        notFoundContent="该分类下暂无模型"
      >
        {visibleModels.map(m => (
          <Option key={m.id} value={m.id} label={m.display_name}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Tooltip title={m.description} placement="right">
                <span>{m.display_name}</span>
              </Tooltip>
              <span>
                {m.task_types.map(t => (
                  <Tag key={t} color={TASK_TAG[t]?.color} style={{ marginLeft: 4, fontSize: 11 }}>
                    {TASK_TAG[t]?.text ?? t}
                  </Tag>
                ))}
              </span>
            </div>
          </Option>
        ))}
      </Select>
    </div>
  );
}
