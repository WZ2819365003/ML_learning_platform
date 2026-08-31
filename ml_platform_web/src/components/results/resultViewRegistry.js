export const RESULT_VIEW_REGISTRY = [
  {
    key: 'logs',
    label: '训练日志',
    families: ['ml', 'dl'],
    taskTypes: ['classification', 'regression'],
    statuses: ['PENDING', 'RUNNING', 'SUCCESS', 'FAILED'],
    renderer: 'logs',
  },
  {
    key: 'visualization',
    label: '训练可视化',
    families: ['ml', 'dl'],
    taskTypes: ['classification', 'regression'],
    statuses: ['SUCCESS'],
    renderer: 'trainingViz',
  },
  {
    key: 'backtest',
    label: '结果回测',
    families: ['ml', 'dl'],
    // Both kinds: a confusion matrix is predictions against truth just as much
    // as a predicted-vs-actual curve is, only asked of labels.
    taskTypes: ['classification', 'regression'],
    statuses: ['SUCCESS'],
    renderer: 'backtest',
  },
  {
    key: 'explain',
    label: '模型解释',
    families: ['ml', 'dl'],
    taskTypes: ['classification', 'regression'],
    statuses: ['SUCCESS'],
    requires: 'explainable',
    renderer: 'explain',
  },
]

export function getResultViewEntries({ family, taskType, status }) {
  return RESULT_VIEW_REGISTRY.filter((entry) => (
    entry.families.includes(family)
    && entry.taskTypes.includes(taskType)
    && entry.statuses.includes(status)
  ))
}
