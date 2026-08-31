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
    // Regression only: lining predictions up against the truth is a numeric
    // comparison. Classification gets its confusion matrix and ROC under
    // 训练可视化 instead.
    taskTypes: ['regression'],
    statuses: ['SUCCESS'],
    renderer: 'backtest',
  },
  {
    key: 'download',
    label: '模型下载',
    // DL only for now: a checkpoint is the thing you take away to serve
    // elsewhere, and the result view had no way to reach it. ML models are
    // already downloadable from 模型管理.
    families: ['dl'],
    taskTypes: ['classification', 'regression'],
    statuses: ['SUCCESS'],
    renderer: 'download',
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
