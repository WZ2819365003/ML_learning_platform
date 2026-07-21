import { describe, expect, it } from 'vitest'
import {
  buildLossHistoryOption,
  buildTaskMetricHistoryOption,
  buildResidualOption,
} from './DLDiagnostics'

const classificationHistory = [
  { epoch: 1, train_loss: 0.8, val_loss: 0.7, val_acc: 0.65, val_f1_macro: 0.61 },
  { epoch: 2, train_loss: 0.5, val_loss: 0.48, val_acc: 0.82, val_f1_macro: 0.79 },
]

describe('DLDiagnostics chart contracts', () => {
  it('keeps loss isolated from task metrics', () => {
    const option = buildLossHistoryOption(classificationHistory)
    expect(option.series.map((series) => series.name)).toEqual(['训练损失', '验证损失'])
  })

  it('shows classification accuracy and F1 without loss or derived error', () => {
    const option = buildTaskMetricHistoryOption(classificationHistory, 'classification')
    expect(option.series.map((series) => series.name)).toEqual(['验证准确率', '验证 F1'])
  })

  it('derives regression residuals from predicted-vs-actual samples', () => {
    const option = buildResidualOption({ actual: [2, 4], predicted: [1.5, 5] })
    expect(option.series[0].data).toEqual([[1.5, 0.5], [5, -1]])
  })
})
