import { describe, expect, it } from 'vitest'

import { buildReportViewModel } from './aiReportViewModel'

describe('buildReportViewModel', () => {
  it('extracts a report-reader summary from a rich AI report payload', () => {
    const vm = buildReportViewModel({
      archive_id: 'fb837adc-7150-4a9a-8b4a-000000000000',
      headline_metrics: [
        { key: 'ai_score', label: 'AI 总分', value: '95/100' },
        { key: 'best_model', label: '最佳模型', value: 'random_forest', detail: 'grid_search' },
      ],
      charts: [{ id: 'training_curves' }],
      tables: [{ id: 'data_profile' }, { id: 'metric_comparison' }],
      markdown: [
        '# AI 建模报告',
        '',
        '## 第一章 结论',
        '',
        '### 1.1 综合判断',
        '',
        '#### 1.1.1 任务结论',
        '**总分：95/100。** 本任务整体完成质量优异。',
        '',
        '## 第二章 过程与评价',
      ].join('\n'),
    }, '复杂训练报告验证')

    expect(vm.title).toBe('AI 建模报告')
    expect(vm.taskName).toBe('复杂训练报告验证')
    expect(vm.score).toBe('95/100')
    expect(vm.archiveLabel).toBe('fb837adc')
    expect(vm.chartCount).toBe(1)
    expect(vm.tableCount).toBe(2)
    expect(vm.summary).toContain('本任务整体完成质量优异')
    expect(vm.metrics).toEqual([
      {
        key: 'best_model',
        label: '最佳模型',
        value: 'random_forest',
        detail: 'grid_search',
        tone: 'default',
      },
    ])
  })

  it('builds the table of contents from visible report blocks when present', () => {
    const vm = buildReportViewModel({
      markdown: [
        '# AI 建模报告',
        '',
        '## 第一章 结论',
        '',
        '## 第二章 原始生成内容',
      ].join('\n'),
      report_blocks: [
        {
          type: 'markdown',
          id: 'conclusion',
          markdown: '# AI 建模报告\n\n## 第一章 结论\n\n### 1.1 综合判断\n\n**总分：90/100。** 任务完成质量较好。',
        },
        {
          type: 'markdown',
          id: 'process_chapter',
          markdown: '## 第二章 过程与评价\n\n**本章围绕结构化证据展开。**',
        },
        {
          type: 'markdown',
          id: 'data_profile_explanation',
          markdown: '### 2.1 数据集概况\n\n#### 2.1.1 数据输入结论\n\n**数据输入明确。** 字段画像可以支撑训练。',
        },
      ],
    }, '目录验证')

    expect(vm.summary).toContain('任务完成质量较好')
    // The table of contents is gone: it duplicated the report's own side rail,
    // and the two together squeezed the body below the width its tables need.
    expect(vm.toc).toBeUndefined()
  })
})
