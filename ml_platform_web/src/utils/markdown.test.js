import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { parseMarkdown, parseInline } from './markdown'

describe('parseMarkdown', () => {
  it('parses the heading levels the report actually emits', () => {
    const blocks = parseMarkdown('# 建模任务报告\n\n## 1. 任务概览\n\n### 4. 评估方法')
    expect(blocks).toEqual([
      { type: 'heading', level: 1, text: '建模任务报告' },
      { type: 'heading', level: 2, text: '1. 任务概览' },
      { type: 'heading', level: 3, text: '4. 评估方法' },
    ])
  })

  it('parses a GFM pipe table', () => {
    const md = ['| 项 | 值 |', '|---|---|', '| 数据集 | churn.csv |', '| 样本量 | 500 |'].join('\n')
    expect(parseMarkdown(md)).toEqual([
      {
        type: 'table',
        headers: ['项', '值'],
        rows: [
          ['数据集', 'churn.csv'],
          ['样本量', '500'],
        ],
      },
    ])
  })

  it('does not treat a stray pipe line as a table', () => {
    // Without the alignment row this is prose, not a table. Guessing wrong
    // would swallow the line into a malformed table and lose the text.
    const blocks = parseMarkdown('| this is not a table')
    expect(blocks).toEqual([{ type: 'paragraph', text: '| this is not a table' }])
  })

  it('joins a multi-line blockquote into one block', () => {
    const md = '> **封存测试集结果，全程仅开启一次。**\n> 这些数据未被使用。'
    expect(parseMarkdown(md)).toEqual([
      {
        type: 'blockquote',
        text: '**封存测试集结果，全程仅开启一次。** 这些数据未被使用。',
      },
    ])
  })

  it('keeps the candidate-table warning intact', () => {
    // This warning is the thing stopping a reader from comparing selection
    // metrics against final-test metrics. If parsing drops it, the rendered
    // report loses its most important sentence while still looking fine.
    const md = '> ⚠️ **以下为模型选择阶段指标**，不可与第 2 节直接比较。'
    const [block] = parseMarkdown(md)
    expect(block.type).toBe('blockquote')
    expect(block.text).toContain('不可与第 2 节直接比较')
    expect(block.text).toContain('⚠️')
  })

  it('separates blocks split by blank lines', () => {
    const blocks = parseMarkdown('第一段。\n\n第二段。')
    expect(blocks).toEqual([
      { type: 'paragraph', text: '第一段。' },
      { type: 'paragraph', text: '第二段。' },
    ])
  })

  it('handles an empty or nullish document', () => {
    expect(parseMarkdown('')).toEqual([])
    expect(parseMarkdown(null)).toEqual([])
    expect(parseMarkdown(undefined)).toEqual([])
  })

  it('handles a table with no data rows', () => {
    const blocks = parseMarkdown('| a | b |\n|---|---|')
    expect(blocks).toEqual([{ type: 'table', headers: ['a', 'b'], rows: [] }])
  })

  it('parses a realistic report fragment end to end', () => {
    const md = [
      '# 建模任务报告 · 客户流失预测',
      '',
      '**结论：选定模型为 `random_forest`。**',
      '',
      '## 2. 最终评估',
      '',
      '> **封存测试集结果。**',
      '',
      '| 指标 | 值 |',
      '|---|---|',
      '| accuracy | 0.8003 |',
    ].join('\n')

    const types = parseMarkdown(md).map((b) => b.type)
    expect(types).toEqual(['heading', 'paragraph', 'heading', 'blockquote', 'table'])
  })
})

describe('parseInline', () => {
  it('splits bold and code out of surrounding text', () => {
    expect(parseInline('选定模型为 `random_forest`，很好')).toEqual([
      { kind: 'text', value: '选定模型为 ' },
      { kind: 'code', value: 'random_forest' },
      { kind: 'text', value: '，很好' },
    ])
  })

  it('handles bold', () => {
    expect(parseInline('**结论**：可用')).toEqual([
      { kind: 'bold', value: '结论', children: [{ kind: 'text', value: '结论' }] },
      { kind: 'text', value: '：可用' },
    ])
  })

  it('returns plain text untouched', () => {
    expect(parseInline('没有标记')).toEqual([{ kind: 'text', value: '没有标记' }])
  })

  it('leaves HTML-looking values as raw text for React to escape', () => {
    // The report interpolates user data (dataset and feature names). The
    // parser must NOT strip or interpret this — it hands the raw string to
    // React, which escapes it. Silently sanitising here would hide the fact
    // that safety depends on the renderer never using innerHTML.
    const segments = parseInline('数据集 `<img src=x onerror=alert(1)>` 已加载')
    expect(segments[1]).toEqual({ kind: 'code', value: '<img src=x onerror=alert(1)>' })
  })
})

describe('parser vs real backend output', () => {
  // Fixture captured from report_service.build_task_report. Hand-written
  // fixtures test the parser against my assumptions; this tests it against
  // what the backend actually emits — the only version that matters.
  const md = readFileSync(
    new URL('./__fixtures__/sample-report.md', import.meta.url),
    'utf8',
  )

  it('consumes every line — nothing silently swallowed', () => {
    const blocks = parseMarkdown(md)
    const sourceLines = md.split('\n').filter((l) => l.trim()).length
    const tableLines = blocks
      .filter((b) => b.type === 'table')
      .reduce((n, b) => n + b.rows.length + 2, 0) // rows + header + alignment
    const otherBlocks = blocks.filter((b) => b.type !== 'table').length
    expect(tableLines + otherBlocks).toBe(sourceLines)
  })

  it('keeps every table structurally sound', () => {
    for (const table of parseMarkdown(md).filter((b) => b.type === 'table')) {
      for (const row of table.rows) {
        expect(row.length, `row does not match header width: ${row}`).toBe(
          table.headers.length,
        )
      }
    }
  })

  it('preserves the do-not-compare warning', () => {
    const warning = parseMarkdown(md).find(
      (b) => b.type === 'blockquote' && b.text.includes('不可与'),
    )
    expect(warning, 'the report lost its most important sentence').toBeTruthy()
  })
})

describe('parseInline nesting', () => {
  it('parses a code span nested inside bold', () => {
    // The report headline is exactly this shape. Rendering the bold body as a
    // flat string left the backticks visible in the UI.
    const [seg] = parseInline('**结论：选定模型为 `xgboost`。**')
    expect(seg.kind).toBe('bold')
    expect(seg.children).toEqual([
      { kind: 'text', value: '结论：选定模型为 ' },
      { kind: 'code', value: 'xgboost' },
      { kind: 'text', value: '。' },
    ])
  })

  it('gives plain bold a children array too, so the renderer needs no branch', () => {
    const [seg] = parseInline('**很重要**')
    expect(seg.children).toEqual([{ kind: 'text', value: '很重要' }])
  })
})
