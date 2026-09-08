import { describe, expect, it } from 'vitest'

import { splitReportOnCharts, stripLeadingTitle } from './ReportDocument'
import { overviewReport } from './__fixtures__/reportCharts.fixtures'

describe('splitReportOnCharts', () => {
  it('splits text around a chart marker', () => {
    const segs = splitReportOnCharts('前文。\n\n{{chart:loss_history}}\n\n后文。')
    expect(segs.map(s => s.kind)).toEqual(['markdown', 'chart', 'markdown'])
    expect(segs[1].value).toBe('loss_history')
  })

  it('returns one span when the model placed nothing', () => {
    // Declining to place a figure is an allowed answer, so a report with no
    // markers is the normal case, not a failure.
    expect(splitReportOnCharts('纯文字报告。')).toEqual([
      { kind: 'markdown', value: '纯文字报告。' },
    ])
  })

  it('handles a marker at the very end', () => {
    const segs = splitReportOnCharts('说明。\n\n{{chart:fold_scores}}')
    expect(segs.map(s => s.kind)).toEqual(['markdown', 'chart'])
  })

  it('keeps several markers in order', () => {
    const segs = splitReportOnCharts('a\n\n{{chart:one}}\n\nb\n\n{{chart:two}}')
    expect(segs.filter(s => s.kind === 'chart').map(s => s.value)).toEqual(['one', 'two'])
  })

  it('normalises case and spacing', () => {
    expect(splitReportOnCharts('{{ Chart : Loss_History }}')[0].value).toBe('loss_history')
  })

  it('returns nothing for empty input', () => {
    expect(splitReportOnCharts('')).toEqual([])
    expect(splitReportOnCharts()).toEqual([])
  })

  it('finds every figure of the overview mock, in reading order', () => {
    const ids = splitReportOnCharts(overviewReport.markdown)
      .filter(s => s.kind === 'chart').map(s => s.value)
    expect(ids).toEqual(['leaderboard', 'fold_dots', 'target_hist', 'field_composition', 'shap_bars'])
    expect(overviewReport.charts.map(c => c.id)).toEqual(ids)
  })
})

describe('stripLeadingTitle', () => {
  it('drops only a leading h1, which the cover already shows', () => {
    expect(stripLeadingTitle('# 报告标题\n\n## 结论\n\n正文')).toBe('## 结论\n\n正文')
    expect(stripLeadingTitle('## 结论\n\n# 不是开头的标题')).toBe('## 结论\n\n# 不是开头的标题')
    expect(stripLeadingTitle('')).toBe('')
    expect(stripLeadingTitle()).toBe('')
  })
})
