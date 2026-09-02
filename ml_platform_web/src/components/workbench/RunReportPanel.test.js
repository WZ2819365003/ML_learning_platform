import { describe, expect, it } from 'vitest'

import { splitReportOnCharts } from './RunReportPanel'

describe('splitReportOnCharts', () => {
  it('splits text around a chart marker', () => {
    const segs = splitReportOnCharts('前文。\n\n{{chart:loss_history}}\n\n后文。')
    expect(segs.map(s => s.kind)).toEqual(['markdown', 'chart', 'markdown'])
    expect(segs[1].value).toBe('loss_history')
    expect(segs[0].value).toBe('前文。')
  })

  it('returns one span when there are no markers', () => {
    const segs = splitReportOnCharts('纯文字报告。')
    expect(segs).toEqual([{ kind: 'markdown', value: '纯文字报告。' }])
  })

  it('handles a marker at the very end', () => {
    // The generator places markers after the paragraph they illustrate, so the
    // last one frequently has nothing following it.
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
})
