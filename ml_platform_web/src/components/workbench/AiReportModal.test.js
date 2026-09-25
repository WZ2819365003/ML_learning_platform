import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { AiReportReader } from './AiReportModal'
import { legacyOptionChart, legacyReport, overviewReport } from './__fixtures__/reportCharts.fixtures'

// Rendered to static markup: the tree either produces a page or throws, which
// is exactly the "does an old archive still open" question. Canvas drawing is
// EChart's job in the browser and is covered by the option tests.
const render = (report, props = {}) => renderToStaticMarkup(
  createElement(AiReportReader, { report, taskName: '任务', ...props }),
)
const count = (html, needle) => html.split(needle).length - 1

describe('AiReportReader · cover', () => {
  it('shows one meta row and the headline, and nothing that counts or scores', () => {
    const html = render(overviewReport)
    expect(html).toContain('测试1-电力负荷预测 · 建模报告')
    expect(count(html, 'class="ai-report-meta-row"')).toBe(1)
    expect(html).toContain('电力负荷预测数据.csv')
    expect(html).toContain('目标列 load')
    expect(html).toContain('7 个 Run / 5 种模型')
    expect(html).toContain('xgboost 胜出，误差不到平均负荷的 1%')
    expect(html).not.toMatch(/\d+\s*\/\s*100/)
    expect(html).not.toMatch(/图表\s*\d+|表格\s*\d+/)
  })

  it('places every figure of the body and folds the appendix shut', () => {
    const html = render(overviewReport)
    overviewReport.charts.forEach((chart) => expect(html).toContain(`data-chart-id="${chart.id}"`))
    expect(html).toContain('附录 · 逐列数据概况 / 参数设置')
    // Folded: the panel body is not in the markup until it is opened.
    expect(html).not.toContain('<h4>逐列数据概况</h4>')
    expect(render(overviewReport, { appendixOpen: true })).toContain('<h4>逐列数据概况</h4>')
  })
})

describe('AiReportReader · legacy archives', () => {
  it('opens an archive that carries ECharts options and tables but no headline or meta', () => {
    const html = render(legacyReport, { taskName: '旧任务' })
    expect(html).toContain('AI 建模报告')
    expect(html).not.toContain('ai-report-meta-row')
    // No headline in the payload: the first paragraph stands in for it.
    expect(html).toContain('当前不存在可直接认定的全局最优模型')
    // The chart has no kind; its ready option passes straight through to EChart.
    expect(html).toContain(`data-chart-id="${legacyOptionChart.id}"`)
    expect(html).toContain(legacyOptionChart.title)
    expect(html).toContain(legacyOptionChart.description)
    // Legacy `tables` fold into the appendix rather than vanishing.
    expect(html).toContain('附录 · 数据概况')
  })

  it('renders a page rather than throwing for thin or malformed payloads', () => {
    expect(() => render(null)).not.toThrow()
    expect(() => render({})).not.toThrow()
    expect(() => render({ markdown: null, charts: null, tables: null })).not.toThrow()
    const html = render({
      markdown: '# 标题\n\n正文。\n\n{{chart:missing}}\n\n{{chart:pie}}',
      charts: [null, { id: 'pie', kind: 'pie' }, { title: '无 id' }, { id: 'ok', option: { series: [] } }],
    })
    // Unknown kinds and missing ids render as nothing; the pass-through chart still draws.
    expect(count(html, 'data-chart-id="')).toBe(1)
    expect(html).toContain('data-chart-id="ok"')
    expect(html).toContain('正文。')
  })
})
