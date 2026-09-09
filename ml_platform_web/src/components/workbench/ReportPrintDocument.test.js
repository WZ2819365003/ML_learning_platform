import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import ReportPrintDocument from './ReportPrintDocument'
import { resolveReportSource } from './reportViewModel'
import { legacyReport, overviewReport } from './__fixtures__/reportCharts.fixtures'

// The print document is rendered to static markup: no DOM, no ECharts init,
// which is enough to check that every figure and every sub-report is on the
// page. EChart mounts a canvas into each figure's div in the browser.
const render = (source) => renderToStaticMarkup(
  createElement(ReportPrintDocument, { source, taskName: '测试1-电力负荷预测' }),
)
const count = (html, needle) => html.split(needle).length - 1

describe('ReportPrintDocument', () => {
  it('draws every figure of the overview and of each sub-report', () => {
    const html = render(resolveReportSource({ archivedAiReport: overviewReport }))

    const overviewIds = overviewReport.charts.map((c) => c.id)
    const runIds = overviewReport.run_reports.flatMap((run) => run.charts.map((c) => c.id))
    expect(count(html, 'data-chart-id="')).toBe(overviewIds.length + runIds.length)
    overviewIds.forEach((id) => expect(html).toContain(`data-chart-id="${id}"`))
    // fold_dots is placed by the overview and again by the xgboost sub-report.
    expect(count(html, 'data-chart-id="fold_dots"')).toBe(2)
    expect(html).toContain('data-chart-id="scatter_pair"')
    expect(html).toContain('data-chart-id="loss_history"')
  })

  it('prints one section per sub-report, in payload order, after the overview', () => {
    const html = render(resolveReportSource({ archivedAiReport: overviewReport }))
    expect(count(html, 'class="report-print-run"')).toBe(overviewReport.run_reports.length)
    const cover = html.indexOf('ai-report-cover')
    const xgb = html.indexOf('xgboost_regressor · 分报告')
    const mlp = html.indexOf('mlp_dl · 分报告')
    expect(cover).toBeGreaterThan(-1)
    expect(xgb).toBeGreaterThan(cover)
    expect(mlp).toBeGreaterThan(xgb)
  })

  it('opens the appendix so the wide tables print instead of a folded header', () => {
    const html = render(resolveReportSource({ archivedAiReport: overviewReport }))
    expect(html).toContain('<h4>逐列数据概况</h4>')
    expect(html).toContain('<h4>参数设置</h4>')
    expect(html).toContain('n_estimators=600, max_depth=6, learning_rate=0.05')
  })

  it('prints a legacy archive with its pass-through chart and its tables', () => {
    const html = render(resolveReportSource({ archivedAiReport: legacyReport }))
    expect(html).toContain('data-chart-id="training_curves"')
    expect(html).toContain('<h4>数据概况</h4>')
  })

  it('prints only the markdown when the task has no AI report', () => {
    const html = render(resolveReportSource({ legacyMarkdown: '# 基础报告\n\n正文。' }))
    expect(html).toContain('基础报告')
    expect(html).not.toContain('data-chart-id=')
    expect(html).not.toContain('ai-report-cover')
  })
})
