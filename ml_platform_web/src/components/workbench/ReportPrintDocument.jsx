/**
 * ReportPrintDocument — the off-screen copy of the report that print / 导出
 * PDF captures (ffe9aa2).
 *
 * It renders the same AiReportReader and RunReportBody as the screen, so every
 * figure the reader saw is drawn again here as a canvas, followed by every
 * sub-report in payload order. The appendix is forced open because a folded
 * panel prints as nothing. A legacy (non-AI) source prints its markdown only.
 */
import React from 'react'

import MarkdownReport from './MarkdownReport'
import { AiReportReader } from './AiReportModal'
import { RunReportBody } from './RunReportPanel'

export default function ReportPrintDocument({ source, taskName }) {
  return (
    <div className="report-print-document" aria-hidden="true">
      {source?.kind === 'ai' ? (
        <>
          <AiReportReader report={source.report} taskName={taskName} appendixOpen />
          {(source.report?.run_reports || []).map((run, index) => (
            <section className="report-print-run" key={run.run_id || run.model_type || index}>
              <RunReportBody report={run} appendixOpen />
            </section>
          ))}
        </>
      ) : (
        <MarkdownReport markdown={source?.markdown || ''} />
      )}
    </div>
  )
}
