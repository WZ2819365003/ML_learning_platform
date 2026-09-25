import React from 'react'

// Shared summary layout from power-trade MetricGrid, with stronger text contrast.
export default function MetricCard({ icon, label, value, color = 'var(--brand-500)' }) {
  const numeric = typeof value === 'number'
  const displayValue = value == null || value === '' ? '—' : numeric ? value.toLocaleString('zh-CN') : value

  return (
    <article className="metric-card" style={{ '--metric-accent': color }}>
      {icon && <div className="metric-icon" aria-hidden="true">{icon}</div>}
      <div className="metric-content">
        <div className="metric-label">{label}</div>
        <div className={`metric-value${numeric ? '' : ' metric-value-text'}`} title={String(displayValue)}>{displayValue}</div>
      </div>
    </article>
  )
}
