import { chartPalette, colors } from './tokens'

const darkText = new Set(['#0f172a', '#172033', '#334155', '#475569', '#333', '#333333', '#666', '#64748b', '#94a3b8'])
const lightLine = new Set(['#e2e8f0', '#cbd5e1', '#eee', '#f0f0f0', '#ddd', 'rgba(15, 23, 42, 0.08)', 'rgba(15, 23, 42, 0.28)'])
const legacyPalette = { '#2563eb': colors.primary, '#3b82f6': colors.primary, '#1d4ed8': colors.primary, '#dc2626': colors.error, '#10b981': colors.success, '#f59e0b': colors.warning }

/** Adapt presentation only. Data, callback functions and semantic series colors
 * retain their identities, so legacy reports and comparison logic stay intact. */
export function darkChartOption(option, resolveVariable = () => '') {
  const visit = (value, path = '') => {
    if (Array.isArray(value)) return value.map(item => visit(item, path))
    if (!value || typeof value !== 'object') {
      if (typeof value !== 'string') return value
      if (value.startsWith('var(')) return resolveVariable(value.slice(4, -1)) || colors.secondary
      if (/(color|fill|stroke)$/i.test(path)) {
        if (legacyPalette[value]) return legacyPalette[value]
        if (darkText.has(value)) return /line|border/i.test(path) ? colors.border : colors.text
        if (lightLine.has(value)) return colors.border
        if (/background/i.test(path) && ['#fff', '#ffffff', 'white'].includes(value)) return colors.surface
      }
      return value
    }
    // Typed arrays and ECharts gradients are not plain option records.
    if (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) return value
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key,
      key === 'data' || key === 'source' || key === 'renderItem' ? item : visit(item, `${path}.${key}`),
    ]))
  }
  const result = visit(option)
  return { color: chartPalette, backgroundColor: 'transparent', ...result,
    textStyle: { color: colors.text, ...result.textStyle },
    tooltip: result.tooltip === false ? false : { backgroundColor: colors.elevated, borderColor: colors.border, ...result.tooltip,
      textStyle: { color: colors.text, ...result.tooltip?.textStyle } },
  }
}
