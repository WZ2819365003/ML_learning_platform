import * as echarts from 'echarts/core'
import {
  BarChart,
  BoxplotChart,
  HeatmapChart,
  LineChart,
  PieChart,
  ScatterChart,
} from 'echarts/charts'
import {
  DataZoomComponent,
  DatasetComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  MarkPointComponent,
  TitleComponent,
  ToolboxComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import { LabelLayout, UniversalTransition } from 'echarts/features'
import { CanvasRenderer } from 'echarts/renderers'
import { darkChartOption } from '../theme/chartOptions'
import { colors, chartPalette } from '../theme/tokens'

echarts.use([
  BarChart,
  BoxplotChart,
  HeatmapChart,
  LineChart,
  PieChart,
  ScatterChart,
  DataZoomComponent,
  DatasetComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  MarkPointComponent,
  TitleComponent,
  ToolboxComponent,
  TooltipComponent,
  VisualMapComponent,
  LabelLayout,
  UniversalTransition,
  CanvasRenderer,
])

echarts.registerTheme('power-trade-dark', {
  color: chartPalette, backgroundColor: 'transparent', textStyle: { color: colors.text },
  title: { textStyle: { color: colors.text }, subtextStyle: { color: colors.secondary } },
  legend: { textStyle: { color: colors.secondary } },
  categoryAxis: { axisLine: { lineStyle: { color: colors.border } }, axisLabel: { color: colors.secondary }, splitLine: { lineStyle: { color: colors.border } } },
  valueAxis: { axisLine: { lineStyle: { color: colors.border } }, axisLabel: { color: colors.secondary }, splitLine: { lineStyle: { color: colors.border } } },
})

export default {
  ...echarts,
  init(element, theme, options) {
    const printing = !!element.closest('.report-print-document')
    const chart = echarts.init(element, theme || (printing ? undefined : 'power-trade-dark'), options)
    const originalSetOption = chart.setOption.bind(chart)
    chart.setOption = (option, ...args) => originalSetOption(printing ? option : darkChartOption(option,
      name => getComputedStyle(element).getPropertyValue(name).trim()), ...args)
    // Covers all direct-init charts as well as EChart. A hidden pane gains its
    // width again when activated, without remounting or losing chart selection.
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => {
      if (element.clientWidth && element.clientHeight && !chart.isDisposed()) chart.resize()
    })
    observer?.observe(element)
    const dispose = chart.dispose.bind(chart)
    chart.dispose = () => { observer?.disconnect(); dispose() }
    return chart
  },
}
