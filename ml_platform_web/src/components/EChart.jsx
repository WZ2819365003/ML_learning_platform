/**
 * Thin ECharts wrapper component.
 * Replaces echarts-for-react — uses the raw `echarts` package already installed.
 *
 * Props:
 *   option  - ECharts option object (reference change triggers setOption)
 *   style   - container style (default: height 300px)
 *   notMerge - passed to setOption (default true)
 */
import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

export default function EChart({ option, style = { height: 300 }, notMerge = true }) {
  const divRef      = useRef(null);
  const chartRef    = useRef(null);

  // Init once
  useEffect(() => {
    if (!divRef.current) return;
    chartRef.current = echarts.init(divRef.current);

    const onResize = () => chartRef.current?.resize();
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      chartRef.current?.dispose();
      chartRef.current = null;
    };
  }, []);

  // Update option whenever it changes
  useEffect(() => {
    if (chartRef.current && option) {
      chartRef.current.setOption(option, notMerge);
    }
  }, [option, notMerge]);

  return <div ref={divRef} style={style} />;
}
