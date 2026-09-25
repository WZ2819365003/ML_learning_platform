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
import echarts from '../utils/echarts';

export default function EChart({ option, style = { height: 300 }, notMerge = true }) {
  const divRef      = useRef(null);
  const chartRef    = useRef(null);

  // Init once
  useEffect(() => {
    if (!divRef.current) return;
    chartRef.current = echarts.init(divRef.current);

    const onResize = () => {
      if (!chartRef.current || !divRef.current) return;
      if (divRef.current.clientWidth > 0 && divRef.current.clientHeight > 0) {
        chartRef.current.resize();
      }
    };
    const resizeObserver = typeof ResizeObserver === 'undefined'
      ? null
      : new ResizeObserver(onResize);
    resizeObserver?.observe(divRef.current);
    window.addEventListener('resize', onResize);
    const resizeFrame = window.requestAnimationFrame(onResize);

    return () => {
      window.cancelAnimationFrame(resizeFrame);
      resizeObserver?.disconnect();
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

  return <div ref={divRef} style={{ width: '100%', ...style }} />;
}
