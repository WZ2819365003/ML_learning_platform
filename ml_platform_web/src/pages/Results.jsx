import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Pagination,
  Row,
  Skeleton,
  Space,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  ArrowLeftOutlined,
  BarChartOutlined,
  BulbOutlined,
  DotChartOutlined,
  EyeOutlined,
  FundOutlined,
  HeatMapOutlined,
  LineChartOutlined,
  ReloadOutlined,
  RiseOutlined,
  SlidersOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import { modelApi } from '../services/api';
import { formatDateTime, formatMetric, metricLabels } from '../utils/formatters';
import EChart from '../components/EChart';
import ShapView from '../components/viz/ShapView';
import TrainingHistoryChart from '../components/viz/TrainingHistoryChart';
import CrossValidationView from '../components/viz/CrossValidationView';
import PerClassMetricsTable from '../components/viz/PerClassMetricsTable';
import PRCurveChart from '../components/viz/PRCurveChart';
import CalibrationCurveChart from '../components/viz/CalibrationCurveChart';
import PredictionDistributionChart from '../components/viz/PredictionDistributionChart';
import ThresholdTuningTable from '../components/viz/ThresholdTuningTable';
import PredictedActualCurve from '../components/viz/PredictedActualCurve';
import {
  deriveRegressionViz,
  getVizEntries,
  getVizEntry,
} from '../components/viz/vizRegistry';
import { classifyVizUnavailable } from '../components/viz/vizAvailability';

const { Paragraph, Text, Title } = Typography;

const REGRESSION_METRIC_KEYS = [
  'rmse', 'mae', 'mse', 'r2',
  'cv_avg_rmse', 'cv_avg_mae', 'cv_avg_mse', 'cv_avg_r2',
];

function getApiErrorText(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map(d => d.msg ?? JSON.stringify(d)).join('；');
  return error?.message || error?.toString?.() || '加载失败';
}

function inferTaskKind(modelType, metrics = {}) {
  const mt = String(modelType || '').toLowerCase();
  if (mt === 'logistic_regression') return 'classification';
  if (mt.includes('regressor') || mt.includes('regression')) return 'regression';
  if (REGRESSION_METRIC_KEYS.some(k => typeof metrics?.[k] === 'number')) return 'regression';
  return 'classification';
}

// Heuristic: is this task a deep-learning task (epoch-style training) vs
// classic ML (K-fold CV). DL model ids in our registry start with `dl_`
// (mlp_dl, lstm_dl, ...); also fall back to detecting `history` in the
// metrics dict (already reduced to per-epoch arrays by the trainer).
function isDLTask(modelType, metrics = {}) {
  const mt = String(modelType || '').toLowerCase();
  if (mt.endsWith('_dl') || mt.startsWith('dl_') || mt === 'mlp' || mt === 'lstm' || mt === 'tcn') return true;
  if (metrics?.history && (
    Array.isArray(metrics.history?.train_loss) ||
    Array.isArray(metrics.history?.val_loss)
  )) return true;
  return false;
}

function isPercentMetric(key) {
  return /(accuracy|acc|precision|recall|auc|f1|error)$/i.test(key);
}

// ─── Regression-error helpers ────────────────────────────────────────────────
//
// Acklam's rational approximation of the inverse standard-normal CDF (Φ⁻¹).
// Accurate to ~1e-9 in the bulk; we only need it for Q-Q plot quantiles so
// that's plenty.  https://web.archive.org/web/20150910044804/http://home.online.no/~pjacklam/notes/invnorm/
function normalInv(p) {
  if (p <= 0 || p >= 1) return p === 0 ? -Infinity : p === 1 ? Infinity : NaN;
  const a = [-3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00];
  const b = [-5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00];
  const d = [ 7.784695709041462e-03,  3.224671290700398e-01,
              2.445134137142996e+00,  3.754408661907416e+00];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q, r;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
          ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

function bin1d(values, binCount = 24) {
  const finite = values.filter(v => Number.isFinite(v));
  if (finite.length === 0) return [];
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (min === max) return [{ x: min, count: finite.length, label: min.toFixed(3) }];
  const w = (max - min) / binCount;
  const bins = Array.from({ length: binCount }, (_, i) => ({
    x: min + (i + 0.5) * w,
    label: `${(min + i * w).toFixed(2)} ~ ${(min + (i + 1) * w).toFixed(2)}`,
    count: 0,
  }));
  finite.forEach(v => {
    let idx = Math.floor((v - min) / w);
    if (idx >= binCount) idx = binCount - 1;
    if (idx < 0) idx = 0;
    bins[idx].count++;
  });
  return bins;
}

function regressionDiagnostics(actual = [], predicted = []) {
  const n = Math.min(actual.length, predicted.length);
  if (n === 0) return null;
  const residuals = [];
  let sumAbs = 0, sumSq = 0, sumActual = 0;
  let mapeSum = 0, mapeN = 0;
  for (let i = 0; i < n; i++) {
    const a = Number(actual[i]);
    const p = Number(predicted[i]);
    if (!Number.isFinite(a) || !Number.isFinite(p)) continue;
    const r = p - a;
    residuals.push(r);
    sumAbs += Math.abs(r);
    sumSq  += r * r;
    sumActual += a;
    if (Math.abs(a) > 1e-9) {
      mapeSum += Math.abs(r / a);
      mapeN++;
    }
  }
  const m = residuals.length;
  if (m === 0) return null;
  const mean = sumActual / m;
  let ssTot = 0;
  for (let i = 0; i < m; i++) {
    const a = Number(actual[i]);
    if (Number.isFinite(a)) ssTot += (a - mean) ** 2;
  }
  const r2 = ssTot > 0 ? 1 - (sumSq / ssTot) : NaN;
  const rmse = Math.sqrt(sumSq / m);
  const mae = sumAbs / m;
  const mape = mapeN > 0 ? (mapeSum / mapeN) * 100 : NaN;
  // residual std (with mean=0 assumption, but compute around actual mean to
  // keep it honest if the model has a systematic bias).
  const residualMean = residuals.reduce((s, v) => s + v, 0) / m;
  const residualVar = residuals.reduce((s, v) => s + (v - residualMean) ** 2, 0) / m;
  const residualStd = Math.sqrt(residualVar);
  return { residuals, rmse, mae, r2, mape, residualMean, residualStd, n: m };
}

// ─── URL helper ──────────────────────────────────────────────────────────────
function useQuery() {
  return new URLSearchParams(useLocation().search);
}

// ─── List view ────────────────────────────────────────────────────────────────
const PAGE_SIZE = 10;

function pickResultListMetrics(record) {
  const metrics = record?.result_metrics ?? {};
  const kind = inferTaskKind(record?.model_type, metrics);
  if (kind === 'regression') {
    return {
      kind,
      primary: ['r2', metrics.r2],
      secondary: ['rmse', metrics.rmse],
    };
  }
  return {
    kind,
    primary: ['accuracy', metrics.accuracy],
    secondary: ['f1', metrics.f1 ?? metrics.f1_macro],
  };
}

function renderListMetric(metric) {
  const [key, value] = metric;
  if (typeof value !== 'number' || Number.isNaN(value)) return <Text type="secondary">-</Text>;
  return (
    <Space size={6}>
      <Text type="secondary" style={{ fontSize: 12 }}>{metricLabels[key] ?? key}</Text>
      <Text strong>{formatMetric(value, { percent: key === 'accuracy' })}</Text>
    </Space>
  );
}

function ResultListView({ navigate }) {
  const [models, setModels] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);

  useEffect(() => { void load(page); }, [page]);

  async function load(p) {
    setLoading(true);
    try {
      const res = await modelApi.listModels({ page: p, page_size: PAGE_SIZE });
      setModels(res.items ?? []);
      setTotal(res.total ?? (res.items ?? []).length);
    } catch {
      message.error('加载模型列表失败');
    } finally {
      setLoading(false);
    }
  }

  const columns = [
    {
      title: '任务名称',
      key: 'name',
      render: (_, r) => r.name ?? <Text type="secondary">{r.task_id?.slice(0, 8)}</Text>,
    },
    {
      title: '数据集',
      dataIndex: 'dataset_name',
      key: 'dataset_name',
      render: (v, r) => v ?? r.dataset_id,
      width: 180,
    },
    {
      title: '模型',
      dataIndex: 'model_type',
      key: 'model_type',
      width: 140,
      render: v => <Tag color="blue">{v}</Tag>,
    },
    {
      title: '任务类型',
      key: 'task_kind',
      width: 96,
      render: (_, r) => {
        const kind = pickResultListMetrics(r).kind;
        return <Tag color={kind === 'regression' ? 'purple' : 'blue'}>{kind === 'regression' ? '回归' : '分类'}</Tag>;
      },
    },
    {
      title: '主指标',
      key: 'primary_metric',
      width: 140,
      render: (_, r) => renderListMetric(pickResultListMetrics(r).primary),
    },
    {
      title: '辅助指标',
      key: 'secondary_metric',
      width: 140,
      render: (_, r) => renderListMetric(pickResultListMetrics(r).secondary),
    },
    {
      title: '完成时间',
      dataIndex: 'finished_at',
      key: 'finished_at',
      width: 160,
      render: v => formatDateTime(v),
    },
    {
      title: '操作',
      key: 'action',
      width: 80,
      render: (_, r) => (
        <Button
          size="small"
          icon={<EyeOutlined />}
          onClick={() => navigate(`/training/results?taskId=${r.task_id}`)}
        >
          查看
        </Button>
      ),
    },
  ];

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>结果可视化</Title>
        <Button icon={<ReloadOutlined />} onClick={() => void load(page)}>刷新</Button>
      </Space>
      <Card>
        <Paragraph type="secondary" style={{ marginBottom: 16 }}>
          已完成训练的模型列表，点击「查看」查看详细可视化分析。
        </Paragraph>
        <Table
          rowKey="task_id"
          dataSource={models}
          columns={columns}
          loading={loading}
          pagination={false}
          size="middle"
          locale={{ emptyText: <Empty description="还没有已完成的训练任务" /> }}
          onRow={r => ({
            style: { cursor: 'pointer' },
            onClick: () => navigate(`/training/results?taskId=${r.task_id}`),
          })}
        />
        <div style={{ marginTop: 16, textAlign: 'right' }}>
          <Pagination
            current={page}
            pageSize={PAGE_SIZE}
            total={total}
            showTotal={t => `共 ${t} 条`}
            onChange={p => setPage(p)}
            showSizeChanger={false}
          />
        </div>
      </Card>
    </div>
  );
}

// ─── Detail view ──────────────────────────────────────────────────────────────
// Four tabs, each driven off an independent Promise.allSettled slot so a
// single endpoint 500 never blanks the page.
function ResultDetailView({ taskId, navigate }) {
  const [models, setModels] = useState([]);
  const [detail, setDetail] = useState(null);
  const [vizState, setVizState] = useState({
    // classification core
    confusionMatrix: null, rocCurve: null,
    // regression core
    residualPlot: null, predictedVsActual: null,
    // shared
    featureImportance: null, learningCurve: null,
    // v3.2 professional set
    perClass: null, prCurve: null, calibration: null,
    threshold: null, distribution: null, shap: null,
    // v3.2.2 — DL epoch history (fetched separately for legacy DL tasks
    // that don't surface metrics.history directly)
    dlEpochs: null,
    taskKind: 'classification',
    isDL: false,
  });
  const [vizErrors, setVizErrors] = useState({});
  const [vizUnavailable, setVizUnavailable] = useState({});
  const [vizPending, setVizPending] = useState({});
  const [loading, setLoading] = useState(true);
  const [vizLoading, setVizLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('performance');

  const loadAllRef = useRef(null);

  useEffect(() => { void loadAllRef.current?.(); }, [taskId]);

  async function loadAll() {
    setLoading(true);
    setDetail(null);
    setVizErrors({});
    setVizUnavailable({});
    setVizPending({});
    setVizState(prev => Object.fromEntries(
      Object.keys(prev).map(key => [key, key === 'taskKind' ? 'classification' : key === 'isDL' ? false : null]),
    ));
    let modelItems = [];
    try {
      const res = await modelApi.listModels({ page_size: 200 });
      modelItems = res.items ?? [];
      setModels(modelItems);
    } catch { /* ignore */ }
    try {
      await loadVisualizations(modelItems, { releasePage: true });
    } finally {
      setLoading(false);
    }
  }

  async function loadVisualizations(modelItems = models, options = {}) {
    const {
      tabKey = null,
      refreshDetail = true,
      keys = null,
      releasePage = false,
    } = options;
    setVizLoading(true);
    const fallbackModel = (modelItems ?? []).find(m => m.task_id === taskId) ?? null;
    let detailPayload = detail;
    let detailError = null;

    if (refreshDetail || !detailPayload) {
      try {
        detailPayload = await modelApi.getModelDetail(taskId);
        setDetail(detailPayload);
      } catch (err) {
        detailError = getApiErrorText(err);
        setDetail(null);
      }
    }

    const metrics = detailPayload?.result_metrics ?? fallbackModel?.result_metrics ?? {};
    const modelType = detailPayload?.model_type ?? fallbackModel?.model_type;
    const taskKind = inferTaskKind(modelType, metrics);
    const dlTask = isDLTask(modelType, metrics);

    const family = dlTask ? 'dl' : 'ml';
    const defaultTab = taskKind === 'regression' ? 'comparison' : 'performance';
    const targetTab = tabKey ?? defaultTab;
    if (tabKey == null) setActiveTab(defaultTab);
    setVizState(prev => ({ ...prev, taskKind, isDL: dlTask }));

    // Detail and task type are enough to render the page shell. Chart requests
    // continue independently, so one slow endpoint no longer holds the whole
    // route behind a 30-second skeleton.
    if (releasePage) setLoading(false);

    let entries = keys
      ? keys.map(getVizEntry).filter(Boolean)
      : getVizEntries({ taskType: taskKind, family, surface: 'results', tab: targetTab });
    if (!keys) entries = entries.filter(entry => entry.loadPolicy !== 'manual');
    if (!entries.length) {
      setVizErrors(prev => ({ ...prev, detail: detailError }));
      setVizLoading(false);
      return;
    }

    const requestKeys = entries.map(entry => entry.key);
    setVizPending(prev => ({
      ...prev,
      ...Object.fromEntries(requestKeys.map(key => [key, true])),
    }));
    const results = await Promise.allSettled(entries.map(entry => entry.fetch(taskId)));
    const nextErrors = { detail: detailError };
    const nextUnavailable = {};
    const payloads = {};
    results.forEach((r, idx) => {
      const k = entries[idx].key;
      if (r.status === 'fulfilled') {
        nextErrors[k] = null;
        nextUnavailable[k] = null;
        payloads[k] = r.value;
      } else {
        nextUnavailable[k] = classifyVizUnavailable(k, r.reason);
        nextErrors[k] = nextUnavailable[k] ? null : getApiErrorText(r.reason);
        payloads[k] = null;
      }
    });
    setVizErrors(prev => ({ ...prev, ...nextErrors }));
    setVizUnavailable(prev => ({ ...prev, ...nextUnavailable }));
    const regressionDerived = deriveRegressionViz(payloads.predictedVsActual);
    setVizState(prev => ({
      ...prev,
      ...payloads,
      ...(regressionDerived.residualPlot ? { residualPlot: regressionDerived.residualPlot } : {}),
      ...(regressionDerived.distribution ? { distribution: regressionDerived.distribution } : {}),
      taskKind,
      isDL: dlTask,
    }));
    // Only surface a toast if every single endpoint failed; otherwise the
    // per-chart inline error is enough (no N-popup spam).
    const allFailed = requestKeys.every(k => nextErrors[k]);
    if (allFailed) message.error('加载可视化详情失败');
    setVizPending(prev => ({
      ...prev,
      ...Object.fromEntries(requestKeys.map(key => [key, false])),
    }));
    setVizLoading(false);
  }

  function handleTabChange(nextTab) {
    setActiveTab(nextTab);
    const entries = getVizEntries({
      taskType: vizState.taskKind,
      family: vizState.isDL ? 'dl' : 'ml',
      surface: 'results',
      tab: nextTab,
    }).filter(entry => entry.loadPolicy !== 'manual');
    const unloaded = entries.filter(entry => (
      !vizState[entry.key] && !vizPending[entry.key] && !vizUnavailable[entry.key]
    ));
    if (unloaded.length) {
      void loadVisualizations(models, {
        tabKey: nextTab,
        refreshDetail: false,
        keys: unloaded.map(entry => entry.key),
      });
    }
  }

  loadAllRef.current = loadAll;

  // Small renderer: shows chart container when data is present, inline error
  // (with retry) when that specific endpoint failed, empty state otherwise.
  function ChartSlot({ errorKey, hasData, emptyText, children }) {
    const err = vizErrors[errorKey];
    if (vizPending[errorKey]) {
      return <Skeleton active paragraph={{ rows: 5 }} style={{ padding: '28px 8px' }} />;
    }
    if (err) {
      return (
        <Alert
          type="error"
          showIcon
          message="图表加载失败"
          description={err}
          action={
            <Button size="small" onClick={() => void loadVisualizations(models, {
              refreshDetail: false,
              keys: [errorKey],
            })} icon={<ReloadOutlined />}>
              重试
            </Button>
          }
          style={{ margin: '40px 0' }}
        />
      );
    }
    if (vizUnavailable[errorKey]) {
      return (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={vizUnavailable[errorKey]}
          style={{ padding: '52px 20px' }}
        />
      );
    }
    if (hasData) return children;
    return <Empty description={emptyText} style={{ padding: '60px 0' }} />;
  }

  const selectedModel = useMemo(
    () => models.find(m => m.task_id === taskId) ?? null,
    [models, taskId],
  );

  const chartOptions = useMemo(() => {
    const confusionMatrix = vizState.confusionMatrix ? {
      tooltip: { position: 'top' },
      grid: { height: '70%', top: 36, left: 72, right: 28, bottom: 40 },
      xAxis: { type: 'category', data: vizState.confusionMatrix.labels, name: '预测类别' },
      yAxis: { type: 'category', data: vizState.confusionMatrix.labels, name: '真实类别' },
      visualMap: {
        min: 0,
        max: Math.max(...vizState.confusionMatrix.matrix.flat()),
        calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
      },
      series: [{
        type: 'heatmap',
        data: vizState.confusionMatrix.matrix.flatMap((row, ri) =>
          row.map((v, ci) => [ci, ri, v])
        ),
        label: { show: true },
      }],
    } : null;

    const rocCurve = vizState.rocCurve ? {
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0 },
      grid: { top: 36, left: 56, right: 20, bottom: 54 },
      xAxis: { type: 'value', min: 0, max: 1, name: 'FPR' },
      yAxis: { type: 'value', min: 0, max: 1, name: 'TPR' },
      series: vizState.rocCurve.multiclass
        ? vizState.rocCurve.curves.map(c => ({
            name: `${c.class} (AUC ${c.auc})`,
            type: 'line', smooth: true,
            data: c.fpr.map((v, i) => [v, c.tpr[i]]),
          }))
        : [
            {
              name: `ROC (AUC ${vizState.rocCurve.auc})`,
              type: 'line', smooth: true,
              data: vizState.rocCurve.fpr.map((v, i) => [v, vizState.rocCurve.tpr[i]]),
            },
            {
              name: '随机基线', type: 'line',
              data: [[0, 0], [1, 1]],
              lineStyle: { type: 'dashed' }, symbol: 'none',
            },
          ],
    } : null;

    const featureImportance = vizState.featureImportance ? {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { top: 24, left: 150, right: 20, bottom: 24 },
      xAxis: { type: 'value', name: '重要性' },
      yAxis: {
        type: 'category',
        data: [...vizState.featureImportance.features].slice(0, 10).reverse(),
      },
      series: [{
        type: 'bar',
        data: [...vizState.featureImportance.importance].slice(0, 10).reverse(),
        itemStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 1, y2: 0,
            colorStops: [{ offset: 0, color: '#0f766e' }, { offset: 1, color: '#38bdf8' }],
          },
        },
      }],
    } : null;

    // NOTE: learningCurve is no longer rendered as an ECharts option here.
    // The K-Fold CV view is now built by <CrossValidationView/> directly
    // off the same payload, with mean ± std summary cards + grouped bar
    // chart + per-fold table. Kept as `null` for shape compatibility with
    // the chartOptions return below.
    const learningCurve = null;

    const predictedVsActual = vizState.predictedVsActual ? (() => {
      const acts = vizState.predictedVsActual.actual ?? [];
      const preds = vizState.predictedVsActual.predicted ?? [];
      const flat = [...acts, ...preds].filter(Number.isFinite);
      const lo = flat.length ? Math.min(...flat) : 0;
      const hi = flat.length ? Math.max(...flat) : 1;
      const pad = (hi - lo) * 0.04 || 1;
      const axisMin = lo - pad;
      const axisMax = hi + pad;
      // ±1σ error band — gives a quick visual feel for how wide the
      // residual cloud is at any given true value. Computed from
      // residuals so it's actually meaningful (not a guess).
      const diag = regressionDiagnostics(acts, preds);
      const sigma = diag?.residualStd ?? 0;
      return {
        tooltip: { trigger: 'item', formatter: p => {
          if (p.seriesType === 'line') return p.seriesName;
          return `真实值: ${p.value[0]?.toFixed?.(4) ?? p.value[0]}<br/>预测值: ${p.value[1]?.toFixed?.(4) ?? p.value[1]}`;
        } },
        legend: { bottom: 0, data: ['样本', 'y = x （理想）', '±1σ 误差带'] },
        grid: { top: 28, left: 70, right: 24, bottom: 56 },
        xAxis: { type: 'value', name: '真实值', min: axisMin, max: axisMax },
        yAxis: { type: 'value', name: '预测值', min: axisMin, max: axisMax },
        series: [
          {
            name: '样本',
            type: 'scatter',
            symbolSize: 6,
            data: acts.map((v, i) => [v, preds[i]]),
            itemStyle: { color: '#2563eb', opacity: 0.6 },
            z: 3,
          },
          {
            name: 'y = x （理想）',
            type: 'line',
            data: [[axisMin, axisMin], [axisMax, axisMax]],
            symbol: 'none',
            lineStyle: { color: '#10b981', width: 2 },
            z: 2,
          },
          ...(sigma > 0 ? [
            {
              name: '±1σ 误差带',
              type: 'line',
              data: [[axisMin, axisMin + sigma], [axisMax, axisMax + sigma]],
              symbol: 'none',
              lineStyle: { color: '#f59e0b', width: 1, type: 'dashed', opacity: 0.7 },
              z: 1,
            },
            {
              name: '±1σ 误差带',
              type: 'line',
              data: [[axisMin, axisMin - sigma], [axisMax, axisMax - sigma]],
              symbol: 'none',
              lineStyle: { color: '#f59e0b', width: 1, type: 'dashed', opacity: 0.7 },
              z: 1,
            },
          ] : []),
        ],
      };
    })() : null;

    const residualPlot = vizState.residualPlot ? {
      tooltip: { trigger: 'item', formatter: p => `预测值: ${p.value[0]}<br/>残差: ${p.value[1]}` },
      grid: { top: 28, left: 70, right: 24, bottom: 56 },
      xAxis: { type: 'value', name: '预测值' },
      yAxis: { type: 'value', name: '残差' },
      series: [{
        name: '残差',
        type: 'scatter',
        symbolSize: 6,
        data: vizState.residualPlot.predicted.map((v, i) => [v, vizState.residualPlot.residuals[i]]),
        itemStyle: { color: '#0f766e', opacity: 0.72 },
        markLine: {
          symbol: 'none',
          lineStyle: { type: 'dashed', color: '#94a3b8' },
          data: [{ yAxis: 0 }],
        },
      }],
    } : null;

    // ── Regression-only error diagnostics (residual histogram + Q-Q) ──
    // Both derive from the residual array, which we recompute here from the
    // (actual, predicted) pair so they stay in sync with predictedVsActual.
    let residualHistogram = null;
    let qqPlot = null;
    if (vizState.taskKind === 'regression' && vizState.predictedVsActual) {
      const diag = regressionDiagnostics(
        vizState.predictedVsActual.actual ?? [],
        vizState.predictedVsActual.predicted ?? [],
      );
      if (diag?.residuals?.length) {
        // Histogram
        const bins = bin1d(diag.residuals, 24);
        residualHistogram = {
          tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
            formatter: ps => {
              const b = bins[ps[0].dataIndex];
              return `区间 <code>${b.label}</code><br/>样本数: <b>${b.count}</b>`;
            },
          },
          grid: { top: 36, left: 56, right: 24, bottom: 50 },
          xAxis: { type: 'category', data: bins.map(b => b.x.toFixed(2)),
            name: '残差', axisLabel: { rotate: 30, fontSize: 10 } },
          yAxis: { type: 'value', name: '样本数' },
          series: [{
            name: '残差分布',
            type: 'bar',
            data: bins.map(b => b.count),
            itemStyle: {
              color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                colorStops: [{ offset: 0, color: '#7c3aed' }, { offset: 1, color: '#a78bfa' }] },
            },
            markLine: {
              symbol: 'none',
              lineStyle: { color: '#10b981', type: 'dashed' },
              label: { show: false },
              data: [{ xAxis: bins.findIndex(b => b.x >= 0) }],
            },
          }],
        };
        // Q-Q plot — sample quantiles vs theoretical normal quantiles
        // Standardise residuals so the reference line is just y = x.
        const std = diag.residualStd > 0 ? diag.residualStd : 1;
        const standardized = diag.residuals
          .map(r => (r - diag.residualMean) / std)
          .sort((a, b) => a - b);
        const qq = standardized.map((s, i) => {
          const p = (i + 0.5) / standardized.length;
          return [normalInv(p), s];
        });
        const qqXs = qq.map(p => p[0]).filter(Number.isFinite);
        const qqMin = qqXs.length ? Math.min(...qqXs) : -3;
        const qqMax = qqXs.length ? Math.max(...qqXs) : 3;
        qqPlot = {
          tooltip: { trigger: 'item', formatter: p => {
            if (p.seriesType === 'line') return p.seriesName;
            return `理论分位: ${p.value[0]?.toFixed(3)}<br/>样本分位: ${p.value[1]?.toFixed(3)}`;
          } },
          legend: { bottom: 0 },
          grid: { top: 28, left: 56, right: 24, bottom: 56 },
          xAxis: { type: 'value', name: '理论正态分位', min: qqMin - 0.3, max: qqMax + 0.3 },
          yAxis: { type: 'value', name: '残差分位（标准化）' },
          series: [
            { name: '样本', type: 'scatter', data: qq, symbolSize: 5,
              itemStyle: { color: '#0ea5e9', opacity: 0.75 } },
            { name: '正态参考线 y = x', type: 'line',
              data: [[qqMin, qqMin], [qqMax, qqMax]],
              symbol: 'none',
              lineStyle: { color: '#10b981', width: 2 } },
          ],
        };
      }
    }

    return { confusionMatrix, rocCurve, featureImportance, learningCurve,
             predictedVsActual, residualPlot, residualHistogram, qqPlot };
  }, [vizState]);

  const resultMetrics = detail?.result_metrics ?? selectedModel?.result_metrics ?? {};
  const taskKind = vizState.taskKind ?? inferTaskKind(detail?.model_type ?? selectedModel?.model_type, resultMetrics);
  const metricItems = Object.entries(resultMetrics)
    .filter(([, v]) => typeof v === 'number' && !Number.isNaN(v))
    .filter(([k]) => k !== 'cv_folds');
  const headlineKeys = taskKind === 'regression'
    ? ['r2', 'rmse', 'mae', 'mse']
    : ['accuracy', 'f1', 'cv_avg_accuracy', 'cv_avg_f1'];
  const headlineMetrics = headlineKeys
    .map(k => [k, resultMetrics?.[k]])
    .filter(([, v]) => typeof v === 'number' && !Number.isNaN(v));

  // ─── Tab content builders ─────────────────────────────────────────────────
  // Two completely independent tab rosters so the user never sees a card
  // that doesn't fit their task type (e.g. ROC on a regression task).

  // Shared CSS-in-JS for tab card visual rhythm — consistent margins,
  // shadow, and a bordered header so the page no longer reads as a wall
  // of look-alike rectangles.
  const cardProps = (accent = '#2563eb') => ({
    variant: 'borderless',
    styles: {
      header: {
        borderBottom: `1px solid ${accent}1f`,
        background: `linear-gradient(90deg, ${accent}08 0%, transparent 100%)`,
        fontSize: 14,
      },
      body: { padding: 16 },
    },
    style: { boxShadow: '0 1px 3px rgba(15, 23, 42, 0.05)', borderRadius: 12 },
  });

  // ── CLASSIFICATION TABS ───────────────────────────────────────────────
  const classificationPerformanceTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}>
          <Card title={<Space><HeatMapOutlined style={{ color: '#2563eb' }} />混淆矩阵</Space>} {...cardProps('#2563eb')}>
            <ChartSlot errorKey="confusionMatrix" hasData={!!vizState.confusionMatrix} emptyText="暂无混淆矩阵数据">
              <EChart option={chartOptions.confusionMatrix} style={{ height: 360 }} />
            </ChartSlot>
          </Card>
        </Col>
        <Col xs={24} xl={12}>
          <Card title={<Space><LineChartOutlined style={{ color: '#7c3aed' }} />ROC 曲线</Space>} {...cardProps('#7c3aed')}>
            <ChartSlot errorKey="rocCurve" hasData={!!vizState.rocCurve} emptyText="暂无 ROC 曲线数据">
              <EChart option={chartOptions.rocCurve} style={{ height: 360 }} />
            </ChartSlot>
          </Card>
        </Col>
        <Col xs={24} xl={12}>
          <Card title={<Space><LineChartOutlined style={{ color: '#0ea5e9' }} />Precision-Recall 曲线</Space>} {...cardProps('#0ea5e9')}>
            <ChartSlot errorKey="prCurve" hasData={!!vizState.prCurve} emptyText="暂无 PR 曲线数据">
              <PRCurveChart payload={vizState.prCurve} height={340} />
            </ChartSlot>
          </Card>
        </Col>
        <Col xs={24}>
          <Card title={<Space><BarChartOutlined style={{ color: '#10b981' }} />逐类指标</Space>} {...cardProps('#10b981')}>
            <ChartSlot errorKey="perClass" hasData={!!vizState.perClass} emptyText="暂无逐类指标">
              <PerClassMetricsTable payload={vizState.perClass} />
            </ChartSlot>
          </Card>
        </Col>
      </Row>
    </Space>
  );

  // ── REGRESSION TABS ───────────────────────────────────────────────────
  const regressionComparisonTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card
        title={<Space><DotChartOutlined style={{ color: '#2563eb' }} />预测 vs 实际（含 y = x 理想线 + ±1σ 误差带）</Space>}
        {...cardProps('#2563eb')}
      >
        <ChartSlot
          errorKey="predictedVsActual"
          hasData={!!vizState.predictedVsActual}
          emptyText="暂无预测-真实值数据（需要回归任务的测试集预测）"
        >
          <EChart option={chartOptions.predictedVsActual} style={{ height: 420 }} />
        </ChartSlot>
        <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
          点应密集地落在绿色 <code>y = x</code> 对角线两侧；超出黄色虚线 ±1σ 的样本属于较大误差。
        </Paragraph>
      </Card>
      <Card
        title={<Space><LineChartOutlined style={{ color: '#0f766e' }} />预测值 vs 实际值（时序曲线）</Space>}
        {...cardProps('#0f766e')}
      >
        <ChartSlot
          errorKey="predictedVsActual"
          hasData={!!vizState.predictedVsActual}
          emptyText="暂无预测-真实值曲线数据"
        >
          <PredictedActualCurve payload={vizState.predictedVsActual} height={360} />
        </ChartSlot>
        <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
          两条线用于定位具体误差区间：可观察峰值是否被削平、预测是否整体滞后。曲线与散点图复用同一次预测结果，不重复计算模型。
        </Paragraph>
      </Card>
    </Space>
  );

  // ── 训练过程 tab — task-shape-aware ────────────────────────────────────
  // DL tasks (epoch history available)  → epoch line chart (loss + metric)
  // ML tasks (K-fold CV)                → per-metric mean ± std + bar + table
  // Both                                → both cards stacked, DL first
  // Neither                             → friendly empty state
  const dlHistory = resultMetrics?.history
    ?? (Array.isArray(vizState.dlEpochs?.items) ? vizState.dlEpochs.items : null);
  const hasDLHistory = !!dlHistory && (
    Array.isArray(dlHistory)
      ? dlHistory.length > 0
      : Object.values(dlHistory).some(v => Array.isArray(v) && v.length > 0)
  );
  const hasCVData = !!vizState.learningCurve && Array.isArray(vizState.learningCurve.steps)
    && vizState.learningCurve.steps.length > 0;

  const regressionTrainingTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Alert
        type="info"
        showIcon
        message="回归训练过程"
        description={
          hasDLHistory
            ? '回归深度学习任务以 epoch 观察 train/validation loss、RMSE/MAE 等误差指标；验证误差回升通常表示过拟合。'
            : '回归机器学习任务用 K-Fold 观察 R²、RMSE、MAE 等误差指标的均值与波动；R² 越高越好，RMSE/MAE 越低越好。'
        }
        style={{ marginBottom: 0 }}
      />

      {hasDLHistory && (
        <Card
          title={<Space><RiseOutlined style={{ color: '#7c3aed' }} />Epoch 训练历史</Space>}
          {...cardProps('#7c3aed')}
        >
          <TrainingHistoryChart
            history={dlHistory}
            taskType={taskKind}
            height={360}
            xAxisName="Epoch"
          />
          <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
            橙/红线为训练 / 验证损失（左轴），绿/蓝线为指标（右轴）。星标处为该曲线的极值。
          </Paragraph>
        </Card>
      )}

      <Card
        title={<Space><RiseOutlined style={{ color: '#7c3aed' }} />回归 K-Fold 交叉验证（R² / RMSE / MAE）</Space>}
        {...cardProps('#7c3aed')}
      >
        <ChartSlot
          errorKey="learningCurve"
          hasData={hasCVData}
          emptyText={
            hasDLHistory
              ? '该任务为深度学习回归训练，未运行 K-Fold CV（请查看上方 epoch 训练历史）'
              : '暂无回归 CV 数据（期望 r2 / rmse / mae / mse）'
          }
        >
          <CrossValidationView
            payload={vizState.learningCurve}
            taskKind="regression"
            height={340}
          />
        </ChartSlot>
      </Card>
    </Space>
  );

  const classificationTrainingTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Alert
        type="info"
        showIcon
        message="分类训练过程"
        description={
          hasDLHistory
            ? '分类深度学习任务以 epoch 观察 train/validation loss、Accuracy、F1 等指标；验证损失回升或指标停滞通常表示过拟合。'
            : '分类机器学习任务用 K-Fold 观察 Accuracy、F1、ROC-AUC 等指标的均值与波动；均值越高且标准差越小，模型越稳定。'
        }
        style={{ marginBottom: 0 }}
      />

      {hasDLHistory && (
        <Card
          title={<Space><RiseOutlined style={{ color: '#7c3aed' }} />Epoch 训练历史</Space>}
          {...cardProps('#7c3aed')}
        >
          <TrainingHistoryChart
            history={dlHistory}
            taskType={taskKind}
            height={360}
            xAxisName="Epoch"
          />
          <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
            橙/红线为训练 / 验证损失（左轴），绿/蓝线为分类指标（右轴）。星标处为该曲线的极值。
          </Paragraph>
        </Card>
      )}

      <Card
        title={<Space><RiseOutlined style={{ color: '#0ea5e9' }} />分类 K-Fold 交叉验证（Accuracy / F1 / ROC-AUC）</Space>}
        {...cardProps('#0ea5e9')}
      >
        <ChartSlot
          errorKey="learningCurve"
          hasData={hasCVData}
          emptyText={
            hasDLHistory
              ? '该任务为深度学习分类训练，未运行 K-Fold CV（请查看上方 epoch 训练历史）'
              : '暂无分类 CV 数据（期望 accuracy / f1 / roc_auc）'
          }
        >
          <CrossValidationView
            payload={vizState.learningCurve}
            taskKind="classification"
            height={340}
          />
        </ChartSlot>
      </Card>
    </Space>
  );

  const explanationTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card title={<Space><BulbOutlined style={{ color: '#f59e0b' }} />SHAP / 特征解释</Space>} {...cardProps('#f59e0b')}>
        {!vizState.shap && !vizErrors.shap && !vizPending.shap ? (
          <Empty
            description="SHAP 需要额外计算，部分模型可能耗时数分钟"
            style={{ padding: '48px 0' }}
          >
            <Button
              type="primary"
              onClick={() => void loadVisualizations(models, {
                refreshDetail: false,
                keys: ['shap'],
              })}
            >
              开始计算 SHAP
            </Button>
          </Empty>
        ) : (
          <ChartSlot errorKey="shap" hasData={!!vizState.shap} emptyText="暂无 SHAP 解释数据">
            <ShapView payload={vizState.shap} />
          </ChartSlot>
        )}
      </Card>
      <Card title={<Space><BarChartOutlined style={{ color: '#10b981' }} />模型原生特征重要性 (Top 10)</Space>} {...cardProps('#10b981')}>
        <ChartSlot errorKey="featureImportance" hasData={!!vizState.featureImportance}
          emptyText="该模型不支持 feature_importances_（仅树模型可用）">
          <EChart option={chartOptions.featureImportance} style={{ height: 360 }} />
        </ChartSlot>
      </Card>
    </Space>
  );

  const classificationThresholdTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card title={<Space><SlidersOutlined style={{ color: '#7c3aed' }} />阈值敏感度分析</Space>} {...cardProps('#7c3aed')}>
        <ChartSlot errorKey="threshold" hasData={!!vizState.threshold}
          emptyText="暂无阈值分析（需要二分类 + predict_proba）">
          <ThresholdTuningTable payload={vizState.threshold} />
        </ChartSlot>
      </Card>
      <Card title={<Space><LineChartOutlined style={{ color: '#0ea5e9' }} />校准曲线</Space>} {...cardProps('#0ea5e9')}>
        <ChartSlot errorKey="calibration" hasData={!!vizState.calibration}
          emptyText="暂无校准曲线数据（需要二分类 + predict_proba）">
          <CalibrationCurveChart payload={vizState.calibration} height={340} />
        </ChartSlot>
      </Card>
      <Card title={<Space><BarChartOutlined style={{ color: '#10b981' }} />预测分布</Space>} {...cardProps('#10b981')}>
        <ChartSlot errorKey="distribution" hasData={!!vizState.distribution}
          emptyText="暂无预测分布数据">
          <PredictionDistributionChart payload={vizState.distribution} height={340} />
        </ChartSlot>
      </Card>
    </Space>
  );

  // ── REGRESSION 误差诊断 ─────────────────────────────────────────────────
  // Built off the (actual, predicted) pair: residual histogram lets you spot
  // skew/bias, residual-vs-predicted reveals heteroscedasticity, Q-Q plot
  // checks normality assumption; the metric strip pins the headline numbers.
  const regressionDiag = useMemo(() => regressionDiagnostics(
    vizState.predictedVsActual?.actual ?? [],
    vizState.predictedVsActual?.predicted ?? [],
  ), [vizState.predictedVsActual]);

  const regressionDiagnosisTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      {/* Headline metric strip — recomputed locally so it stays in sync
          with whatever the model actually predicted on the test set. */}
      {regressionDiag && (
        <Row gutter={[12, 12]}>
          <Col xs={12} md={6}>
            <Card {...cardProps('#10b981')}>
              <Statistic title="R²" value={regressionDiag.r2}
                valueStyle={{ color: regressionDiag.r2 >= 0.8 ? '#10b981' : regressionDiag.r2 >= 0.5 ? '#f59e0b' : '#ef4444' }}
                precision={4} />
            </Card>
          </Col>
          <Col xs={12} md={6}>
            <Card {...cardProps('#7c3aed')}>
              <Statistic title="RMSE" value={regressionDiag.rmse} precision={4}
                valueStyle={{ color: '#7c3aed' }} />
            </Card>
          </Col>
          <Col xs={12} md={6}>
            <Card {...cardProps('#2563eb')}>
              <Statistic title="MAE" value={regressionDiag.mae} precision={4}
                valueStyle={{ color: '#2563eb' }} />
            </Card>
          </Col>
          <Col xs={12} md={6}>
            <Card {...cardProps('#f59e0b')}>
              <Statistic title="MAPE (%)" value={regressionDiag.mape}
                precision={2} suffix="%" valueStyle={{ color: '#f59e0b' }}
                formatter={v => Number.isFinite(Number(v)) ? Number(v).toFixed(2) : '—'} />
            </Card>
          </Col>
        </Row>
      )}

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}>
          <Card title={<Space><BarChartOutlined style={{ color: '#7c3aed' }} />残差直方图</Space>} {...cardProps('#7c3aed')}>
            <ChartSlot errorKey="predictedVsActual" hasData={!!chartOptions.residualHistogram}
              emptyText="暂无残差数据">
              <EChart option={chartOptions.residualHistogram} style={{ height: 320 }} />
            </ChartSlot>
            <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
              理想形态：以 0 为中心、近似正态钟形。明显偏斜或多峰提示模型存在系统性偏差。
            </Paragraph>
          </Card>
        </Col>
        <Col xs={24} xl={12}>
          <Card title={<Space><LineChartOutlined style={{ color: '#0ea5e9' }} />残差 vs 预测值（异方差检查）</Space>} {...cardProps('#0ea5e9')}>
            <ChartSlot errorKey="residualPlot" hasData={!!vizState.residualPlot} emptyText="暂无残差数据">
              <EChart option={chartOptions.residualPlot} style={{ height: 320 }} />
            </ChartSlot>
            <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
              点应在 y = 0 横线上下随机分布。出现喇叭口或曲线模式说明残差方差随预测值变化（异方差）。
            </Paragraph>
          </Card>
        </Col>
        <Col xs={24} xl={12}>
          <Card title={<Space><DotChartOutlined style={{ color: '#10b981' }} />Q-Q 正态性检验</Space>} {...cardProps('#10b981')}>
            <ChartSlot errorKey="predictedVsActual" hasData={!!chartOptions.qqPlot}
              emptyText="暂无残差数据">
              <EChart option={chartOptions.qqPlot} style={{ height: 320 }} />
            </ChartSlot>
            <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
              点越贴近绿色对角线，残差越接近正态分布。两端翘起表示重尾。
            </Paragraph>
          </Card>
        </Col>
        <Col xs={24} xl={12}>
          <Card title={<Space><FundOutlined style={{ color: '#f59e0b' }} />预测分布</Space>} {...cardProps('#f59e0b')}>
            <ChartSlot errorKey="distribution" hasData={!!vizState.distribution} emptyText="暂无预测分布数据">
              <PredictionDistributionChart payload={vizState.distribution} height={320} />
            </ChartSlot>
          </Card>
        </Col>
      </Row>
    </Space>
  );

  // ── Final tab roster, by task type ─────────────────────────────────────
  const tabItems = taskKind === 'regression' ? [
    {
      key: 'comparison',
      label: <Space><DotChartOutlined />预测对比</Space>,
      children: regressionComparisonTab,
    },
    {
      key: 'training',
      label: <Space><RiseOutlined />训练过程</Space>,
      children: regressionTrainingTab,
    },
    {
      key: 'explain',
      label: <Space><BulbOutlined />解释性</Space>,
      children: explanationTab,
    },
    {
      key: 'diagnosis',
      label: <Space><WarningOutlined />误差诊断</Space>,
      children: regressionDiagnosisTab,
    },
  ] : [
    {
      key: 'performance',
      label: <Space><HeatMapOutlined />性能</Space>,
      children: classificationPerformanceTab,
    },
    {
      key: 'training',
      label: <Space><RiseOutlined />训练过程</Space>,
      children: classificationTrainingTab,
    },
    {
      key: 'explain',
      label: <Space><BulbOutlined />解释性</Space>,
      children: explanationTab,
    },
    {
      key: 'threshold',
      label: <Space><SlidersOutlined />阈值与分布</Space>,
      children: classificationThresholdTab,
    },
  ];

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/training/results')}>
            返回列表
          </Button>
          <Title level={2} style={{ margin: 0 }}>结果可视化详情</Title>
        </Space>
        <Button icon={<ReloadOutlined />} loading={vizLoading} onClick={() => void loadVisualizations(models, {
          tabKey: activeTab,
        })}>
          刷新
        </Button>
      </Space>

      {loading ? (
        <Card><Skeleton active paragraph={{ rows: 4 }} /></Card>
      ) : (
        <>
          {headlineMetrics.length > 0 && (
            <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
              {headlineMetrics.map(([k, v]) => (
                <Col xs={24} md={12} xl={6} key={k}>
                  <Card>
                    <Statistic
                      title={metricLabels[k] ?? k}
                      value={isPercentMetric(k) ? Number(v) * 100 : Number(v)}
                      suffix={isPercentMetric(k) ? '%' : undefined}
                      precision={isPercentMetric(k) ? 2 : 4}
                    />
                  </Card>
                </Col>
              ))}
            </Row>
          )}

          <Card style={{ marginBottom: 16 }}>
            <Descriptions bordered column={{ xs: 1, md: 2, xl: 4 }}>
              <Descriptions.Item label="数据集">{detail?.dataset?.name ?? selectedModel?.dataset_name ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="模型">{detail?.model_type ?? selectedModel?.model_type ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="目标列">{detail?.target_column ?? selectedModel?.target_column ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="完成时间">{formatDateTime(detail?.finished_at ?? selectedModel?.finished_at)}</Descriptions.Item>
              <Descriptions.Item label="测试集比例">{formatMetric(detail?.test_size ?? 0.2, { digits: 2 })}</Descriptions.Item>
              <Descriptions.Item label="任务类型">
                <Tag color={taskKind === 'regression' ? 'purple' : 'blue'}>
                  {taskKind === 'regression' ? '回归' : '分类'}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="任务 ID" span={2}><Text code>{taskId}</Text></Descriptions.Item>
            </Descriptions>
            {vizLoading && <Text type="secondary" style={{ display: 'block', marginTop: 12 }}>正在加载图表数据…</Text>}
          </Card>

          {metricItems.length > 0 && (
            <Card style={{ marginBottom: 16 }}>
              <Space wrap size={[12, 12]}>
                {metricItems.map(([k, v]) => (
                  <Tag key={k} color="blue" style={{ padding: '6px 10px', borderRadius: 999 }}>
                    {metricLabels[k] ?? k}: {k.includes('accuracy') ? formatMetric(v, { percent: true }) : formatMetric(v)}
                  </Tag>
                ))}
              </Space>
            </Card>
          )}

          <Card styles={{ body: { padding: '0 24px 24px' } }}>
            <Tabs
              // Re-keying on taskKind forces ant-design Tabs to forget the
              // previous taskKind's active key (e.g. 'performance' from a
              // classification view) when switching to a regression layout
              // whose roster doesn't include that key.
              key={taskKind}
              activeKey={activeTab}
              onChange={handleTabChange}
              items={tabItems}
            />
          </Card>
        </>
      )}
    </div>
  );
}

// ─── Root ─────────────────────────────────────────────────────────────────────
const Results = () => {
  const query = useQuery();
  const navigate = useNavigate();
  const taskId = query.get('taskId');

  if (taskId) return <ResultDetailView taskId={taskId} navigate={navigate} />;
  return <ResultListView navigate={navigate} />;
};

export default Results;
