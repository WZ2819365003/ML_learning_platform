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
  EyeOutlined,
  HeatMapOutlined,
  LineChartOutlined,
  ReloadOutlined,
  RiseOutlined,
  SlidersOutlined,
} from '@ant-design/icons';
import * as echarts from 'echarts';
import { modelApi, vizApi } from '../services/api';
import { formatDateTime, formatMetric, metricLabels } from '../utils/formatters';
import ShapView from '../components/viz/ShapView';
import TrainingHistoryChart from '../components/viz/TrainingHistoryChart';
import PerClassMetricsTable from '../components/viz/PerClassMetricsTable';
import PRCurveChart from '../components/viz/PRCurveChart';
import CalibrationCurveChart from '../components/viz/CalibrationCurveChart';
import PredictionDistributionChart from '../components/viz/PredictionDistributionChart';
import ThresholdTuningTable from '../components/viz/ThresholdTuningTable';

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
  if (mt.includes('regressor') || mt.includes('regression')) return 'regression';
  if (REGRESSION_METRIC_KEYS.some(k => typeof metrics?.[k] === 'number')) return 'regression';
  return 'classification';
}

function isPercentMetric(key) {
  return /(accuracy|acc|precision|recall|auc|f1|error)$/i.test(key);
}

// ─── URL helper ──────────────────────────────────────────────────────────────
function useQuery() {
  return new URLSearchParams(useLocation().search);
}

// ─── ECharts hook ─────────────────────────────────────────────────────────────
function useChart(ref, option) {
  useEffect(() => {
    if (!ref.current || !option) return undefined;
    const instance = echarts.init(ref.current);
    instance.setOption(option, true);
    const onResize = () => instance.resize();
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      instance.dispose();
    };
  }, [option, ref]);
}

// ─── List view ────────────────────────────────────────────────────────────────
const PAGE_SIZE = 10;

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
      title: '准确率',
      key: 'accuracy',
      width: 90,
      render: (_, r) => formatMetric(r.result_metrics?.accuracy, { percent: true }),
    },
    {
      title: 'F1',
      key: 'f1',
      width: 80,
      render: (_, r) => formatMetric(r.result_metrics?.f1),
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
    taskKind: 'classification',
  });
  const [vizErrors, setVizErrors] = useState({});
  const [loading, setLoading] = useState(true);
  const [vizLoading, setVizLoading] = useState(false);

  const confusionMatrixRef = useRef(null);
  const rocCurveRef       = useRef(null);
  const featureImportanceRef = useRef(null);
  const learningCurveRef  = useRef(null);
  const residualPlotRef = useRef(null);
  const predictedVsActualRef = useRef(null);

  useEffect(() => { void loadAll(); }, [taskId]);

  async function loadAll() {
    setLoading(true);
    let modelItems = [];
    try {
      const res = await modelApi.listModels({ page_size: 200 });
      modelItems = res.items ?? [];
      setModels(modelItems);
    } catch { /* ignore */ }
    setLoading(false);
    void loadVisualizations(modelItems);
  }

  async function loadVisualizations(modelItems = models) {
    setVizLoading(true);
    const fallbackModel = (modelItems ?? []).find(m => m.task_id === taskId) ?? null;
    let detailPayload = null;
    let detailError = null;

    try {
      detailPayload = await modelApi.getModelDetail(taskId);
      setDetail(detailPayload);
    } catch (err) {
      detailError = getApiErrorText(err);
      setDetail(null);
    }

    const metrics = detailPayload?.result_metrics ?? fallbackModel?.result_metrics ?? {};
    const taskKind = inferTaskKind(
      detailPayload?.model_type ?? fallbackModel?.model_type,
      metrics,
    );

    // Build the fetcher list dynamically by task kind so we don't spam the
    // backend with endpoints that will cleanly 400 with task-kind guards.
    const baseRequests = [
      ['featureImportance', () => vizApi.getFeatureImportance(taskId)],
      ['learningCurve', () => vizApi.getLearningCurve(taskId)],
      ['distribution', () => vizApi.getDistribution(taskId)],
      ['shap', () => vizApi.getShapSummary(taskId)],
    ];
    const classificationRequests = [
      ['confusionMatrix', () => vizApi.getConfusionMatrix(taskId)],
      ['rocCurve', () => vizApi.getRocCurve(taskId)],
      ['perClass', () => vizApi.getPerClass(taskId)],
      ['prCurve', () => vizApi.getPrCurve(taskId)],
      ['calibration', () => vizApi.getCalibration(taskId)],
      ['threshold', () => vizApi.getThreshold(taskId)],
    ];
    const regressionRequests = [
      ['residualPlot', () => vizApi.getResidualPlot(taskId)],
      ['predictedVsActual', () => vizApi.getPredictedVsActual(taskId)],
    ];
    const chartRequests = [
      ...baseRequests,
      ...(taskKind === 'regression' ? regressionRequests : classificationRequests),
    ];

    const results = await Promise.allSettled(chartRequests.map(([, fn]) => fn()));
    const nextErrors = { detail: detailError };
    const payloads = {};
    results.forEach((r, idx) => {
      const k = chartRequests[idx][0];
      if (r.status === 'fulfilled') {
        nextErrors[k] = null;
        payloads[k] = r.value;
      } else {
        nextErrors[k] = getApiErrorText(r.reason);
        payloads[k] = null;
      }
    });
    setVizErrors(nextErrors);
    setVizState(prev => ({
      ...prev,
      confusionMatrix: payloads.confusionMatrix ?? null,
      rocCurve: payloads.rocCurve ?? null,
      featureImportance: payloads.featureImportance ?? null,
      learningCurve: payloads.learningCurve ?? null,
      residualPlot: payloads.residualPlot ?? null,
      predictedVsActual: payloads.predictedVsActual ?? null,
      perClass: payloads.perClass ?? null,
      prCurve: payloads.prCurve ?? null,
      calibration: payloads.calibration ?? null,
      threshold: payloads.threshold ?? null,
      distribution: payloads.distribution ?? null,
      shap: payloads.shap ?? null,
      taskKind,
    }));
    // Only surface a toast if every single endpoint failed; otherwise the
    // per-chart inline error is enough (no N-popup spam).
    const chartKeys = chartRequests.map(([k]) => k);
    const allFailed = chartKeys.every(k => nextErrors[k]);
    if (allFailed) message.error('加载可视化详情失败');
    setVizLoading(false);
  }

  // Small renderer: shows chart container when data is present, inline error
  // (with retry) when that specific endpoint failed, empty state otherwise.
  function ChartSlot({ errorKey, hasData, emptyText, children }) {
    const err = vizErrors[errorKey];
    if (err) {
      return (
        <Alert
          type="error"
          showIcon
          message="图表加载失败"
          description={err}
          action={
            <Button size="small" onClick={() => void loadVisualizations()} icon={<ReloadOutlined />}>
              重试
            </Button>
          }
          style={{ margin: '40px 0' }}
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

    const learningCurve = vizState.learningCurve ? {
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0 },
      grid: { top: 24, left: 56, right: 20, bottom: 52 },
      xAxis: {
        type: 'category',
        data: vizState.learningCurve.steps.map(s => `Fold ${s.step}`),
      },
      yAxis: {
        type: 'value',
        ...(vizState.taskKind === 'classification' ? { min: 0, max: 1 } : {}),
      },
      series: [
        ...(vizState.taskKind === 'regression'
          ? ['rmse', 'mae', 'mse', 'r2']
          : ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']),
        ...Object.keys(vizState.learningCurve.steps[0]?.metrics ?? {}).filter(k => k !== 'fold'),
      ]
        .filter((k, idx, arr) => arr.indexOf(k) === idx)
        .filter(k => vizState.learningCurve.steps.some(s => typeof s.metrics?.[k] === 'number'))
        .map(k => ({
          name: metricLabels[k] ?? k,
          type: 'line', smooth: true,
          data: vizState.learningCurve.steps.map(s => s.metrics?.[k] ?? null),
        })),
    } : null;

    const predictedVsActual = vizState.predictedVsActual ? {
      tooltip: { trigger: 'item', formatter: p => `真实值: ${p.value[0]}<br/>预测值: ${p.value[1]}` },
      grid: { top: 28, left: 70, right: 24, bottom: 56 },
      xAxis: { type: 'value', name: '真实值' },
      yAxis: { type: 'value', name: '预测值' },
      series: [
        {
          name: '样本',
          type: 'scatter',
          symbolSize: 6,
          data: vizState.predictedVsActual.actual.map((v, i) => [v, vizState.predictedVsActual.predicted[i]]),
          itemStyle: { color: '#2563eb', opacity: 0.7 },
        },
      ],
    } : null;

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

    return { confusionMatrix, rocCurve, featureImportance, learningCurve, predictedVsActual, residualPlot };
  }, [vizState]);

  useChart(confusionMatrixRef, chartOptions.confusionMatrix);
  useChart(rocCurveRef, chartOptions.rocCurve);
  useChart(featureImportanceRef, chartOptions.featureImportance);
  useChart(learningCurveRef, chartOptions.learningCurve);
  useChart(predictedVsActualRef, chartOptions.predictedVsActual);
  useChart(residualPlotRef, chartOptions.residualPlot);

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

  const performanceTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Row gutter={[16, 16]}>
        {taskKind === 'regression' ? (
          <>
            <Col xs={24} xl={12}>
              <Card title={<Space><BarChartOutlined />预测值 vs 真实值</Space>} variant="borderless">
                <ChartSlot errorKey="predictedVsActual" hasData={!!vizState.predictedVsActual} emptyText="暂无预测-真实值数据">
                  <div ref={predictedVsActualRef} style={{ width: '100%', height: 360 }} />
                </ChartSlot>
              </Card>
            </Col>
            <Col xs={24} xl={12}>
              <Card title={<Space><LineChartOutlined />残差散点</Space>} variant="borderless">
                <ChartSlot errorKey="residualPlot" hasData={!!vizState.residualPlot} emptyText="暂无残差数据">
                  <div ref={residualPlotRef} style={{ width: '100%', height: 360 }} />
                </ChartSlot>
              </Card>
            </Col>
          </>
        ) : (
          <>
            <Col xs={24} xl={12}>
              <Card title={<Space><HeatMapOutlined />混淆矩阵</Space>} variant="borderless">
                <ChartSlot errorKey="confusionMatrix" hasData={!!vizState.confusionMatrix} emptyText="暂无混淆矩阵数据">
                  <div ref={confusionMatrixRef} style={{ width: '100%', height: 360 }} />
                </ChartSlot>
              </Card>
            </Col>
            <Col xs={24} xl={12}>
              <Card title={<Space><LineChartOutlined />ROC 曲线</Space>} variant="borderless">
                <ChartSlot errorKey="rocCurve" hasData={!!vizState.rocCurve} emptyText="暂无 ROC 曲线数据">
                  <div ref={rocCurveRef} style={{ width: '100%', height: 360 }} />
                </ChartSlot>
              </Card>
            </Col>
            <Col xs={24} xl={12}>
              <Card title={<Space><LineChartOutlined />Precision-Recall 曲线</Space>} variant="borderless">
                <ChartSlot errorKey="prCurve" hasData={!!vizState.prCurve} emptyText="暂无 PR 曲线数据">
                  <PRCurveChart payload={vizState.prCurve} height={340} />
                </ChartSlot>
              </Card>
            </Col>
            <Col xs={24} xl={12}>
              <Card title={<Space><LineChartOutlined />校准曲线</Space>} variant="borderless">
                <ChartSlot errorKey="calibration" hasData={!!vizState.calibration} emptyText="暂无校准曲线数据（需要二分类 + predict_proba）">
                  <CalibrationCurveChart payload={vizState.calibration} height={340} />
                </ChartSlot>
              </Card>
            </Col>
            <Col xs={24}>
              <Card title={<Space><BarChartOutlined />逐类指标</Space>} variant="borderless">
                <ChartSlot errorKey="perClass" hasData={!!vizState.perClass} emptyText="暂无逐类指标">
                  <PerClassMetricsTable payload={vizState.perClass} />
                </ChartSlot>
              </Card>
            </Col>
          </>
        )}
      </Row>
    </Space>
  );

  const trainingTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card title={<Space><RiseOutlined />交叉验证曲线</Space>} variant="borderless">
        <ChartSlot errorKey="learningCurve" hasData={!!vizState.learningCurve} emptyText="暂无 CV 数据（仅 K-Fold 训练任务可用）">
          <div ref={learningCurveRef} style={{ width: '100%', height: 360 }} />
        </ChartSlot>
      </Card>
      {resultMetrics?.history && (
        <Card title={<Space><RiseOutlined />Epoch 训练历史</Space>} variant="borderless">
          <TrainingHistoryChart
            history={resultMetrics.history}
            taskType={taskKind}
            height={360}
          />
        </Card>
      )}
    </Space>
  );

  const explanationTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card title={<Space><BulbOutlined />SHAP / 特征解释</Space>} variant="borderless">
        <ChartSlot errorKey="shap" hasData={!!vizState.shap} emptyText="暂无 SHAP 解释数据">
          <ShapView payload={vizState.shap} />
        </ChartSlot>
      </Card>
      <Card title={<Space><BarChartOutlined />模型原生特征重要性 (Top 10)</Space>} variant="borderless">
        <ChartSlot errorKey="featureImportance" hasData={!!vizState.featureImportance} emptyText="该模型不支持 feature_importances_（仅树模型可用）">
          <div ref={featureImportanceRef} style={{ width: '100%', height: 360 }} />
        </ChartSlot>
      </Card>
    </Space>
  );

  const thresholdTab = (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      {taskKind !== 'regression' && (
        <Card title={<Space><SlidersOutlined />阈值敏感度分析</Space>} variant="borderless">
          <ChartSlot errorKey="threshold" hasData={!!vizState.threshold} emptyText="暂无阈值分析（需要二分类 + predict_proba）">
            <ThresholdTuningTable payload={vizState.threshold} />
          </ChartSlot>
        </Card>
      )}
      <Card title={<Space><BarChartOutlined />预测分布</Space>} variant="borderless">
        <ChartSlot errorKey="distribution" hasData={!!vizState.distribution} emptyText="暂无预测分布数据">
          <PredictionDistributionChart payload={vizState.distribution} height={340} />
        </ChartSlot>
      </Card>
    </Space>
  );

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/training/results')}>
            返回列表
          </Button>
          <Title level={2} style={{ margin: 0 }}>结果可视化详情</Title>
        </Space>
        <Button icon={<ReloadOutlined />} loading={vizLoading} onClick={() => void loadVisualizations()}>
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
              defaultActiveKey="performance"
              items={[
                {
                  key: 'performance',
                  label: <Space><HeatMapOutlined />性能</Space>,
                  children: performanceTab,
                },
                {
                  key: 'training',
                  label: <Space><RiseOutlined />训练过程</Space>,
                  children: trainingTab,
                },
                {
                  key: 'explain',
                  label: <Space><BulbOutlined />解释性</Space>,
                  children: explanationTab,
                },
                {
                  key: 'threshold',
                  label: <Space><SlidersOutlined />阈值 &amp; 分布</Space>,
                  children: thresholdTab,
                },
              ]}
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
