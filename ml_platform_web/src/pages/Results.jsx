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
} from '@ant-design/icons';
import * as echarts from 'echarts';
import { modelApi, vizApi } from '../services/api';
import { formatDateTime, formatMetric, metricLabels } from '../utils/formatters';

const { Paragraph, Text, Title } = Typography;

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
function ResultDetailView({ taskId, navigate }) {
  const [models, setModels] = useState([]);
  const [detail, setDetail] = useState(null);
  const [vizState, setVizState] = useState({
    confusionMatrix: null, rocCurve: null,
    featureImportance: null, learningCurve: null,
  });
  const [loading, setLoading] = useState(true);
  const [vizLoading, setVizLoading] = useState(false);

  const confusionMatrixRef = useRef(null);
  const rocCurveRef       = useRef(null);
  const featureImportanceRef = useRef(null);
  const learningCurveRef  = useRef(null);

  useEffect(() => { void loadAll(); }, [taskId]);

  async function loadAll() {
    setLoading(true);
    try {
      const res = await modelApi.listModels({ page_size: 200 });
      setModels(res.items ?? []);
    } catch { /* ignore */ }
    setLoading(false);
    void loadVisualizations();
  }

  async function loadVisualizations() {
    setVizLoading(true);
    try {
      const [det, cm, roc, fi, lc] = await Promise.all([
        modelApi.getModelDetail(taskId),
        vizApi.getConfusionMatrix(taskId),
        vizApi.getRocCurve(taskId),
        vizApi.getFeatureImportance(taskId),
        vizApi.getLearningCurve(taskId),
      ]);
      setDetail(det);
      setVizState({ confusionMatrix: cm, rocCurve: roc, featureImportance: fi, learningCurve: lc });
    } catch {
      message.error('加载可视化详情失败');
    } finally {
      setVizLoading(false);
    }
  }

  const selectedModel = useMemo(
    () => models.find(m => m.task_id === taskId) ?? null,
    [models, taskId]
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
      yAxis: { type: 'value', min: 0, max: 1 },
      series: ['accuracy', 'f1']
        .filter(k => vizState.learningCurve.steps.some(s => typeof s.metrics?.[k] === 'number'))
        .map(k => ({
          name: metricLabels[k] ?? k,
          type: 'line', smooth: true,
          data: vizState.learningCurve.steps.map(s => s.metrics?.[k] ?? null),
        })),
    } : null;

    return { confusionMatrix, rocCurve, featureImportance, learningCurve };
  }, [vizState]);

  useChart(confusionMatrixRef, chartOptions.confusionMatrix);
  useChart(rocCurveRef, chartOptions.rocCurve);
  useChart(featureImportanceRef, chartOptions.featureImportance);
  useChart(learningCurveRef, chartOptions.learningCurve);

  const resultMetrics = detail?.result_metrics ?? selectedModel?.result_metrics ?? {};
  const metricItems = Object.entries(resultMetrics)
    .filter(([, v]) => typeof v === 'number' && !Number.isNaN(v))
    .filter(([k]) => k !== 'cv_folds');

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
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col xs={24} md={12} xl={6}>
              <Card><Statistic title="准确率" value={Number(resultMetrics.accuracy ?? 0) * 100} suffix="%" precision={2} /></Card>
            </Col>
            <Col xs={24} md={12} xl={6}>
              <Card><Statistic title="F1" value={resultMetrics.f1 ?? 0} precision={4} /></Card>
            </Col>
            <Col xs={24} md={12} xl={6}>
              <Card><Statistic title="交叉验证准确率" value={Number(resultMetrics.cv_avg_accuracy ?? 0) * 100} suffix="%" precision={2} /></Card>
            </Col>
            <Col xs={24} md={12} xl={6}>
              <Card><Statistic title="交叉验证 F1" value={resultMetrics.cv_avg_f1 ?? 0} precision={4} /></Card>
            </Col>
          </Row>

          <Card style={{ marginBottom: 16 }}>
            <Descriptions bordered column={{ xs: 1, md: 2, xl: 4 }}>
              <Descriptions.Item label="数据集">{detail?.dataset?.name ?? selectedModel?.dataset_name ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="模型">{detail?.model_type ?? selectedModel?.model_type ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="目标列">{detail?.target_column ?? selectedModel?.target_column ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="完成时间">{formatDateTime(detail?.finished_at ?? selectedModel?.finished_at)}</Descriptions.Item>
              <Descriptions.Item label="测试集比例">{formatMetric(detail?.test_size ?? 0.2, { digits: 2 })}</Descriptions.Item>
              <Descriptions.Item label="任务 ID"><Text code>{taskId}</Text></Descriptions.Item>
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

          <Card bodyStyle={{ padding: '0 24px 24px' }}>
            <Tabs
              defaultActiveKey="performance"
              items={[
                {
                  key: 'performance',
                  label: <Space><HeatMapOutlined />性能图表</Space>,
                  children: (
                    <Row gutter={[16, 16]}>
                      <Col xs={24} xl={12}>
                        <Card
                          title={<Space><HeatMapOutlined /> 混淆矩阵</Space>}
                          bordered={false}
                        >
                          {vizState.confusionMatrix
                            ? <div ref={confusionMatrixRef} style={{ width: '100%', height: 360 }} />
                            : <Empty description="暂无混淆矩阵数据" style={{ padding: '60px 0' }} />
                          }
                        </Card>
                      </Col>
                      <Col xs={24} xl={12}>
                        <Card
                          title={<Space><LineChartOutlined /> ROC 曲线</Space>}
                          bordered={false}
                        >
                          {vizState.rocCurve
                            ? <div ref={rocCurveRef} style={{ width: '100%', height: 360 }} />
                            : <Empty description="暂无 ROC 曲线数据（仅分类任务）" style={{ padding: '60px 0' }} />
                          }
                        </Card>
                      </Col>
                    </Row>
                  ),
                },
                {
                  key: 'explain',
                  label: <Space><BulbOutlined />模型解释</Space>,
                  children: (
                    <div>
                      {vizState.featureImportance ? (
                        <>
                          <div style={{ color: '#64748b', fontSize: 12, marginBottom: 12 }}>
                            模型原生特征重要性（基于 sklearn
                            <Tag style={{ marginLeft: 6, fontSize: 10 }}>feature_importances_</Tag>
                            属性，前 10 个特征）
                          </div>
                          <div ref={featureImportanceRef} style={{ width: '100%', height: 420 }} />
                        </>
                      ) : (
                        <Empty
                          description="暂无特征重要性数据（树模型训练后自动生成）"
                          style={{ padding: '60px 0' }}
                        />
                      )}
                      <Alert
                        type="info"
                        showIcon
                        icon={<BulbOutlined />}
                        style={{ marginTop: 16 }}
                        message={'如需 SHAP 级别解释，请在「实验管理」中使用 AutoML 训练，并在实验详情页触发 SHAP 分析。'}
                      />
                    </div>
                  ),
                },
                {
                  key: 'learning',
                  label: <Space><RiseOutlined />训练过程</Space>,
                  children: vizState.learningCurve
                    ? <div ref={learningCurveRef} style={{ width: '100%', height: 360 }} />
                    : <Empty description="暂无训练过程数据" style={{ padding: '60px 0' }} />,
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
