import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import {
  Alert,
  Card,
  Col,
  Descriptions,
  Empty,
  Row,
  Select,
  Skeleton,
  Space,
  Statistic,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  BarChartOutlined,
  HeatMapOutlined,
  LineChartOutlined,
  RiseOutlined,
} from '@ant-design/icons';
import * as echarts from 'echarts';
import { modelApi, vizApi } from '../services/api';
import { formatDateTime, formatMetric, metricLabels } from '../utils/formatters';

const { Paragraph, Text, Title } = Typography;

function useChart(ref, option) {
  useEffect(() => {
    if (!ref.current || !option) {
      return undefined;
    }

    const instance = echarts.init(ref.current);
    instance.setOption(option, true);

    const handleResize = () => instance.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      instance.dispose();
    };
  }, [option, ref]);
}

const Results = () => {
  const location = useLocation();
  const [models, setModels] = useState([]);
  const [selectedTaskId, setSelectedTaskId] = useState('');
  const [detail, setDetail] = useState(null);
  const [vizState, setVizState] = useState({
    confusionMatrix: null,
    rocCurve: null,
    featureImportance: null,
    learningCurve: null,
  });
  const [loading, setLoading] = useState(true);
  const [vizLoading, setVizLoading] = useState(false);

  const confusionMatrixRef = useRef(null);
  const rocCurveRef = useRef(null);
  const featureImportanceRef = useRef(null);
  const learningCurveRef = useRef(null);

  useEffect(() => {
    void loadModels();
  }, []);

  useEffect(() => {
    if (!selectedTaskId) {
      return;
    }

    void loadVisualizations(selectedTaskId);
  }, [selectedTaskId]);

  const selectedModel = useMemo(
    () => models.find((item) => item.task_id === selectedTaskId) ?? null,
    [models, selectedTaskId]
  );

  const chartOptions = useMemo(() => {
    const confusionMatrix = vizState.confusionMatrix
      ? {
          tooltip: { position: 'top' },
          grid: { height: '70%', top: 36, left: 72, right: 28, bottom: 40 },
          xAxis: {
            type: 'category',
            data: vizState.confusionMatrix.labels,
            name: '预测类别',
          },
          yAxis: {
            type: 'category',
            data: vizState.confusionMatrix.labels,
            name: '真实类别',
          },
          visualMap: {
            min: 0,
            max: Math.max(...vizState.confusionMatrix.matrix.flat()),
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: 0,
          },
          series: [
            {
              type: 'heatmap',
              data: vizState.confusionMatrix.matrix.flatMap((row, rowIndex) =>
                row.map((value, columnIndex) => [columnIndex, rowIndex, value])
              ),
              label: { show: true },
            },
          ],
        }
      : null;

    const rocCurve = vizState.rocCurve
      ? {
          tooltip: { trigger: 'axis' },
          legend: { bottom: 0 },
          grid: { top: 36, left: 56, right: 20, bottom: 54 },
          xAxis: { type: 'value', min: 0, max: 1, name: 'FPR' },
          yAxis: { type: 'value', min: 0, max: 1, name: 'TPR' },
          series: vizState.rocCurve.multiclass
            ? vizState.rocCurve.curves.map((curve) => ({
                name: `${curve.class} (AUC ${curve.auc})`,
                type: 'line',
                smooth: true,
                data: curve.fpr.map((value, index) => [value, curve.tpr[index]]),
              }))
            : [
                {
                  name: `ROC (AUC ${vizState.rocCurve.auc})`,
                  type: 'line',
                  smooth: true,
                  data: vizState.rocCurve.fpr.map((value, index) => [value, vizState.rocCurve.tpr[index]]),
                },
                {
                  name: '随机基线',
                  type: 'line',
                  data: [
                    [0, 0],
                    [1, 1],
                  ],
                  lineStyle: { type: 'dashed' },
                  symbol: 'none',
                },
              ],
        }
      : null;

    const featureImportance = vizState.featureImportance
      ? {
          tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
          grid: { top: 24, left: 150, right: 20, bottom: 24 },
          xAxis: { type: 'value', name: '重要性' },
          yAxis: {
            type: 'category',
            data: [...vizState.featureImportance.features].slice(0, 10).reverse(),
          },
          series: [
            {
              type: 'bar',
              data: [...vizState.featureImportance.importance].slice(0, 10).reverse(),
              itemStyle: {
                color: {
                  type: 'linear',
                  x: 0,
                  y: 0,
                  x2: 1,
                  y2: 0,
                  colorStops: [
                    { offset: 0, color: '#0f766e' },
                    { offset: 1, color: '#38bdf8' },
                  ],
                },
              },
            },
          ],
        }
      : null;

    const learningCurve = vizState.learningCurve
      ? {
          tooltip: { trigger: 'axis' },
          legend: { bottom: 0 },
          grid: { top: 24, left: 56, right: 20, bottom: 52 },
          xAxis: {
            type: 'category',
            data: vizState.learningCurve.steps.map((step) => `Fold ${step.step}`),
          },
          yAxis: { type: 'value', min: 0, max: 1 },
          series: ['accuracy', 'f1']
            .filter((metricName) =>
              vizState.learningCurve.steps.some((step) => typeof step.metrics?.[metricName] === 'number')
            )
            .map((metricName) => ({
              name: metricLabels[metricName] ?? metricName,
              type: 'line',
              smooth: true,
              data: vizState.learningCurve.steps.map((step) => step.metrics?.[metricName] ?? null),
            })),
        }
      : null;

    return {
      confusionMatrix,
      rocCurve,
      featureImportance,
      learningCurve,
    };
  }, [vizState]);

  useChart(confusionMatrixRef, chartOptions.confusionMatrix);
  useChart(rocCurveRef, chartOptions.rocCurve);
  useChart(featureImportanceRef, chartOptions.featureImportance);
  useChart(learningCurveRef, chartOptions.learningCurve);

  async function loadModels() {
    setLoading(true);
    try {
      const response = await modelApi.listModels({ page_size: 50 });
      const items = response.items ?? [];
      setModels(items);

      const preferredTaskId = new URLSearchParams(location.search).get('taskId');
      const defaultTaskId =
        (preferredTaskId && items.some((item) => item.task_id === preferredTaskId) && preferredTaskId) ||
        items[0]?.task_id ||
        '';
      setSelectedTaskId(defaultTaskId);
    } catch (error) {
      console.error('加载可视化列表失败:', error);
      message.error('加载可视化列表失败');
    } finally {
      setLoading(false);
    }
  }

  async function loadVisualizations(taskId) {
    setVizLoading(true);
    try {
      const [detailResponse, confusionMatrix, rocCurve, featureImportance, learningCurve] = await Promise.all([
        modelApi.getModelDetail(taskId),
        vizApi.getConfusionMatrix(taskId),
        vizApi.getRocCurve(taskId),
        vizApi.getFeatureImportance(taskId),
        vizApi.getLearningCurve(taskId),
      ]);

      setDetail(detailResponse);
      setVizState({
        confusionMatrix,
        rocCurve,
        featureImportance,
        learningCurve,
      });
    } catch (error) {
      console.error('加载可视化详情失败:', error);
      message.error('加载可视化详情失败');
    } finally {
      setVizLoading(false);
    }
  }

  const resultMetrics = detail?.result_metrics ?? selectedModel?.result_metrics ?? {};

  const metricItems = Object.entries(resultMetrics)
    .filter(([, value]) => typeof value === 'number' && !Number.isNaN(value))
    .filter(([key]) => key !== 'cv_folds');

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      <Card>
        <Space direction="vertical" size={12} style={{ width: '100%' }}>
          <Title level={2} style={{ margin: 0 }}>
            结果可视化
          </Title>
          <Paragraph type="secondary" style={{ marginBottom: 0 }}>
            每张图都直接读取后端训练任务与可视化接口，适合验证训练结果、解释模型表现和排查异常。
          </Paragraph>
          {loading ? (
            <Skeleton active paragraph={{ rows: 1 }} />
          ) : models.length === 0 ? (
            <Empty description="还没有可视化结果，请先完成一次训练。" />
          ) : (
            <Select
              value={selectedTaskId || undefined}
              onChange={setSelectedTaskId}
              style={{ maxWidth: 520 }}
              data-testid="results-task-select"
              options={models.map((item) => ({
                label: `${item.dataset_name ?? item.dataset_id} · ${item.model_type}`,
                value: item.task_id,
              }))}
            />
          )}
        </Space>
      </Card>

      {selectedModel ? (
        <>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12} xl={6}>
              <Card>
                <Statistic title="准确率" value={Number(resultMetrics.accuracy ?? 0) * 100} suffix="%" precision={2} />
              </Card>
            </Col>
            <Col xs={24} md={12} xl={6}>
              <Card>
                <Statistic title="F1" value={resultMetrics.f1 ?? 0} precision={4} />
              </Card>
            </Col>
            <Col xs={24} md={12} xl={6}>
              <Card>
                <Statistic title="交叉验证准确率" value={Number(resultMetrics.cv_avg_accuracy ?? 0) * 100} suffix="%" precision={2} />
              </Card>
            </Col>
            <Col xs={24} md={12} xl={6}>
              <Card>
                <Statistic title="交叉验证 F1" value={resultMetrics.cv_avg_f1 ?? 0} precision={4} />
              </Card>
            </Col>
          </Row>

          <Card>
            <Descriptions bordered column={{ xs: 1, md: 2, xl: 4 }}>
              <Descriptions.Item label="数据集">{detail?.dataset?.name ?? selectedModel.dataset_name ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="模型">{detail?.model_type ?? selectedModel.model_type}</Descriptions.Item>
              <Descriptions.Item label="目标列">{detail?.target_column ?? selectedModel.target_column}</Descriptions.Item>
              <Descriptions.Item label="完成时间">{formatDateTime(detail?.finished_at ?? selectedModel.finished_at)}</Descriptions.Item>
              <Descriptions.Item label="测试集比例">{formatMetric(detail?.test_size ?? 0.2, { digits: 2 })}</Descriptions.Item>
              <Descriptions.Item label="模型大小">{selectedModel.model_size ? `${Math.round(selectedModel.model_size / 1024)} KB` : '-'}</Descriptions.Item>
              <Descriptions.Item label="任务 ID">{detail?.task_id ?? selectedModel.task_id}</Descriptions.Item>
              <Descriptions.Item label="状态">{detail?.status ?? 'SUCCESS'}</Descriptions.Item>
            </Descriptions>
            {vizLoading ? (
              <Text type="secondary" style={{ display: 'block', marginTop: 12 }}>
                正在加载图表数据...
              </Text>
            ) : null}
          </Card>

          {metricItems.length > 0 ? (
            <Card>
              <Space wrap size={[12, 12]}>
                {metricItems.map(([key, value]) => (
                  <Tag key={key} color="blue" style={{ padding: '6px 10px', borderRadius: 999 }}>
                    {metricLabels[key] ?? key}: {key.includes('accuracy') ? formatMetric(value, { percent: true }) : formatMetric(value)}
                  </Tag>
                ))}
              </Space>
            </Card>
          ) : null}

          <Row gutter={[16, 16]}>
            <Col xs={24} xl={12}>
              <Card title={<Space><HeatMapOutlined /> 混淆矩阵</Space>}>
                <div ref={confusionMatrixRef} data-testid="confusion-matrix-chart" style={{ width: '100%', height: 360 }} />
              </Card>
            </Col>
            <Col xs={24} xl={12}>
              <Card title={<Space><LineChartOutlined /> ROC 曲线</Space>}>
                <div ref={rocCurveRef} data-testid="roc-curve-chart" style={{ width: '100%', height: 360 }} />
              </Card>
            </Col>
            <Col xs={24} xl={12}>
              <Card title={<Space><BarChartOutlined /> 特征重要性</Space>}>
                <div ref={featureImportanceRef} data-testid="feature-importance-chart" style={{ width: '100%', height: 360 }} />
              </Card>
            </Col>
            <Col xs={24} xl={12}>
              <Card title={<Space><RiseOutlined /> 训练过程曲线</Space>}>
                <div ref={learningCurveRef} data-testid="learning-curve-chart" style={{ width: '100%', height: 360 }} />
              </Card>
            </Col>
          </Row>
        </>
      ) : loading ? (
        <Card>
          <Skeleton active paragraph={{ rows: 8 }} />
        </Card>
      ) : (
        <Alert type="info" showIcon message="请先选择一个已完成的训练任务。" />
      )}
    </Space>
  );
};

export default Results;
