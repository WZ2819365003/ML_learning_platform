import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  Progress,
  Row,
  Skeleton,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  ArrowRightOutlined,
  CheckCircleOutlined,
  DatabaseOutlined,
  LineChartOutlined,
  RocketOutlined,
  TrophyOutlined,
} from '@ant-design/icons';
import { dataApi, modelApi, trainingApi } from '../services/api';
import { formatDateTime, formatMetric, metricLabels, pickPrimaryMetric } from '../utils/formatters';

const { Paragraph, Text, Title } = Typography;

const gradientCardStyle = {
  border: 'none',
  background:
    'linear-gradient(135deg, rgba(12, 74, 110, 0.95) 0%, rgba(8, 47, 73, 0.96) 50%, rgba(15, 23, 42, 0.98) 100%)',
  color: '#f8fafc',
};

function getStatusTag(status) {
  const normalized = String(status ?? '').toUpperCase();
  const colorMap = {
    SUCCESS: 'success',
    RUNNING: 'processing',
    FAILED: 'error',
    PENDING: 'warning',
  };

  return <Tag color={colorMap[normalized] ?? 'default'}>{normalized || 'UNKNOWN'}</Tag>;
}

const Dashboard = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState({
    datasets: [],
    models: [],
    recentTasks: [],
    totals: {
      datasetCount: 0,
      taskCount: 0,
      modelCount: 0,
      successCount: 0,
      runningCount: 0,
      failedCount: 0,
    },
  });

  useEffect(() => {
    void loadDashboardData();
  }, []);

  const successRate = useMemo(() => {
    if (dashboardData.totals.taskCount === 0) {
      return 0;
    }

    return dashboardData.totals.successCount / dashboardData.totals.taskCount;
  }, [dashboardData.totals]);

  const bestModel = useMemo(() => {
    const candidates = [...dashboardData.models];
    candidates.sort((left, right) => {
      const leftMetric = pickPrimaryMetric(left.result_metrics ?? {})?.value ?? -1;
      const rightMetric = pickPrimaryMetric(right.result_metrics ?? {})?.value ?? -1;
      return rightMetric - leftMetric;
    });
    return candidates[0] ?? null;
  }, [dashboardData.models]);

  async function loadDashboardData() {
    setLoading(true);
    try {
      const [
        datasetResponse,
        taskResponse,
        successResponse,
        runningResponse,
        failedResponse,
        modelResponse,
      ] = await Promise.all([
        dataApi.listDatasets({ page_size: 50 }),
        trainingApi.listTasks({ page_size: 20 }),
        trainingApi.listTasks({ page_size: 1, status: 'SUCCESS' }),
        trainingApi.listTasks({ page_size: 1, status: 'RUNNING' }),
        trainingApi.listTasks({ page_size: 1, status: 'FAILED' }),
        modelApi.listModels({ page_size: 20 }),
      ]);

      const datasets = datasetResponse.items ?? [];
      const models = modelResponse.items ?? [];
      const datasetMap = new Map(datasets.map((dataset) => [dataset.id, dataset.name]));

      const recentTasks = (taskResponse.items ?? []).map((task) => {
        const primaryMetric = pickPrimaryMetric(task.result_metrics ?? {});
        return {
          key: task.id,
          id: task.id,
          model_type: task.model_type,
          dataset_name: datasetMap.get(task.dataset_id) ?? task.dataset_id,
          status: task.status,
          progress: task.progress ?? 0,
          metric_key: primaryMetric?.key ?? null,
          metric_value: primaryMetric?.value ?? null,
          created_at: task.created_at,
          finished_at: task.finished_at,
        };
      });

      setDashboardData({
        datasets,
        models,
        recentTasks,
        totals: {
          datasetCount: datasetResponse.total ?? datasets.length,
          taskCount: taskResponse.total ?? recentTasks.length,
          modelCount: modelResponse.total ?? models.length,
          successCount: successResponse.total ?? 0,
          runningCount: runningResponse.total ?? 0,
          failedCount: failedResponse.total ?? 0,
        },
      });
    } catch (error) {
      console.error('加载仪表盘失败:', error);
      message.error('加载仪表盘失败');
    } finally {
      setLoading(false);
    }
  }

  const latestDataset = dashboardData.datasets[0] ?? null;

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      <Card bordered={false} style={gradientCardStyle} bodyStyle={{ padding: 28 }}>
        <Row gutter={[24, 24]} align="middle">
          <Col xs={24} lg={15}>
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Tag color="cyan" style={{ width: 'fit-content', marginInlineEnd: 0 }}>
                已接通真实后端
              </Tag>
              <Title level={2} style={{ color: '#f8fafc', margin: 0 }}>
                训练平台总览
              </Title>
              <Paragraph style={{ color: 'rgba(248, 250, 252, 0.78)', marginBottom: 0 }}>
                这里展示的数量、训练状态和模型摘要都直接来自后端接口，不再使用任何演示数据。
              </Paragraph>
              <Space wrap>
                <Button type="primary" icon={<RocketOutlined />} onClick={() => navigate('/training/config')}>
                  新建训练任务
                </Button>
                <Button ghost icon={<LineChartOutlined />} onClick={() => navigate('/results')}>
                  查看可视化结果
                </Button>
              </Space>
            </Space>
          </Col>
          <Col xs={24} lg={9}>
            <Card
              bordered={false}
              bodyStyle={{ padding: 20 }}
              style={{ background: 'rgba(248, 250, 252, 0.08)', borderRadius: 20 }}
            >
              <Space direction="vertical" size={10} style={{ width: '100%' }}>
                <Text style={{ color: 'rgba(248, 250, 252, 0.72)' }}>当前表现最好的已保存模型</Text>
                {loading ? (
                  <Skeleton active title={false} paragraph={{ rows: 2 }} />
                ) : bestModel ? (
                  <>
                    <Text
                      strong
                      style={{ color: '#f8fafc', fontSize: 18 }}
                      data-testid="dashboard-model-name"
                    >
                      {bestModel.model_type}
                    </Text>
                    <Text style={{ color: 'rgba(248, 250, 252, 0.72)' }}>
                      数据集：{bestModel.dataset_name ?? bestModel.dataset_id}
                    </Text>
                    <Text style={{ color: '#7dd3fc' }}>
                      {metricLabels[pickPrimaryMetric(bestModel.result_metrics ?? {})?.key ?? 'accuracy'] ?? '指标'}
                      ：{formatMetric(pickPrimaryMetric(bestModel.result_metrics ?? {})?.value, { percent: true })}
                    </Text>
                  </>
                ) : (
                  <Text style={{ color: 'rgba(248, 250, 252, 0.72)' }}>还没有可用模型</Text>
                )}
              </Space>
            </Card>
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} xl={6}>
          <Card>
            <Statistic title="数据集" value={dashboardData.totals.datasetCount} prefix={<DatabaseOutlined />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} xl={6}>
          <Card>
            <Statistic title="训练任务" value={dashboardData.totals.taskCount} prefix={<RocketOutlined />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} xl={6}>
          <Card>
            <Statistic title="已保存模型" value={dashboardData.totals.modelCount} prefix={<TrophyOutlined />} />
          </Card>
        </Col>
        <Col xs={24} sm={12} xl={6}>
          <Card>
            <Statistic
              title="成功率"
              value={successRate * 100}
              suffix="%"
              precision={1}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={9}>
          <Card title="运行状态分布" extra={<Button type="link" onClick={() => navigate('/training/monitor')}>查看任务</Button>}>
            {loading ? (
              <Skeleton active paragraph={{ rows: 4 }} />
            ) : (
              <Space direction="vertical" size={18} style={{ width: '100%' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Text>成功</Text>
                    <Text>{dashboardData.totals.successCount}</Text>
                  </div>
                  <Progress
                    percent={dashboardData.totals.taskCount ? (dashboardData.totals.successCount / dashboardData.totals.taskCount) * 100 : 0}
                    strokeColor="#16a34a"
                    showInfo={false}
                  />
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Text>运行中</Text>
                    <Text>{dashboardData.totals.runningCount}</Text>
                  </div>
                  <Progress
                    percent={dashboardData.totals.taskCount ? (dashboardData.totals.runningCount / dashboardData.totals.taskCount) * 100 : 0}
                    strokeColor="#0284c7"
                    showInfo={false}
                  />
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Text>失败</Text>
                    <Text>{dashboardData.totals.failedCount}</Text>
                  </div>
                  <Progress
                    percent={dashboardData.totals.taskCount ? (dashboardData.totals.failedCount / dashboardData.totals.taskCount) * 100 : 0}
                    strokeColor="#dc2626"
                    showInfo={false}
                  />
                </div>
              </Space>
            )}
          </Card>
        </Col>
        <Col xs={24} lg={15}>
          <Card title="最近训练任务">
            {loading ? (
              <Skeleton active paragraph={{ rows: 6 }} />
            ) : dashboardData.recentTasks.length === 0 ? (
              <Empty description="还没有训练任务" />
            ) : (
              <Table
                rowKey="id"
                pagination={false}
                dataSource={dashboardData.recentTasks}
                columns={[
                  {
                    title: '数据集',
                    dataIndex: 'dataset_name',
                    render: (value) => (
                      <Text strong={value === latestDataset?.name} data-testid={value === latestDataset?.name ? 'dashboard-dataset-name' : undefined}>
                        {value}
                      </Text>
                    ),
                  },
                  {
                    title: '模型',
                    dataIndex: 'model_type',
                    render: (value) => <Tag color="blue">{value}</Tag>,
                  },
                  {
                    title: '状态',
                    dataIndex: 'status',
                    render: (value) => getStatusTag(value),
                  },
                  {
                    title: '进度',
                    dataIndex: 'progress',
                    render: (value) => <Progress percent={Math.round(value ?? 0)} size="small" />,
                  },
                  {
                    title: '核心指标',
                    render: (_, record) =>
                      record.metric_key ? (
                        <Text>
                          {metricLabels[record.metric_key] ?? record.metric_key}:&nbsp;
                          {formatMetric(record.metric_value, { percent: true })}
                        </Text>
                      ) : (
                        '-'
                      ),
                  },
                  {
                    title: '完成时间',
                    dataIndex: 'finished_at',
                    render: (value, record) => formatDateTime(value ?? record.created_at),
                  },
                ]}
              />
            )}
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card
            title="最新数据集"
            extra={
              <Button type="link" icon={<ArrowRightOutlined />} onClick={() => navigate('/data')}>
                前往数据管理
              </Button>
            }
          >
            {loading ? (
              <Skeleton active paragraph={{ rows: 3 }} />
            ) : latestDataset ? (
              <Space direction="vertical" size={10}>
                <Text strong>{latestDataset.name}</Text>
                <Text type="secondary">
                  {latestDataset.row_count ?? 0} 行 / {latestDataset.column_count ?? 0} 列
                </Text>
                <Text type="secondary">上传时间：{formatDateTime(latestDataset.created_at)}</Text>
              </Space>
            ) : (
              <Empty description="还没有上传数据集" />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card
            title="模型产出"
            extra={
              <Button type="link" icon={<ArrowRightOutlined />} onClick={() => navigate('/models')}>
                前往模型管理
              </Button>
            }
          >
            {loading ? (
              <Skeleton active paragraph={{ rows: 3 }} />
            ) : bestModel ? (
              <Space direction="vertical" size={10}>
                <Text strong>{bestModel.model_type}</Text>
                <Text type="secondary">来源数据集：{bestModel.dataset_name ?? bestModel.dataset_id}</Text>
                <Text type="secondary">
                  保存时间：{formatDateTime(bestModel.finished_at ?? bestModel.started_at)}
                </Text>
              </Space>
            ) : (
              <Alert type="info" showIcon message="当前还没有可用于预测的已保存模型。" />
            )}
          </Card>
        </Col>
      </Row>
    </Space>
  );
};

export default Dashboard;
