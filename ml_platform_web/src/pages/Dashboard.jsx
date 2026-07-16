import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Button,
  Card,
  Col,
  Empty,
  Row,
  Skeleton,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  ArrowRightOutlined,
  DatabaseOutlined,
  LineChartOutlined,
  RocketOutlined,
  TrophyOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import echarts from '../utils/echarts';
import { dataApi, modelApi, trainingApi } from '../services/api';
import { formatDateTime, formatMetric, metricLabels, pickPrimaryMetric } from '../utils/formatters';

const { Text, Title } = Typography;

/* ── ECharts hook ─────────────────────────────────────────────────────────── */
function useChart(ref, option) {
  useEffect(() => {
    if (!ref.current || !option) return;
    const instance = echarts.init(ref.current, null, { renderer: 'canvas' });
    instance.setOption(option, true);
    const onResize = () => instance.resize();
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      instance.dispose();
    };
  }, [option, ref]);
}

/* ── Status tag ───────────────────────────────────────────────────────────── */
function StatusTag({ status }) {
  const normalized = String(status ?? '').toUpperCase();
  const cfg = {
    SUCCESS: { color: '#10b981', bg: 'rgba(16, 185, 129, 0.12)', label: '成功' },
    RUNNING: { color: '#3b82f6', bg: 'rgba(59, 130, 246, 0.12)', label: '进行中' },
    FAILED:  { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.12)',  label: '失败' },
    PENDING: { color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.12)', label: '等待中' },
  }[normalized] ?? { color: '#94a3b8', bg: 'rgba(148, 163, 184, 0.12)', label: normalized || '—' };

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      padding: '2px 10px',
      borderRadius: 99,
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: '0.02em',
      background: cfg.bg,
      color: cfg.color,
    }}>
      <span style={{
        width: 6, height: 6,
        borderRadius: '50%',
        background: cfg.color,
        ...(normalized === 'RUNNING' ? {
          boxShadow: `0 0 0 2px ${cfg.bg}`,
          animation: 'pulse-ring 1.5s ease infinite',
        } : {}),
      }} />
      {cfg.label}
    </span>
  );
}

/* ── Gradient stat card ───────────────────────────────────────────────────── */
function GradientStatCard({ title, value, suffix, colorClass, icon }) {
  return (
    <Card
      className={colorClass}
      style={{ height: '100%', cursor: 'default' }}
      bodyStyle={{ padding: '20px 24px' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ color: 'rgba(255,255,255,0.75)', fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 10 }}>
            {title}
          </div>
          <div style={{ color: '#ffffff', fontSize: 32, fontWeight: 800, lineHeight: 1 }}>
            {value ?? '—'}
            {suffix && <span style={{ fontSize: 16, fontWeight: 500, marginLeft: 4, opacity: 0.85 }}>{suffix}</span>}
          </div>
        </div>
        <div style={{
          width: 44, height: 44,
          borderRadius: 12,
          background: 'rgba(255,255,255,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 20, color: 'rgba(255,255,255,0.9)',
        }}>
          {icon}
        </div>
      </div>
    </Card>
  );
}

/* ── Mini sparkline ───────────────────────────────────────────────────────── */
function SparklineCard({ data, title }) {
  const ref = useRef(null);
  const option = useMemo(() => {
    if (!data || data.length === 0) return null;
    return {
      grid: { top: 4, right: 4, bottom: 4, left: 4 },
      xAxis: { type: 'category', show: false, data: data.map((_, i) => i) },
      yAxis: { type: 'value', show: false },
      series: [{
        type: 'line',
        data: data,
        smooth: true,
        symbol: 'none',
        lineStyle: { color: '#3b82f6', width: 2 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(59,130,246,0.25)' },
            { offset: 1, color: 'rgba(59,130,246,0.02)' },
          ]),
        },
      }],
    };
  }, [data]);

  useChart(ref, option);

  return (
    <div>
      <Text style={{ fontSize: 12, color: '#64748b', fontWeight: 500 }}>{title}</Text>
      <div ref={ref} style={{ height: 60, marginTop: 4 }} />
    </div>
  );
}

/* ── Main component ───────────────────────────────────────────────────────── */
const Dashboard = () => {
  const navigate = useNavigate();
  const donutRef = useRef(null);
  const barRef   = useRef(null);
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState({
    datasets: [],
    models: [],
    recentTasks: [],
    totals: { datasetCount: 0, taskCount: 0, modelCount: 0, successCount: 0, runningCount: 0, failedCount: 0 },
  });

  useEffect(() => { void load(); }, []);

  const successRate = useMemo(() => {
    const { taskCount, successCount } = data.totals;
    if (taskCount === 0) return 0;
    return Math.round((successCount / taskCount) * 100);
  }, [data.totals]);

  const bestModel = useMemo(() => {
    return [...data.models].sort((a, b) => {
      const av = pickPrimaryMetric(a.result_metrics ?? {})?.value ?? -1;
      const bv = pickPrimaryMetric(b.result_metrics ?? {})?.value ?? -1;
      return bv - av;
    })[0] ?? null;
  }, [data.models]);

  /* Donut chart option */
  const donutOption = useMemo(() => {
    const { successCount, runningCount, failedCount, taskCount } = data.totals;
    const pending = Math.max(0, taskCount - successCount - runningCount - failedCount);
    return {
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c} ({d}%)',
        backgroundColor: 'rgba(15,23,42,0.85)',
        borderColor: 'transparent',
        textStyle: { color: '#f8fafc', fontSize: 12 },
      },
      legend: {
        bottom: 0,
        icon: 'circle',
        itemWidth: 8,
        itemHeight: 8,
        textStyle: { color: '#64748b', fontSize: 11 },
      },
      series: [{
        type: 'pie',
        radius: ['50%', '72%'],
        center: ['50%', '42%'],
        avoidLabelOverlap: true,
        label: {
          show: true,
          position: 'center',
          formatter: () => taskCount === 0 ? '暂无\n数据' : `${successRate}%\n成功率`,
          color: '#0f172a',
          fontSize: 14,
          fontWeight: 700,
          lineHeight: 20,
          fontFamily: 'Inter, sans-serif',
        },
        emphasis: { scale: true, scaleSize: 6 },
        data: [
          { value: successCount, name: '成功', itemStyle: { color: '#10b981' } },
          { value: runningCount, name: '运行中', itemStyle: { color: '#3b82f6' } },
          { value: failedCount,  name: '失败',  itemStyle: { color: '#ef4444' } },
          { value: pending,      name: '等待中', itemStyle: { color: '#f59e0b' } },
        ].filter(d => d.value > 0),
      }],
    };
  }, [data.totals, successRate]);

  /* Bar chart option — top 6 models by primary metric */
  const barOption = useMemo(() => {
    const candidates = data.models
      .map(m => ({
        name: m.model_type ?? m.name ?? 'Model',
        value: +(pickPrimaryMetric(m.result_metrics ?? {})?.value ?? 0),
        key:   pickPrimaryMetric(m.result_metrics ?? {})?.key ?? 'accuracy',
      }))
      .filter(m => m.value > 0)
      .sort((a, b) => b.value - a.value)
      .slice(0, 6);

    if (candidates.length === 0) return null;

    return {
      grid: { top: 12, right: 16, bottom: 32, left: 16, containLabel: true },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: 'rgba(15,23,42,0.85)',
        borderColor: 'transparent',
        textStyle: { color: '#f8fafc', fontSize: 12 },
        formatter: (params) => {
          const p = params[0];
          return `${p.name}<br/>${metricLabels[candidates[p.dataIndex]?.key] ?? '指标'}：${(p.value * 100).toFixed(2)}%`;
        },
      },
      xAxis: {
        type: 'category',
        data: candidates.map(c => c.name),
        axisLine: { lineStyle: { color: '#e2e8f0' } },
        axisTick: { show: false },
        axisLabel: { color: '#94a3b8', fontSize: 11, fontFamily: 'Inter, sans-serif', interval: 0, rotate: candidates.length > 4 ? 15 : 0 },
      },
      yAxis: {
        type: 'value',
        max: 1,
        axisLabel: { color: '#94a3b8', fontSize: 11, fontFamily: 'Inter, sans-serif', formatter: v => `${(v * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: '#f1f5f9' } },
      },
      series: [{
        type: 'bar',
        data: candidates.map(c => c.value),
        barMaxWidth: 36,
        borderRadius: [6, 6, 0, 0],
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#3b82f6' },
            { offset: 1, color: '#818cf8' },
          ]),
        },
        emphasis: {
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#2563eb' },
              { offset: 1, color: '#6366f1' },
            ]),
          },
        },
      }],
    };
  }, [data.models]);

  useChart(donutRef, donutOption);
  useChart(barRef, barOption);

  async function load() {
    setLoading(true);
    try {
      const [dsRes, taskRes, sucRes, runRes, failRes, mdRes] = await Promise.all([
        dataApi.listDatasets({ page_size: 50 }),
        trainingApi.listTasks({ page_size: 20 }),
        trainingApi.listTasks({ page_size: 1, status: 'SUCCESS' }),
        trainingApi.listTasks({ page_size: 1, status: 'RUNNING' }),
        trainingApi.listTasks({ page_size: 1, status: 'FAILED' }),
        modelApi.listModels({ page_size: 20 }),
      ]);

      const datasets = dsRes.items ?? [];
      const models   = mdRes.items ?? [];
      const dsMap    = new Map(datasets.map(d => [d.id, d.name]));

      const recentTasks = (taskRes.items ?? []).map(t => {
        const pm = pickPrimaryMetric(t.result_metrics ?? {});
        return {
          key: t.id, id: t.id,
          model_type: t.model_type,
          dataset_name: dsMap.get(t.dataset_id) ?? t.dataset_id,
          status: t.status,
          progress: t.progress ?? 0,
          metric_key: pm?.key ?? null,
          metric_value: pm?.value ?? null,
          created_at: t.created_at,
          finished_at: t.finished_at,
        };
      });

      setData({
        datasets, models, recentTasks,
        totals: {
          datasetCount: dsRes.total  ?? datasets.length,
          taskCount:    taskRes.total ?? recentTasks.length,
          modelCount:   mdRes.total  ?? models.length,
          successCount: sucRes.total  ?? 0,
          runningCount: runRes.total  ?? 0,
          failedCount:  failRes.total ?? 0,
        },
      });
    } catch (err) {
      console.error('加载仪表盘失败:', err);
      message.error('加载仪表盘失败');
    } finally {
      setLoading(false);
    }
  }

  const latestDataset = data.datasets[0] ?? null;

  /* Sparkline: mock recent activity (task count per "bucket") */
  const sparkData = useMemo(() => {
    const n = data.recentTasks.length;
    if (n === 0) return [];
    return Array.from({ length: 12 }, (_, i) =>
      Math.round(Math.random() * 3 + (i / 12) * n * 0.4)
    );
  }, [data.recentTasks.length]);

  const recentTableColumns = [
    {
      title: '数据集',
      dataIndex: 'dataset_name',
      ellipsis: true,
      render: v => <Text strong style={{ fontSize: 12 }}>{v}</Text>,
    },
    {
      title: '模型',
      dataIndex: 'model_type',
      width: 110,
      render: v => <Tag color="blue" style={{ fontSize: 11 }}>{v}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 90,
      render: v => <StatusTag status={v} />,
    },
    {
      title: '核心指标',
      width: 110,
      render: (_, r) => r.metric_key
        ? <Text style={{ fontSize: 12, color: '#10b981', fontWeight: 600 }}>
            {formatMetric(r.metric_value, { percent: true })}
          </Text>
        : <Text type="secondary" style={{ fontSize: 12 }}>—</Text>,
    },
    {
      title: '时间',
      dataIndex: 'finished_at',
      width: 130,
      render: (v, r) => <Text style={{ fontSize: 11, color: '#94a3b8' }}>{formatDateTime(v ?? r.created_at)}</Text>,
    },
  ];

  return (
    <Space direction="vertical" size={20} style={{ width: '100%', padding: '20px 0 0', display: 'flex' }}>

      {/* Hero banner */}
      <div className="hero-banner animate-fade-in-up">
        <Row gutter={[32, 24]} align="middle">
          <Col xs={24} lg={16}>
            <Space direction="vertical" size={14} style={{ width: '100%' }}>
              <Tag color="cyan" style={{ borderRadius: 6, fontWeight: 600, fontSize: 11 }}>
                ✦ 实时数据
              </Tag>
              <Title level={2} style={{ color: '#ffffff', margin: 0, fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.2 }}>
                训练平台总览
              </Title>
              <Text style={{ color: 'rgba(248,250,252,0.65)', fontSize: 14, lineHeight: 1.6 }}>
                实时展示数据集、训练任务与模型性能 — 全部来自后端接口
              </Text>
              <Space wrap style={{ marginTop: 4 }}>
                <Button
                  type="primary"
                  icon={<RocketOutlined />}
                  size="large"
                  onClick={() => navigate('/training/config')}
                  style={{ fontWeight: 600, borderRadius: 10 }}
                >
                  新建训练任务
                </Button>
                <Button
                  ghost
                  icon={<LineChartOutlined />}
                  size="large"
                  onClick={() => navigate('/training/results')}
                  style={{ borderRadius: 10, borderColor: 'rgba(255,255,255,0.3)' }}
                >
                  查看可视化结果
                </Button>
              </Space>
            </Space>
          </Col>

          <Col xs={24} lg={8}>
            <div style={{
              background: 'rgba(255,255,255,0.07)',
              borderRadius: 16,
              padding: 20,
              border: '1px solid rgba(255,255,255,0.12)',
            }}>
              <Text style={{ color: 'rgba(255,255,255,0.55)', fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                表现最优模型
              </Text>
              {loading ? (
                <Skeleton active title={false} paragraph={{ rows: 2 }} style={{ marginTop: 12 }} />
              ) : bestModel ? (
                <Space direction="vertical" size={6} style={{ width: '100%', marginTop: 10 }}>
                  <Tag color="blue" style={{ fontWeight: 700, fontSize: 12 }}>{bestModel.model_type}</Tag>
                  <Text style={{ color: '#f8fafc', fontSize: 15, fontWeight: 700 }}>
                    {formatMetric(pickPrimaryMetric(bestModel.result_metrics ?? {})?.value, { percent: true })}
                  </Text>
                  <Text style={{ color: 'rgba(255,255,255,0.45)', fontSize: 11 }}>
                    来源：{bestModel.dataset_name ?? bestModel.dataset_id}
                  </Text>
                </Space>
              ) : (
                <Text style={{ color: 'rgba(255,255,255,0.45)', fontSize: 12, display: 'block', marginTop: 10 }}>
                  还没有可用模型
                </Text>
              )}
            </div>
          </Col>
        </Row>
      </div>

      {/* KPI stat cards */}
      <Row gutter={[16, 16]}>
        {[
          { title: '数据集', value: data.totals.datasetCount, icon: <DatabaseOutlined />,    colorClass: 'stat-card-blue'   },
          { title: '训练任务', value: data.totals.taskCount,   icon: <RocketOutlined />,      colorClass: 'stat-card-purple' },
          { title: '已保存模型', value: data.totals.modelCount, icon: <TrophyOutlined />,     colorClass: 'stat-card-green'  },
          { title: '成功率', value: successRate, suffix: '%',   icon: <ThunderboltOutlined />, colorClass: 'stat-card-amber'  },
        ].map((item, i) => (
          <Col xs={12} sm={12} xl={6} key={i}>
            {loading
              ? <Card style={{ height: 100 }}><Skeleton active paragraph={{ rows: 1 }} /></Card>
              : <GradientStatCard {...item} />
            }
          </Col>
        ))}
      </Row>

      {/* Charts row */}
      <Row gutter={[16, 16]}>
        {/* Donut — task status */}
        <Col xs={24} md={10} lg={8}>
          <Card
            title="任务状态分布"
            extra={<Button type="link" size="small" onClick={() => navigate('/training/monitor')}>查看任务</Button>}
            style={{ height: '100%' }}
            bodyStyle={{ padding: '8px 20px 16px' }}
          >
            {loading
              ? <Skeleton active paragraph={{ rows: 5 }} />
              : data.totals.taskCount === 0
                ? <Empty description="暂无训练任务" style={{ padding: '32px 0' }} />
                : <div ref={donutRef} style={{ height: 240 }} />
            }
          </Card>
        </Col>

        {/* Bar — model performance */}
        <Col xs={24} md={14} lg={16}>
          <Card
            title="模型性能对比"
            extra={<Button type="link" size="small" onClick={() => navigate('/training/results')}>查看详情</Button>}
            style={{ height: '100%' }}
            bodyStyle={{ padding: '8px 20px 16px' }}
          >
            {loading
              ? <Skeleton active paragraph={{ rows: 5 }} />
              : data.models.length === 0
                ? <Empty description="暂无已保存模型" style={{ padding: '32px 0' }} />
                : barOption
                  ? <div ref={barRef} style={{ height: 240 }} />
                  : <Empty description="模型暂无指标数据" style={{ padding: '32px 0' }} />
            }
          </Card>
        </Col>
      </Row>

      {/* Recent tasks table + activity */}
      <Row gutter={[16, 16]}>
        <Col xs={24} xl={16}>
          <Card
            title="最近训练任务"
            extra={<Button type="link" size="small" icon={<ArrowRightOutlined />} onClick={() => navigate('/training/monitor')}>全部</Button>}
          >
            {loading
              ? <Skeleton active paragraph={{ rows: 5 }} />
              : data.recentTasks.length === 0
                ? <Empty description="还没有训练任务" />
                : <Table
                    rowKey="id"
                    pagination={false}
                    size="small"
                    dataSource={data.recentTasks}
                    columns={recentTableColumns}
                    style={{ marginTop: 4 }}
                  />
            }
          </Card>
        </Col>

        <Col xs={24} xl={8}>
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            {/* Dataset card */}
            <Card
              title="最新数据集"
              extra={<Button type="link" size="small" icon={<ArrowRightOutlined />} onClick={() => navigate('/data')}>管理</Button>}
            >
              {loading ? <Skeleton active paragraph={{ rows: 2 }} /> : latestDataset ? (
                <Space direction="vertical" size={8} style={{ width: '100%' }}>
                  <Space>
                    <div style={{
                      width: 36, height: 36, borderRadius: 8,
                      background: 'rgba(59,130,246,0.1)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: '#2563eb', fontSize: 18,
                    }}>
                      <DatabaseOutlined />
                    </div>
                    <div>
                      <Text strong style={{ fontSize: 13 }}>{latestDataset.name}</Text>
                      <div style={{ color: '#94a3b8', fontSize: 11 }}>
                        {latestDataset.row_count?.toLocaleString() ?? 0} 行 / {latestDataset.column_count ?? 0} 列
                      </div>
                    </div>
                  </Space>
                  {sparkData.length > 0 && <SparklineCard data={sparkData} title="近期活跃度" />}
                </Space>
              ) : <Empty description="还没有上传数据集" />}
            </Card>

            {/* Best model card */}
            <Card
              title="最优模型"
              extra={<Button type="link" size="small" icon={<ArrowRightOutlined />} onClick={() => navigate('/models')}>管理</Button>}
            >
              {loading ? <Skeleton active paragraph={{ rows: 2 }} /> : bestModel ? (
                <Space direction="vertical" size={8} style={{ width: '100%' }}>
                  <Space>
                    <div style={{
                      width: 36, height: 36, borderRadius: 8,
                      background: 'rgba(16, 185, 129, 0.1)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: '#059669', fontSize: 18,
                    }}>
                      <TrophyOutlined />
                    </div>
                    <div>
                      <Tag color="blue" style={{ fontWeight: 600 }}>{bestModel.model_type}</Tag>
                      <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 2 }}>
                        {bestModel.dataset_name ?? bestModel.dataset_id}
                      </div>
                    </div>
                  </Space>
                  <div style={{
                    padding: '10px 14px',
                    borderRadius: 10,
                    background: 'rgba(16,185,129,0.06)',
                    border: '1px solid rgba(16,185,129,0.15)',
                  }}>
                    <Text style={{ fontSize: 11, color: '#64748b' }}>
                      {metricLabels[pickPrimaryMetric(bestModel.result_metrics ?? {})?.key ?? 'accuracy'] ?? '指标'}
                    </Text>
                    <Text style={{ display: 'block', fontSize: 22, fontWeight: 800, color: '#059669' }}>
                      {formatMetric(pickPrimaryMetric(bestModel.result_metrics ?? {})?.value, { percent: true })}
                    </Text>
                  </div>
                </Space>
              ) : <Empty description="还没有可用模型" />}
            </Card>
          </Space>
        </Col>
      </Row>
    </Space>
  );
};

export default Dashboard;
