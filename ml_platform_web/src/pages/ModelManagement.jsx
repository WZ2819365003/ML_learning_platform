import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Form,
  Input,
  Modal,
  Pagination,
  Popconfirm,
  Row,
  Select,
  Space,
  Spin,
  Table,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  ApiOutlined,
  BarChartOutlined,
  CloudUploadOutlined,
  CopyOutlined,
  DeleteOutlined,
  DownloadOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  RocketOutlined,
  TagOutlined,
  TrophyOutlined,
} from '@ant-design/icons';
import echarts from '../utils/echarts';
import api, { dataApi, deployApi, dlApi, modelApi, timesfmApi, trainingApi } from '../services/api';
import { formatBytes, formatDateTime, formatMetric, metricLabels } from '../utils/formatters';
import { buildResultsUrl } from '../utils/resultRoutes';

const { Text, Title } = Typography;
const { TextArea } = Input;

const PAGE_SIZE = 10;

const ML_METRIC_ORDER = [
  'accuracy', 'f1', 'precision', 'recall', 'roc_auc',
  'cv_avg_accuracy', 'cv_avg_f1',
];
const DL_METRIC_ORDER = [
  'val_acc', 'val_f1_macro', 'val_auc_roc', 'val_precision', 'val_recall',
  'val_rmse', 'val_mae', 'val_r2', 'val_mape', 'best_val_loss', 'val_loss',
];

function sanitizePreviewRow(row, targetColumn) {
  if (!row) return {};
  const next = { ...row };
  delete next[targetColumn];
  return next;
}

function renderMetricCards(metrics, keys) {
  return keys
    .filter((k) => typeof metrics?.[k] === 'number')
    .slice(0, 6)
    .map((k) => (
      <Col xs={24} sm={12} xl={8} key={k}>
        <Card size="small">
          <Text type="secondary">{metricLabels[k] ?? k}</Text>
          <Title level={4} style={{ marginTop: 4, marginBottom: 0 }}>
            {formatMetric(metrics?.[k], { percent: k.includes('acc') || k === 'accuracy' })}
          </Title>
        </Card>
      </Col>
    ));
}

// ── Tags display (read-only) ──────────────────────────────────────────────────
function TagsDisplay({ tags, notes }) {
  if (!tags?.length && !notes) return null;
  return (
    <Space direction="vertical" size={4} style={{ width: '100%' }}>
      {tags?.length > 0 && (
        <Space wrap size={4}>
          {tags.map((t) => (
            <Tag key={t} color="geekblue">{t}</Tag>
          ))}
        </Space>
      )}
      {notes && <Text type="secondary" style={{ fontSize: 12 }}>{notes}</Text>}
    </Space>
  );
}

// ── Tags + Notes edit modal (shared) ─────────────────────────────────────────
function TagsNotesModal({ open, asset, tagLibrary, saving, onSave, onCancel }) {
  const [tags, setTags] = useState([]);
  const [notes, setNotes] = useState('');

  useEffect(() => {
    if (asset) {
      setTags(asset.tags ?? []);
      setNotes(asset.notes ?? '');
    }
  }, [asset]);

  // Group tags by dimension for optgroup display
  const grouped = {};
  for (const t of tagLibrary) {
    const dim = (typeof t === 'object' ? t.dimension : null) || '其他';
    (grouped[dim] = grouped[dim] ?? []).push(t);
  }
  const libraryOptions = Object.entries(grouped).map(([dim, items]) => ({
    label: dim,
    options: items.map(t => {
      const name = typeof t === 'object' ? t.name : t;
      return { value: name, label: name };
    }),
  }));

  return (
    <Modal
      title={
        <Space>
          <TagOutlined />
          {`编辑标签与备注 — ${asset?.model_type ?? asset?.name ?? ''}`}
        </Space>
      }
      open={open}
      onCancel={onCancel}
      onOk={() => onSave(tags, notes)}
      okText="保存"
      cancelText="取消"
      confirmLoading={saving}
      destroyOnHidden
      width={520}
    >
      <Space direction="vertical" style={{ width: '100%' }} size={16}>
        <div>
          <Text strong style={{ display: 'block', marginBottom: 6 }}>标签</Text>
          <Select
            mode="tags"
            value={tags}
            onChange={setTags}
            style={{ width: '100%' }}
            placeholder="从标签库选择，或输入新标签后按回车"
            tokenSeparators={[',']}
            options={libraryOptions}
            allowClear
          />
          {tagLibrary.length > 0 && (
            <Text type="secondary" style={{ fontSize: 11, marginTop: 4, display: 'block' }}>
              共 {tagLibrary.length} 个标签可选，输入新标签后按回车可直接创建
            </Text>
          )}
        </div>
        <div>
          <Text strong style={{ display: 'block', marginBottom: 6 }}>备注</Text>
          <TextArea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={4}
            placeholder="添加备注说明…"
          />
        </div>
      </Space>
    </Modal>
  );
}

// ── ML Tab ────────────────────────────────────────────────────────────────────
function MLModelTab({ openDeployModal, openTagsModal }) {
  const navigate = useNavigate();
  const [models, setModels] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedKeys, setSelectedKeys] = useState([]);
  const [compareData, setCompareData] = useState([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const compareChartRef = useRef(null);

  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [predictionPayload, setPredictionPayload] = useState('[]');
  const [predictionResult, setPredictionResult] = useState('');
  const [predictionRunning, setPredictionRunning] = useState(false);

  useEffect(() => { void loadModels(page); }, [page]);

  useEffect(() => {
    if (!compareOpen || !compareChartRef.current || compareData.length === 0) return undefined;
    const chart = echarts.init(compareChartRef.current);
    const metricKeys = ML_METRIC_ORDER.filter((k) =>
      compareData.some((item) => typeof item.result_metrics?.[k] === 'number')
    );
    chart.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { bottom: 0 },
      grid: { top: 24, left: 56, right: 20, bottom: 56 },
      xAxis: { type: 'category', data: metricKeys.map((k) => metricLabels[k] ?? k) },
      yAxis: { type: 'value', min: 0, max: 1 },
      series: compareData.map((item) => ({
        name: item.model_type,
        type: 'bar',
        data: metricKeys.map((k) => item.result_metrics?.[k] ?? 0),
      })),
    });
    const onResize = () => chart.resize();
    window.addEventListener('resize', onResize);
    return () => { window.removeEventListener('resize', onResize); chart.dispose(); };
  }, [compareData, compareOpen]);

  async function loadModels(p) {
    setLoading(true);
    try {
      const res = await modelApi.listAssets({ runtime_type: 'ml', page: p, page_size: PAGE_SIZE });
      setModels(res.items ?? []);
      setTotal(res.total ?? 0);
    } catch {
      message.error('加载机器学习模型失败');
    } finally {
      setLoading(false);
    }
  }

  async function openDetail(record) {
    setSelectedAsset(record);
    setDetailOpen(true);
    setDetailLoading(true);
    setPredictionResult('');
    setDetail(null);
    try {
      const [det, preview] = await Promise.all([
        modelApi.getModelDetail(record.task_id),
        dataApi.previewDataset(record.dataset_id).catch(() => null),
      ]);
      setDetail(det);
      if (preview?.rows?.[0]) {
        setPredictionPayload(
          JSON.stringify([sanitizePreviewRow(preview.rows[0], det.target_column)], null, 2)
        );
      }
    } catch {
      message.error('加载模型详情失败');
    } finally {
      setDetailLoading(false);
    }
  }

  async function handleDelete(taskId) {
    try {
      await modelApi.deleteModel(taskId);
      message.success('已删除');
      if (selectedAsset?.task_id === taskId) setDetailOpen(false);
      void loadModels(page);
    } catch {
      message.error('删除失败');
    }
  }

  async function handleCompare() {
    if (selectedKeys.length < 2) { message.warning('请至少选择两个模型'); return; }
    try {
      const res = await modelApi.compareModels(selectedKeys);
      setCompareData(res);
      setCompareOpen(true);
    } catch {
      message.error('加载对比失败');
    }
  }

  async function handlePredict() {
    if (!detail) return;
    let rows;
    try { rows = JSON.parse(predictionPayload); } catch { message.error('JSON 格式错误'); return; }
    if (!Array.isArray(rows) || rows.length === 0) { message.error('请至少提供一行数据'); return; }
    setPredictionRunning(true);
    try {
      const res = await modelApi.predict(detail.task_id, { rows, include_probabilities: true });
      setPredictionResult(JSON.stringify(res, null, 2));
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '预测失败');
    } finally {
      setPredictionRunning(false);
    }
  }

  const predictionUrl = detail ? `${api.defaults.baseURL}/models/${detail.task_id}/predict` : '';

  const columns = [
    {
      title: '任务名称',
      key: 'name',
      render: (_, r) => r.name ?? <Text type="secondary">{r.task_id.slice(0, 8)}</Text>,
    },
    { title: '数据集', dataIndex: 'dataset_name', render: (v, r) => v ?? r.dataset_id },
    { title: '模型', dataIndex: 'model_type', render: (v) => <Tag color="blue">{v}</Tag> },
    {
      title: '标签',
      dataIndex: 'tags',
      render: (tags) =>
        tags?.length
          ? tags.map((t) => <Tag key={t} color="geekblue" style={{ marginBottom: 2 }}>{t}</Tag>)
          : <Text type="secondary">—</Text>,
    },
    {
      title: '主要指标',
      key: 'metric',
      render: (_, r) => {
        const { primary_metric_name: name, primary_metric_value: val } = r.metrics_summary ?? {};
        if (!name) return '—';
        return (
          <span>
            {metricLabels[name] ?? name}:{' '}
            <strong>{formatMetric(val, { percent: name.includes('acc') || name === 'accuracy' })}</strong>
          </span>
        );
      },
    },
    { title: '完成时间', dataIndex: 'finished_at', render: (v) => formatDateTime(v), width: 155 },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      render: (_, r) => (
        <Space size={4} onClick={(e) => e.stopPropagation()}>
          <Button size="small" icon={<EyeOutlined />} onClick={() => void openDetail(r)}>详情</Button>
          <Button size="small" icon={<TagOutlined />} onClick={() => openTagsModal(r, 'ml')}>标签</Button>
          <Button size="small" icon={<CloudUploadOutlined />} onClick={() => openDeployModal({ ...r, runtime_type: 'ml' })} />
          <Button size="small" danger icon={<DeleteOutlined />} onClick={() => void handleDelete(r.task_id)} />
        </Space>
      ),
    },
  ];

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Space wrap style={{ width: '100%', justifyContent: 'space-between' }}>
        <Space>
          <Button
            type="primary"
            icon={<BarChartOutlined />}
            disabled={selectedKeys.length < 2}
            onClick={() => void handleCompare()}
          >
            对比已选 ({selectedKeys.length}/3)
          </Button>
          <Text type="secondary">勾选最多 3 个模型进行指标对比</Text>
        </Space>
        <Button onClick={() => void loadModels(page)}>刷新</Button>
      </Space>

      <Card>
        <Table
          rowKey="asset_id"
          dataSource={models}
          columns={columns}
          loading={loading}
          pagination={false}
          size="middle"
          locale={{ emptyText: <Empty description="还没有训练完成的机器学习模型" /> }}
          rowSelection={{
            selectedRowKeys: selectedKeys,
            onChange: (keys) => {
              if (keys.length > 3) { message.warning('最多选择 3 个'); return; }
              setSelectedKeys(keys);
            },
          }}
          onRow={(r) => ({
            style: { cursor: 'pointer' },
            onClick: () => void openDetail(r),
          })}
        />
        <div style={{ marginTop: 12, textAlign: 'right' }}>
          <Pagination
            current={page}
            pageSize={PAGE_SIZE}
            total={total}
            onChange={setPage}
            showTotal={(t) => `共 ${t} 条`}
            showSizeChanger={false}
          />
        </div>
      </Card>

      {/* Detail modal */}
      <Modal
        title={`模型详情 — ${selectedAsset?.model_type ?? ''}`}
        open={detailOpen}
        onCancel={() => { setDetailOpen(false); setDetail(null); }}
        footer={null}
        width={1000}
        destroyOnHidden
      >
        {detailLoading || !detail ? (
          <div style={{ textAlign: 'center', padding: 40 }}><Spin size="large" /></div>
        ) : (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <TagsDisplay tags={selectedAsset?.tags} notes={selectedAsset?.notes} />

            <Descriptions bordered column={{ xs: 1, sm: 2, xl: 3 }} size="small">
              <Descriptions.Item label="数据集">{detail.dataset?.name ?? '—'}</Descriptions.Item>
              <Descriptions.Item label="模型">{detail.model_type}</Descriptions.Item>
              <Descriptions.Item label="目标列">{detail.target_column}</Descriptions.Item>
              <Descriptions.Item label="状态">{detail.status}</Descriptions.Item>
              <Descriptions.Item label="完成时间">{formatDateTime(detail.finished_at)}</Descriptions.Item>
              <Descriptions.Item label="模型大小">{formatBytes(detail.model_size)}</Descriptions.Item>
            </Descriptions>

            <Row gutter={[12, 12]}>
              {renderMetricCards(detail.result_metrics, ML_METRIC_ORDER)}
            </Row>

            <Card size="small" title="快捷操作">
              <Space wrap>
                <Button
                  type="primary"
                  icon={<TrophyOutlined />}
                  onClick={() => { setDetailOpen(false); navigate(`/training/results?taskId=${detail.task_id}`); }}
                >
                  查看结果可视化
                </Button>
                <Button
                  icon={<DownloadOutlined />}
                  onClick={() => {
                    const a = document.createElement('a');
                    a.href = modelApi.downloadModelUrl(detail.task_id);
                    a.download = '';
                    a.click();
                  }}
                >
                  下载模型文件
                </Button>
              </Space>
            </Card>

            <Card
              size="small"
              title={<Space><ApiOutlined />预测接口</Space>}
              extra={
                <Button
                  icon={<CopyOutlined />}
                  size="small"
                  onClick={async () => { await navigator.clipboard.writeText(predictionUrl); message.success('已复制'); }}
                >
                  复制 URL
                </Button>
              }
            >
              <Space direction="vertical" size={12} style={{ width: '100%' }}>
                <Alert type="info" showIcon message="向下方 URL 发送 POST 请求（body 传 rows 数组）即可获得预测结果。" />
                <Text code style={{ display: 'block', wordBreak: 'break-all' }}>{predictionUrl}</Text>
                <Text strong>请求体 (rows 数组)</Text>
                <TextArea
                  value={predictionPayload}
                  onChange={(e) => setPredictionPayload(e.target.value)}
                  rows={8}
                />
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  loading={predictionRunning}
                  onClick={() => void handlePredict()}
                >
                  运行预测
                </Button>
                <Text strong>预测结果</Text>
                <pre
                  style={{
                    margin: 0, padding: 12, borderRadius: 8,
                    background: '#0f172a', color: '#e2e8f0',
                    minHeight: 80, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12,
                  }}
                >
                  {predictionResult || '点击「运行预测」后结果显示在这里'}
                </pre>
              </Space>
            </Card>
          </Space>
        )}
      </Modal>


      {/* Compare modal */}
      <Modal
        title="模型指标对比"
        open={compareOpen}
        onCancel={() => setCompareOpen(false)}
        footer={null}
        width={900}
        destroyOnHidden
      >
        {compareData.length === 0
          ? <Alert type="info" showIcon message="暂无对比数据" />
          : <div ref={compareChartRef} style={{ width: '100%', height: 420 }} />}
      </Modal>
    </Space>
  );
}

// ── DL Tab ────────────────────────────────────────────────────────────────────
function DLModelTab({ openDeployModal, openTagsModal }) {
  const navigate = useNavigate();
  const [models, setModels] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [predictionPayload, setPredictionPayload] = useState('[]');
  const [predictionResult, setPredictionResult] = useState('');
  const [predictionRunning, setPredictionRunning] = useState(false);

  useEffect(() => { void loadModels(page); }, [page]);

  async function loadModels(p) {
    setLoading(true);
    try {
      const res = await modelApi.listAssets({ runtime_type: 'dl', page: p, page_size: PAGE_SIZE });
      setModels(res.items ?? []);
      setTotal(res.total ?? 0);
    } catch {
      message.error('加载深度学习模型失败');
    } finally {
      setLoading(false);
    }
  }

  async function openDetail(record) {
    setSelectedAsset(record);
    setDetailOpen(true);
    setDetailLoading(true);
    setDetail(null);
    setPredictionResult('');
    try {
      const [det, preview] = await Promise.all([
        dlApi.getStatus(record.task_id),
        dataApi.previewDataset(record.dataset_id).catch(() => null),
      ]);
      const enriched = {
        ...det,
        dataset_name: record.dataset_name,
        dataset_id: record.dataset_id,
      };
      setDetail(enriched);
      if (preview?.rows?.[0]) {
        setPredictionPayload(
          JSON.stringify([sanitizePreviewRow(preview.rows[0], det.target_column)], null, 2)
        );
      }
    } catch {
      message.error('加载模型详情失败');
    } finally {
      setDetailLoading(false);
    }
  }

  async function handleDelete(taskId) {
    try {
      await dlApi.deleteTask(taskId);
      message.success('已删除');
      if (selectedAsset?.task_id === taskId) setDetailOpen(false);
      void loadModels(page);
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '删除失败');
    }
  }

  const dlPredictionUrl = detail ? `${api.defaults.baseURL}/dl/${detail.id}/predict` : '';

  async function handleDLPredict() {
    if (!detail) return;
    let rows;
    try { rows = JSON.parse(predictionPayload); } catch { message.error('JSON 格式错误'); return; }
    if (!Array.isArray(rows) || rows.length === 0) { message.error('请至少提供一行数据'); return; }
    setPredictionRunning(true);
    try {
      const res = await dlApi.predictTask(detail.id, { rows });
      setPredictionResult(JSON.stringify(res, null, 2));
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '预测失败');
    } finally {
      setPredictionRunning(false);
    }
  }

  const columns = [
    {
      title: '任务名称',
      key: 'name',
      render: (_, r) => r.name ?? <Text type="secondary">{r.task_id.slice(0, 8)}</Text>,
    },
    { title: '数据集', dataIndex: 'dataset_name', render: (v, r) => v ?? r.dataset_id },
    { title: '架构', dataIndex: 'model_type', render: (v) => <Tag color="purple">{v}</Tag> },
    { title: '任务类型', dataIndex: 'task_type', width: 100 },
    {
      title: '标签',
      dataIndex: 'tags',
      render: (tags) =>
        tags?.length
          ? tags.map((t) => <Tag key={t} color="geekblue" style={{ marginBottom: 2 }}>{t}</Tag>)
          : <Text type="secondary">—</Text>,
    },
    {
      title: '主要指标',
      key: 'metric',
      render: (_, r) => {
        const { primary_metric_name: name, primary_metric_value: val } = r.metrics_summary ?? {};
        if (!name) return '—';
        return (
          <span>
            {metricLabels[name] ?? name}:{' '}
            <strong>{formatMetric(val, { percent: name === 'val_acc' })}</strong>
          </span>
        );
      },
    },
    { title: '完成时间', dataIndex: 'finished_at', render: (v) => formatDateTime(v), width: 155 },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      render: (_, r) => (
        <Space size={4} onClick={(e) => e.stopPropagation()}>
          <Button size="small" icon={<EyeOutlined />} onClick={() => void openDetail(r)}>详情</Button>
          <Button size="small" icon={<TagOutlined />} onClick={() => openTagsModal(r, 'dl')}>标签</Button>
          <Button size="small" icon={<CloudUploadOutlined />} onClick={() => openDeployModal(r)} />
          <Button size="small" danger icon={<DeleteOutlined />} onClick={() => void handleDelete(r.task_id)} />
        </Space>
      ),
    },
  ];

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Space style={{ width: '100%', justifyContent: 'space-between' }}>
        <Text type="secondary">点击行查看训练详情；点击「标签」按钮管理标签和备注。</Text>
        <Button onClick={() => void loadModels(page)}>刷新</Button>
      </Space>

      <Card>
        <Table
          rowKey="asset_id"
          dataSource={models}
          columns={columns}
          loading={loading}
          pagination={false}
          size="middle"
          locale={{ emptyText: <Empty description="还没有训练完成的深度学习模型" /> }}
          onRow={(r) => ({
            style: { cursor: 'pointer' },
            onClick: () => void openDetail(r),
          })}
        />
        <div style={{ marginTop: 12, textAlign: 'right' }}>
          <Pagination
            current={page}
            pageSize={PAGE_SIZE}
            total={total}
            onChange={setPage}
            showTotal={(t) => `共 ${t} 条`}
            showSizeChanger={false}
          />
        </div>
      </Card>

      {/* Detail modal */}
      <Modal
        title={`深度学习模型详情 — ${selectedAsset?.model_type ?? ''}`}
        open={detailOpen}
        onCancel={() => { setDetailOpen(false); setDetail(null); }}
        footer={null}
        width={1000}
        destroyOnHidden
      >
        {detailLoading || !detail ? (
          <div style={{ textAlign: 'center', padding: 40 }}><Spin size="large" /></div>
        ) : (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <TagsDisplay tags={selectedAsset?.tags} notes={selectedAsset?.notes} />

            <Descriptions bordered column={{ xs: 1, sm: 2, xl: 3 }} size="small">
              <Descriptions.Item label="数据集">{detail.dataset_name ?? detail.dataset_id ?? '—'}</Descriptions.Item>
              <Descriptions.Item label="架构">{detail.model_type}</Descriptions.Item>
              <Descriptions.Item label="目标列">{detail.target_column}</Descriptions.Item>
              <Descriptions.Item label="任务类型">{detail.task_type}</Descriptions.Item>
              <Descriptions.Item label="训练进度">{detail.current_epoch} / {detail.total_epochs} Epoch</Descriptions.Item>
              <Descriptions.Item label="完成时间">{formatDateTime(detail.finished_at)}</Descriptions.Item>
            </Descriptions>

            <Row gutter={[12, 12]}>
              {renderMetricCards(detail.result_metrics, DL_METRIC_ORDER)}
            </Row>

            <Card size="small" title="快捷操作">
              <Space wrap>
                <Button
                  icon={<RocketOutlined />}
                  onClick={() => { setDetailOpen(false); navigate(`/dl/monitor?taskId=${detail.id}`); }}
                >
                  查看训练监控
                </Button>
                <Button
                  type="primary"
                  icon={<TrophyOutlined />}
                  onClick={() => {
                    setDetailOpen(false);
                    navigate(buildResultsUrl({ family: 'dl', taskId: detail.id }));
                  }}
                >
                  查看结果可视化
                </Button>
              </Space>
            </Card>

            <Card
              size="small"
              title={<Space><ApiOutlined />预测接口</Space>}
              extra={
                <Button
                  icon={<CopyOutlined />}
                  size="small"
                  onClick={async () => { await navigator.clipboard.writeText(dlPredictionUrl); message.success('已复制'); }}
                >
                  复制 URL
                </Button>
              }
            >
              <Space direction="vertical" size={12} style={{ width: '100%' }}>
                <Alert type="info" showIcon message="向下方 URL 发送 POST 请求（body 传 rows 数组）即可获得预测结果。" />
                <Text code style={{ display: 'block', wordBreak: 'break-all' }}>{dlPredictionUrl}</Text>
                <Text strong>请求体 (rows 数组)</Text>
                <TextArea
                  value={predictionPayload}
                  onChange={(e) => setPredictionPayload(e.target.value)}
                  rows={8}
                />
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  loading={predictionRunning}
                  onClick={() => void handleDLPredict()}
                >
                  运行预测
                </Button>
                <Text strong>预测结果</Text>
                <pre
                  style={{
                    margin: 0, padding: 12, borderRadius: 8,
                    background: '#0f172a', color: '#e2e8f0',
                    minHeight: 80, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12,
                  }}
                >
                  {predictionResult || '点击「运行预测」后结果显示在这里'}
                </pre>
              </Space>
            </Card>
          </Space>
        )}
      </Modal>
    </Space>
  );
}

// ── Universal (TimesFM) Tab ───────────────────────────────────────────────────
const FREQ_LABELS = { high: '高频', medium: '中频', low: '低频' };
const TS_STATUS_LABEL = { PENDING: '等待中', RUNNING: '运行中', SUCCESS: '完成', FAILED: '失败' };

function UniversalModelTab() {
  const navigate = useNavigate();
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState(null);

  useEffect(() => { void fetchTasks(1, null); }, []);

  async function fetchTasks(p, sf) {
    setLoading(true);
    try {
      const params = { page: p, page_size: PAGE_SIZE };
      if (sf) params.status = sf;
      const res = await timesfmApi.listForecasts(params);
      setTasks(res.items ?? []);
      setTotal(res.total ?? 0);
      setPage(p);
    } catch {
      message.error('加载时序任务失败');
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id) {
    try {
      await timesfmApi.deleteForecast(id);
      message.success('已删除');
      void fetchTasks(page, statusFilter);
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '删除失败');
    }
  }

  const columns = [
    {
      title: '数据集 / 预测目标',
      key: 'name',
      render: (_, r) => (
        <Space direction="vertical" size={0}>
          <Text strong>{r.dataset_name ?? r.dataset_id?.slice(0, 8)}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>{r.value_column}</Text>
        </Space>
      ),
    },
    {
      title: '预测步数',
      dataIndex: 'horizon',
      render: (v) => `${v} 步`,
      width: 90,
    },
    {
      title: '频率',
      dataIndex: 'frequency',
      render: (v) => FREQ_LABELS[v] ?? v,
      width: 80,
    },
    {
      title: '模型',
      dataIndex: 'model_name',
      render: (v) => <Tag color="purple">{v?.split('/').pop() ?? v}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 90,
      render: (s) => (
        <Tag color={
          s === 'SUCCESS' ? 'success' :
          s === 'RUNNING' ? 'processing' :
          s === 'FAILED'  ? 'error' : 'default'
        }>
          {TS_STATUS_LABEL[s] ?? s}
        </Tag>
      ),
    },
    {
      title: '提交时间',
      dataIndex: 'created_at',
      render: (v) => formatDateTime(v),
      width: 140,
    },
    {
      title: '操作',
      key: 'actions',
      width: 130,
      render: (_, r) => (
        <Space size={4} onClick={(e) => e.stopPropagation()}>
          <Button
            size="small"
            icon={<EyeOutlined />}
            disabled={r.status !== 'SUCCESS'}
            onClick={() => navigate(`/ts/tasks/${r.id}`)}
          >
            查看结果
          </Button>
          <Popconfirm
            title="确认删除此预测记录？"
            onConfirm={() => void handleDelete(r.id)}
            disabled={r.status === 'RUNNING'}
          >
            <Button size="small" danger icon={<DeleteOutlined />} disabled={r.status === 'RUNNING'} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      {/* Quick nav */}
      <Card size="small">
        <Space wrap>
          <Text type="secondary">Amazon Chronos 零样本时序预测：</Text>
          <Button size="small" type="primary" onClick={() => navigate('/ts/config')}>新建预测任务</Button>
          <Button size="small" onClick={() => navigate('/ts/monitor')}>任务监控</Button>
          <Button size="small" onClick={() => navigate('/ts/results')}>结果可视化</Button>
        </Space>
      </Card>

      {/* Task list */}
      <Card
        title="时序预测任务"
        extra={
          <Space size={8}>
            <Select
              size="small"
              style={{ width: 90 }}
              placeholder="状态"
              allowClear
              value={statusFilter}
              onChange={(v) => { setStatusFilter(v ?? null); void fetchTasks(1, v ?? null); }}
              options={[
                { value: 'SUCCESS', label: '完成' },
                { value: 'RUNNING', label: '运行中' },
                { value: 'PENDING', label: '等待中' },
                { value: 'FAILED', label: '失败' },
              ]}
            />
            <Button size="small" onClick={() => void fetchTasks(page, statusFilter)}>刷新</Button>
          </Space>
        }
      >
        <Table
          rowKey="id"
          dataSource={tasks}
          columns={columns}
          loading={loading}
          size="middle"
          pagination={{
            current: page,
            pageSize: PAGE_SIZE,
            total,
            onChange: (p) => void fetchTasks(p, statusFilter),
            showTotal: (t) => `共 ${t} 条`,
            showSizeChanger: false,
          }}
          locale={{ emptyText: <Empty description="暂无时序预测记录" /> }}
          onRow={(r) => ({
            style: r.status === 'SUCCESS' ? { cursor: 'pointer' } : {},
            onClick: () => r.status === 'SUCCESS' && navigate(`/ts/results?id=${r.id}`),
          })}
        />
      </Card>
    </Space>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function ModelManagement() {
  const [deployTarget, setDeployTarget] = useState(null);
  const [deployLoading, setDeployLoading] = useState(false);
  const [deployForm] = Form.useForm();

  // Tag library
  const [tagLibrary, setTagLibrary] = useState([]);

  // Shared tags/notes modal state
  const [tagsModal, setTagsModal] = useState({ open: false, asset: null, runtimeType: null });
  const [tagsSaving, setTagsSaving] = useState(false);

  // Reload triggers so tabs know when tags were updated externally
  const [mlReloadKey, setMlReloadKey] = useState(0);
  const [dlReloadKey, setDlReloadKey] = useState(0);

  useEffect(() => { void fetchTagLibrary(); }, []);

  async function fetchTagLibrary() {
    try {
      const res = await modelApi.listTags();
      setTagLibrary(res.tags ?? []);
    } catch { /* non-critical */ }
  }

  function openDeployModal(target) {
    setDeployTarget(target);
    deployForm.setFieldValue('name', `${target.model_type ?? 'model'}-deploy`);
  }

  function openTagsModal(asset, runtimeType) {
    setTagsModal({ open: true, asset, runtimeType });
  }

  async function handleSaveTags(tags, notes) {
    const { asset, runtimeType } = tagsModal;
    if (!asset) return;
    setTagsSaving(true);
    try {
      if (runtimeType === 'dl') {
        await dlApi.updateMeta(asset.task_id, { notes, tags });
      } else {
        await trainingApi.updateMeta(asset.task_id, { notes, tags });
      }
      // Sync new tags to the library (best-effort)
      if (tags?.length > 0) {
        const res = await modelApi.syncTags(tags).catch(() => null);
        if (res?.tags) setTagLibrary(res.tags);
      }
      message.success('保存成功');
      setTagsModal({ open: false, asset: null, runtimeType: null });
      // Trigger table reload in the relevant tab
      if (runtimeType === 'dl') setDlReloadKey((k) => k + 1);
      else setMlReloadKey((k) => k + 1);
    } catch {
      message.error('保存失败');
    } finally {
      setTagsSaving(false);
    }
  }

  async function handleDeploy(values) {
    if (!deployTarget) return;
    setDeployLoading(true);
    try {
      if (deployTarget.runtime_type === 'dl') {
        await dlApi.createDeployment(deployTarget.task_id, {
          name: values.name,
          description: values.description || null,
        });
      } else {
        await deployApi.createDeployment(deployTarget.task_id, {
          name: values.name,
          description: values.description || null,
          max_batch_size: 100,
        });
      }
      message.success('部署成功，请前往「模型部署」页查看接口');
      setDeployTarget(null);
      deployForm.resetFields();
    } catch (err) {
      message.error(err?.response?.data?.detail ?? '部署失败');
    } finally {
      setDeployLoading(false);
    }
  }

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      <Title level={2} style={{ margin: 0 }}>模型管理</Title>

      <Tabs
        defaultActiveKey="ml"
        items={[
          {
            key: 'ml',
            label: '机器学习模型',
            children: (
              <MLModelTab
                key={mlReloadKey}
                openDeployModal={openDeployModal}
                openTagsModal={openTagsModal}
              />
            ),
          },
          {
            key: 'dl',
            label: '深度学习模型',
            children: (
              <DLModelTab
                key={dlReloadKey}
                openDeployModal={openDeployModal}
                openTagsModal={openTagsModal}
              />
            ),
          },
          {
            key: 'universal',
            label: '通用模型',
            children: <UniversalModelTab />,
          },
        ]}
      />

      {/* Shared tags/notes modal */}
      <TagsNotesModal
        open={tagsModal.open}
        asset={tagsModal.asset}
        tagLibrary={tagLibrary}
        saving={tagsSaving}
        onSave={handleSaveTags}
        onCancel={() => setTagsModal({ open: false, asset: null, runtimeType: null })}
      />

      {/* Shared deploy modal */}
      <Modal
        title={<Space><CloudUploadOutlined />部署模型</Space>}
        open={!!deployTarget}
        onCancel={() => { setDeployTarget(null); deployForm.resetFields(); }}
        onOk={() => deployForm.submit()}
        okText="确认部署"
        confirmLoading={deployLoading}
        destroyOnHidden
      >
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message={
            deployTarget?.runtime_type === 'dl'
              ? `将为深度学习模型 ${deployTarget?.model_type ?? ''} 创建部署`
              : `将为机器学习模型 ${deployTarget?.model_type ?? ''} 创建部署，部署后可在「模型部署」页测试接口`
          }
        />
        <Form form={deployForm} layout="vertical" onFinish={(v) => void handleDeploy(v)}>
          <Form.Item
            label="部署名称"
            name="name"
            rules={[{ required: true, message: '请输入部署名称' }]}
          >
            <Input placeholder="例：生产预测服务-v1" />
          </Form.Item>
          <Form.Item label="描述（可选）" name="description">
            <Input.TextArea rows={2} placeholder="简短描述此部署的用途" />
          </Form.Item>
        </Form>
      </Modal>
    </Space>
  );
}
