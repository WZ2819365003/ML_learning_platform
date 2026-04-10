import React, { useEffect, useState } from 'react';
import {
  Badge, Button, Card, Col, Divider, Empty, Input, message, Modal, Row,
  Skeleton, Space, Spin, Statistic, Table, Tabs, Tag, Tooltip, Typography,
} from 'antd';
import {
  CopyOutlined, DeleteOutlined, LinkOutlined,
  PauseCircleOutlined, PlayCircleOutlined,
} from '@ant-design/icons';
import { deployApi } from '../services/api';

const { Text, Paragraph } = Typography;

// ---------------------------------------------------------------------------
// URL display + one-click copy
// ---------------------------------------------------------------------------
const CopyableUrl = ({ url, label }) => (
  <div style={{ marginBottom: 8 }}>
    <Text type="secondary" style={{ display: 'block', marginBottom: 4 }}>{label}</Text>
    <Space style={{ width: '100%' }} align="start">
      <Text code style={{ wordBreak: 'break-all', flex: 1 }}>{url}</Text>
      <Tooltip title="复制">
        <Button
          icon={<CopyOutlined />}
          size="small"
          onClick={() => {
            navigator.clipboard.writeText(url);
            message.success('已复制');
          }}
        />
      </Tooltip>
    </Space>
  </div>
);

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function ModelDeploy() {
  const [deployments, setDeployments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const [testInput, setTestInput] = useState('[\n  {"feature1": 1.0, "feature2": "A"}\n]');
  const [testResult, setTestResult] = useState(null);
  const [testLoading, setTestLoading] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);

  const fetchDeployments = async () => {
    setLoading(true);
    try {
      const data = await deployApi.listDeployments();
      setDeployments(data.deployments || []);
    } catch {
      message.error('加载部署列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchDeployments(); }, []);

  const handleDelete = async (id) => {
    try {
      await deployApi.deleteDeployment(id);
      message.success('已删除');
      setDeleteTarget(null);
      if (selected?.deployment_id === id) setSelected(null);
      fetchDeployments();
    } catch { message.error('删除失败'); }
  };

  const handleToggleStatus = async (deployment) => {
    const next = deployment.status === 'active' ? 'paused' : 'active';
    try {
      await deployApi.updateStatus(deployment.deployment_id, next);
      message.success(next === 'active' ? '已恢复' : '已暂停');
      // Refresh selected
      if (selected?.deployment_id === deployment.deployment_id) {
        setSelected((prev) => prev ? { ...prev, status: next } : prev);
      }
      fetchDeployments();
    } catch { message.error('操作失败'); }
  };

  const handleTest = async () => {
    if (!selected) return;
    setTestLoading(true);
    setTestResult(null);
    try {
      let rows;
      try { rows = JSON.parse(testInput); }
      catch { message.error('JSON 格式错误'); setTestLoading(false); return; }
      if (!Array.isArray(rows)) rows = [rows];
      const result = await deployApi.predict(selected.deployment_id, {
        rows,
        include_probabilities: true,
      });
      setTestResult(result);
    } catch (e) {
      message.error('预测失败: ' + (e?.response?.data?.detail || e?.message || '未知错误'));
    } finally { setTestLoading(false); }
  };

  // Aggregated stats
  const totalDeployments = deployments.length;
  const activeCount = deployments.filter((d) => d.status === 'active').length;
  const totalCalls = deployments.reduce((acc, d) => acc + (d.request_count || 0), 0);

  const columns = [
    {
      title: '部署名称', dataIndex: 'name', key: 'name',
      render: (v, r) => (
        <a onClick={() => setSelected(r)} style={{ fontWeight: selected?.deployment_id === r.deployment_id ? 600 : 400 }}>
          {v}
        </a>
      ),
    },
    {
      title: '状态', dataIndex: 'status', key: 'status',
      render: (s) => (
        <Badge
          status={s === 'active' ? 'success' : 'default'}
          text={s === 'active' ? '活跃' : '暂停'}
        />
      ),
    },
    { title: '调用次数', dataIndex: 'request_count', key: 'request_count' },
    {
      title: '创建时间', dataIndex: 'created_at', key: 'created_at',
      render: (v) => new Date(v).toLocaleString('zh-CN'),
    },
    {
      title: '操作', key: 'action',
      render: (_, r) => (
        <Space>
          <Tooltip title={r.status === 'active' ? '暂停' : '恢复'}>
            <Button
              size="small"
              icon={r.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={() => handleToggleStatus(r)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => setDeleteTarget(r)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const mlContent = (
    <div>
      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card><Statistic title="部署总数" value={totalDeployments} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="活跃部署" value={activeCount} valueStyle={{ color: '#3f8600' }} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="总调用次数" value={totalCalls} /></Card>
        </Col>
      </Row>

      <Card title="部署列表" style={{ marginBottom: 24 }}>
        <Table
          columns={columns}
          dataSource={deployments}
          rowKey="deployment_id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>

      {selected && (
        <Card
          title={
            <Space>
              <LinkOutlined />
              <span>{selected.name} — 接口详情</span>
              <Tag color={selected.status === 'active' ? 'green' : 'default'}>
                {selected.status === 'active' ? '活跃' : '暂停'}
              </Tag>
            </Space>
          }
        >
          <Row gutter={24}>
            <Col xs={24} md={12}>
              <Text strong>预测 URL (url1 — POST):</Text>
              <div style={{ marginTop: 8, marginBottom: 16 }}>
                <CopyableUrl url={selected.endpoints?.predict || ''} label="提交预测请求" />
              </div>
              <Text strong>结果 URL (url2 — GET):</Text>
              <div style={{ marginTop: 8 }}>
                <CopyableUrl url={selected.endpoints?.result || ''} label="查询预测结果" />
              </div>
            </Col>
            <Col xs={24} md={12}>
              <Text strong>在线测试</Text>
              <div style={{ marginTop: 8 }}>
                <Input.TextArea
                  value={testInput}
                  onChange={(e) => setTestInput(e.target.value)}
                  rows={5}
                  style={{ fontFamily: 'monospace', marginBottom: 8 }}
                  placeholder='[{"feature1": 1.0, "feature2": "A"}]'
                />
                <Button
                  type="primary"
                  onClick={handleTest}
                  loading={testLoading}
                  disabled={selected.status !== 'active'}
                >
                  发送预测
                </Button>
                {selected.status !== 'active' && (
                  <Text type="secondary" style={{ marginLeft: 8 }}>部署已暂停</Text>
                )}
              </div>
              {testResult && (
                <>
                  <Divider />
                  <Text strong>预测结果:</Text>
                  <pre style={{ background: '#f6f8fa', padding: 12, borderRadius: 4, marginTop: 8, fontSize: 12, overflowX: 'auto' }}>
                    {JSON.stringify(testResult, null, 2)}
                  </pre>
                </>
              )}
            </Col>
          </Row>
        </Card>
      )}

      <Modal
        open={!!deleteTarget}
        title="确认删除"
        onOk={() => deleteTarget && handleDelete(deleteTarget.deployment_id)}
        onCancel={() => setDeleteTarget(null)}
        okText="删除"
        okButtonProps={{ danger: true }}
      >
        <p>确定要删除部署 <strong>{deleteTarget?.name}</strong> 吗？此操作不可撤销。</p>
      </Modal>
    </div>
  );

  return (
    <div>
      <Typography.Title level={2} style={{ marginBottom: 16 }}>模型部署</Typography.Title>
      <Tabs
        defaultActiveKey="ml"
        items={[
          { key: 'ml', label: '机器学习部署', children: mlContent },
          {
            key: 'dl',
            label: '深度学习部署',
            children: (
              <Card>
                <Empty description="深度学习模型部署功能即将上线，敬请期待。" />
              </Card>
            ),
          },
          {
            key: 'universal',
            label: '通用模型部署',
            children: (
              <Card>
                <Empty description="通用模型（如 Google TimesFM）部署功能即将上线，敬请期待。" />
              </Card>
            ),
          },
        ]}
      />
    </div>
  );
}
