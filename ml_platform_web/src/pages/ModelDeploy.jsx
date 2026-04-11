import React, { useEffect, useMemo, useState } from 'react';
import {
  Badge,
  Button,
  Card,
  Col,
  Empty,
  Input,
  Modal,
  Row,
  Space,
  Statistic,
  Table,
  Tabs,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd';
import {
  CopyOutlined,
  DeleteOutlined,
  LinkOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons';
import api, { dataApi, deployApi, dlApi, modelApi } from '../services/api';

const { Text } = Typography;

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

export default function ModelDeploy() {
  const [mlDeployments, setMlDeployments] = useState([]);
  const [mlLoading, setMlLoading] = useState(false);
  const [selectedMlDeployment, setSelectedMlDeployment] = useState(null);
  const [mlTestInput, setMlTestInput] = useState('[\n  {}\n]');
  const [mlTestResult, setMlTestResult] = useState(null);
  const [mlTestLoading, setMlTestLoading] = useState(false);
  const [mlDeleteTarget, setMlDeleteTarget] = useState(null);

  const [dlDeployments, setDlDeployments] = useState([]);
  const [dlLoading, setDlLoading] = useState(false);
  const [selectedDlDeployment, setSelectedDlDeployment] = useState(null);
  const [dlTestInput, setDlTestInput] = useState('[\n  {}\n]');
  const [dlTestResult, setDlTestResult] = useState(null);
  const [dlTestLoading, setDlTestLoading] = useState(false);
  const [dlDeleteTarget, setDlDeleteTarget] = useState(null);

  useEffect(() => {
    void fetchMlDeployments();
    void fetchDlDeployments();
  }, []);

  // Auto-populate test input with real feature columns when ML deployment selected
  useEffect(() => {
    if (!selectedMlDeployment?.task_id) return;
    void (async () => {
      try {
        const detail = await modelApi.getModelDetail(selectedMlDeployment.task_id);
        const preview = await dataApi.previewDataset(detail.dataset?.id ?? detail.dataset_id).catch(() => null);
        if (preview?.rows?.[0] && detail.target_column) {
          const row = { ...preview.rows[0] };
          delete row[detail.target_column];
          setMlTestInput(JSON.stringify([row], null, 2));
        }
      } catch {
        // ignore — keep default
      }
    })();
  }, [selectedMlDeployment?.task_id]);

  // Auto-populate test input with real feature columns when DL deployment selected
  useEffect(() => {
    if (!selectedDlDeployment?.dl_task_id) return;
    void (async () => {
      try {
        const task = await dlApi.getStatus(selectedDlDeployment.dl_task_id);
        const preview = await dataApi.previewDataset(task.dataset_id).catch(() => null);
        if (preview?.rows?.[0] && task.target_column) {
          const row = { ...preview.rows[0] };
          delete row[task.target_column];
          setDlTestInput(JSON.stringify([row], null, 2));
        }
      } catch {
        // ignore
      }
    })();
  }, [selectedDlDeployment?.dl_task_id]);

  const mlStats = useMemo(() => ({
    total: mlDeployments.length,
    active: mlDeployments.filter((deployment) => deployment.status === 'active').length,
    calls: mlDeployments.reduce((sum, deployment) => sum + (deployment.request_count || 0), 0),
  }), [mlDeployments]);

  const dlStats = useMemo(() => ({
    total: dlDeployments.length,
    active: dlDeployments.filter((deployment) => deployment.status === 'active').length,
    calls: dlDeployments.reduce((sum, deployment) => sum + (deployment.request_count || 0), 0),
  }), [dlDeployments]);

  async function fetchMlDeployments() {
    setMlLoading(true);
    try {
      const data = await deployApi.listDeployments();
      const items = data.deployments || [];
      setMlDeployments(items);
      setSelectedMlDeployment((current) =>
        items.find((item) => item.deployment_id === current?.deployment_id) ?? null
      );
    } catch (error) {
      console.error('加载机器学习部署列表失败:', error);
      message.error('加载机器学习部署列表失败');
    } finally {
      setMlLoading(false);
    }
  }

  async function fetchDlDeployments() {
    setDlLoading(true);
    try {
      const data = await dlApi.listDeployments();
      const items = data.deployments || [];
      setDlDeployments(items);
      setSelectedDlDeployment((current) =>
        items.find((item) => item.id === current?.id) ?? null
      );
    } catch (error) {
      console.error('加载深度学习部署列表失败:', error);
      message.error('加载深度学习部署列表失败');
    } finally {
      setDlLoading(false);
    }
  }

  async function handleDeleteMlDeployment(id) {
    try {
      await deployApi.deleteDeployment(id);
      message.success('已删除');
      setMlDeleteTarget(null);
      if (selectedMlDeployment?.deployment_id === id) {
        setSelectedMlDeployment(null);
      }
      await fetchMlDeployments();
    } catch (error) {
      console.error('删除机器学习部署失败:', error);
      message.error(error?.response?.data?.detail ?? '删除失败');
    }
  }

  async function handleDeleteDlDeployment(id) {
    try {
      await dlApi.deleteDeployment(id);
      message.success('已删除');
      setDlDeleteTarget(null);
      if (selectedDlDeployment?.id === id) {
        setSelectedDlDeployment(null);
      }
      await fetchDlDeployments();
    } catch (error) {
      console.error('删除深度学习部署失败:', error);
      message.error(error?.response?.data?.detail ?? '删除失败');
    }
  }

  async function handleToggleMlStatus(deployment) {
    const nextStatus = deployment.status === 'active' ? 'paused' : 'active';
    try {
      await deployApi.updateStatus(deployment.deployment_id, nextStatus);
      message.success(nextStatus === 'active' ? '已恢复' : '已暂停');
      if (selectedMlDeployment?.deployment_id === deployment.deployment_id) {
        setSelectedMlDeployment((current) => current ? { ...current, status: nextStatus } : current);
      }
      await fetchMlDeployments();
    } catch (error) {
      console.error('更新机器学习部署状态失败:', error);
      message.error(error?.response?.data?.detail ?? '操作失败');
    }
  }

  async function handleToggleDlStatus(deployment) {
    const nextStatus = deployment.status === 'active' ? 'paused' : 'active';
    try {
      await dlApi.toggleDeployment(deployment.id, nextStatus);
      message.success(nextStatus === 'active' ? '已恢复' : '已暂停');
      if (selectedDlDeployment?.id === deployment.id) {
        setSelectedDlDeployment((current) => current ? { ...current, status: nextStatus } : current);
      }
      await fetchDlDeployments();
    } catch (error) {
      console.error('更新深度学习部署状态失败:', error);
      message.error(error?.response?.data?.detail ?? '操作失败');
    }
  }

  async function handleTestMlDeployment() {
    if (!selectedMlDeployment) return;
    setMlTestLoading(true);
    setMlTestResult(null);
    try {
      let rows;
      try {
        rows = JSON.parse(mlTestInput);
      } catch {
        message.error('JSON 格式错误');
        setMlTestLoading(false);
        return;
      }
      if (!Array.isArray(rows)) rows = [rows];
      const result = await deployApi.predict(selectedMlDeployment.deployment_id, {
        rows,
        include_probabilities: true,
      });
      setMlTestResult(result);
    } catch (error) {
      message.error(`预测失败: ${error?.response?.data?.detail || error?.message || '未知错误'}`);
    } finally {
      setMlTestLoading(false);
    }
  }

  async function handleTestDlDeployment() {
    if (!selectedDlDeployment) return;
    setDlTestLoading(true);
    setDlTestResult(null);
    try {
      let rows;
      try {
        rows = JSON.parse(dlTestInput);
      } catch {
        message.error('JSON 格式错误');
        setDlTestLoading(false);
        return;
      }
      if (!Array.isArray(rows)) rows = [rows];
      const result = await dlApi.predictDeployment(selectedDlDeployment.id, { rows });
      setDlTestResult(result);
    } catch (error) {
      message.error(`预测失败: ${error?.response?.data?.detail || error?.message || '未知错误'}`);
    } finally {
      setDlTestLoading(false);
    }
  }

  const mlColumns = [
    {
      title: '部署名称',
      dataIndex: 'name',
      key: 'name',
      render: (value, record) => (
        <a
          onClick={() => setSelectedMlDeployment(record)}
          style={{ fontWeight: selectedMlDeployment?.deployment_id === record.deployment_id ? 600 : 400 }}
        >
          {value}
        </a>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Badge
          status={status === 'active' ? 'success' : 'default'}
          text={status === 'active' ? '活跃' : '暂停'}
        />
      ),
    },
    { title: '调用次数', dataIndex: 'request_count', key: 'request_count' },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (value) => new Date(value).toLocaleString('zh-CN'),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Tooltip title={record.status === 'active' ? '暂停' : '恢复'}>
            <Button
              size="small"
              icon={record.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={() => void handleToggleMlStatus(record)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => setMlDeleteTarget(record)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const dlColumns = [
    {
      title: '部署名称',
      dataIndex: 'name',
      key: 'name',
      render: (value, record) => (
        <a
          onClick={() => setSelectedDlDeployment(record)}
          style={{ fontWeight: selectedDlDeployment?.id === record.id ? 600 : 400 }}
        >
          {value}
        </a>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Badge
          status={status === 'active' ? 'success' : 'default'}
          text={status === 'active' ? '活跃' : '暂停'}
        />
      ),
    },
    { title: '调用次数', dataIndex: 'request_count', key: 'request_count' },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (value) => new Date(value).toLocaleString('zh-CN'),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Tooltip title={record.status === 'active' ? '暂停' : '恢复'}>
            <Button
              size="small"
              icon={record.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={() => void handleToggleDlStatus(record)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => setDlDeleteTarget(record)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const dlPredictUrl = selectedDlDeployment
    ? `${api.defaults.baseURL}/dl/deployments/${selectedDlDeployment.id}/predict`
    : '';

  return (
    <div>
      <Typography.Title level={2} style={{ marginBottom: 16 }}>模型部署</Typography.Title>
      <Tabs
        defaultActiveKey="ml"
        items={[
          {
            key: 'ml',
            label: '机器学习部署',
            children: (
              <DeploymentTabContent
                title="机器学习部署"
                stats={mlStats}
                deployments={mlDeployments}
                loading={mlLoading}
                columns={mlColumns}
                onRefresh={fetchMlDeployments}
                selected={selectedMlDeployment}
                predictUrl={selectedMlDeployment?.endpoints?.predict || ''}
                resultUrl={selectedMlDeployment?.endpoints?.result || ''}
                supportsResultUrl
                testInput={mlTestInput}
                setTestInput={setMlTestInput}
                onTest={handleTestMlDeployment}
                testLoading={mlTestLoading}
                testResult={mlTestResult}
              />
            ),
          },
          {
            key: 'dl',
            label: '深度学习部署',
            children: (
              <DeploymentTabContent
                title="深度学习部署"
                stats={dlStats}
                deployments={dlDeployments}
                loading={dlLoading}
                columns={dlColumns}
                onRefresh={fetchDlDeployments}
                selected={selectedDlDeployment ? {
                  ...selectedDlDeployment,
                  deployment_id: selectedDlDeployment.id,
                  name: selectedDlDeployment.name,
                  status: selectedDlDeployment.status,
                } : null}
                predictUrl={dlPredictUrl}
                resultUrl=""
                supportsResultUrl={false}
                testInput={dlTestInput}
                setTestInput={setDlTestInput}
                onTest={handleTestDlDeployment}
                testLoading={dlTestLoading}
                testResult={dlTestResult}
              />
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

      <Modal
        open={!!mlDeleteTarget}
        title="确认删除"
        onOk={() => mlDeleteTarget && handleDeleteMlDeployment(mlDeleteTarget.deployment_id)}
        onCancel={() => setMlDeleteTarget(null)}
        okText="删除"
        okButtonProps={{ danger: true }}
      >
        <p>确定要删除部署 <strong>{mlDeleteTarget?.name}</strong> 吗？此操作不可撤销。</p>
      </Modal>

      <Modal
        open={!!dlDeleteTarget}
        title="确认删除"
        onOk={() => dlDeleteTarget && handleDeleteDlDeployment(dlDeleteTarget.id)}
        onCancel={() => setDlDeleteTarget(null)}
        okText="删除"
        okButtonProps={{ danger: true }}
      >
        <p>确定要删除部署 <strong>{dlDeleteTarget?.name}</strong> 吗？此操作不可撤销。</p>
      </Modal>
    </div>
  );
}

function DeploymentTabContent({
  title,
  stats,
  deployments,
  loading,
  columns,
  onRefresh,
  selected,
  predictUrl,
  resultUrl,
  supportsResultUrl,
  testInput,
  setTestInput,
  onTest,
  testLoading,
  testResult,
}) {
  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Text type="secondary">部署创建入口保留在模型管理页，这里集中做查看、测试和启停。</Text>
        <Button onClick={() => void onRefresh()}>刷新</Button>
      </Space>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card><Statistic title="部署总数" value={stats.total} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="活跃部署" value={stats.active} valueStyle={{ color: '#3f8600' }} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="总调用次数" value={stats.calls} /></Card>
        </Col>
      </Row>

      <Card title={`${title}列表`} style={{ marginBottom: 24 }}>
        <Table
          columns={columns}
          dataSource={deployments}
          rowKey={(record) => record.deployment_id ?? record.id}
          loading={loading}
          pagination={{ pageSize: 10 }}
          size="small"
          locale={{ emptyText: <Empty description="还没有已创建的部署" /> }}
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
              <Text strong>预测 URL:</Text>
              <div style={{ marginTop: 8, marginBottom: 16 }}>
                <CopyableUrl url={predictUrl} label="提交预测请求" />
              </div>
              <Text strong>结果 URL:</Text>
              <div style={{ marginTop: 8 }}>
                {supportsResultUrl ? (
                  <CopyableUrl url={resultUrl} label="查询预测结果" />
                ) : (
                  <Text type="secondary">该部署返回即时结果，不提供轮询查询 URL。</Text>
                )}
              </div>
            </Col>
            <Col xs={24} md={12}>
              <Text strong>在线测试</Text>
              <div style={{ marginTop: 8 }}>
                <Input.TextArea
                  value={testInput}
                  onChange={(event) => setTestInput(event.target.value)}
                  rows={5}
                  style={{ fontFamily: 'monospace', marginBottom: 8 }}
                />
                <Button
                  type="primary"
                  onClick={() => void onTest()}
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
                <pre style={{ background: '#f6f8fa', padding: 12, borderRadius: 4, marginTop: 16, fontSize: 12, overflowX: 'auto' }}>
                  {JSON.stringify(testResult, null, 2)}
                </pre>
              )}
            </Col>
          </Row>
        </Card>
      )}
    </div>
  );
}
