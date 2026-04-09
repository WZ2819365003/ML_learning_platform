import React, { useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { Button, Card, Descriptions, Space, Table, Tag, Typography, message } from 'antd';
import { ReloadOutlined, StopOutlined } from '@ant-design/icons';
import { dataApi, logsApi, trainingApi } from '../services/api';

const { Paragraph, Text, Title } = Typography;

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

function formatStatus(status) {
  return (status ?? 'UNKNOWN').toUpperCase();
}

const statusColors = {
  PENDING: 'default',
  RUNNING: 'processing',
  SUCCESS: 'success',
  FAILED: 'error',
};

const TrainingMonitor = () => {
  const query = useQuery();
  const focusTaskId = query.get('taskId');
  const [tasks, setTasks] = useState([]);
  const [datasetsById, setDatasetsById] = useState({});
  const [selectedTaskId, setSelectedTaskId] = useState(focusTaskId);
  const [logText, setLogText] = useState('');

  const selectedTask = useMemo(
    () => tasks.find((task) => task.id === selectedTaskId) ?? null,
    [selectedTaskId, tasks]
  );

  useEffect(() => {
    void refreshAll();
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void refreshAll();
    }, 2000);

    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (focusTaskId) {
      setSelectedTaskId(focusTaskId);
    } else if (!selectedTaskId && tasks.length > 0) {
      setSelectedTaskId(tasks[0].id);
    }
  }, [focusTaskId, selectedTaskId, tasks]);

  useEffect(() => {
    if (!selectedTaskId) {
      setLogText('');
      return;
    }

    void loadLogs(selectedTaskId);
  }, [selectedTaskId]);

  async function refreshAll() {
    try {
      const [taskResponse, datasetResponse] = await Promise.all([
        trainingApi.listTasks(),
        dataApi.listDatasets(),
      ]);

      setTasks(taskResponse.items ?? []);
      setDatasetsById(
        Object.fromEntries((datasetResponse.items ?? []).map((dataset) => [dataset.id, dataset.name]))
      );
    } catch (error) {
      console.error('刷新训练任务失败:', error);
      message.error('刷新训练任务失败');
    }
  }

  async function loadLogs(taskId) {
    try {
      const logResponse = await logsApi.getLogs(taskId, { page_size: 20 });
      const text = (logResponse.entries ?? [])
        .map((entry) => `${entry.created_at} | ${entry.level} | ${entry.message}`)
        .join('\n');
      setLogText(text);
    } catch (error) {
      setLogText('当前任务还没有日志输出。');
    }
  }

  async function stopTask(taskId) {
    try {
      await trainingApi.stopTraining(taskId);
      message.success('训练任务已停止');
      await refreshAll();
    } catch (error) {
      console.error('停止训练任务失败:', error);
      message.error('停止训练任务失败');
    }
  }

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>
          训练监控
        </Title>
        <Button icon={<ReloadOutlined />} onClick={() => void refreshAll()}>
          刷新
        </Button>
      </Space>

      <Card style={{ marginBottom: 24 }}>
        <Paragraph type="secondary" style={{ marginBottom: 0 }}>
          页面会自动轮询任务状态。成功后会展示最终指标，失败时会展示错误消息。
        </Paragraph>
      </Card>

      <Card title="训练任务列表" style={{ marginBottom: 24 }}>
        <Table
          rowKey="id"
          dataSource={tasks}
          pagination={{ pageSize: 10 }}
          onRow={(record) => ({
            onClick: () => setSelectedTaskId(record.id),
          })}
          columns={[
            {
              title: '任务 ID',
              dataIndex: 'id',
              key: 'id',
              render: (value) => <Text code>{value.slice(0, 8)}</Text>,
            },
            {
              title: '数据集',
              dataIndex: 'dataset_id',
              key: 'dataset_id',
              render: (value) => datasetsById[value] ?? value,
            },
            {
              title: '模型',
              dataIndex: 'model_type',
              key: 'model_type',
            },
            {
              title: '状态',
              dataIndex: 'status',
              key: 'status',
              render: (value) => <Tag color={statusColors[formatStatus(value)]}>{formatStatus(value)}</Tag>,
            },
            {
              title: '进度',
              dataIndex: 'progress',
              key: 'progress',
              render: (value) => `${Math.round(value ?? 0)}%`,
            },
            {
              title: '创建时间',
              dataIndex: 'created_at',
              key: 'created_at',
              render: (value) => new Date(value).toLocaleString('zh-CN'),
            },
            {
              title: '操作',
              key: 'actions',
              render: (_, record) =>
                formatStatus(record.status) === 'RUNNING' ? (
                  <Button
                    danger
                    size="small"
                    icon={<StopOutlined />}
                    onClick={(event) => {
                      event.stopPropagation();
                      void stopTask(record.id);
                    }}
                  >
                    停止
                  </Button>
                ) : null,
            },
          ]}
        />
      </Card>

      <Card title="任务详情">
        {selectedTask ? (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <Descriptions bordered column={2} size="small">
              <Descriptions.Item label="数据集">
                {datasetsById[selectedTask.dataset_id] ?? selectedTask.dataset_id}
              </Descriptions.Item>
              <Descriptions.Item label="模型">{selectedTask.model_type}</Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={statusColors[formatStatus(selectedTask.status)]}>{formatStatus(selectedTask.status)}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="进度">{Math.round(selectedTask.progress ?? 0)}%</Descriptions.Item>
              <Descriptions.Item label="开始时间">
                {selectedTask.started_at ? new Date(selectedTask.started_at).toLocaleString('zh-CN') : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="结束时间">
                {selectedTask.finished_at ? new Date(selectedTask.finished_at).toLocaleString('zh-CN') : '-'}
              </Descriptions.Item>
            </Descriptions>

            {selectedTask.result_metrics ? (
              <Descriptions bordered column={2} size="small" title="训练指标">
                {Object.entries(selectedTask.result_metrics).map(([key, value]) => (
                  <Descriptions.Item key={key} label={key}>
                    {String(value)}
                  </Descriptions.Item>
                ))}
              </Descriptions>
            ) : null}

            {selectedTask.error_message ? (
              <Card size="small" title="错误信息">
                <Text type="danger">{selectedTask.error_message}</Text>
              </Card>
            ) : null}

            <Card size="small" title="最近日志">
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{logText || '当前任务还没有日志输出。'}</pre>
            </Card>
          </Space>
        ) : (
          <Text type="secondary">暂无训练任务。</Text>
        )}
      </Card>
    </div>
  );
};

export default TrainingMonitor;
