import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Button,
  Card,
  Descriptions,
  message,
  Modal,
  Progress,
  Space,
  Table,
  Tag,
  Typography,
  Upload,
} from 'antd';
import { DeleteOutlined, EyeOutlined, FileTextOutlined, RocketOutlined } from '@ant-design/icons';
import { InboxOutlined } from '@ant-design/icons';
import { dataApi } from '../services/api';

const { Dragger } = Upload;
const { Text, Title } = Typography;

function formatDataset(dataset) {
  return {
    ...dataset,
    key: dataset.id,
    sizeLabel: `${Math.round(dataset.file_size / 1024)} KB`,
    typeLabel: dataset.name.split('.').pop()?.toUpperCase() ?? 'FILE',
    createdLabel: new Date(dataset.created_at).toLocaleString('zh-CN'),
  };
}

function buildPreviewColumns(rows) {
  if (!rows.length) {
    return [];
  }

  return Object.keys(rows[0]).map((key) => ({
    title: key,
    dataIndex: key,
    key,
    ellipsis: true,
  }));
}

const DataManagement = () => {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);
  const [previewDataset, setPreviewDataset] = useState(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId]
  );

  useEffect(() => {
    void fetchDatasets();
  }, []);

  async function fetchDatasets() {
    try {
      const response = await dataApi.listDatasets();
      const items = (response.items ?? []).map(formatDataset);
      setDatasets(items);
      if (!selectedDatasetId && items.length > 0) {
        setSelectedDatasetId(items[0].id);
      }
    } catch (error) {
      console.error('获取数据集列表失败:', error);
      message.error('获取数据集列表失败');
    }
  }

  async function handleUpload(file) {
    setUploading(true);
    setUploadProgress(0);

    try {
      const uploaded = await dataApi.uploadDataset(file, (progressEvent) => {
        if (!progressEvent.total) {
          return;
        }
        setUploadProgress(Math.round((progressEvent.loaded * 100) / progressEvent.total));
      });

      const formatted = formatDataset(uploaded);
      setDatasets((current) => {
        const rest = current.filter((item) => item.id !== formatted.id);
        return [formatted, ...rest];
      });
      setSelectedDatasetId(formatted.id);
      message.success(`${file.name} 上传成功`);
    } catch (error) {
      console.error('上传文件失败:', error);
      message.error('上传文件失败');
    } finally {
      setUploading(false);
    }

    return false;
  }

  async function handlePreview(dataset) {
    try {
      const response = await dataApi.previewDataset(dataset.id);
      setPreviewDataset(response);
      setPreviewOpen(true);
      setSelectedDatasetId(dataset.id);
    } catch (error) {
      console.error('获取数据预览失败:', error);
      message.error('获取数据预览失败');
    }
  }

  async function handleDelete(datasetId) {
    try {
      await dataApi.deleteDataset(datasetId);
      setDatasets((current) => current.filter((item) => item.id !== datasetId));
      if (selectedDatasetId === datasetId) {
        setSelectedDatasetId(null);
      }
      message.success('数据集删除成功');
    } catch (error) {
      console.error('删除数据集失败:', error);
      message.error('删除数据集失败');
    }
  }

  function goToTraining() {
    if (!selectedDataset) {
      message.warning('请先上传或选择一个数据集');
      return;
    }

    navigate('/training/config', {
      state: {
        preferredDatasetId: selectedDataset.id,
      },
    });
  }

  const previewColumns = buildPreviewColumns(previewDataset?.rows ?? []);

  return (
    <div>
      <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>
          数据管理
        </Title>
        <Button
          type="primary"
          icon={<RocketOutlined />}
          data-testid="configure-training-button"
          disabled={!selectedDataset}
          onClick={goToTraining}
        >
          配置训练
        </Button>
      </Space>

      <Card style={{ marginBottom: 24 }}>
        <Dragger
          name="file"
          multiple={false}
          accept=".csv,.parquet,.xlsx"
          beforeUpload={handleUpload}
          showUploadList={false}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ fontSize: 48, color: '#1677ff' }} />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">支持 CSV、Parquet、Excel，单文件最大 200MB</p>
          {uploading ? <Progress percent={uploadProgress} status="active" /> : null}
        </Dragger>
      </Card>

      <Card title="数据集列表">
        <Table
          rowKey="id"
          dataSource={datasets}
          pagination={{ pageSize: 10 }}
          rowSelection={{
            type: 'radio',
            selectedRowKeys: selectedDatasetId ? [selectedDatasetId] : [],
            onChange: (keys) => setSelectedDatasetId(keys[0] ?? null),
          }}
          columns={[
            {
              title: '名称',
              dataIndex: 'name',
              key: 'name',
              render: (name) => (
                <Space>
                  <FileTextOutlined style={{ color: '#1677ff' }} />
                  <Text strong>{name}</Text>
                </Space>
              ),
            },
            {
              title: '类型',
              dataIndex: 'typeLabel',
              key: 'typeLabel',
              render: (type) => <Tag color="blue">{type}</Tag>,
            },
            {
              title: '大小',
              dataIndex: 'sizeLabel',
              key: 'sizeLabel',
            },
            {
              title: '行数',
              dataIndex: 'row_count',
              key: 'row_count',
            },
            {
              title: '列数',
              dataIndex: 'column_count',
              key: 'column_count',
            },
            {
              title: '创建时间',
              dataIndex: 'createdLabel',
              key: 'createdLabel',
            },
            {
              title: '操作',
              key: 'action',
              render: (_, record) => (
                <Space>
                  <Button type="text" icon={<EyeOutlined />} onClick={() => void handlePreview(record)}>
                    预览
                  </Button>
                  <Button type="text" icon={<RocketOutlined />} onClick={() => {
                    setSelectedDatasetId(record.id);
                    navigate('/training/config', { state: { preferredDatasetId: record.id } });
                  }}>
                    训练
                  </Button>
                  <Button danger type="text" icon={<DeleteOutlined />} onClick={() => void handleDelete(record.id)}>
                    删除
                  </Button>
                </Space>
              ),
            },
          ]}
        />
      </Card>

      <Modal
        title={previewDataset ? `数据预览 - ${previewDataset.name}` : '数据预览'}
        open={previewOpen}
        onCancel={() => setPreviewOpen(false)}
        footer={null}
        width={960}
      >
        {previewDataset ? (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="文件名">{previewDataset.name}</Descriptions.Item>
              <Descriptions.Item label="文件大小">{Math.round(previewDataset.file_size / 1024)} KB</Descriptions.Item>
              <Descriptions.Item label="行数">{previewDataset.row_count}</Descriptions.Item>
              <Descriptions.Item label="列数">{previewDataset.column_count}</Descriptions.Item>
            </Descriptions>
            <Table
              size="small"
              rowKey={(_, index) => `${previewDataset.id}-${index}`}
              dataSource={previewDataset.rows.slice(0, 5)}
              columns={previewColumns}
              pagination={false}
              scroll={{ x: 'max-content' }}
            />
          </Space>
        ) : null}
      </Modal>
    </div>
  );
};

export default DataManagement;
