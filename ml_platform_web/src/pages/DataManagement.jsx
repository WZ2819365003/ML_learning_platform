import React, { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Table,
  Tag,
  Modal,
  message,
  Typography,
  Space,
  Progress,
  Descriptions,
  Statistic,
  Row,
  Col,
  Upload,
  Popconfirm,
} from 'antd'
import {
  UploadOutlined,
  DeleteOutlined,
  EyeOutlined,
  FileTextOutlined,
  InboxOutlined,
} from '@ant-design/icons'
import { dataApi } from '../services/api'

const { Title, Text } = Typography
const { Dragger } = Upload

/** 格式化字节数为可读字符串 */
function formatBytes(bytes) {
  if (bytes == null) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

/** 格式化 ISO 时间为本地可读时间 */
function formatDate(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('zh-CN', { hour12: false })
}

const DataManagement = () => {
  const [datasets, setDatasets]           = useState([])
  const [loading, setLoading]             = useState(false)
  const [uploading, setUploading]         = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [previewModal, setPreviewModal]   = useState(false)
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [previewData, setPreviewData]     = useState({ rows: [], statistics: {} })
  const [previewLoading, setPreviewLoading] = useState(false)

  useEffect(() => { fetchDatasets() }, [])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const res = await dataApi.listDatasets({ page: 1, page_size: 100 })
      // axios interceptor already unwraps response.data → res is {items, total, …}
      setDatasets((res.items || []).map((ds, i) => ({ ...ds, key: String(i + 1) })))
    } catch (err) {
      console.error('获取数据集列表失败:', err)
      message.error('获取数据集列表失败')
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = (file) => {
    setUploading(true)
    setUploadProgress(0)
    dataApi.uploadDataset(file, (e) => {
      if (e.total) setUploadProgress(Math.round((e.loaded * 100) / e.total))
    })
      .then(() => {
        message.success(`${file.name} 上传成功`)
        fetchDatasets()
      })
      .catch((err) => {
        console.error('上传文件失败:', err)
        message.error('上传文件失败')
      })
      .finally(() => {
        setUploading(false)
        setUploadProgress(0)
      })
    return false // 阻止 antd Upload 默认行为
  }

  const handlePreview = async (dataset) => {
    setSelectedDataset(dataset)
    setPreviewData({ rows: [], statistics: {} })
    setPreviewModal(true)
    setPreviewLoading(true)
    try {
      // axios interceptor unwraps → res is {id, name, rows, statistics, columns_info, …}
      const res = await dataApi.previewDataset(dataset.id)
      setPreviewData({ rows: res.rows || [], statistics: res.statistics || {} })
    } catch (err) {
      console.error('获取数据预览失败:', err)
      message.error('获取数据预览失败')
    } finally {
      setPreviewLoading(false)
    }
  }

  const handleDelete = async (id) => {
    try {
      await dataApi.deleteDataset(id)
      message.success('数据集删除成功')
      fetchDatasets()
    } catch (err) {
      console.error('删除数据集失败:', err)
      message.error('删除数据集失败')
    }
  }

  /* ── 列表列定义 ── */
  const columns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (name) => (
        <Space>
          <FileTextOutlined style={{ color: '#1890ff' }} />
          <Text strong>{name}</Text>
        </Space>
      ),
    },
    {
      title: '大小',
      dataIndex: 'file_size',
      key: 'file_size',
      render: (v) => formatBytes(v),
    },
    {
      title: '行数',
      dataIndex: 'row_count',
      key: 'row_count',
      render: (v) => v?.toLocaleString() ?? '—',
    },
    {
      title: '列数',
      dataIndex: 'column_count',
      key: 'column_count',
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (v) => formatDate(v),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="middle">
          <Button
            type="text"
            icon={<EyeOutlined />}
            onClick={() => handlePreview(record)}
          >
            预览
          </Button>
          <Popconfirm
            title="确认删除该数据集？"
            onConfirm={() => handleDelete(record.id)}
            okText="删除"
            cancelText="取消"
          >
            <Button type="text" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  /* ── 预览表格列（动态从第一行生成） ── */
  const previewColumns =
    previewData.rows.length > 0
      ? Object.keys(previewData.rows[0]).map((key) => ({
          title: key,
          dataIndex: key,
          key,
          ellipsis: true,
          render: (v) => (v == null ? <Text type="secondary">null</Text> : String(v)),
        }))
      : []

  /* ── 统计卡片（取缺失值最多的列展示） ── */
  const statsEntries = Object.entries(previewData.statistics)
  const totalMissing = statsEntries.reduce((sum, [, s]) => sum + (s.missing_count ?? 0), 0)
  const numericCols  = statsEntries.filter(([, s]) => s.mean != null)
  const avgMean      = numericCols.length
    ? (numericCols.reduce((s, [, v]) => s + v.mean, 0) / numericCols.length).toFixed(4)
    : '—'
  const avgStd       = numericCols.length
    ? (numericCols.reduce((s, [, v]) => s + (v.std ?? 0), 0) / numericCols.length).toFixed(4)
    : '—'

  return (
    <div>
      <Title level={2}>数据管理</Title>

      {/* 上传区域 */}
      <Card className="mb-6" hoverable>
        <Dragger
          name="file"
          multiple={false}
          accept=".csv,.parquet,.xlsx"
          beforeUpload={handleUpload}
          showUploadList={false}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ fontSize: 48, color: '#1890ff' }} />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">支持 CSV、Parquet、Excel 文件，单个文件最大 200MB</p>
          {uploading && (
            <div className="mt-4">
              <Progress percent={uploadProgress} status="active" />
            </div>
          )}
        </Dragger>
      </Card>

      {/* 数据集列表 */}
      <Card title="数据集列表" hoverable>
        <Table
          dataSource={datasets}
          columns={columns}
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* 预览弹窗 */}
      <Modal
        title={`数据预览 — ${selectedDataset?.name}`}
        open={previewModal}
        onCancel={() => setPreviewModal(false)}
        width={900}
        footer={[
          <Button key="close" onClick={() => setPreviewModal(false)}>
            关闭
          </Button>,
        ]}
      >
        {selectedDataset && (
          <div>
            <Descriptions bordered size="small" className="mb-4" column={3}>
              <Descriptions.Item label="文件名">{selectedDataset.name}</Descriptions.Item>
              <Descriptions.Item label="大小">{formatBytes(selectedDataset.file_size)}</Descriptions.Item>
              <Descriptions.Item label="行数">{selectedDataset.row_count?.toLocaleString()}</Descriptions.Item>
              <Descriptions.Item label="列数">{selectedDataset.column_count}</Descriptions.Item>
              <Descriptions.Item label="创建时间" span={2}>{formatDate(selectedDataset.created_at)}</Descriptions.Item>
            </Descriptions>

            <Title level={5}>数据预览（前 100 行）</Title>
            <Table
              dataSource={(previewData.rows || []).map((r, i) => ({ ...r, _key: i }))}
              rowKey="_key"
              columns={previewColumns}
              pagination={{ pageSize: 5, size: 'small' }}
              size="small"
              loading={previewLoading}
              scroll={{ x: 'max-content' }}
            />

            {statsEntries.length > 0 && (
              <>
                <Title level={5} className="mt-4">基本统计（数值列均值）</Title>
                <Row gutter={16}>
                  <Col span={8}>
                    <Card size="small">
                      <Statistic title="数值列均值（平均）" value={avgMean} />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card size="small">
                      <Statistic title="数值列标准差（平均）" value={avgStd} />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card size="small">
                      <Statistic title="总缺失值数" value={totalMissing} />
                    </Card>
                  </Col>
                </Row>
              </>
            )}
          </div>
        )}
      </Modal>
    </div>
  )
}

export default DataManagement
