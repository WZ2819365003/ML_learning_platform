import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Button, 
  Table, 
  Tag, 
  Modal, 
  Upload, 
  message, 
  Typography, 
  Space, 
  Progress,
  Descriptions,
  Statistic,
  Row,
  Col
} from 'antd'
import { 
  UploadOutlined, 
  DeleteOutlined, 
  EyeOutlined, 
  FileTextOutlined, 
  DatabaseOutlined 
} from '@ant-design/icons'
import { InboxOutlined } from '@ant-design/icons'
import api from '../services/api'

const { Title, Text } = Typography
const { Dragger } = Upload

const DataManagement = () => {
  const [datasets, setDatasets] = useState([])
  const [previewModal, setPreviewModal] = useState(false)
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [previewData, setPreviewData] = useState([])

  // 获取数据集列表
  useEffect(() => {
    fetchDatasets()
  }, [])

  const fetchDatasets = async () => {
    try {
      const response = await api.get('/data/datasets')
      setDatasets(response.map((dataset, index) => ({
        ...dataset,
        key: String(index + 1)
      })))
    } catch (error) {
      console.error('获取数据集列表失败:', error)
      message.error('获取数据集列表失败')
    }
  }

  const handleUpload = async (file) => {
    setUploading(true)
    setUploadProgress(0)
    
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const response = await api.post('/data/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          setUploadProgress(percentCompleted)
        }
      })
      
      setUploading(false)
      message.success(`${file.name} 上传成功`)
      
      // 重新获取数据集列表
      fetchDatasets()
    } catch (error) {
      console.error('上传文件失败:', error)
      message.error('上传文件失败')
      setUploading(false)
    }

    return false
  }

  const handlePreview = async (dataset) => {
    setSelectedDataset(dataset)
    
    try {
      const response = await api.get(`/data/preview/${dataset.id}`)
      setPreviewData(response.data)
      setPreviewModal(true)
    } catch (error) {
      console.error('获取数据预览失败:', error)
      message.error('获取数据预览失败')
    }
  }

  const handleDelete = async (id) => {
    try {
      await api.delete(`/data/delete/${id}`)
      message.success('数据集删除成功')
      // 重新获取数据集列表
      fetchDatasets()
    } catch (error) {
      console.error('删除数据集失败:', error)
      message.error('删除数据集失败')
    }
  }

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: '.csv,.parquet',
    beforeUpload: handleUpload,
    showUploadList: false
  }

  return (
    <div>
      <Title level={2}>数据管理</Title>
      
      {/* 上传区域 */}
      <Card className="mb-6" hoverable>
        <Dragger {...uploadProps}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ fontSize: 48, color: '#1890ff' }} />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">
            支持 CSV 和 Parquet 文件，单个文件最大 200MB
          </p>
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
          columns={[
            {
              title: '名称',
              dataIndex: 'name',
              key: 'name',
              render: (name) => (
                <Space>
                  <FileTextOutlined style={{ color: '#1890ff' }} />
                  <Text strong>{name}</Text>
                </Space>
              )
            },
            {
              title: '类型',
              dataIndex: 'type',
              key: 'type',
              render: (type) => <Tag color="blue">{type}</Tag>
            },
            {
              title: '大小',
              dataIndex: 'size',
              key: 'size'
            },
            {
              title: '行数',
              dataIndex: 'rows',
              key: 'rows'
            },
            {
              title: '列数',
              dataIndex: 'columns',
              key: 'columns'
            },
            {
              title: '创建时间',
              dataIndex: 'created',
              key: 'created'
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
                  <Button 
                    type="text" 
                    danger 
                    icon={<DeleteOutlined />} 
                    onClick={() => handleDelete(record.id)}
                  >
                    删除
                  </Button>
                </Space>
              )
            }
          ]}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* 预览模态框 */}
      <Modal
        title={`数据预览 - ${selectedDataset?.name}`}
        open={previewModal}
        onCancel={() => setPreviewModal(false)}
        width={800}
        footer={[
          <Button key="close" onClick={() => setPreviewModal(false)}>
            关闭
          </Button>
        ]}
      >
        {selectedDataset && (
          <div>
            <Descriptions bordered className="mb-4">
              <Descriptions.Item label="文件名称">{selectedDataset.name}</Descriptions.Item>
              <Descriptions.Item label="文件类型">{selectedDataset.type}</Descriptions.Item>
              <Descriptions.Item label="文件大小">{selectedDataset.size}</Descriptions.Item>
              <Descriptions.Item label="行数">{selectedDataset.rows}</Descriptions.Item>
              <Descriptions.Item label="列数">{selectedDataset.columns}</Descriptions.Item>
              <Descriptions.Item label="创建时间">{selectedDataset.created}</Descriptions.Item>
            </Descriptions>
            
            <Title level={4}>数据预览（前5行）</Title>
            <Table 
              dataSource={previewData} 
              columns={Object.keys(previewData[0]).map(key => ({
                title: key,
                dataIndex: key,
                key: key
              }))} 
              pagination={false}
              size="small"
            />
            
            <div className="mt-4">
              <Title level={4}>基本统计信息</Title>
              <Row gutter={16}>
                <Col span={8}>
                  <Card>
                    <Statistic title="均值" value={0.5} />
                  </Card>
                </Col>
                <Col span={8}>
                  <Card>
                    <Statistic title="标准差" value={0.3} />
                  </Card>
                </Col>
                <Col span={8}>
                  <Card>
                    <Statistic title="缺失值" value="0%" />
                  </Card>
                </Col>
              </Row>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default DataManagement