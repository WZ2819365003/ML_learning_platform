# Step 2：可视化API + 前端开发（第2周）

## 目标

完成后端评估可视化 API（SHAP、混淆矩阵、ROC、学习曲线等），模型版本管理 API，然后搭建完整的 React 前端，对接所有后端功能。

---

## 1. 后端：评估可视化 API

### 1.1 新增 API 路由 `app/api/routes/visualization.py`

- **GET /api/viz/{task_id}/confusion_matrix** — 混淆矩阵数据
  - 返回 `{labels: [...], matrix: [[...], ...]}`，前端直接渲染热力图

- **GET /api/viz/{task_id}/roc_curve** — ROC 曲线数据
  - 返回 `{fpr: [...], tpr: [...], auc: float}`

- **GET /api/viz/{task_id}/feature_importance** — 特征重要性
  - 返回 `{features: [...], importance: [...]}`
  - 基于模型内置 feature_importances_ 或 coef_

- **GET /api/viz/{task_id}/learning_curve** — 学习曲线数据
  - 直接从 metrics JSON 读取已有的 per-fold 数据

- **GET /api/viz/{task_id}/shap_summary** — SHAP 可解释性数据
  - 返回 `{features: [...], shap_values: [[...]], feature_names: [...]}`
  - 计算量大，限制样本数

### 1.2 新增服务 `app/services/viz_service.py`

核心逻辑：
- 加载已保存模型（joblib）
- 加载原始数据集
- 用模型+数据计算各类可视化数据
- SHAP 限制 100 个样本防止卡死

### 1.3 数据结构

混淆矩阵响应：
```json
{
  "labels": ["class_0", "class_1"],
  "matrix": [[45, 5], [8, 42]],
  "normalize": false
}
```

ROC 曲线响应：
```json
{
  "fpr": [0, 0.1, 0.2, ...],
  "tpr": [0, 0.5, 0.8, ...],
  "auc": 0.923,
  "thresholds": [1.0, 0.9, ...]
}
```

---

## 2. 后端：模型版本管理

### 2.1 API `app/api/routes/model_mgmt.py`

- **GET /api/models/list** — 已保存模型列表
  - 从 training_tasks 表中筛选 status=SUCCESS 的任务
  - 返回模型信息 + 训练指标

- **GET /api/models/{task_id}/detail** — 模型详情
  - 模型类型、超参数、训练指标、文件大小、创建时间

- **DELETE /api/models/{task_id}** — 删除模型文件

- **GET /api/models/compare** — 多模型对比
  - Query params: task_ids (逗号分隔)
  - 返回多个模型的指标对比数据

---

## 3. 前端：React + Vite + Ant Design + ECharts

### 3.1 技术栈

```
frontend/
├── package.json
├── vite.config.js
├── index.html
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── api/                    # API 调用封装
│   │   ├── index.js            # axios 实例
│   │   ├── data.js             # 数据管理 API
│   │   ├── training.js         # 训练 API
│   │   ├── logs.js             # 日志 API
│   │   └── visualization.js    # 可视化 API
│   ├── pages/
│   │   ├── Dashboard.jsx       # 总览面板
│   │   ├── DataManagement.jsx  # 数据上传/预览
│   │   ├── TrainingConfig.jsx  # 模型配置+启动
│   │   ├── TrainingMonitor.jsx # 实时监控
│   │   ├── Results.jsx         # 结果可视化
│   │   └── ModelManagement.jsx # 模型管理
│   ├── components/
│   │   ├── Layout.jsx          # 侧边栏布局
│   │   ├── DataPreview.jsx     # 数据表格预览
│   │   ├── ModelSelector.jsx   # 模型选择器
│   │   ├── HyperParamForm.jsx  # 超参数表单
│   │   ├── MetricsChart.jsx    # 实时指标图
│   │   ├── ConfusionMatrix.jsx # 混淆矩阵
│   │   ├── ROCCurve.jsx        # ROC 曲线
│   │   ├── FeatureImportance.jsx # 特征重要性
│   │   ├── ShapSummary.jsx     # SHAP 图
│   │   └── LogViewer.jsx       # 日志查看器
│   └── hooks/
│       ├── useWebSocket.js     # WebSocket hook
│       └── useTraining.js      # 训练状态管理
└── public/
```

### 3.2 页面设计

**Dashboard** — 系统总览
- 数据集数量、训练任务数量、成功/失败统计
- 最近训练任务列表

**DataManagement** — 数据管理
- 拖拽上传区域
- 数据集列表（卡片视图）
- 点击查看数据预览表格 + 列统计信息

**TrainingConfig** — 训练配置
- Step 1: 选择数据集
- Step 2: 选择目标列
- Step 3: 选择模型类型（卡片选择器）
- Step 4: 配置超参数（动态表单）
- Step 5: 选择评估指标
- 提交按钮 → 启动训练

**TrainingMonitor** — 训练监控
- 训练任务列表 + 状态标签
- 点击进入详情：
  - 实时 loss/accuracy 折线图 (WebSocket)
  - 进度条
  - 实时日志流
  - 停止按钮

**Results** — 结果可视化
- Tab 1: 混淆矩阵（热力图）
- Tab 2: ROC 曲线
- Tab 3: 特征重要性（柱状图）
- Tab 4: SHAP Summary（蜂群图）
- Tab 5: 学习曲线（折/步曲线）

**ModelManagement** — 模型管理
- 模型列表（表格）
- 多模型对比（雷达图/柱状图）
- 删除模型

---

## 4. 每日计划

| 日期 | 任务 | 验收标准 |
|------|------|---------|
| Day 8 | 评估可视化 API (confusion_matrix, roc, feature_importance) | Swagger 测试返回正确数据 |
| Day 9 | SHAP API + 模型管理 API | SHAP 数据可查询，模型可列表/对比/删除 |
| Day 10 | 前端项目搭建 + Layout + Dashboard | `npm run dev` 启动，看到侧边栏布局 |
| Day 11 | 数据管理页 + 训练配置页 | 能上传文件、选模型、配参数、启动训练 |
| Day 12 | 训练监控页 (WebSocket 实时图表) | 看到实时更新的 loss 曲线 |
| Day 13 | 结果可视化页 (混淆矩阵/ROC/SHAP) | 5 种可视化图表正确渲染 |
| Day 14 | 模型管理页 + 整体联调 + 样式优化 | 完整流程可走通 |

---

## 5. 注意事项

1. **SHAP 计算限制**: 最多用 100 个样本，防止计算超时
2. **前端用 Ant Design**: 快速搭建 UI，不花时间在样式上
3. **ECharts 图表**: 混淆矩阵用 heatmap，ROC 用 line，特征重要性用 bar，SHAP 用 scatter
4. **WebSocket 断线重连**: 前端 hook 需要处理重连逻辑
5. **API 调用统一封装**: axios 实例统一设置 baseURL 和错误处理
