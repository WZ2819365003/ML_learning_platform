# DL Monitor And Deploy Recovery Design

**Context**

当前深度学习训练监控和模型部署经历过一次统一化改造后，出现了三个方向性回退：

1. 模型部署详情不再保持 `v2.0.0` 那种“点击列表项后，在当前页面展开详细组件”的交互。
2. 深度学习训练监控页的图表布局、Epoch 表格和日志颗粒度不符合使用预期。
3. 训练过程中的 Epoch/日志状态缺少数据库持久化，关闭页面后无法恢复。

**Goals**

1. 恢复部署页的旧版详细组件体验，同时保留 ML / DL / 通用模型三 tab 结构。
2. 让 DL 监控页支持每个 epoch 持久化、重进恢复、分页浏览和更稳定的布局。
3. 在不引入重型迁移框架的前提下，完成最小可维护的数据持久化方案。
4. 防止同一份数据集被重复上传后产生重复数据记录。
5. 让模型管理页能查看和编辑标签、备注，并为表格统一分页。

**Non-Goals**

1. 不重构整套模型管理/部署后端契约。
2. 不引入新的数据库迁移工具链。
3. 不改变训练核心算法、验证逻辑或测试集拆分策略。

**Design**

## 1. DL 训练监控持久化

新增两张表：

- `dl_training_epochs`
  - `id`
  - `task_id`
  - `epoch`
  - `total_epochs`
  - `train_loss`
  - `val_loss`
  - `val_acc`
  - `val_f1_macro`
  - `val_rmse`
  - `val_mae`
  - `val_r2`
  - `lr`
  - `created_at`
- `dl_training_logs`
  - `id`
  - `task_id`
  - `level`
  - `message`
  - `extra`
  - `created_at`

DL 训练线程在每个 epoch 回调时：

- 持久化当前 epoch 记录到 `dl_training_epochs`
- 写一条结构化日志到 `dl_training_logs`
- 继续保留现有文件日志和 WebSocket 推送

监控页重新进入时：

- 先调用 REST 读取已落库的 epoch 历史和日志
- 再接 WebSocket 增量更新

## 2. DL 监控页交互

- 图表区改为 `Row/Col` 响应式布局，避免两个图被压成窄条
- 图表高度固定为 320px
- Epoch 表格增加分页，默认 10 条一页
- Epoch 表格增加固定高度滚动区域
- 日志改成每个 epoch 一条，不再每 5 个 epoch 才输出
- 日志内容必须直接展示关键指标，例如 `val_loss`、`val_acc`、`val_f1_macro`

## 3. 部署页详情回退

- 保持三 tab：机器学习部署 / 深度学习部署 / 通用模型部署
- ML tab 的详细组件结构回到 `v2.0.0` 样式
- DL tab 复用同一交互模式：
  - 上方统计卡片
  - 中间部署表格
  - 点击部署项后，下方出现接口详情组件

## 4. 模型管理页增强

- ML / DL 两个 tab 的列表都显示标签
- 增加备注列或备注预览
- 提供编辑标签/备注的入口
- 所有表格显式设置分页

## 5. 数据集去重

上传文件时计算内容 hash，并与已存在数据集文件逐一比较：

- 若文件内容一致，则直接复用已有 `Dataset` 记录
- 不新建新的数据集记录

该方案避免修改现有 `datasets` 表结构，也不依赖数据库迁移。

**Testing**

1. `pytest`
   - DL epoch/log 持久化
   - DL epoch/log 分页读取
   - 数据集去重上传
2. `playwright`
   - 部署详情组件回归
   - DL 监控页刷新后恢复历史 epoch
   - DL 监控页表格分页
   - 模型管理标签/备注可见
