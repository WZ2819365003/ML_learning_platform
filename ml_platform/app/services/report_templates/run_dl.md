# {{run.model}} · 分报告

## 结论

{{headline.sentence}}{{gap.sentence}}{{train.verdict_sentence}}

## 训练过程

{{train.plan_note}}实际训练 {{train.actual_epochs}} 轮，{{train.stop_reason}}。

第 {{train.best_epoch}} 轮取得最优验证 {{train.metric}} {{train.best_value}}，此后 {{train.patience_used}} 轮未再刷新。{{train.overfit_note}}<<结合下面的曲线说明收敛过程与模型容量是否合适，两句>>

{{chart:loss_history}}

{{#if train.lr_note}}
{{train.lr_note}}
{{/if}}

{{chart:lr_history}}

## 训练结果

{{metrics.sentence}}{{#if error_shape.sentence}}{{error_shape.sentence}}{{/if}}<<这个成绩相对最优模型处在什么位置、值不值得继续投入，一到两句>>

{{chart:prediction_curve}}

{{#if gap.caveat}}
{{gap.caveat}}<<这个口径差异对读者意味着什么，一句>>
{{/if}}
