# {{run.model}} · 分报告

## 结论

{{headline.sentence}}{{gap.sentence}}{{train.verdict_sentence}}

{{#if gap.caveat}}
{{gap.caveat}}
{{/if}}

## 训练过程

{{run.arch_note}}{{train.plan_note}}实际训练 {{train.actual_epochs}} 轮，{{train.stop_reason}}。

{{chart:loss_history}}

第 {{train.best_epoch}} 轮取得最优验证{{train.metric_phrase}}{{train.best_value}}，此后 {{train.patience_used}} 轮未再刷新。{{train.overfit_note}}<<结合上面的曲线说明收敛过程与模型容量是否合适，两句>>

{{chart:lr_history}}

## 训练结果

{{chart:pred_vs_actual}}

{{metrics.sentence}}{{#if error_shape.sentence}}{{error_shape.sentence}}{{/if}}
