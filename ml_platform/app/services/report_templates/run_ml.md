# {{run.model}} · 分报告

## 结论

{{headline.sentence}}{{gap.sentence}}{{cv.verdict_sentence}}

{{#if shap.top_feature}}
主要部署风险来自特征依赖：模型高度依赖 {{shap.top_feature}}（详见训练结果）。<<这个依赖在什么场景下会失效，一句>>
{{/if}}

## 训练过程

采用 {{run.strategy}} 策略、{{run.params_note}}、{{cv.scheme}}。树模型不按 epoch 迭代，无逐轮记录，故以折间表现替代收敛曲线。

五折 {{cv.metric}} 极差 {{cv.range}}，为均值的 {{cv.range_pct}}，变异系数 {{cv.cv_pct}}。最差折（{{cv.worst_fold}} {{cv.worst_value}}）与最好折（{{cv.best_fold}} {{cv.best_value}}）之间{{cv.spread_note}}。<<这说明数据划分与模型稳定性如何，一句>>

{{tables.folds}}

{{chart:fold_scores}}

## 训练结果

{{metrics.sentence}}{{#if error_shape.sentence}}{{error_shape.sentence}}{{/if}}

{{chart:prediction_curve}}

{{#if shap.concentration_sentence}}
{{shap.concentration_sentence}}<<模型实质上在依赖什么信号、这决定了怎样的适用边界，两句>>
{{/if}}

{{#if shap.vs_best_sentence}}
{{shap.vs_best_sentence}}
{{/if}}

{{tables.shap}}
