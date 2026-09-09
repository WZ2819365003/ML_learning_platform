# {{run.model}} · 分报告

## 结论

{{headline.sentence}}{{gap.sentence}}{{#if cv.verdict_sentence}}{{cv.verdict_sentence}}{{/if}}

{{#if gap.caveat}}
{{gap.caveat}}
{{/if}}

{{#if shap.top_feature}}
主要部署风险来自特征依赖：模型高度依赖 {{shap.top_feature}}。<<这个依赖在什么场景下会失效，一句>>
{{/if}}

## 训练过程

采用 {{run.strategy}} 策略，{{run.params_note}}。这类模型不按轮次迭代，没有收敛曲线，稳定性看各折。

{{#if cv.scheme}}
{{chart:fold_scores}}

{{cv.summary_sentence}}<<这说明数据划分与模型稳定性如何，一句>>
{{/if}}

{{#if validation.summary_sentence}}
{{validation.summary_sentence}}
{{/if}}

## 训练结果

{{chart:pred_vs_actual}}

{{metrics.sentence}}{{#if error_shape.sentence}}{{error_shape.sentence}}{{/if}}

{{#if shap.concentration_sentence}}
{{chart:shap_bars}}

{{shap.concentration_sentence}}<<模型实质上在依赖什么信号、这决定了怎样的适用边界，两句>>
{{/if}}

{{#if shap.vs_best_sentence}}
{{shap.vs_best_sentence}}
{{/if}}
