# {{task.name}} · 建模报告

## 结论

{{conclusion.verdict}}{{conclusion.runner}}{{conclusion.others}}

{{risk.sentence}}<<用一句话写下一步该做什么，以“下一步”开头；只说动作，不评价优先级或值不值得>>

{{chart:leaderboard_bars}}

## 模型差距

{{chart:fold_dots}}

{{gaps.sentence}}<<除了误差，据此选型还看什么，一句>>

## 数据集

{{ds.shape_sentence}}

{{chart:target_hist}}

{{chart:field_composition}}

{{#if fields.has_groups}}
<<这批构造特征在解决什么问题，两句>>
{{/if}}

{{#if shap.lead_sentence}}
## 特征依赖

{{chart:shap_bars}}

{{shap.lead_sentence}}<<这带来什么部署风险，一句>>
{{/if}}
