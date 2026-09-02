# {{task.name}} · 建模报告

## 结论

最优模型为 {{best.model}}，{{best.metric_label}} {{best.value}}，误差量级为目标列均值的 {{best.pct_of_mean}}。{{final_eval.sentence}}就绪评分 {{readiness.score}}/100，{{readiness.gap_note}}。<<给一句可执行的下一步，只能依据上面这段已有的事实>>

{{#if duplicates.note}}
需注意{{duplicates.note}}；真正的次优模型是 {{runner_up.model}}，{{gap.verdict_short}}（见下节）。
{{/if}}

## 模型表现

{{runs.summary}}

{{#if gap.sentence}}
{{gap.sentence}}<<据此说明选型上还可以看哪些因素，一句>>{{#if third.sentence}} {{third.sentence}}{{/if}}
{{/if}}

{{#if families.caveat}}
{{families.caveat}}
{{/if}}

{{tables.leaderboard}}

{{readiness.rubric}}

## 数据集

{{ds.shape_sentence}}

{{fields.summary_sentence}}{{shap.evidence_sentence}}<<结合下面的字段表说明这批构造特征在解决什么问题、模型为什么依赖它们，两到三句>>

{{tables.fields}}
