---
description: Pangu 下一交易日 Top20 决策候选（A 股）
subagent: false
---
使用 Pangu MCP 服务器（pangu）完成以下工作流，禁止 spawns 子代理，全部在当前会话执行：

1. 调用 MCP 工具 `pangu_recommend`（limit=20，参数：$ARGUMENTS 为空则用默认；若参数是 6 位股票代码则改用 `pangu_analyze_stock`；参数为 `refresh` 则先 `pangu_refresh`；参数为 `status` 则用 `pangu_sources`）。
2. 默认输出（Top20 极简格式，禁止展开全部原始因子/source_status/策略池）：

```
分析时点：<asof>
目标交易日：<execution_date>
市场状态：<market_conclusion>
数据状态：<data_status 正常|DEGRADED>

Top 20（BUY 可少于 20，不足 20 只时如实说明）
# | 股票 | 决策 | 入场区 | 持有 | 止损 | 核心理由 | 最大风险
```

3. 单票模式（$ARGUMENTS 为代码）只展开该票：结论 / 下一交易日计划 / 建议持有时间 / 3 个最重要理由 / 2 个最大风险 / 关键新闻公告 / 失效条件。
4. 数据状态为 DEGRADED/failed 时必须如实标注；failed 时不存在 BUY 决策。
5. 未经校准的分数一律称"原始分/证据置信"，禁止表述为"上涨概率"。
