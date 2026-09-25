---
description: Pangu 下一交易日 Top20 决策候选（通过 pangu MCP 调用，当前会话执行）
---
通过 pangu MCP 工具完成用户请求（$ARGUMENTS）：

- 空参数 → `pangu_recommend`（limit=20），按极简格式输出 Top20 表
- 6 位代码 → `pangu_analyze_stock`，只展开该票
- `refresh` → 先 `pangu_refresh`
- `status` → `pangu_sources`

输出纪律：BUY 可少于 20；数据 DEGRADED 如实标注；未经校准分数禁止称"上涨概率"。
禁止 spawns 任何子代理；全部在当前会话完成。
