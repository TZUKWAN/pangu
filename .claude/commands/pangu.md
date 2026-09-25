---
description: Pangu 下一交易日 Top20 决策候选（A 股）。通过 pangu MCP 调用，当前会话执行，禁止子代理。
---
用户输入：$ARGUMENTS

工作流（全部在当前会话完成，禁止 spawns 子代理）：
1. 空输入 → 调用 MCP 工具 `pangu_recommend`（limit=20）
2. 6 位代码 → `pangu_analyze_stock`
3. `refresh` → `pangu_refresh`；`status` → `pangu_sources`
4. 输出格式与诚实纪律：BUY 可少于 20；DEGRADED 如实标注；未校准分数不称概率；每票必有持有周期与止损。
