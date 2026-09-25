---
description: Pangu 下一交易日 Top20 决策候选（A 股）。通过 pangu MCP 调用，当前会话执行，subagent: false。
---
用户输入：$ARGUMENTS

工作流（全部在当前会话完成，禁止子代理/子智能体）：
1. 空输入 → MCP 工具 `pangu_recommend`（limit=20）
2. 6 位代码 → `pangu_analyze_stock`
3. `refresh` → `pangu_refresh`；`status` → `pangu_sources`
4. 极简输出：分析时点/目标交易日/市场状态/数据状态 + Top20 表（# | 股票 | 决策 | 入场区 | 持有 | 止损 | 核心理由 | 最大风险）
5. 诚实纪律：BUY 可少于 20；failed 数据无 BUY；未校准分数不称概率。
