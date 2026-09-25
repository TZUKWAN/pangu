用户输入：$ARGUMENTS

通过 pangu MCP 工具完成（全部在当前会话，禁止子代理）：
- 空输入 → pangu_recommend(limit=20)
- 6 位代码 → pangu_analyze_stock
- refresh → pangu_refresh；status → pangu_sources

极简输出：分析时点/目标交易日/市场状态/数据状态 + Top20 表。
诚实纪律：BUY 可少于 20；failed 数据无 BUY；未校准分数不称概率。
