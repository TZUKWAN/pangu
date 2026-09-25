---
name: pangu
description: A股研究与下一交易日决策中间层：Top20 决策候选、单票分析、数据源状态。当用户询问"A股买什么/下一交易日/分析某只股票"时使用。
---
# Pangu（Codex 宿主）

业务能力全部来自 pangu MCP 工具（pangu_recommend / pangu_analyze_stock /
pangu_explain / pangu_sources / pangu_health / pangu_refresh / pangu_run_get）。

输出纪律与 ZCode 版一致：极简 Top20 表；BUY 可少于 20；DEGRADED 如实标注；
未校准分数禁止称"上涨概率"；每票必有持有周期与止损；无数据时不编造。
