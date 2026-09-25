# Pangu 3.0 用户指南

## /pangu（默认：下一交易日 Top20 决策候选）

```text
分析时点：2026-09-26 12:00（上海）
目标交易日：2026-09-28
市场状态：震荡，反转方向占优
数据状态：正常

Top 20（BUY 可少于 20；不足 20 只时如实说明）
# | 股票 | 决策 | 入场 | 持有 | 止损 | 核心理由 | 最大风险
```

## /pangu 600519（单票）

只展开该票：结论 / 下一交易日计划 / 建议持有时间 / 3 个最重要理由 /
2 个最大风险 / 关键新闻公告 / 失效条件。

## /pangu why <run_id> <code>

展开完整证据链：当时的数据新鲜度、来源健康、分数分解、因子/事件/市场/流动性
证据、全部退出条件、置信类型。

## /pangu status

行情/新闻/公告更新时间、各源 age 与 quality、PIT 档案最新日、MCP 状态。

## /pangu refresh

强制清缓存并增量刷新各数据源。

## 诚实约定

1. 决策枚举只有 BUY/WATCH/AVOID/BLOCKED；
2. BUY 少于 20 是正常且诚实的，不足会说明原因；
3. 数据状态 DEGRADED/failed 显式标注；failed 时不存在 BUY；
4. `confidence_type=raw|evidence` 的分数是证据置信，不是"上涨概率"；
   只有 `calibrated=true` 才允许概率表述；
5. 每只股票必有持有周期与止损；无数据时明确说"无数据"。

## 本地 CLI（不经宿主 Agent 直接用）

```bash
.venv/Scripts/python -m engine.decision.cli recommend --limit 20
.venv/Scripts/python -m engine.decision.cli analyze 600519
.venv/Scripts/python -m engine.decision.cli why <run_id> 600519
.venv/Scripts/python -m engine.decision.cli status
```

## 宿主接入

见 `docs/pangu3/HOST_COMPATIBILITY.md`（MCP 注册 + 各宿主命令文件 +
实机测试状态）。
