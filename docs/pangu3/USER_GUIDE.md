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

## "持有期结束 ≥5%"如何得到保证（诚实机制）

系统**不承诺**收益。它给出的是可验证的历史证据：

1. 每个候选按 `市场状态 × 反转z分位` 匹配历史同类 setup（研究窗 2022→2025，
   1340 万样本次决策）；
2. 统计该画像在推荐持有窗内、同等止损纪律下**触及 +5% 目标的频率**，并给出
   95% Wilson 置信下界；
3. 持有期=满足"下界 ≥30%"的**最小**窗口（5/10/20 日自动选）；
4. BUY 证据线：命中表有数据时，无 ≥30% 下界支撑的候选**自动降为 WATCH**；
5. OOS 校准：2025 年样本上预测/实际平均绝对误差 4.7 个百分点（已登记实验）。
   基础频率参考：5 日窗 ≈29.6%、10 日 ≈35.1%、20 日 ≈34.0%。

## 每日自动扫描

交易日 15:05 由调度器自动运行 `pangu_decision` 步骤（config/settings.yaml →
`decision.enabled: true`）：生成并持久化当日 DecisionRun（含尾盘/次日开盘
入场判定）。手动即时触发：`python -m engine.decision.cli recommend`。

## 插件自测（不启动任何外部 AGENT）

```bash
python -m engine.mcp.selftest --fast
```
按 5 个宿主适配清单逐一拉起 pangu MCP server 并完成协议全流程，
`5/5 PASS` 即代表任一宿主按同一清单配置即可用。

## 宿主接入

见 `docs/pangu3/HOST_COMPATIBILITY.md`（MCP 注册 + 各宿主命令文件 +
实机测试状态）。
