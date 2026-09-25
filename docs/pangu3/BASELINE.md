# Pangu 3.0 基线审计（Phase 0 / Task 0.2-0.3）

> 分支：`pangu-3-plugin-rebuild`（自 `pangu-2-rebuild@37e3fdb` 创建）
> 执行时间：2026-09-26（周六，非交易日，网络部分降级）
> 本文档为**本轮实际执行**结果，非沿用 Pangu 2.0 报告。

## pytest 基线（重新执行）

```
.venv/Scripts/python -m pytest engine/tests -q
→ 794 passed, 2 warnings in 908.80s (0:15:08)
passed=794 failed=0 skipped=0
```

## CLI doctor（实时，周六）

| 检查 | 状态 |
|---|---|
| all_spot | ok（同花顺 5209 只）|
| daily_kline | ok |
| limit_up_pool | degraded（非交易日，空池属预期）|
| limit_down_pool | ok |
| fund_flow | ok |
| rps_table | **failed**（RPS 表需 rps-build 重建；历史 RPS 由本地档案计算正常）|
| llm_config | degraded（无 base_url/model，Pangu 3.0 起 LLM 非必需）|
| config / trading_day / strategy_pools | ok |
| overall | failed（由 rps_table 一票导致）|

## Pipeline smoke（PIT replay 20260715→20260725）

- 8 个交易日逐日复放完成，exit=0；20260720 单日：final=1 / watch=187 / candidates=100。
- 复放回测汇总：成交 2 笔（insufficient_sample 如实标注），PF 0.0，MDD 18.4%。

## News smoke

- CLI 无独立 `news` 子命令（新闻由 pipeline 新闻阶段与 web `/api/news/*` 提供）。
- replay 中新闻阶段正常（13 题材/180 只）。
- 实时新闻抓取当日部分源超时（sina 重试用尽）——记录为当前网络环境限制。

## Web API smoke（端口 8690 新起进程）

| 端点 | 结果 |
|---|---|
| /api/execution/overview | OK |
| /api/strategies | OK（7 池 research + uat_manual_paper paper）|
| /api/research/experiments | OK（79 实验/29 因子）|
| /api/latest | **timeout**（周六预热实时行情阻塞；交易日正常，Pangu 3.0 Phase 2 的 query-time refresh 将重构此路径）|

## 行为快照 fixtures（Task 0.3）

`engine/tests/fixtures/pangu3_snapshot/`（固定历史日期 20260720，PIT replay）：

- pipeline_replay_20260720.json（完整输出结构）
- final_recommendations / watchlist / candidate_evidence / source_status / strategy_signals / market_phase 切片
- strategy_registry.json（注册表状态：7 池 research + uat_manual_paper paper）

真实当前数据行为**不入 fixture**（每日变化），由显式 live smoke 记录于本文件。
