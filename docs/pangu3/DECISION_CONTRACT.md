# 决策契约（Decision Contract）

> 权威定义：`engine/decision/contracts.py`（SCHEMA_VERSION = `pangu3.decision.v1`）
> 测试：`engine/tests/test_decision_contracts.py`（完整对象/缺失字段/None/非法
> enum/roundtrip/向后兼容）

## DecisionRequest
| 字段 | 类型 | 说明 |
|---|---|---|
| query_timestamp | ISO8601? | 查询时刻（缺省=now，上海时区）|
| asof | ISO8601? | 信息截止时点 |
| market | str | 仅支持 "CN" |
| limit | int 1..100 | 默认 20 |
| requested_execution_date | YYYY-MM-DD? | 一般留空=下一交易日 |
| risk_profile | str | 默认 default |
| force_refresh | bool | 强制清缓存增量刷新 |
| codes | list[str]? | 限定股票池（单票分析）|
| include_watchlist | bool | 默认 True |

## StockDecision（关键字段）
- `decision`: **BUY | WATCH | AVOID | BLOCKED**（唯一输出枚举；final/candidate/
  watch/kept 等历史状态不外露）
- `confidence_type`: **raw | evidence | calibrated**；raw/evidence 禁止称概率
- `execution_date`：下一交易日；`entry_zone`/`stop_loss`/`target_zone`
- `expected_holding_days` + `holding_range`：每票独立
- `exit_conditions`：hard_stop/target/trailing/time_stop/news/market/sector/factor
- `factor_evidence / event_evidence / market_evidence / fundamental_evidence /
  liquidity_evidence / risks / evidence_ids / score_breakdown / freshness`

## Top20DecisionSet
- ≤ limit 个决策，rank 连续唯一；不足 20 附 `note` 说明（不硬凑）
- `data_status`: ok | degraded | failed；**failed 时禁止 BUY**（构造器强制）

## DecisionRun
run_id / query_timestamp / decision_date / execution_date / asof_timestamp /
market_status / data_freshness / source_health / recommendations / warnings /
blocked_reasons / model_version（pangu_ranker_v1）/ evidence_version / latency /
request —— 全量 JSON roundtrip 稳定，`/pangu why` 据此重建当时推荐依据。
