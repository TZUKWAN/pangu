# Pangu 2.0 数据字典与 PIT 规范

## 1. 核心数据事件结构（P1-001）

任何进入研究/交易的数据都必须可回答：**决策时刻，市场参与者最早何时可知？**

| 字段 | 含义 |
|---|---|
| instrument | 证券代码（baostock 风格 sh.600000 / sz.000001） |
| field | 字段名（close / volume / announcement / news / …） |
| value | 值 |
| event_time | 事件发生时间（如财报报告期、K线日期） |
| effective_at | 经济含义生效时间（如除权除息日） |
| available_at | 市场参与者最早可知时间 |
| received_at | 本系统实际收到时间 |
| source / source_version / revision / checksum / quality | 溯源五元组 |

回测读取约束：`available_at <= decision_time`，由 PITStore 查询层面强制。

## 2. 日线表 `breadth_raw`

来源：baostock 未复权日线（回补自 2022-01-01；原档案自 2025-12-15）。

| 列 | 说明 |
|---|---|
| date | 交易日期（决策时点 15:05 后当日行可用） |
| code | sh.6XXXXX / sz.0XXXXX / sz.3XXXXX（含已退市代码） |
| open/high/low/close | 未复权价格 |
| preclose | **除权除息调整后**前收盘（baostock 口径） |
| pct_change | 当日收益（%，基于调整后 preclose）——**收益率唯一合法来源** |
| volume / amount | 成交量（股）/成交额（元） |
| turnover | 换手率（%） |
| is_st | 当日 ST 标记（PIT） |

## 3. 每日成员 `breadth_universe(date, code, name)`

PIT 成员表：当日实际可出现的股票（含后来退市者）。停牌推断 = 成员但无当日 bar。

## 4. 行业 `industry_membership`

可用性：2026-01-28 起 ok；此前 **degraded**（不得进入严格历史因果回测的行业约束/中性化）。

## 5. 指数 `index_daily`

sh.000001 / sh.000300 / sh.000905 / sh.000852 / sz.399001 / sz.399006 日线，作基准与状态变量。

## 6. 公告 `data/announcement_archive/YYYYMMDD.json`

cninfo 事件，`published_at` 通常仅日期粒度（00:00:00）。保守约定：**T 日公告在 T 日 15:05 决策点一律视为不可用，最早 T+1 决策点可用**（宁可延迟，不可泄漏）。

## 7. 新闻 `data/wscn_news_archive/YYYYMMDD.json`

精确 `published_at` 时间戳：`published_at <= decision_time` 才可用。

## 8. 除权除息

`corporate_actions.sqlite`（推断事件，诊断用途）。收益计算**禁止**使用静态全历史前复权价；统一用 `pct_change`。

## 9. 质量与快照

- `data/pit/quality/YYYY-MM-DD.json`：completeness/freshness/missing_ratio/pit_safe；`pit_safe=false` → 当日不交易。
- `data/pit/manifests/YYYY-MM-DD.json`：数据+配置+代码 SHA 快照清单（不可变审计锚点）。

## 10. 执行域

- `data/execution.db`：orders / order_events / reconcile_runs / strategy_registry(+history) / portfolio 状态 / risk 状态 / compliance_state
- `data/paper_broker.db`：PaperBroker 账户/持仓/委托/成交/资金流水
- `data/experiments/registry.jsonl`：实验登记（append-only）
- `data/audit/`：compliance_log.jsonl、程序化交易报告、基线与审计文档
