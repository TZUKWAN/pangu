# 数据新鲜度（Data Freshness）

> 实现：`engine/decision/freshness.py` + `engine/decision/asof.py`
> 测试：`engine/tests/test_p3_clock_asof_freshness.py`（任务书 10 场景全覆盖）

## 每源 SLA

| 源 | SLA | 语义 |
|---|---|---|
| realtime_quote | 180s | 分钟级实时行情 |
| news | 600s | 分钟级新闻快讯 |
| announcements | 86400s | 当天公告 |
| daily_kline | 86400s | 须覆盖最近已完成交易日 |
| industry_concept | 7d | 最近有效快照 |
| financials | 90d | 最新有效报告期 |

## freshness 元数据（每个证据必带）

`source, fetched_at, published_at, effective_at, asof, age_seconds, stale,
quality(ok|degraded|failed), fallback_used, latency_seconds`
—— 禁止只返回 `"status": "ok"`。

## 刷新语义（query-time refresh）

1. 缓存 fetched_at 距 asof 未超 SLA → 直接用缓存；
2. 超期 → 调用该源增量刷新（**绝不重建全量档案**）；
3. 刷新异常 → quality=failed + fallback_used=true，不中断整体决策；
4. 关键源（realtime_quote / daily_kline）failed 或 stale 且无缓存 →
   `data_status=failed` → **本轮禁止 BUY**（仅 WATCH/AVOID/BLOCKED）。

## 交易日历

优先级：本地缓存 `data/pit/calendar.sqlite` → baostock 在线（写缓存）→
工作日规则回退（**显式 `calendar_estimated=true` + warning**，禁止假装确定）。
周五盘后→下周一；节假日前→节后首个交易日；非交易日→吸附最近收盘信息为决策日。

## 已验证场景

交易日上午/下午/收盘后、周末、法定节假日（节前与节中询问）、数据源断开、
缓存 29 分钟、缓存 31 分钟、系统时钟跨日、北京时间↔UTC、未来数据泄漏守卫。
