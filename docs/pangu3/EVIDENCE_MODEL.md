# 证据模型（Evidence Fabric）

> 实现：`engine/evidence/model.py`（EvidenceItem）、`taxonomy.py`（25 类事件）、
> `engine.py`（分类/聚类/衰减/实体链接）
> 测试：`engine/tests/test_evidence_fabric.py`

## EvidenceItem

每个证据必带：evidence_id、entity_type(stock/sector/market)、entity_id、
category（price/technical/factor/liquidity/capital_flow/announcement/news/
financial/sector/macro/risk/execution）、event_type、direction（positive/
negative/neutral/risk）、magnitude(0-1)、source、**source_tier**、
published_at、effective_at、fetched_at、expiry_at、confidence、
**directness**（direct / sector_inherited / market，严格区分）、
corroboration_count、content_hash、raw_reference、summary。

## 事件分类（25 类）

earnings_beat / earnings_miss / profit_warning / order_win / contract / buyback /
insider_increase / insider_reduce / restructuring / acquisition /
regulatory_investigation / litigation / product_release / policy_support /
policy_restriction / supply_disruption / price_increase / capacity_expansion /
capital_raise / dividend / analyst_upgrade / analyst_downgrade /
industry_catalyst / macro_event / rumor / unknown。

每类携带：方向、默认半衰期（天，7~30 不等）、风险标记、触发 BUY 的最低来源层级、
印证要求。分类基于有序关键词规则；禁止把新闻当笼统正/负关键词计数。

## 来源层级（SourceTier）

| Tier | 覆盖 | 权重 | 限制 |
|---|---|---|---|
| A | 交易所/巨潮公告、监管机构 | 1.0 | — |
| B | 高质量财经媒体、结构化行情 | 0.75 | — |
| C | 新闻聚合、二手转载 | 0.5 | — |
| D | 社交热度、未确认消息 | 0.25 | **单一 D 源事件禁止触发 BUY**；rumor 需 A 级印证 |

## 去重与跨源印证

聚类键 = (实体, 事件类型, 发布日) + 标题相似度（去标点字符集 Jaccard ≥ 0.45）。
同一事件 5 家转载 = 1 个事件，`corroboration_count=5`；簇代表取层级最高
（其次最新）。置信 = 层级权重 × 直接性(direct 1.0 / sector_inherited 0.6 /
market 0.4) × min(1, 1+0.15×(印证数−1))。

## 时间衰减

有效强度 = magnitude × 0.5^(age_days / half_life_days)；按事件类别差异化
（涨价 7d、订单 14d、回购/增持 20d、重组/并购 30d…）。昨天的普通利好 ≠
5 分钟前的重大公告。

## 实体链接

代码直match、全名、去 ST 简称、末二字昵称（歧义不注册）。标题只含板块词 →
`sector_inherited`（置信打折，严格区别于 direct）。新闻发布时间晚于 asof 的
一律禁用（未来新闻泄漏守卫，测试覆盖）。
