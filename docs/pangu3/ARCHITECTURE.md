# Pangu 3.0 架构（对话式选股中间层）

## 产品定位变化

Pangu 2.0：独立 Web 驾驶舱 + 内嵌 LLM 解读。
**Pangu 3.0：股票研究与下一交易日决策中间层**，由宿主 Agent（Claude Code /
Codex / OpenCode / Kimi / ZCode）通过 MCP 调用。宿主负责自然语言；Pangu 负责
数据、证据、排序、决策、持有周期、退出条件与审计。

```text
宿主 Agent（理解问题 → 调 MCP 工具 → 翻译证据）
        │ stdio / JSON-RPC
        ▼
engine/mcp/server.py ── 7 tools + 3 resources + 1 prompt（唯一业务入口）
        ▼
engine/decision/service.py  recommend_next_session()（无 LLM 可完整运行）
   ├─ engine/decision/clock.py        Clock 抽象（禁 datetime.now 散落）
   ├─ engine/decision/asof.py         AsOfContext + A股交易日历（禁 +1day）
   ├─ engine/decision/freshness.py    每源 SLA + freshness 元数据 + 增量刷新
   ├─ engine/evidence/*               统一证据面（事件分类/层级/聚类/衰减/实体链接）
   ├─ engine/decision/ranker.py       Top20 排序（登记驱动因子集成+事件alpha+regime）
   ├─ engine/decision/holding.py      每票独立持有周期 + 7 类退出条件
   └─ engine/decision/runstore.py     run 持久化（/pangu why 可重建）
        ▼
engine/data/pit_store.py（PIT 档案 2022→2026-09，562 万行）
```

## 决策链路（每次 /pangu）

1. `build_asof_context`：分析时点 → 决策日（非交易日向前吸附）→ **严格下一交易日**；
2. `MarketContextRefresher.refresh_context`：各源 freshness 检查，过期才增量刷新；
   刷新失败 → quality=failed/fallback=true（关键源失败 → 本轮禁止 BUY）；
3. 证据面：近 N 天 WSCN 新闻（精确时间戳）+ 巨潮公告（Tier A）→ 事件分类 →
   跨源聚类 → 半衰期衰减 → 实体链接（direct vs sector_inherited）；
   `published_at > asof` 的一律禁用；
4. 排序：登记驱动因子集成（rev5/amihud/lowvol/lowturnover/vpr 等，方向与权重
   来自 data/experiments 研究产物）× regime 权重剖面 + 事件 alpha + 流动性可行性
   − 波动/风险惩罚 → 风险调整分（含 breakdown）；
5. 决策：BUY / WATCH / AVOID / BLOCKED（数据 failed 无 BUY；CRISIS 最多 WATCH；
   Tier D 单源事件不能制造 BUY）；
6. 持有计划：horizon 由事件半衰期/波动/regime 决定（1/3/5/10/20D），输出
   hard stop / target / trailing / time stop / news/market/sector/factor 失效；
7. `DecisionRun` 持久化 → `run_id` 可完整重建当时为什么推荐。

## 诚实输出纪律（Phase 13）

- `confidence_type`: raw / evidence / calibrated——未校准分数禁止称"上涨概率"；
- DEGRADED/failed 显式展示；关键源 failed → 无 BUY，仅 WATCH；
- 不足 20 只时如实返回不足数量并说明，禁止硬凑。

## 与 Pangu 2.0 的关系

- PIT 底座/回测引擎/实验登记/OMS/PaperBroker/组合风控 **原样复用**；
- 7 策略池保留为 research signal（研究证据显示动量/趋势方向为负 IC），
  不再等价于正式推荐；
- `engine/agent/core.py`（独立 LLM Agent 循环）标记 deprecated，默认不加载；
- Web 驾驶舱降级为辅助界面：默认导航 今日Top20 / 个股 / 系统，
  执行/策略/研究 收进"高级"分组（研究能力不删除）。
