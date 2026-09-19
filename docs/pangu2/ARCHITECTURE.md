# Pangu 2.0 架构与模块地图

> 重构目标：真实数据、严格 PIT、可验证策略、可控执行、全链路审计。
> 本文随实施更新。契约唯一来源：`engine/contracts.py`。

## 模块布局（新增部分）

```text
engine/
├── contracts.py               # 共享类型：Order/状态机/ExecutionMode/Compliance/PortfolioTarget
├── data/                      # P1 PIT 数据底座
│   ├── pit_store.py           # PITStore：日线面板/每日 universe/指数/公告/新闻（strict asof）
│   ├── universe.py            # PIT 可交易池构建 + manifest
│   ├── corporate_actions.py   # 除权除息推断（诊断用；收益一律用 pct_change）
│   ├── quality.py             # 数据质量评分（P1-008），不合格→不交易
│   └── lookahead.py           # 未来数据断言 + 静态扫描
├── pipeline_factory.py        # P0-005 唯一 Pipeline 构造入口
├── research/                  # P2/P3 研究平台
│   ├── data_interface.py      # ResearchData 协议 + PIT 适配 + 合成数据
│   ├── factors/               # Factor API + 因子库（动量/反转/低波/流动性/量价/涨停结构）
│   └── univariate.py          # IC/RankIC/ICIR/分位/中性化/状态拆分
├── validation/                # P4/P5 回测与统计验证
│   ├── backtest_v2.py         # 事件驱动引擎（T+1/涨跌停/停牌/费用/滑点/容量）
│   ├── exec_model.py          # 成交模型纯函数（保守路径）
│   ├── walk_forward.py        # purge+embargo 切分 + HoldoutPolicy
│   ├── statistics.py          # DSR/Bootstrap/PBO-CSCV/多重比较
│   ├── robustness.py          # 参数邻域/成本压力/延迟压力/子区间
│   ├── cross_check.py         # 双引擎交叉复核
│   ├── leakage.py             # 泄漏静态审计 + 运行期守卫
│   └── experiment_registry.py # 实验 JSONL 登记（缺字段不得晋级）
├── strategies/                # P6 注册表与生命周期
│   ├── registry.py / manifest.py / gates.py / manifests/
├── portfolio_engine/          # P7 组合与风险
│   ├── constructor.py / risk.py / state.py
├── execution/                 # P8/P9 执行
│   ├── broker.py              # BrokerAdapter 抽象
│   ├── paper.py               # PaperBroker（真实约束模拟）
│   ├── oms.py                 # 订单状态机/幂等/对账/UNKNOWN
│   ├── ths_easytrader.py      # 同花顺 easytrader 适配（未安装则如实不可用）
│   ├── miniqmt.py             # miniQMT 适配（同上）
│   ├── risk_controls.py       # 重复/偏离/资金/持仓/限速/日损熔断/KillSwitch
│   └── reconcile.py
├── compliance/                # P10 程序化交易合规闸门
│   └── program_trading.py     # ComplianceState/LiveGate/软件信息导出
└── web/                       # P11 Web 产品（FastAPI + SPA）
```

数据存储：
- `data/market_breadth/raw.sqlite3`：breadth_raw（日线）、breadth_universe（PIT 成员）、industry_membership、index_daily（回补 2022 起）
- `data/execution.db`：OMS 订单/对账/风控/注册表/组合状态
- `data/paper_broker.db`：PaperBroker 账户
- `data/experiments/`：registry.jsonl + 因子/策略报告
- `data/pit/`：corporate_actions.sqlite、universe_manifest.jsonl、quality/、manifests/

## 执行模式与准入门

`DISABLED / PAPER / SHADOW / MANUAL_CONFIRM / LIVE`，默认 PAPER。LIVE 需同时：
validated→paper→shadow 证据链、broker adapter 通过、compliance=CONFIRMED、reconcile 通过、KillSwitch 可用、LiveGate 全检通过（fails-closed）。

## 与旧系统的关系

- 原 7 策略池降级为实验策略（registry status=research），人工评分拆解为可验证因子（engine/research/factors）。
- replay_backtest / short_term_replay 保留为兼容链路；最终验证以 validation/backtest_v2 + 双引擎复核为准。
- LLM 只做解读/复核（现状保持），不参与选股决策、不临场决定风控动作。
