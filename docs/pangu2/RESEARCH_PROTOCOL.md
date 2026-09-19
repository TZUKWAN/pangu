# Pangu 2.0 研究协议（强制执行）

> 本文件是策略研究的纪律文件。所有研究型子智能体与后续运行脚本必须遵守。
> 违反任意一条，对应实验作废并登记为 `voided`。

## 1. 数据与 PIT 纪律

1. 唯一研究数据面：`engine.data.pit_store.PITStore`（SQLite `data/market_breadth/raw.sqlite3`）。
2. 收益率一律使用 `pct_change`（来源已做除权口径调整的日收益，单位 %）。禁止用 `close_t / close_{t-1} - 1` 直接计算（会在除权除息日失真）。
3. 任何特征矩阵在决策日 `t` 只能包含 `date <= t` 的行；运行期由 `HistoryView`（engine/validation）与 `assert_no_future_rows`（engine/data/lookahead.py）双重强制。
4. 财务/价值/质量类因子当前**无 PIT 财报源**，注册为 `availability="degraded_no_source"`，不得进入任何严格历史回测，也不得作为准入门证据。
5. 行业成员 `industry_membership` 仅 2026-01-28 之后可用；此前日期行业中性化/行业约束记 `degraded`，不得据此宣称"行业中性"。
6. 股票池必须使用 `breadth_universe` 的 PIT 每日成员（含后来退市的股票），禁止用当前上市清单回放历史。

## 2. 时间切分与 Holdout

数据范围：回补后为 2022-01-04 → 2026-09-04（约 1100 个交易日）。在此之前仅有 2025-12-15 起的数据。

- **Holdout（最终测试段）**：`2026-06-01 → 2026-09-04`（约 60 个交易日）。在 `holdout_audit.jsonl` 记录一次性解锁之前，任何研究代码不得读取该段数据。解锁只发生一次：所有训练/验证/参数选择结束后，由主智能体执行。
- **研究段**：`2022-01-04 → 2026-05-29`，walk-forward：
  - expanding 模式：train ≥ 500 交易日，validation 60 交易日，test 60 交易日，step 60 交易日，embargo 5 交易日（≥ 最长标签窗口 5d）。
- 允许的调参集合 = train + validation；test（含 walk-forward 各段 test 与最终 holdout）只能评估一次，禁止依据 test 结果回改参数。
- 若 holdout 失败：不得回改参数后重新宣称同一 holdout；该段降级为 validation，等待未来新数据成为新 holdout。

## 3. 成本与执行假设（全部实验统一）

- 决策时点：T 日 15:05（盘后）；最早成交 T+1。
- 佣金 0.03%（最低 5 元）、印花税 0.05%（卖出）、滑点 10bp（base）/20bp（2x 压力）/30bp（3x 压力）。
- 涨停价买入不可成交、跌停价卖出不可成交、停牌不可交易——由 `exec_model` 强制，未成交事件计入执行率。
- T+1：当日买入不可当日卖出。
- 容量：单笔订单金额 ≤ 2% × 当日成交额。

## 4. 实验登记

- 每个实验（含失败与被否决的）必须经 `engine.validation.experiment_registry.ExperimentRegistry.register()` 写入 `data/experiments/registry.jsonl`，字段缺失即抛错（缺字段不得晋级）。
- 失败实验的结论字段必须写明失败原因（如 `oos_pf_below_1`、`cost_stress_failed`、`ic_not_significant`）。
- 多重比较：登记数即 `n_trials`，最终策略的 Sharpe 必须给出 Deflated Sharpe；≥20 个同族变体时附 PBO/CSCV 估计。

## 5. 准入门（P5-011）

- 基本门槛：OOS 扣费后期望收益 > 0；OOS PF > 1；优于随机排序对照与等权基准；MDD 在风险预算内；样本量足够（≥30 个独立交易日且 ≥100 笔成交用于短线；日频组合 ≥120 个交易日）；参数 ±10%/±20% 不脆弱；分市场阶段不塌陷；无泄漏（红队审计通过）；双引擎净值一致（相对差 < 1e-6）。
- `paper_candidate` 建议线：PF ≥ 1.2、Sharpe ≥ 1.0、MDD ≤ 15%、成交 ≥ 200 笔、有效交易日 ≥ 120、2x 成本仍正期望。
- "5%/日"或"每日盈利"**不是**任何门槛；仅作为观察字段（positive_day_ratio 等）如实统计。

## 6. 红队否决权

独立审计（不参与研发的检查）覆盖：shift(-n) 特征污染、未来 join、全期标准化、全期因子筛选、幸存者池、报告期误用、测试集泄漏、对未来公司行动的静态前复权。任一发现即否决该实验的晋级资格。
