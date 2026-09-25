# pangu_ranker_v1 排序器验证报告（Phase 6）

> 窗口：研究窗 2022-01-04→2025-12-31（166 个周度调仓日，holdout 未触碰）
> 权重：预注册（来自 Pangu 2.0 因子研究登记），窗内零拟合
> 执行：BacktestV2 全成本（佣金 3bp/印花 5bp/滑点 10bp/T+1/涨跌停不可成交/
> 停牌/2% ADV 容量），BUY 子集交易
> 数据：`data/experiments/strategies/pangu_ranker_v1_validation.json`
> 登记：`data/experiments/registry.jsonl` → `ranker_pangu_ranker_v1_oos`

## 1. Pick 层（5 日前向收益，决策后严格前向）

| 深度 | n | mean | median | hit rate | 胜全市场中位数 |
|---|---|---|---|---|---|
| Top5 | 830 | +0.23% | -0.44% | 46.4% | 47.1% |
| Top10 | 1660 | +0.13% | -0.46% | 46.2% | 46.4% |
| Top20 | 3320 | +0.15% | -0.42% | 46.7% | 47.0% |

- BUY 子集（233 笔）**+0.34%** > WATCH 子集（3087 笔）+0.13% → 风险调整分层有区分度
- Top20 相对全市场等权的超额（regret 口径）：**+0.18%/5日**

## 2. 组合级（扣全部成本后）

| 指标 | 值 |
|---|---|
| 总收益 | **-17.8%** |
| Sharpe | -0.08 |
| Profit Factor | 0.73 |
| MDD | 55.4% |
| 执行率 | 82.5% |
| 年化换手 | 5.23 |

- DSR 0.417（<0.5）；bootstrap Sharpe CI [-0.0011, +0.0010]（含 0）
- horizon accuracy：51.5%（微弱高于随机）

## 3. 新闻增量价值（Task 6.5 ablation）

**如实登记为 deferred**：PIT 新闻档案（WSCN/公告）仅覆盖 2026 年起，
研究窗内无可用事件证据 → C 臂（quant+structured events）无法在窗内评估；
A 臂（quant only）即本报告。后续以每日 journal 前瞻积累 B/C 臂样本，
在未产生稳定 OOS 增量前**不宣传"新闻/AI 增强提高收益"**。

## 4. 结论（诚实）

1. Pick 层存在微弱正 alpha（BUY>WATCH、regret>0、Top5>Top20 单调），方向可辨；
2. **组合级扣成本后为负**：+0.15%/5日 的 pick 优势被 5.2 倍换手的交易成本吞没；
3. rank IC -0.027：整体截面排序能力弱，头部有效性非单调；
4. **不批准任何实盘/纸面自动交易升级**；ranker 保持 `research`。

## 5. 下一轮改进方向（预注册，尚未实施）

- 降换手：月度调仓或 20% 缓冲带（只有跌出 Top40 才卖出），目标换手 <1.5；
- cost-aware：期望超额 < 成本估计的候选直接不交易；
- 只交易 BUY 子集 + regime 危机时空仓（已实现，待组合级复验）；
- 事件臂增量验证待 2026 年 journal 数据积累后复跑 A/B/C。
