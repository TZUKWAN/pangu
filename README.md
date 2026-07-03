# 盘古 Pangu · A股选股决策辅助系统

> 一个基于真实 A 股数据的盘后选股原型系统。**规则选股为主，LLM 只做解读/复核**。
> 本系统不承诺收益，不自动交易，所有输出仅作为次日开盘前的观察参考。

## ⚠️ 重要声明

- **本系统是盘后决策辅助工具，不保证收益，不自动交易。**
- 选股以**规则 + 真实数据约束**为主，LLM 仅用于报告解读、复核和摘要，**不参与选股决策**。
- 所有概率/推荐度字段在未经过样本外回测校准前，**不得视为真实胜率**。
- 当关键数据缺失或市场状态不适合交易时，系统会明确输出“今日无正式推荐”，不会硬凑股票。

---

## 核心设计

系统从“单一涨幅 TopN + RPS + 均线突破”重构为：

```text
市场状态（market_phase）
    ↓
7 类策略池（strategy_pools）
    ↓
量化护栏（QuantGuard）
    ↓
最终推荐闸门（RecommendationGate）
    ↓
final_recommendations / watchlist / rejected
```

只有同时通过**数据完整性、市场状态、板块共振、个股地位、交易计划、风险过滤**六道闸门的股票，才能进入 `final_recommendations`。

---

## 市场状态（6 阶段）

`engine/market_phase.py` 根据涨停/跌停、炸板率、连板高度、昨日涨停今日表现、市场宽度等判断：

- **冰点期**：空仓/极轻仓，只允许低位修复观察和红利低波防守
- **修复期**：轻仓试错，允许低位反转、首板启动、强趋势回踩
- **主升期**：积极仓位，允许题材龙头、中军趋势、补涨扩散、连板核心
- **高潮期**：轻仓只参与核心，禁止后排追涨
- **分歧期**：控制仓位，只看龙头承接/中军抗跌
- **退潮期**：空仓或防守仓位，禁止短线进攻

---

## 7 类策略池

| 策略池 | 模块 | 说明 |
|--------|------|------|
| 题材龙头 | `ThemeLeaderPool` | 先识别主线题材，再识别龙头/中军/补涨角色 |
| 连板梯队 | `LimitUpPool` | 连板质量与涨停结构，一字板只作情绪锚点 |
| 趋势回踩 | `TrendPullbackPool` | 强趋势股缩量回踩 MA10/MA20 或平台突破后承接 |
| 超跌反弹 | `OversoldReboundPool` | 冰点修复期专用，跌深企稳信号 |
| 小盘优质 | `SmallQualityPool` | 小市值+质量+流动性，默认只进观察池 |
| 红利低波 | `DividendLowVolPool` | 防守池，弱市时替代展示 |
| 事件驱动 | `EventDrivenPool` | 龙虎榜/公告/事件催化，目前基于龙虎榜上榜 |

每个策略池独立产出 `StrategySignal`，不受其他池污染。

---

## 最终推荐闸门（6 道硬闸门）

`engine/recommendation_gate.py`：

1. **数据完整性**：真实 RPS、完整 K 线、资金流可解释、买卖点可计算
2. **市场状态**：当前阶段允许该策略（冰点禁追涨、退潮禁进攻等）
3. **板块共振**：有明确题材/板块，非孤立上涨
4. **个股地位**：角色明确（龙头/中军/补涨/首板观察/趋势核心），后排剔除
5. **交易计划**：有触发条件、止损、目标/止盈、盈亏比
6. **风险过滤**：通过 QuantGuard（ST/财务/估值/退市/一字板等）

任一闸门不通过即降级到 `watchlist` 或 `rejected`。

---

## 快速开始

### 1. 安装依赖

```bash
cd D:/交易智能体
pip install -r requirements.txt
```

### 2. 配置 LLM（可选，仅用于报告解读/复核）

**不要把 API key 写入配置文件提交仓库。** 通过环境变量传入：

```bash
# Linux / macOS
export PANGU_LLM_API_KEY="sk-..."
export PANGU_LLM_BASE_URL="https://api.example.com/v1"
export PANGU_LLM_MODEL="gpt-4o-mini"

# Windows CMD
set PANGU_LLM_API_KEY=sk-...
set PANGU_LLM_BASE_URL=https://api.example.com/v1
set PANGU_LLM_MODEL=gpt-4o-mini
```

参考 `.env.example`。

### 3. 预计算真实 RPS（首次必做，盘后跑一次即可）

```bash
python -m engine.cli rps-build --workers 10
```

> 未运行 RPS 预计算时，系统不会生成正式推荐，只会输出观察池。

### 4. 验证引擎

```bash
python -m engine.cli sentiment              # 情绪温度
python -m engine.cli market-phase           # 市场阶段/情绪周期
python -m engine.cli pools                  # 七大策略池原始信号
python -m engine.cli scan                   # 完整选股链路 → JSON
python -m engine.cli report                 # 生成 Markdown 简报 → data/reports/
```

### 5. 每日盘后一键调度

```bash
python -m engine.cli daily                  # RPS → 快照 → 扫描 → 报告
python -m engine.cli daily --dry-run        # 只检查配置/通知
```

### 6. Web 看板

```bash
python -m engine.web                        # http://127.0.0.1:8000
```

页面展示：
- 情绪温度计与市场阶段
- 热门板块与短线连板梯队
- 严格候选 / 观察池 / 最终推荐 分离展示
- AI 摘要/解读（仅解读，不参与选股）

---

## 核心输出字段

最终推荐股票包含：

```json
{
  "code": "000000",
  "name": "示例股份",
  "strategy": "题材龙头",
  "theme": "机器人",
  "role": "中军",
  "market_phase_fit": true,
  "rps_mode": "real",
  "fund_flow_status": "ok",
  "entry_condition": "...",
  "invalid_condition": "...",
  "stop_loss": "...",
  "risk_flags": [],
  "gate_status": "final"
}
```

---

## 关键约束（系统强制）

1. **观察池不进入最终推荐**
2. **风险票不进入最终推荐**
3. **真实 RPS 缺失时不生成正式推荐**
4. **关键数据源失败时不生成正式推荐**
5. **LLM 不可用时明确标记为规则验证/降级，不伪装成 AI 辩论**
6. **AI 摘要只叫摘要，不叫 AI 决策**
7. **未校准概率不视为真实胜率**

---

## 项目结构

```
D:/交易智能体/
├── engine/
│   ├── data_loader.py        数据层（akshare + 多源降级）
│   ├── sentiment_meter.py    情绪温度计
│   ├── market_phase.py       市场状态/情绪周期识别（新增）
│   ├── strategy_pools.py     7 类策略池（新增）
│   ├── recommendation_gate.py 最终推荐闸门（新增）
│   ├── trend_scanner.py      趋势扫描与真实 RPS 查表
│   ├── quant_guard.py        量化护栏（kept/watch/rejected）
│   ├── entry_exit.py         买卖点引擎
│   ├── pipeline.py           主链路
│   ├── agent/                LLM 客户端 / 辩论 / Agent 复核
│   ├── web/                  FastAPI 看板
│   └── tests/                单元测试
├── config/settings.yaml      全部阈值与策略开关
├── .env.example              LLM 环境变量示例
└── data/                     报告、缓存、快照
```

---

## 配置调参

所有阈值在 `config/settings.yaml`：

- `trend.rps.require_real` / `allow_approx`：RPS 硬前置
- `strategy_framework.enabled`：启用 7 大策略池 + 推荐闸门
- `strategy_framework.small_quality.*` / `dividend_low_vol.*`：策略池参数
- `guard.*`：QuantGuard 风险阈值
- `entry_exit.*`：买卖点参数

---

## 测试

```bash
# 纯逻辑单元测试（不含真实数据 live 测试）
python -m pytest engine/tests/ -q --ignore=engine/tests/test_pipeline_live.py

# 真实数据冒烟测试（需网络）
python -m pytest engine/tests/test_pipeline_live.py -q
```

当前状态：**277 个纯逻辑单元测试全过**。

---

## 已知限制与后续 TODO

- 部分策略池（事件驱动、红利低波）目前为原型实现，依赖可用数据源。
- 题材龙头池的板块持续性、新闻催化强度尚未完全量化。
- 事件驱动池目前主要基于龙虎榜，业绩预增/回购等事件需后续接入公告解析。
- 未校准的推荐度/概率字段已标记 `calibrated: false`，不代表真实胜率。
- `python -m engine.cli doctor` 数据源健康检查命令待补充。

---

*盘古 Pangu · 用真实数据和明确规则，把 A 股选股变成可解释、可审计、可拒绝的过程。*
