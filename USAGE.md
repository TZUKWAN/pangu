# 盘古 Pangu · 使用指南

> A股盘后选股决策辅助系统 · 规则选股为主，LLM 只做解读/复核

## 启动

```bash
cd D:/交易智能体
python -m engine.repl
```

## 日常使用流程

### 第一次用 / 每个交易日盘后

```bash
python -m engine.cli rps-build --workers 10    # 预计算全市场真实 RPS（必做）
```

> 未运行 RPS 预计算时，系统只会输出观察池，不会生成正式推荐。

### 查看市场状态

```bash
python -m engine.cli sentiment                 # 情绪温度
python -m engine.cli market-phase              # 市场阶段/情绪周期
```

### 查看策略池信号

```bash
python -m engine.cli pools                     # 七大策略池原始信号
```

### 系统健康检查

```bash
python -m engine.cli doctor                    # 检查 all_spot / kline / RPS / fund_flow / LLM / 策略池
```

### 跑完整选股链路

```bash
python -m engine.cli scan                      # 输出 JSON（含 source_status / final_recommendations / watchlist）
python -m engine.cli report                    # 生成 Markdown 简报
```

输出结构：

```json
{
  "date": "20250115",
  "sentiment": {...},
  "market_modules": {
    "market_phase": {
      "market_phase": "主升期",
      "phase_score": 80,
      "allowed_strategies": [...],
      "forbidden_strategies": [...],
      "position_advice": "积极仓位，聚焦主线"
    },
    ...
  },
  "candidates": [...],            # 严格候选（含 debate / xuanwu）
  "watchlist": [...],             # 观察池，不进入最终推荐
  "rejected": [...],              # 被护栏剔除
  "final_recommendations": [...], # 通过六道闸门的最终推荐
  "source_status": {...},         # 各数据源健康状态
  "recommendation_allowed": true,
  "warnings": [...]
}
```

### 每日盘后一键调度

```bash
python -m engine.cli daily                     # RPS → 快照 → 扫描 → 报告
python -m engine.cli daily --dry-run           # 只检查配置/通知
```

## 推荐引擎说明

系统通过以下流程生成最终推荐：

1. **市场状态识别**（market_phase）
2. **7 类策略池产出候选信号**（strategy_pools）
3. **量化护栏过滤风险票**（quant_guard：kept / watch / rejected）
4. **最终推荐闸门六道检查**（recommendation_gate）
5. **LLM 解读/复核**（仅当配置启用且候选通过闸门后）

只有同时通过数据、市场、题材、个股地位、交易计划、风险六道闸门的股票，才会进入 `final_recommendations`。

当以下任一情况发生时，系统会输出“今日无正式推荐”：

- 真实 RPS 缺失
- 关键数据源失败
- 市场状态为冰点期/退潮期
- 没有候选通过全部闸门
- LLM 复核明确拒绝

## 配置调参

`config/settings.yaml` 可调所有阈值：

- `trend.rps.require_real` / `allow_approx`：RPS 硬前置
- `strategy_framework.enabled`：启用策略框架
- `strategy_framework.pools`：启用的策略池列表
- `guard.*`：风险护栏阈值
- `entry_exit.*`：买卖点参数
- `llm.*`：LLM 环境变量配置

**安全提示**：不要把真实 API key 写入配置文件，使用环境变量 `PANGU_LLM_API_KEY` / `PANGU_LLM_BASE_URL` / `PANGU_LLM_MODEL`。

## 验证状态

- 277 个纯逻辑单元测试全过：`python -m pytest engine/tests/ -q --ignore=engine/tests/test_pipeline_live.py`
- Live 冒烟测试：`python -m pytest engine/tests/test_pipeline_live.py -q`（需网络）

## 重要提醒

- 系统是**决策辅助工具，不保证收益，不自动交易**。
- 所有概率/推荐度字段未经过样本外回测校准时，**不得视为真实胜率**。
- 短线个股天然高风险，所有输出仅作为次日开盘前的观察参考。
- 每个交易日盘后务必跑 `rps-build` 更新真实 RPS。

## 命令速查

| 命令 | 作用 |
|------|------|
| `python -m engine.repl` | 交互式菜单 |
| `python -m engine.cli rps-build` | 预计算真实 RPS |
| `python -m engine.cli sentiment` | 情绪温度 |
| `python -m engine.cli market-phase` | 市场阶段 |
| `python -m engine.cli pools` | 七大策略池 |
| `python -m engine.cli doctor` | 数据源与系统健康检查 |
| `python -m engine.cli scan` | 完整选股链路 → JSON |
| `python -m engine.cli report` | 生成 Markdown 报告 |
| `python -m engine.cli daily` | 每日盘后一键调度 |
| `python -m engine.web` | 启动 Web 看板 |

---

*盘古 Pangu · 用真实数据和明确规则，把 A 股选股变成可解释、可审计、可拒绝的过程。*
