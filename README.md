# 盘古 Pangu · A股短线「情绪+趋势」选股 Agent

> 一个独立、干净的 A 股短线选股决策助手。**盘后分析，告诉你明天关注什么**——情绪+趋势主导，量化做辅助护栏，用独立 Python Agent 做 LLM 综合，多数据源（同花顺/腾讯/新浪/adata）免 token 稳定取数。

## ⚠️ 风险声明

本系统是**盘后决策辅助工具，不保证收益，不自动交易**。它基于当日收盘数据，为**次日开盘**筛选关注标的。月 13-20% 目标是短线高波动风格，盈亏取决于你的执行与风控。回测≠未来。所有输出末尾均带免责提示。

---

## 它解决什么问题

你要的不是又一个回测框架，而是一个**每天收盘后告诉你"明天该关注什么"**的助手。它的工作方式：

```
① 情绪温度计 (0-100)   →  判定次日攻防姿态
       <40 冰点观望 / 40-85 正常选股 / >85 亢奋警惕
② 趋势扫描            →  热门板块 + 个股形态 + 主力资金流入
③ 量化护栏            →  剔 ST/财务风险/估值泡沫（排雷）
④ LLM 综合 (独立 Agent)  →  结合新闻简报，输出明日 3-5 只 + 买点/止损/风险
```

情绪和趋势是**引擎**（做对的事），量化是**护栏**（不做坏事）—— 这正是你想要的"量化只做辅助"。

## 架构（4 层解耦，独立干净）

```
engine/      Python 选股引擎（akshare 数据 + 情绪计 + 趋势计 + 护栏）
engine/agent/  独立 Agent（LLM 工具调用循环 + OpenAI 兼容 client）
agent/        旧 pi Extension（已弃用，移到 agent/legacy/）
skills/       capitalise-finnews（已装，新闻简报源）
config/       settings.yaml（所有阈值可调）
data/         SQLite + Markdown 报告
```

## 快速开始

### 1. 装依赖
```bash
cd D:/交易智能体
pip install -r requirements.txt              # Python 引擎 + 独立 Agent
```

### 2. 配置 LLM（可选，Agent 综合分析需要）
**不要把真实 api_key 写进 `config/settings.yaml` 提交仓库。** 推荐通过环境变量传入：
```bash
export PANGU_LLM_API_KEY="sk-..."           # Linux/macOS
set PANGU_LLM_API_KEY=sk-...                # Windows CMD
$env:PANGU_LLM_API_KEY="sk-..."             # PowerShell
```
`config/settings.yaml` 中 `llm.api_key` 保持空字符串即可；也支持填写 `${ENV:YOUR_KEY_NAME}` 占位符。
支持 DeepSeek / Qwen / GLM / 豆包 / Kimi / OpenAI 等所有 OpenAI 兼容接口。

### 3. 预计算真实 RPS（首次必做，盘后跑一次即可）
```bash
python -m engine.cli rps-build --workers 10   # 算全市场5500+只真实20日RPS存库（约1-2分钟）
```
> 没跑这步，scan 会用失真的近似RPS并打印警告。建议每个交易日15:30后跑一次。

### 4. 验证引擎（无需 LLM key，用真实 A 股数据）
```bash
python -m engine.cli sentiment               # 看情绪温度（增强版含市场结构/广度/历史分位）
python -m engine.cli scan                    # 跑完整选股链路（真实RPS+趋势+护栏+买卖点）
python -m engine.cli report                  # 生成 Markdown 选股简报 → data/reports/
```

### 5. 每日盘后一键调度（推荐）
```bash
python -m engine.cli daily                   # RPS → 快照 → 扫描 → 报告 → 通知（如已配置）
python -m engine.cli daily --dry-run         # 只检查配置/通知，不执行耗时取数
python -m engine.cli daily --skip-rps        # 跳过 RPS 预计算
```
调度状态写入 `data/scheduler/YYYYMMDD_status.json` 与 `data/scheduler/scheduler.log`。

### 5. 跑 Agent（LLM 综合选股）
```bash
# 交互式 REPL（菜单 3. AI选股）
python -m engine.repl

# 或命令行一次性问答
python -m engine.agent.cli "明天 A 股帮我关注 3-5 只短线票"
# 或
python -m engine.cli agent "000001 平安银行明天能关注吗"

# Agent 会调 get_sentiment → scan_trend → get_news_briefing → debate_stock → 输出买卖点报告
```

### 6. Web 看板（暗色交互 UI，推荐）
不想看终端 JSON？启动网页看板，所有重要数据一目了然：
```bash
python -m engine.web            # 默认 http://127.0.0.1:8000
python -m engine.web --port 9000 --host 0.0.0.0   # 自定义端口/可外部访问
```
浏览器打开后能看到：
- **情绪温度计**（环形进度 + 姿态 + 15 项分项条形图 + 历史分位/动量/见顶见底信号）
- **热门板块**（涨幅+资金双因子排名表）
- **候选股卡片**（推荐度/等级SABC/上涨概率/预测涨幅，点击展开买卖点+6维雷达图）
- **🤖 AI 明日展望**（点「生成」流式输出 LLM 对次日操作的解读，需配好 llm key）
- **生成明日报告**（后台异步跑 Pipeline，右下角浮层显示进度日志）
- **历史报告**（下拉切换历史日期）

> 首次无数据时点「📊 生成明日报告」，约 1-3 分钟（多源取数 + 选股）。

### 7. 独立桌面 GUI（PySide6，白色主题）
如果你更喜欢原生桌面窗口，启动独立 GUI：
```bash
python -m engine.gui            # 默认后端 127.0.0.1:18421
python -m engine.gui --port 9000 --host 0.0.0.0   # 自定义后端端口
```
界面布局：
- **顶部工具栏**：标题 / 日期选择 / 自动刷新开关 / 「生成明日报告」 / 「AI 明日展望」
- **上半区**：情绪温度圆环 + 市场信号 + 情绪分项横向条形图；市场结构 + 热门板块表格 + 盘后新闻
- **下半区**：明日关注池候选股卡片（推荐度 / 等级 / 买卖点 / 6 维雷达图 / 相关新闻）；AI 明日展望文本区（SSE 流式输出）
- **底部提示条**：风险提示/系统提示

> 关闭窗口时自动停止内置 FastAPI 后端线程。

## 实测结果（2026-06-26 数据）

```
情绪温度 58.2（正常）｜涨停60 连板6 炸板率36.8% 跌停30 涨790/跌4676
热门板块：昨日打二板 / 工业气体 / 玻璃基板 / 纳米银
候选 10 只（扫描 98 只 → 护栏剔除 29 → 保留 39 → 取前 10）
  兴业科技 / 航天工程 / 超声电子 / 艾华集团 / 长信科技 ...
```
每只都带：均线多头 + 突破平台 + 放量 + RPS相对强势 + 主力资金流入 的入选理由。

## 工具命令

| 命令 | 作用 |
|------|------|
| `python -m engine.cli rps-build` | **预计算全市场真实RPS**（盘后跑一次，1-2分钟） |
| `python -m engine.cli sentiment` | 当日情绪温度（增强版含市场结构/广度/历史分位） |
| `python -m engine.cli scan` | 完整选股链路 → JSON（含买卖点） |
| `python -m engine.cli report` | 链路 → Markdown 简报（存盘） |
| `pytest -k "not live"` | 纯逻辑单测（38 个，秒级） |
| `pytest` | 含真实数据冒烟测试（需网络） |
| `python -m engine.agent.cli "问题"` | 独立 Agent 综合选股（命令行） |
| `python -m engine.repl` → 菜单 3 | 交互式 Agent |

## 配置调参

所有策略阈值在 `config/settings.yaml`，改文件即可，不动代码：
- 情绪各分项权重与锚点
- 趋势：均线周期/突破回看/量比/RPS下限/市值区间/资金连续流入天数
- 护栏：PE/PB/负债率上限、是否剔ST/次新

## 选股方法论（写在 `engine/agent/prompts.py`，LLM 必须遵守）

1. **情绪定调**：永远先看情绪温度。冰点不选股，亢奋提示追高风险。
2. **趋势选股**：均线多头 + 突破 + 放量 + RPS强势 + 主力资金流入。
3. **新闻催化**：结合 capitalise-finnews 简报，有题材共振的优先。
4. **量化排雷**：候选已过 ST/估值/财务护栏，仍要警惕追高接力。
5. **风控前置**：每只必给具体买点、止损位、目标位（盈亏比≥2:1）。

## 关键技术决策

- **数据层**：akshare 单库无 token 5/5 覆盖（含涨停板全家桶，akshare 独家）。`engine/data_loader.py` 带重试+缓存+降级。
- **qlib**：Phase 1 不接（你说量化只做辅助，规则化护栏够用），留扩展点。
- **情绪源**：纯量化指标起步（涨停/资金/板块），社交舆情留 Phase 3。
- **Agent**：独立 Python Agent，直接调用 engine Python API 作为 LLM 工具，不依赖 pi。
- **已知坑点**（已在代码处理）：北向资金 `stock_hsgt_*` 自 2024-08 失效，改用主力资金流；个股资金流对坏代码会抛错，已加校验。

## 项目结构

```
D:/交易智能体/
├── engine/           选股引擎（Python）
│   ├── data_loader.py      akshare 取数（重试/缓存/降级/财务内存缓存）
│   ├── sentiment_meter.py  情绪温度计（基础版）
│   ├── market_structure.py 增强情绪+市场结构（含历史分位/广度/见顶信号）
│   ├── trend_scanner.py    趋势选股（真实RPS查表）
│   ├── quant_guard.py      量化护栏（PE/PB/财务/ST/次新）
│   ├── entry_exit.py       买卖点引擎（ATR止损/盈亏比止盈/1%风险仓位）
│   ├── rps.py              全市场20日真实RPS预计算（根治失真）
│   ├── pipeline.py         主链路串联
│   ├── report.py           Markdown 简报
│   ├── cli.py              命令行
│   └── tests/              38 个逻辑测试 + 冒烟测试
├── engine/agent/     独立 Agent（LLM client + 工具循环 + prompts）
├── agent/            旧 pi Extension（已弃用，见 agent/legacy/）
├── config/settings.yaml   所有阈值
├── skills/capitalise-finnews/   新闻简报技能（已装）
├── data/             报告 + 缓存
└── _refs/            参考项目（pi, TradingAgents-CN）
```

## 路线图

- **Phase 1（已完成）**：后端主链路 ✅ — engine + 独立 Agent，139 测试，真实数据验证。
- **Phase 1.5（已完成）**：深度审计修复 ✅ — kimi 协同审计 4 路 + 主控根治：
  - 真实 RPS（5507只，根治0.19相关失真）✓
  - 买卖点引擎（ATR止损/盈亏比/仓位）✓
  - 增强情绪（市场结构/广度/见顶信号）✓
  - 修复 7 个 bug（涨停解析/find_col/资金流NaN/PE死分支等）✓
  - LLM 完整闭环跑通（独立 Agent + deep_pick → 专业选股报告）✓
- **Phase 2**：React+Vite+UnoCSS 仪表盘（KLineCharts K线 + ECharts 情绪热力图），gateway 长驻服务。
- **Phase 3**：社交舆情（股吧/微博散户情绪）接入，情绪刻画更立体。

## 验证状态（全部实测通过）

- ✅ 139 个纯逻辑单测全过（pytest -k "not live"）
- ✅ 真实 RPS：全市场 5507 只，64.8s 计算，均值50.0分布正确
- ✅ 完整选股链路：真实RPS+增强情绪+趋势+护栏+买卖点，选出10只带交易计划
- ✅ LLM 闭环：独立 Agent 调 deep_pick → 输出专业选股报告（含买卖点/止损/排除说明）
- ✅ 独立 Agent：不依赖 pi，支持 OpenAI 兼容接口（DeepSeek/Qwen/GLM/豆包/Kimi/OpenAI）

---

*盘古 Pangu · 用情绪和趋势，在 A 股短线里找到该出手的时刻。*
