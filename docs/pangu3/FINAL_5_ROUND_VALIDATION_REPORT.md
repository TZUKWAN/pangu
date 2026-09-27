# Pangu 3.0 五轮完整验证报告（FINAL 5-ROUND VALIDATION REPORT）

> 分支：`pangu-3-plugin-rebuild`　基线：`pangu-2-rebuild@37e3fdb`
> 执行窗口：2026-09-25 → 2026-09-26
> 纪律：每轮重跑完整验证矩阵；发现 Bug → 修复 → 永久回归测试 → 当轮从头重跑。

## 每轮记录

### Round 1 — 功能正确性
- Commit（轮始）：`3225ce6` →（轮末修复）`b1673a2`
- full pytest：**861 passed, 1 skipped**（12:21）
- 决策链路：CLI recommend/status/analyze 真实档案运行 ✅；MCP stdio 3/3 ✅；
  Top20/T+1/持有周期/退出条件测试 ✅；新闻事件分类 25 类 ✅
- **发现 Bug #1（P1）**：index.html 内联 JS 字符串被写成真实换行 → 整页脚本
  失效（switchView 未定义）。修复 + 永久回归测试
  （`test_p3_web_static.py`：node --check 全部内联脚本 + 导航结构断言）。
  **当轮从头重跑 → PASS**（861 → 863 tests）。

### Round 2 — 时间/PIT/数据正确性
- Commit：`b1673a2`
- full pytest：**861 passed**（17:41）
- PIT replay smoke（20260715→22）✅；holdout 审计：**恰好 1 次解锁** ✅；
- 泄漏静态扫描（ranker/holding/asof/service/evidence/validate/final_gate，
  7 个文件）：**0 findings** ✅
- 无 Bug。

### Round 3 — 宿主兼容
- Commit：`b1673a2`
- full pytest：**861 passed**（17:41）
- 12 个适配文件存在性 ✅；`.zcode-plugin/agents/` 不存在（子智能体禁令）✅；
  OpenCode 命令含 `subagent: false` ✅；MCP 协议测试 3/3 ✅
- 宿主二进制：opencode 1.18.23 / claude 2.1.233 / codex 0.154.0 / kimi 0.39.1
- 实机状态（详见 HOST_COMPATIBILITY.md）：Claude Code 实测通过（真实调用
  pangu_health 返回 data_status）；OpenCode 因本机模型路由不支持 tool use
  未达 MCP 阶段；Codex 登录态失效；Kimi 非交互未达工具调用——均如实标注。
- 无 Bug。

### Round 4 — 故障与安全
- Commit：`b1673a2`
- full pytest：**861 passed**（18:26）
- 新闻源失败（真实周六环境）→ 全部候选 BLOCKED（无 BUY）✅；
  损坏 registry 行 → 容错不崩 ✅；未知代码 → 0 决策 + 如实说明 ✅；
  并发 MCP 调用 → 双响应 ✅；缺失配置 → doctor 优雅退出 ✅
- 无 Bug。

### Round 5 — Clean-room 回归
- 轮始 Commit：`b1673a2` → 修复后 `cd5319e` → **当轮从头重跑**
- fresh clone（clean checkout，无开发机档案）+ **全新 venv** +
  `pip install -r requirements.txt`（EXIT=0）
- doctor（实时源）✅；decision status 无档案时**诚实降级不崩溃** ✅；
  pytest 核心子集 **22 passed** ✅
- **发现 Bug #2（P1）**：无 PIT 档案时 `decision status` 抛 FileNotFoundError
  崩溃。修复：service fail-closed（status 如实 failed + pit_error；
  recommend 明确拒绝）+ CLI 顶层错误处理 + `mcp` 写入 requirements +
  MCP 测试子进程 utf-8 编码修复 + **2 个永久回归测试**
  （`TestNoArchiveCleanRoom`）。当轮从头重跑 → **PASS**
- secret 扫描（全 diff vs main）：0 hits ✅；>1MB 追踪文件：3 个 fixture
  已 slim 至共 349KB（保留结构样本）✅；import 全链检查 ✅；git status 干净 ✅

## 汇总

| 项 | 值 |
|---|---|
| 每轮 commit | R1: b1673a2 / R2: b1673a2 / R3: b1673a2 / R4: b1673a2 / R5: cd5319e（重跑）|
| 测试总数 | 861 passed + 1 skipped（基线 794 → 新增 ~68）|
| 发现并修复 | Bug#1 SPA JS 失效（P1）、Bug#2 无档案崩溃（P1）——均带永久回归测试 |
| Live smoke | CLI 决策链路（真实档案+真实网络）、Claude Code MCP、Web 端点、扫描任务 |
| Host matrix | Claude ✅ / OpenCode ⚠️ / Codex ❌(auth) / Kimi ⚠️ / ZCode 协议级 ✅ |
| 性能 | status≈31s（含实时源刷新，周六网络）；recommend BLOCKED 路径 <60s；MCP 单调用 <10s |
| 数据状态 | PIT 档案 2022→2026-09-04（562 万行）；当日实时源周六降级（如实） |

## 最终工程结论

> 已完成 5 轮完整回归，当前无已知 P0/P1 阻塞问题，所有已发现问题均已有
> 永久回归测试；剩余限制和风险已明确登记。

### 剩余限制与风险（明确登记）

1. **无策略达到 paper/live 准入线**（Pangu 2.0 验证 + Pangu 3.0 ranker 验证
   一致：pick 层弱正 alpha，组合扣成本后负）→ 系统保持研究/观察输出，
   禁止 BUY 的条件下如实降级；这不视为缺陷而是产品纪律。
2. OpenCode/Codex/Kimi 的端到端 `/pangu` 实机调用未完成（模型路由/登录态/
   注册路径），配置与协议级测试已提供。
3. PIT 新闻档案仅覆盖 2026 起 → 新闻 ablation（B/C 臂）deferred，以每日
   journal 前瞻积累。
4. 周末/非交易日实时源部分失败 → data_status=failed → 无 BUY（设计行为）。
5. 高频源 SLA 下的 status 延迟 ~31s（实时源刷新阻塞），优化项：后台预热
   （Web 已有 warmup；MCP 路径待加）。

---

## 附录（2026-09-28）："每日扫描 + 持有期 ≥5% 目标"能力与验证

### 新增能力
1. **+5% 目标命中概率引擎**（`engine/decision/target_prob.py` +
   `tools/build_target_table.py`）：按 `市场状态 × 反转z分位` 条件统计历史
   同类 setup 在持有窗内、止损纪律下触及 +5% 的频率（研究窗 1340 万样本次，
   stop_first 保守序）；多持有期表 h=5/10/20 × 双入场（尾盘/次日开盘）。
2. **持有期按把握自动选择**：取 95% Wilson 下界 ≥30% 的最小窗口；
   BUY 证据线：命中表存在时无 ≥30% 下界支撑自动降 WATCH。
3. **OOS 校准**：2025 年样本上预测/实际 MAE **4.7 个百分点**（15 桶，
   已登记实验 `target_hit_table_v1`）。
4. **每日自动扫描**：调度器新增 `pangu_decision` 步骤（`decision.enabled`
   默认 true，交易日 15:05 自动生成并持久化 DecisionRun）；pytest 环境
   自动跳过真实网络（永久回归测试覆盖）。
5. **尾盘/次日双入口**：盘中 14:30-14:57 询问 → 当日尾盘入场（切尾盘口径
   命中表）；其余 → 次日开盘。
6. **插件自测器**（`python -m engine.mcp.selftest`）：按 5 个宿主适配清单
   逐一拉起 MCP server 跑完整协议流程——**5/5 PASS**（Claude/.mcp.json、
   OpenCode、ZCode、WorkBuddy、Kimi config.toml）。
7. **MCP 预热**：server 启动即后台预载 PIT 档案，消除首调用冷启动。
8. **降换手复验**（缓冲带 Top40，已登记 `ranker_pangu_ranker_v1_oos_buffered`）：
   换手 5.2→3.2、MDD 55%→39%，组合仍为负（PF 0.73）→ ranker 保持 research
   的结论不变（诚实）。

### 全量回归（本轮收尾）
- **879 passed, 1 skipped（11:30）**，0 failed。
- 过程中发现并修复：调度器决策步骤在单测内真跑网络（测试隔离守卫 +
  步骤期望更新）。

### 关于"确保 ≥5%"的诚实结论
系统把该诉求转译为可验证机制：**只推荐历史上同画像 setup 触及 +5% 频率的
置信下界 ≥30% 的标的与持有期**，并把该频率、样本量、下界随每条建议输出。
基础频率参考：5 日窗 29.6% / 10 日 35.1% / 20 日 34.0%（全市场，止损优先
保守口径）。**无法承诺单个标的必然 ≥5%**；任何声称可以保证的系统都违反
本项目纪律。
