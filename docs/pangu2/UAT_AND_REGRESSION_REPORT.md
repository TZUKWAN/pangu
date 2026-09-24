# Pangu 2.0 UAT 覆盖矩阵与最终回归报告（2026-09-25）

> 测试方式：ZCode 内置浏览器（IAB）真实点击/输入/刷新，禁止系统默认浏览器。
> 服务：`python -m engine.web --port 8688`（FastAPI + SPA）。
> 两轮回归；第二轮从首页重新完整走查。

## 1. 覆盖矩阵（两轮合计）

| 模块/功能 | 操作 | 结果 | 备注 |
|---|---|---|---|
| 首页今日视图 | 加载/刷新/KPI/市场总览/板块热力图/排行榜四标签 | ✅ | 真实行情数据渲染 |
| 板块下拉选择器 | 连续切换/选择 | ✅ | 跨境电商等真实板块 |
| 刷新扫描按钮 | 真实触发后台任务并轮询 | ✅（修复后） | 发现并修复 P1 Bug #1 |
| 设置面板 | 打开/输入/保存 | ✅ | "已保存"回执 |
| 执行视图-总览 | 模式徽章 PAPER/合规/通道/对账时间/累计订单 | ✅ | 默认 PAPER，fail-closed |
| 执行视图-策略下拉 | 仅列可执行状态策略 | ✅ | research 池如实显示"禁止下单" |
| 手动纸面下单 | 正常单/空代码/非法代码/超大数量 | ✅ | 校验文案清晰，无崩溃 |
| 订单状态机 | CREATED→PENDING→SUBMITTED→ACK/REJECTED/UNKNOWN | ✅ | 全时间线展示 |
| 市场性保护 | 限价低于当日最低价 → not_marketable_below_low | ✅ | 红队修复生效 |
| 立即对账 | API+UI | ✅ | matched=1, reconcile_ok |
| Kill Switch | UI 双确认熔断→409 拒单→填写原因恢复 | ✅ | 全审计留痕 |
| UNKNOWN 恢复 | 熔断取消未确认单→UNKNOWN→reconcile→解除阻断 | ✅ | Scenario D 闭环 |
| LIVE 准入门 | 切换 LIVE → 409 + failed_checks 4 项 | ✅ | Scenario G/H |
| 策略视图 | 注册表/生命周期/证据链接/非实盘标注 | ✅ | research 全部如实标注 |
| 研究视图 | 实验登记簿 79 条/因子库 29 个/失败可见声明 | ✅ | 失败实验全部可见 |
| 刷新持久化 | F5 后订单/实验仍在 | ✅ | |
| 导航往返 | 今日↔执行↔策略↔研究 循环切换 | ✅ | 无报错 |
| 控制台/网络 | 两轮持续收集 | ✅ | 0 JS 错误，0 5xx |
| 数据质量闸门 | 档案外日期日常链路 → blocked(data_quality) | ✅ | Scenario F（真实数据） |
| 无可执行策略 | 日常链路 → no_executable_strategy | ✅ | 不硬凑交易 |

## 2. 发现并修复的 Bug

| # | 级别 | 问题 | 根因 | 修复 |
|---|---|---|---|---|
| 1 | P1 | Web 扫描任务 100% 失败 `No module named 'engine.web.pipeline_factory'` | server.py 相对导入层级错误（`from .pipeline_factory` 应为 `from ..pipeline_factory`），既有 web 测试未覆盖该惰性导入路径 | a7acae1 修复导入；修复后端到端扫描成功（Pipeline 8 分钟跑完，候选 100 只，degraded 如实不更新 latest）|
| 2 | P2 | 旧进程持坏代码对象：改源码后旧 uvicorn 仍报错 | 惰性导入在启动时已编译 | 用 netstat 找 PID taskkill 重启（运维项，已记录一键恢复文档要求）|

## 3. 最终全量回归

- `pytest engine/tests -q`：**794 passed / 0 failed**（12 分 18 秒）。
- Web API 抽查：/api/latest、/api/execution/*、/api/strategies、/api/research/experiments、/api/system/data-quality 全 200。
- 端到端扫描：`POST /api/scan` → done（候选 100，degraded 如实降级，正式推荐 0）。

## 4. 诚实的边界（未宣称项）

- 原生同花顺客户端：easytrader 未安装 → 适配器 health() 如实返回不可用；
  本次未做受控实机测试（无真实客户端环境与账户权限）。
- Shadow 模式：依赖真实券商只读连接，同上不可用 → 未产生 Shadow 报告。
- LIVE：合规状态 UNKNOWN + 无达标策略 → 全链路 fail-closed，从未开启。
