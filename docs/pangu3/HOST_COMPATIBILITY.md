# 宿主兼容矩阵（Pangu 3.0 / Phase 9）

> 实测时间：2026-09-26。原则：MCP server 只有一份（`engine/mcp/server.py`）；
> 各宿主仅薄适配。禁止伪造"已实测"——下表 Live tested 列如实标注。

| Host | 二进制版本 | MCP | Command/Skill 适配文件 | 触发方式 | Live tested |
|---|---|---|---|---|---|
| **Claude Code** | 2.1.233 | ✅ 项目级 `.mcp.json` 自动注册 | `.claude/commands/pangu.md` | `/pangu` | ✅ 实测通过：`claude -p --allowedTools mcp__pangu__pangu_health` 真实调用返回 `data_status`（无头模式需 allowedTools；交互模式走权限弹窗） |
| **OpenCode** | 1.18.23 | ✅ `opencode.json`（type: local） | `.opencode/command/pangu.md`（subagent: false） | `/pangu` | ⚠️ 部分：MCP 已注册，但本机 openrouter 模型路由不支持 tool use（"No endpoints found that support tool use"），未达 MCP 调用阶段；属宿主模型路由问题，与本仓库无关 |
| **Codex CLI** | 0.154.0 | ⚠️ 需写入用户级 `~/.codex/config.toml`（`[mcp_servers.pangu]`，说明见 `.codex-plugin/plugin.json`） | `.codex-plugin/prompts/pangu.md` + `skills/pangu/SKILL.md` | `/prompts:pangu` 或 `$pangu`（随版本而异） | ❌ 未实测：本机 codex 登录态失效（401 refresh token revoked），无法执行；注册说明已提供，未替用户改全局配置 |
| **Kimi Code** | 0.39.1 | ⚠️ MCP 注册路径以 kimi 当前版本为准（提供 `.kimi/commands/pangu.md`） | `.kimi/commands/pangu.md` | `/pangu` | ⚠️ 部分：二进制可运行（会话正常创建）；非交互 `-p` 未达 MCP 工具调用阶段，未获真实返回，不宣称通过 |
| **ZCode** | 本会话宿主 | ✅ `.zcode-plugin/plugin.json`（mcpServers 注册） | `.zcode-plugin/commands/pangu.md` + `skills/pangu/SKILL.md` | `/pangu` | ✅ 协议级实测：stdio JSON-RPC initialize/tools list/call/resources/prompts 全通过（engine/tests/test_p3_mcp.py） |

## MCP 协议级测试（与宿主无关，全部通过）

- stdio transport：initialize → notifications/initialized → tools/list → tools/call
- 7 个工具：pangu_health / pangu_refresh / pangu_recommend / pangu_analyze_stock /
  pangu_explain / pangu_sources / pangu_run_get
- 3 个 resource：pangu://recommendation/latest、pangu://sources/health、pangu://methodology
- 1 个 prompt：pangu（参数 limit，arguments 按规范为字符串）
- 异常路径：malformed arguments → JSON-RPC error（-32602 / isError）

## 未宣称事项

- OpenCode/Kimi/Codex 的端到端 `/pangu` 实机调用未全部完成，原因如上（模型路由/
  登录态/MCP 注册路径），均已提供配置文件与说明；
- 子智能体策略：所有适配文件均显式 `subagent: false` 或等价声明；ZCode 插件
  不含 `agents/` 组件；OpenCode 命令标记当前会话执行。
