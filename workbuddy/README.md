# WorkBuddy（CodeBuddy IDE）适配

Pangu 的业务能力全部来自 pangu MCP server（stdio）。WorkBuddy 是 IDE 形态，
MCP 通过 IDE 内置面板注册（无命令行自动化入口），因此本目录提供**可直接
导入的配置**与操作步骤。

## 接入步骤

1. 用 WorkBuddy 打开 Pangu 仓库根目录；
2. 打开 MCP 面板（扩展/工具 → MCP → 添加服务器），类型选 **stdio**；
3. 按下表填写（或直接粘贴 `pangu.mcp.json` 对应字段）：

| 字段 | 值 |
|---|---|
| 名称 | pangu |
| 类型 | stdio |
| 命令 | `.venv/Scripts/python.exe`（建议改为绝对路径）|
| 参数 | `-m engine.mcp.server` |
| 工作目录 | Pangu 仓库根目录（绝对路径）|

4. 启用后即可在 Craft/对话中调用 `pangu_recommend`、`pangu_analyze_stock` 等 7 个工具。

## `/pangu` 快捷指令

将 `.zcode-plugin/commands/pangu.md` 的内容粘贴到 WorkBuddy 的自定义提示词/
规则中（CodeBuddy 支持项目规则文件），即可实现等价的 `/pangu` 工作流：
空输入 → Top20；代码 → 单票分析；status → 数据源状态。

## 实测状态（诚实）

- ✅ MCP server 协议级验证通过（stdio JSON-RPC，见 `engine/tests/test_p3_mcp.py`）
- ⚠️ WorkBuddy IDE 内端到端调用**未实测**：IDE 的 MCP 注册存储于内部状态库
  （无文件级/CLI 级自动化入口），无法由本仓库自动完成；按上表手动配置后
  即与已验证的同一 server 通信。
