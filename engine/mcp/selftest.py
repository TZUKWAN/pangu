"""插件自测器：扮演宿主 Agent，按各适配清单逐个拉起 pangu MCP server 验证。

不启动 KIMI / WorkBuddy / Claude / Codex 等任何外部 AGENT——只按各清单里
写的 command/args 直接启动 Pangu 自己的 MCP server（stdio），完成
initialize → tools/list → tools/call 全流程。清单能跑通 = 宿主按同一清单
配置即可用。

用法：python -m engine.mcp.selftest [--fast]
  --fast: tools/call 用 pangu_run_get（无实时源依赖，秒级）
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def _manifests() -> list[tuple[str, dict]]:
    """(宿主名, {command, args, cwd}) 列表 —— 全部从仓库内的适配清单解析。"""
    out = []

    def add(name, cmd, args):
        out.append((name, {"command": cmd, "args": args}))

    def read_json(p: str) -> dict:
        return json.loads((ROOT / p).read_text(encoding="utf-8"))

    try:
        d = read_json(".mcp.json")["mcpServers"]["pangu"]
        add("ClaudeCode(.mcp.json)", d["command"], d.get("args", []))
    except Exception as e:  # noqa: BLE001
        out.append(("ClaudeCode(.mcp.json)", {"_error": repr(e)}))
    try:
        d = read_json("opencode.json")["mcp"]["pangu"]
        add("OpenCode(opencode.json)", d["command"][0], d["command"][1:])
    except Exception as e:  # noqa: BLE001
        out.append(("OpenCode(opencode.json)", {"_error": repr(e)}))
    try:
        d = read_json(".zcode-plugin/plugin.json")["mcpServers"]["pangu"]
        add("ZCode(.zcode-plugin)", d["command"], d.get("args", []))
    except Exception as e:  # noqa: BLE001
        out.append(("ZCode(.zcode-plugin)", {"_error": repr(e)}))
    try:
        d = read_json("workbuddy/pangu.mcp.json")["mcpServers"]["pangu"]
        add("WorkBuddy(workbuddy/)", d["command"], d.get("args", []))
    except Exception as e:  # noqa: BLE001
        out.append(("WorkBuddy(workbuddy/)", {"_error": repr(e)}))
    try:
        import tomllib
        cfg = tomllib.loads((Path.home() / ".kimi-code" / "config.toml")
                            .read_text(encoding="utf-8"))
        d = cfg["mcp_servers"]["pangu"]
        add("KimiCode(~/.kimi-code/config.toml)", d["command"], d.get("args", []))
    except Exception as e:  # noqa: BLE001
        out.append(("KimiCode(config.toml)", {"_error": repr(e)}))
    return out


def _launch(cfg: dict):
    cmd = [cfg["command"], *cfg.get("args", [])]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", cwd=str(ROOT),
                            env={"PYTHONPATH": str(ROOT),
                                 "SYSTEMROOT": r"C:\Windows"})


def _rpc(proc, id_, method, params=None):
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": id_,
                                 "method": method,
                                 "params": params or {}}) + "\n")
    proc.stdin.flush()


class _Session:
    """单 reader 线程收集全部 stdout；按 id 检索响应（响应可能乱序）。"""

    def __init__(self, proc):
        self.proc = proc
        self.lines: list[str] = []
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self):
        for line in self.proc.stdout:
            self.lines.append(line)

    def wait_result(self, id_, timeout=90) -> Optional[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            for line in self.lines:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("id") == id_:
                    return o
            time.sleep(0.05)
        return None


def _wait_result(proc, id_, timeout=90) -> Optional[dict]:
    """兼容旧签名：使用 session 级共享收集器。"""
    sess = getattr(proc, "_pangu_session", None)
    if sess is None:
        sess = _Session(proc)
        proc._pangu_session = sess
    return sess.wait_result(id_, timeout)


def selftest(fast: bool = True, per_timeout: int = 120) -> tuple[int, list[str]]:
    """返回 (通过清单数, 失败明细)。"""
    ok, fails = 0, []
    rpc_id = 0
    for name, cfg in _manifests():
        rpc_id = 0
        try:
            if "_error" in cfg:
                raise RuntimeError(f"manifest 解析失败: {cfg['_error']}")
            proc = _launch(cfg)
            try:
                rpc_id += 1
                _rpc(proc, rpc_id, "initialize",
                     {"protocolVersion": "2024-11-05", "capabilities": {},
                      "clientInfo": {"name": "plugin-selftest", "version": "0"}})
                r = _wait_result(proc, rpc_id, per_timeout)
                if r is None or "error" in r:
                    raise RuntimeError(f"initialize 失败: {r}")
                proc.stdin.write(json.dumps(
                    {"jsonrpc": "2.0",
                     "method": "notifications/initialized"}) + "\n")
                proc.stdin.flush()
                rpc_id += 1
                _rpc(proc, rpc_id, "tools/list")
                r = _wait_result(proc, rpc_id, per_timeout)
                tools = [t["name"] for t in r["result"]["tools"]]
                need = {"pangu_health", "pangu_refresh", "pangu_recommend",
                        "pangu_analyze_stock", "pangu_explain", "pangu_sources",
                        "pangu_run_get"}
                if not need.issubset(set(tools)):
                    raise RuntimeError(f"工具缺失: {need - set(tools)}")
                rpc_id += 1
                call_tool = "pangu_run_get" if fast else "pangu_recommend"
                _rpc(proc, rpc_id, "tools/call",
                     {"name": call_tool,
                      "arguments": {"run_id": "selftest-nonexistent"}
                      if fast else {"limit": 3}})
                r = _wait_result(proc, rpc_id, per_timeout)
                if r is None or "error" in r:
                    raise RuntimeError(f"tools/call 失败: {r}")
            finally:
                try:
                    proc.stdin.close()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    proc.wait(timeout=15)
                except Exception:  # noqa: BLE001
                    proc.kill()
            ok += 1
            print(f"PASS  {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            fails.append(f"{name}: {e!r}"[:220])
            print(f"FAIL  {name}: {e!r}", flush=True)
    return ok, fails


def main() -> int:
    fast = "--fast" in sys.argv or "--full" not in sys.argv
    print(f"插件自测（按清单拉起 pangu MCP，fast={fast}）")
    ok, fails = selftest(fast=fast)
    print(f"\n通过 {ok} 个清单；失败 {len(fails)} 个")
    for f in fails:
        print("  -", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
