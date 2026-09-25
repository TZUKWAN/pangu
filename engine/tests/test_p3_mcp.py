"""MCP server 协议级测试（stdio JSON-RPC，注入 Fake service）。"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from engine.decision.contracts import DecisionAction, MarketStatus
from engine.decision.contracts import (DecisionRequest, DecisionRun,
                                       StockDecision, Top20DecisionSet)

ROOT = str(Path(__file__).parent.parent.parent)


class FakeService:
    def status(self):
        return {"asof": "2026-08-13T15:05:00+08:00",
                "decision_date": "2026-08-13", "execution_date": "2026-08-14",
                "market_status": "closed_after", "data_status": "ok",
                "sources": {"realtime_quote": {"quality": "ok", "stale": False}},
                "pit_archive": {"max_date": "2026-08-13"},
                "mcp": {"available": True}}

    def recommend_next_session(self, asof=None, limit=20, force_refresh=False,
                               persist=True):
        dec = StockDecision(rank=1, code="600519", name="贵州茅台",
                            decision=DecisionAction.BUY, score=70,
                            confidence=0.7,
                            confidence_type="evidence",
                            execution_date="2026-08-14",
                            entry_condition="x")
        tset = Top20DecisionSet(asof="a", execution_date="2026-08-14",
                                decisions=[dec], market_status=MarketStatus.CLOSED_AFTER)
        return DecisionRun(
            run_id="pangu-20260814-test0001",
            query_timestamp="2026-08-13T15:05:00+08:00",
            decision_date="2026-08-13", execution_date="2026-08-14",
            asof_timestamp="2026-08-13T15:05:00+08:00",
            market_status=MarketStatus.CLOSED_AFTER,
            recommendations=tset, model_version="pangu_ranker_v1")

    def analyze_stock(self, code, asof=None):
        return self.recommend_next_session(limit=1)

    def explain(self, run_id, code):
        return None

    def runstore(self):
        class _RS:
            def load(self, rid):
                return None

            def latest(self, execution_date=None):
                return None
        return _RS()


def _mk_rpc(id_, method, params=None):
    return json.dumps({"jsonrpc": "2.0", "id": id_, "method": method,
                       "params": params or {}})


NOTIF_INITIALIZED = json.dumps({"jsonrpc": "2.0",
                                "method": "notifications/initialized"})


class ProcResult:
    def __init__(self, lines, stderr):
        self.stdout = "\n".join(lines)
        self.stderr = stderr


def _rpc_round(shim: Path, messages, timeout=90):
    """写全部请求；后台线程收集 stdout（响应可能乱序）；
    收齐全部 id 响应或超时后关闭。"""
    py = sys.executable
    proc = subprocess.Popen([py, str(shim)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace",
                            cwd=ROOT,
                            env={"PYTHONPATH": ROOT, "SYSTEMROOT": r"C:\Windows"})
    lines = []
    n_ids = sum(1 for m in messages if '"id"' in m)

    def reader():
        for line in proc.stdout:
            lines.append(line)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        for msg in messages:
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            got = sum(1 for l in lines if l.strip().startswith("{"))
            if got >= n_ids:
                break
            time.sleep(0.05)
        proc.stdin.close()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
    return ProcResult(lines, proc.stderr.read() if proc.stderr else "")


def _extract(proc, id_):
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("id") == id_:
            return obj
    return None


@pytest.fixture()
def server_env():
    shim = Path(ROOT) / "tmp" / "mcp_shim.py"
    shim.parent.mkdir(exist_ok=True)
    shim.write_text(
        "import sys\n"
        "sys.path.insert(0, r'%s')\n"
        "sys.path.insert(0, r'%s')\n"
        "from test_p3_mcp import FakeService\n"
        "import engine.mcp.server as srv\n"
        "srv.set_service(FakeService())\n"
        "srv.main()\n" % (str(Path(__file__).parent), str(Path(__file__).parent)),
        encoding="utf-8")
    return shim


def _init_and(msgs_after):
    return [_mk_rpc(1, "initialize", {"protocolVersion": "2024-11-05",
                                      "capabilities": {},
                                      "clientInfo": {"name": "t", "version": "0"}}),
            NOTIF_INITIALIZED] + msgs_after


class TestMcpStdio:
    def test_initialize_tools_list_and_call(self, server_env):
        proc = _rpc_round(server_env, _init_and([
            _mk_rpc(2, "tools/list"),
            _mk_rpc(3, "tools/call", {"name": "pangu_recommend",
                                      "arguments": {"limit": "20"}}),
        ]))
        init = _extract(proc, 1)
        assert init is not None and "error" not in init
        tools = _extract(proc, 2)
        names = [t["name"] for t in tools["result"]["tools"]]
        for expected in ("pangu_health", "pangu_refresh", "pangu_recommend",
                         "pangu_analyze_stock", "pangu_explain", "pangu_sources",
                         "pangu_run_get"):
            assert expected in names, f"missing tool {expected}"
        call = _extract(proc, 3)
        assert call is not None and "error" not in call
        content = call["result"]["content"][0]["text"]
        payload = json.loads(content)
        assert payload["recommendations"]["decisions"][0]["code"] == "600519"

    def test_resources_and_prompts(self, server_env):
        proc = _rpc_round(server_env, _init_and([
            _mk_rpc(2, "resources/list"),
            _mk_rpc(3, "resources/read", {"uri": "pangu://methodology"}),
            _mk_rpc(4, "prompts/list"),
            _mk_rpc(5, "prompts/get", {"name": "pangu", "arguments": {"limit": "20"}}),
        ]))
        res = _extract(proc, 2)
        uris = [r["uri"] for r in res["result"]["resources"]]
        assert "pangu://recommendation/latest" in uris
        read = _extract(proc, 3)
        assert "pangu_ranker_v1" in read["result"]["contents"][0]["text"]
        prompts = _extract(proc, 4)
        assert any(p["name"] == "pangu" for p in prompts["result"]["prompts"])
        got = _extract(proc, 5)
        assert got is not None, "prompts/get 未响应"
        assert "Top20" in json.dumps(got["result"], ensure_ascii=False)

    def test_malformed_arguments_return_json_error(self, server_env):
        proc = _rpc_round(server_env, _init_and([
            _mk_rpc(2, "tools/call", {"name": "pangu_recommend",
                                      "arguments": {"limit": "not-an-int"}}),
        ]))
        call = _extract(proc, 2)
        assert call is not None
        assert call.get("error") or "isError" in json.dumps(call)
