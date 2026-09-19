"""Phase 11 Web 执行/策略/研究 API 测试。

镜像 test_web_logic.py 的模式：FastAPI TestClient + tmp sqlite 库 +
stub 报价源，不触任何真实行情/LLM。所有子系统落库路径经环境变量指到
tmp 目录，每个用例前后 reset 单例，避免跨用例串库。

覆盖：
- overview 默认 PAPER、trading_allowed
- LIVE 门禁 fail-closed（Scenario G/H）：409 + failed_checks 非空
- kill switch 阻断下单（409）与恢复
- PAPER 下单经 PaperBrokerAsAdapter → ACKNOWLEDGED/FILLED + timeline
- 策略注册表 7 个研究池可见且标注"未验证/研究"
- 实验登记簿失败实验永远可见 + 因子报告端点
- 组合状态/数据质量优雅降级、设置摘要、今日执行链路条
- 旧端点不回归（另跑 test_web_logic.py 全量）
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

fastapi = pytest.importorskip("fastapi")  # 没装 fastapi 则整体跳过
from fastapi.testclient import TestClient

from engine.contracts import BrokerUnavailable, StrategyStatus
from engine.strategies.gates import PromotionContext
from engine.strategies.manifest import StrategyManifest


# ---------------------------------------------------------------------- #
# 夹具：tmp 库 + stub 报价
# ---------------------------------------------------------------------- #
def _stub_quote(symbol: str, date: str):
    """固定报价：preclose=10，任何 10 元附近限价单都能在纸面撮合成交。"""
    return {"open": 10.0, "high": 10.4, "low": 9.9, "close": 10.0,
            "preclose": 10.0, "is_st": 0, "source": "stub"}


PASSING_METRICS = {
    "oos_profit_factor": 1.5, "oos_sharpe": 1.3, "oos_max_drawdown": 0.08,
    "trade_count": 300, "trade_date_count": 150, "cost2x_positive": 1,
    "parameter_stable": 1, "leakage_audit_passed": 1,
    "independent_backtest_consistent": 1, "oos_net_expectancy": 0.02,
}


@pytest.fixture()
def exec_env(monkeypatch, tmp_path):
    from engine.web import execution_api as exapi

    paths = {
        "PANGU_EXECUTION_DB": str(tmp_path / "execution.db"),
        "PANGU_PAPER_BROKER_DB": str(tmp_path / "paper_broker.db"),
        "PANGU_COMPLIANCE_DB": str(tmp_path / "compliance.db"),
        "PANGU_COMPLIANCE_AUDIT_DIR": str(tmp_path / "audit"),
        "PANGU_PORTFOLIO_DB": str(tmp_path / "portfolio_state.db"),
        "PANGU_EXPERIMENTS_REGISTRY": str(tmp_path / "experiments" / "registry.jsonl"),
        "PANGU_FACTOR_INDEX": str(tmp_path / "experiments" / "factor_index.jsonl"),
        "PANGU_FACTORS_DIR": str(tmp_path / "experiments" / "factors"),
        "PANGU_PIT_DB": str(tmp_path / "missing_pit.sqlite3"),
    }
    for key, value in paths.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(exapi, "_quote_provider_factory", lambda: _stub_quote)
    exapi.reset_execution_singletons()
    yield exapi
    exapi.reset_execution_singletons()


@pytest.fixture()
def client(exec_env):
    from engine.web import server
    return TestClient(server.app)


# ---------------------------------------------------------------------- #
# 种子工具
# ---------------------------------------------------------------------- #
def seed_research_pools(exapi, n: int = 7):
    """注册 n 个 research 状态策略（模拟 seed_registry 的七大池）。"""
    registry = exapi.get_registry()
    for i in range(n):
        registry.register(StrategyManifest(
            strategy_id=f"pool_{i}", version="1.0", code_sha="a" * 64,
            data_snapshot="replay_test", approval_state=StrategyStatus.RESEARCH,
            notes="人工评分池，未通过独立样本外验证，禁止实盘"))
    return registry


def seed_executable_strategy(exapi, strategy_id: str = "paper_strat"):
    """注册一个 validated 策略并过闸升到 paper（可执行）。"""
    registry = exapi.get_registry()
    registry.register(StrategyManifest(
        strategy_id=strategy_id, version="1.0", code_sha="b" * 64,
        data_snapshot="replay_test", approval_state=StrategyStatus.VALIDATED,
        metrics=dict(PASSING_METRICS)))
    ctx = PromotionContext(
        validation_metrics=dict(PASSING_METRICS),
        paper_report_ref="paper_report.json", shadow_report_ref="shadow_report.json",
        compliance_ok=True, broker_ok=True, reconcile_ok=True, kill_switch_ready=True)
    return registry.transition(strategy_id, "paper", operator="test",
                               reason="test seeding", checks=ctx)


def seed_failed_experiment(exapi):
    from engine.validation.experiment_registry import ExperimentRegistry
    reg = ExperimentRegistry(os.environ["PANGU_EXPERIMENTS_REGISTRY"])
    reg.register({
        "experiment_id": "EXP-001", "hypothesis": "动量因子在样本外有超额",
        "economic_rationale": "动量溢价的持续", "data": "pit_daily",
        "pit_status": "ok", "universe": "hs300", "decision_time": "15:00",
        "execution_time": "open", "features": ["mom9"], "label": "ret_5d",
        "train_range": "2020-2022", "validation_range": "2023", "test_range": "2024",
        "costs": "2x", "slippage": "10bps", "baseline": "buy_hold",
        "parameters": {}, "optimization_method": "none", "n_variants_tried": 1,
        "metrics": {"oos_profit_factor": 0.8, "oos_sharpe": -0.2},
        "leakage_audit": "passed", "independent_backtest": "consistent",
        "conclusion": "OOS 亏损，假设不成立，已放弃",
        "status": "running",
    })
    reg.mark_failed("EXP-001", "oos_drawdown_breach")


def seed_factor_index(exapi):
    idx = os.environ["PANGU_FACTOR_INDEX"]
    os.makedirs(os.path.dirname(idx), exist_ok=True)
    with open(idx, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": "mom9", "version": "1.0", "family": "momentum",
                             "horizons": ["5d"], "status": "evaluated",
                             "ts": "2026-09-19T10:00:00+08:00"}) + "\n")


# ---------------------------------------------------------------------- #
# 总览 / 模式 / 门禁
# ---------------------------------------------------------------------- #
def test_overview_defaults_to_paper_and_allows_trading(client, exec_env):
    r = client.get("/api/execution/overview")
    assert r.status_code == 200
    d = r.json()
    assert d["execution_mode"] == "PAPER"
    assert d["trading_allowed"] is True
    assert d["kill_switch"] is False
    assert d["manual_intervention_required"] is False
    assert d["last_reconcile"] is None
    assert d["compliance_state"] == "UNKNOWN"
    assert isinstance(d["adapter_health"], dict)
    assert d["adapter_health"]["ok"] is True  # PaperBroker 已连接


def test_switch_to_live_refused_with_failed_checks(client, exec_env):
    seed_research_pools(exec_env, n=7)
    r = client.post("/api/execution/mode", json={"mode": "LIVE"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert isinstance(detail["failed_checks"], list) and detail["failed_checks"]
    # Scenario G/H：拒绝后模式保持 PAPER
    assert client.get("/api/execution/overview").json()["execution_mode"] == "PAPER"
    # 研究池不能触发 live-capable 放行
    assert "strategy_status_not_live_capable:research" in detail["failed_checks"] or \
        any(s.startswith("strategy_status_not_live_capable") for s in detail["failed_checks"])
    # 拒绝也要留审计痕
    with sqlite3.connect(os.environ["PANGU_EXECUTION_DB"]) as conn:
        rows = conn.execute(
            "SELECT status, detail FROM order_events WHERE kind='execution_mode'").fetchall()
    assert rows and rows[-1][0] == "LIVE_REFUSED" and "LIVE refused" in rows[-1][1]


def test_mode_validates_against_enum(client, exec_env):
    r = client.post("/api/execution/mode", json={"mode": "MOON"})
    assert r.status_code == 400
    assert "未知执行模式" in r.json()["detail"]


def test_mode_switch_to_shadow_is_audited(client, exec_env):
    r = client.post("/api/execution/mode", json={"mode": "SHADOW", "reason": "影子盘观察"})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True and d["execution_mode"] == "SHADOW" and d["previous"] == "PAPER"
    assert client.get("/api/execution/overview").json()["execution_mode"] == "SHADOW"
    # 审计必须落库（order_events, kind=execution_mode）
    with sqlite3.connect(os.environ["PANGU_EXECUTION_DB"]) as conn:
        rows = conn.execute(
            "SELECT kind, status, detail FROM order_events WHERE kind='execution_mode'").fetchall()
    assert rows and rows[-1][1] == "SHADOW" and "PAPER -> SHADOW" in rows[-1][2]


# ---------------------------------------------------------------------- #
# 熔断 / 下单 / 订单列表
# ---------------------------------------------------------------------- #
def test_kill_switch_requires_reason_blocks_orders_then_recovers(client, exec_env):
    seed_executable_strategy(exec_env)
    # 无原因拒绝（审计要求）
    r = client.post("/api/execution/kill-switch", json={"action": "engage"})
    assert r.status_code == 400
    r = client.post("/api/execution/kill-switch", json={"action": "explode", "reason": "x"})
    assert r.status_code == 400

    r = client.post("/api/execution/kill-switch",
                    json={"action": "engage", "reason": "盘中异常波动"})
    assert r.status_code == 200
    assert r.json()["trading_allowed"] is False

    # 熔断期间下单 → 409
    r = client.post("/api/execution/orders", json={
        "strategy_id": "paper_strat", "symbol": "600000", "side": "BUY",
        "qty": 100, "limit_price": 10.5})
    assert r.status_code == 409
    assert "kill_switch" in r.json()["detail"]["blocked_by"]

    r = client.post("/api/execution/kill-switch",
                    json={"action": "disengage", "reason": "人工核查完毕"})
    assert r.status_code == 200
    assert r.json()["trading_allowed"] is True


def test_place_order_paper_fills_via_paper_adapter(client, exec_env):
    seed_executable_strategy(exec_env)
    r = client.post("/api/execution/orders", json={
        "strategy_id": "paper_strat", "symbol": "600000", "side": "BUY",
        "qty": 200, "limit_price": 10.5})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True and d["submitted"] is True
    order = d["order"]
    # click-success != success：OMS 状态机到 ACKNOWLEDGED（纸面撮合 FILLED 后 broker 确认）
    assert order["status"] in ("ACKNOWLEDGED", "FILLED")
    assert str(order["broker_order_id"]).startswith("PAPER-")
    timeline_statuses = [e["status"] for e in order["timeline"]]
    assert "CREATED" in timeline_statuses and "SUBMITTED" in timeline_statuses
    assert "ACKNOWLEDGED" in timeline_statuses


def test_restage_same_decision_is_rejected_not_duplicated(client, exec_env):
    seed_executable_strategy(exec_env)
    payload = {"strategy_id": "paper_strat", "symbol": "600000", "side": "BUY",
               "qty": 100, "limit_price": 10.2, "decision_id": "DEC-001"}
    r1 = client.post("/api/execution/orders", json=payload)
    assert r1.status_code == 200
    r2 = client.post("/api/execution/orders", json=payload)
    # 已提交订单不允许重复 submit（NeedsReconciliation 防线），绝不产生第二笔
    assert r2.status_code == 409
    orders = client.get("/api/execution/orders").json()["orders"]
    assert len([o for o in orders if o["decision_id"] == "DEC-001"]) == 1


def test_order_rejects_unregistered_and_research_strategies(client, exec_env):
    seed_research_pools(exec_env, n=2)
    payload = {"symbol": "600000", "side": "BUY", "qty": 100, "limit_price": 10.5}
    r = client.post("/api/execution/orders", json={**payload, "strategy_id": "ghost"})
    assert r.status_code == 400
    r = client.post("/api/execution/orders", json={**payload, "strategy_id": "pool_0"})
    assert r.status_code == 403
    assert "不可执行下单" in r.json()["detail"]


def test_order_validates_payload(client, exec_env):
    seed_executable_strategy(exec_env)
    base = {"strategy_id": "paper_strat", "qty": 100, "limit_price": 10.5}
    assert client.post("/api/execution/orders", json={**base, "symbol": "", "side": "BUY"}).status_code == 400
    assert client.post("/api/execution/orders", json={**base, "symbol": "600000", "side": "UP"}).status_code == 400
    assert client.post("/api/execution/orders", json={**base, "symbol": "600000", "side": "BUY", "qty": 0}).status_code == 400
    assert client.post("/api/execution/orders", json={**base, "symbol": "600000", "side": "BUY", "limit_price": 0}).status_code == 400


def test_orders_list_newest_first_with_timeline(client, exec_env):
    seed_executable_strategy(exec_env)
    for i, dec in enumerate(("DEC-A", "DEC-B")):
        r = client.post("/api/execution/orders", json={
            "strategy_id": "paper_strat", "symbol": f"00000{i+1}", "side": "BUY",
            "qty": 100, "limit_price": 10.5, "decision_id": dec})
        assert r.status_code == 200
    d = client.get("/api/execution/orders?limit=10").json()
    assert d["count"] == 2
    ids = [o["decision_id"] for o in d["orders"]]
    assert ids == ["DEC-B", "DEC-A"]  # newest first
    for o in d["orders"]:
        assert isinstance(o["timeline"], list) and o["timeline"]


def test_reconcile_endpoint_returns_report(client, exec_env):
    seed_executable_strategy(exec_env)
    r = client.post("/api/execution/orders", json={
        "strategy_id": "paper_strat", "symbol": "600000", "side": "BUY",
        "qty": 100, "limit_price": 10.5})
    assert r.status_code == 200
    r = client.post("/api/execution/reconcile")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    report = d["report"]
    assert report["matched"] == 1 and report["reconcile_ok"] is True
    overview = client.get("/api/execution/overview").json()
    assert overview["last_reconcile"] is not None
    assert overview["last_reconcile"]["matched"] == 1


# ---------------------------------------------------------------------- #
# 策略注册表
# ---------------------------------------------------------------------- #
def test_strategies_lists_seeded_pools_as_research(client, exec_env):
    seed_research_pools(exec_env, n=7)
    d = client.get("/api/strategies").json()
    assert d["count"] == 7
    for s in d["strategies"]:
        assert s["status"] == "research"
        assert s["research_only"] is True and s["validated"] is False
        assert s["executable"] is False
        assert "未验证" in s["advice"] or "研究" in s["advice"]
        assert "实盘" not in s["advice"].split("——")[0] or "不构成实盘建议" in s["advice"]


def test_strategies_live_capable_only_for_live_statuses(client, exec_env):
    seed_executable_strategy(exec_env, "paper_only_strat")
    d = client.get("/api/strategies").json()
    row = next(s for s in d["strategies"] if s["strategy_id"] == "paper_only_strat")
    assert row["status"] == "paper" and row["executable"] is True
    assert row["live_capable"] is False  # paper 阶段绝不显示可实盘


# ---------------------------------------------------------------------- #
# 研究登记簿 / 因子
# ---------------------------------------------------------------------- #
def test_research_endpoint_returns_failed_experiments(client, exec_env):
    seed_failed_experiment(exec_env)
    seed_factor_index(exec_env)
    d = client.get("/api/research/experiments").json()
    assert d["total"] == 1
    assert d["failed_count"] == 1  # 失败实验不能隐藏
    exp = d["experiments"][0]
    assert exp["status"] == "failed"
    assert exp["status_update_reason"] == "oos_drawdown_breach"
    assert exp["metrics"]["oos_profit_factor"] == 0.8
    assert d["factors"][0]["name"] == "mom9"


def test_research_endpoint_graceful_when_registry_missing(client, exec_env):
    d = client.get("/api/research/experiments").json()
    assert d["total"] == 0 and d["experiments"] == []
    assert d["registry_exists"] is False
    assert any("因子索引不存在" in w for w in d["warnings"])


def test_factor_report_endpoint(client, exec_env):
    factors_dir = os.environ["PANGU_FACTORS_DIR"]
    os.makedirs(factors_dir, exist_ok=True)
    with open(os.path.join(factors_dir, "mom9.json"), "w", encoding="utf-8") as fh:
        json.dump({"name": "mom9", "family": "momentum", "ic_rank_mean": 0.02}, fh)
    r = client.get("/api/research/factors/mom9")
    assert r.status_code == 200
    assert r.json()["ic_rank_mean"] == 0.02
    assert client.get("/api/research/factors/ghost9").status_code == 404
    assert client.get("/api/research/factors/..%5Cevil").status_code in (400, 404)


# ---------------------------------------------------------------------- #
# 组合 / 数据质量 / 设置 / 链路条
# ---------------------------------------------------------------------- #
def test_portfolio_latest_graceful_empty_then_plan(client, exec_env):
    r = client.get("/api/portfolio/latest")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "empty" and d["plan"] is None

    from engine.portfolio_engine.constructor import PortfolioPlan
    from engine.portfolio_engine.state import PortfolioState
    state = PortfolioState(db_path=os.environ["PANGU_PORTFOLIO_DB"])
    plan = PortfolioPlan(decision_date="2026-09-19", cash_weight=0.4,
                         allocations=[{"symbol": "600000", "weight": 0.6}])
    state.record_plan(plan)

    d = client.get("/api/portfolio/latest").json()
    assert d["status"] == "ok"
    assert d["plan"]["cash_weight"] == 0.4
    assert d["plan"]["allocations"][0]["symbol"] == "600000"


def test_data_quality_unavailable_without_pit_archive(client, exec_env):
    r = client.get("/api/system/data-quality")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "unavailable"
    assert "PIT 档案不存在" in d["reason"]


def test_settings_includes_execution_summary(client, exec_env):
    r = client.get("/api/settings")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["execution"]["execution_mode"] == "PAPER"
    assert d["execution"]["rate_limits"]["max_per_day"] >= 1
    assert d["execution"]["editable"] is False


def test_pipeline_strip_tracks_order_stages(client, exec_env, monkeypatch):
    from engine.web import server
    report = {
        "date": "20260919",
        "final_recommendations": [{"code": "600000", "name": "测试股"}],
    }
    monkeypatch.setattr(server, "_latest_result", report)
    monkeypatch.setattr(server, "_find_latest_report", lambda: None)

    d = client.get("/api/execution/pipeline-strip").json()
    assert d["count"] == 1
    item = d["items"][0]
    assert item["signal"] is True
    assert item["plan_staged"] is False and item["filled"] is False

    seed_executable_strategy(exec_env)
    r = client.post("/api/execution/orders", json={
        "strategy_id": "paper_strat", "symbol": "600000", "side": "BUY",
        "qty": 100, "limit_price": 10.5})
    assert r.status_code == 200

    item = client.get("/api/execution/pipeline-strip").json()["items"][0]
    assert item["plan_staged"] is True
    assert item["order_submitted"] is True
    assert item["acknowledged"] is True
    assert item["filled"] is False  # OMS 确认后需对账才置 FILLED
    assert item["client_order_id"]


def test_pipeline_strip_empty_without_report(client, exec_env, monkeypatch):
    from engine.web import server
    monkeypatch.setattr(server, "_latest_result", None)
    monkeypatch.setattr(server, "_find_latest_report", lambda: None)
    d = client.get("/api/execution/pipeline-strip").json()
    assert d["items"] == [] and d["count"] == 0 and d["has_report"] is False


def test_overview_reflects_order_count_and_manual_intervention(client, exec_env, monkeypatch):
    seed_executable_strategy(exec_env)
    # 制造一次 UNKNOWN：适配器 submit 抛 BrokerError 后需人工介入
    from engine.web import execution_api as exapi
    oms = exapi.get_oms()
    order = oms.stage_order("paper_strat", "DEC-UNKNOWN", "600000", "BUY", 100, 10.5)

    class BoomAdapter:
        def get_balance(self):
            raise BrokerUnavailable("down")

        def get_positions(self):
            raise BrokerUnavailable("down")

        def submit_order(self, o):
            raise BrokerUnavailable("circuit broken")

    # 直接通过 OMS 内部路径模拟：submit 前临时换适配器
    real_adapter = oms.adapter
    oms.adapter = BoomAdapter()
    try:
        # trading_allowed 先为 True；submit 因 BrokerError→UNKNOWN
        submitted = oms.submit(order.client_order_id)
    finally:
        oms.adapter = real_adapter
    assert submitted.status.value == "UNKNOWN"

    overview = client.get("/api/execution/overview").json()
    assert overview["manual_intervention_required"] is True
    assert overview["trading_allowed"] is False
    assert overview["order_count"] == 1
    # 阻断期间 API 下单同样 409
    r = client.post("/api/execution/orders", json={
        "strategy_id": "paper_strat", "symbol": "000001", "side": "BUY",
        "qty": 100, "limit_price": 10.5})
    assert r.status_code == 409
    assert "manual_intervention" in r.json()["detail"]["blocked_by"]
