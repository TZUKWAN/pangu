"""Pangu 2.0 每日交易循环（daily loop）测试：全部离线，tmp db + 注入式行情。

覆盖场景：
- H 诚实默认：无可执行策略 → no_executable_strategy，orders db 无订单
- PAPER 主路径：validated→paper 的假策略 + 注入 pipeline_result → 下单成交
- F 数据质量阻断：pit_safe=False → blocked / data_quality
- I 风控否决：daily_pnl 超硬限 → risk_decisions 显示 block，无订单
- G/H LIVE + compliance UNKNOWN → live_blocked，failed_checks 含 compliance
- kill switch 拨下 → trading_allowed False → blocked
- SHADOW 模式：只挂单（CREATED）不提交
- scheduler 步骤：默认关闭时步骤列表不变；开启时调用 run_daily_loop
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import engine.daily_loop as daily_loop_mod
from engine.contracts import StrategyStatus
from engine.daily_loop import (
    DailyExecutionLoop,
    ExecutionLoopConfig,
    LoopResult,
    run_daily_loop,
)
from engine.scheduler import DailyScheduler
from engine.strategies.manifest import StrategyManifest

DATE = "20260914"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

class _FakeDQ:
    """DataQualityReport 替身：monkeypatch 到 daily_loop 命名空间。"""

    pit_safe = True

    @classmethod
    def build(cls, store, date, out_dir=None):
        return {"pit_safe": cls.pit_safe, "date": date, "data_completeness": 1.0,
                "freshness": 1.0, "source_count": 4, "missing_ratio": 0.0}


@pytest.fixture()
def paths(tmp_path):
    return {
        "db_path": str(tmp_path / "exec.db"),
        "paper_db_path": str(tmp_path / "paper.db"),
        "portfolio_state_path": str(tmp_path / "state.db"),
        "pit_db_path": str(tmp_path / "missing_pit.sqlite3"),
        "compliance_db_path": str(tmp_path / "compliance.db"),
        "compliance_audit_dir": str(tmp_path / "audit"),
        "output_dir": str(tmp_path / "execution_out"),
    }


def make_cfg(paths, **overrides):
    exec_cfg = {
        "enabled": True,
        "execution_mode": "PAPER",
        "rate_limits": {"max_per_day": 200, "min_interval_seconds": 0},
        **paths,
    }
    exec_cfg.update(overrides)
    return {"execution": exec_cfg}


def stub_quote(symbol, date):
    return {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0,
            "preclose": 10.0, "is_st": False}


PASSING_PAPER_CHECKS = {
    "validation_metrics": {
        "oos_profit_factor": 1.5,
        "oos_sharpe": 1.2,
        "oos_max_drawdown": 0.10,
        "trade_count": 300,
        "trade_date_count": 150,
        "cost2x_positive": True,
        "parameter_stable": True,
        "leakage_audit_passed": True,
        "independent_backtest_consistent": True,
        "oos_net_expectancy": 0.05,
    },
    "paper_report_ref": "test://paper",
    "shadow_report_ref": "test://shadow",
    "compliance_ok": True,
    "broker_ok": True,
    "reconcile_ok": True,
    "kill_switch_ready": True,
}


def register_paper_strategy(db_path: str, strategy_id: str = "fake_alpha") -> None:
    """注册 validated 策略并凭合格证据升入 paper（执行态）。"""
    from engine.strategies.registry import StrategyRegistry

    reg = StrategyRegistry(db_path)
    reg.register(StrategyManifest(
        strategy_id=strategy_id, version="1.0", code_sha="deadbeef",
        data_snapshot="test", approval_state=StrategyStatus.VALIDATED))
    reg.transition(strategy_id, StrategyStatus.PAPER, operator="test",
                   reason="test promotion", checks=dict(PASSING_PAPER_CHECKS))


def make_pipeline_result(strategy_id="fake_alpha", code="600000", score=88.0):
    return {
        "date": "2026-09-14",
        "data_quality": "ok",
        "tradable": True,
        "final_recommendations": [{
            "code": code, "name": "测试股", "strategy": strategy_id,
            "score": score, "gate_status": "final",
        }],
    }


def make_loop(cfg, pipeline_result=None, monkeypatch=None, pit_safe=True):
    if monkeypatch is not None:
        _FakeDQ.pit_safe = pit_safe
        monkeypatch.setattr(daily_loop_mod, "DataQualityReport", _FakeDQ)
    loop = DailyExecutionLoop.from_config(
        cfg, pipeline_result=pipeline_result, quote_provider=stub_quote)
    if monkeypatch is not None:
        monkeypatch.setattr(loop, "_pit_adv",
                            lambda symbols: {s: 20_000_000.0 for s in symbols})
    return loop


def count_orders(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0])


# ---------------------------------------------------------------------------
# Scenario H：无可执行策略（诚实默认）
# ---------------------------------------------------------------------------

def test_no_executable_strategy_places_nothing(paths, tmp_path, monkeypatch):
    """空 registry（仅种子研究态策略）→ no_executable_strategy，orders db 无订单。"""
    cfg = make_cfg(paths)
    pr = make_pipeline_result(strategy_id="趋势回踩")  # 研究态池
    loop = make_loop(cfg, pr, monkeypatch)
    result = loop.run(DATE)

    assert isinstance(result, LoopResult)
    assert result.status == "no_executable_strategy"
    assert result.orders == []
    assert result.decision_date == DATE
    assert count_orders(paths["db_path"]) == 0
    # 诚实注记应说明哪些策略是什么状态
    assert any("趋势回踩" in n for n in result.notes)


# ---------------------------------------------------------------------------
# PAPER 主路径
# ---------------------------------------------------------------------------

def test_paper_mode_full_flow_fills_order(paths, monkeypatch):
    register_paper_strategy(paths["db_path"], "fake_alpha")
    cfg = make_cfg(paths)
    loop = make_loop(cfg, make_pipeline_result("fake_alpha"), monkeypatch)
    result = loop.run(DATE)

    assert result.status == "ok", result.notes
    assert len(result.orders) == 1
    order = result.orders[0]
    assert order["symbol"] == "600000"
    assert order["strategy_id"] == "fake_alpha"
    assert order["status"] in ("ACKNOWLEDGED", "FILLED")
    # PAPER broker 立即成交 + 收盘 reconcile → FILLED
    assert order["status"] == "FILLED"
    assert order["filled_qty"] == order["qty"] > 0
    assert result.reconcile.get("reconcile_ok") is True
    assert result.reconcile.get("matched", 0) >= 1
    # broker 成交记录在 paper db
    with sqlite3.connect(paths["paper_db_path"]) as conn:
        trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert trades == 1
    # 计划与风控决策均已落库
    with sqlite3.connect(paths["portfolio_state_path"]) as conn:
        plans = conn.execute("SELECT COUNT(*) FROM portfolio_plans").fetchone()[0]
        risks = conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0]
    assert plans >= 1 and risks >= 1
    assert result.risk_decisions[0]["approved"] is True
    # LoopResult 可序列化
    json.dumps(result.to_dict(), ensure_ascii=False)


def test_repeated_run_is_idempotent(paths, monkeypatch):
    """同日重跑：确定性 client_order_id 命中已有订单，不会重复下单。"""
    register_paper_strategy(paths["db_path"], "fake_alpha")
    cfg = make_cfg(paths)
    loop = make_loop(cfg, make_pipeline_result("fake_alpha"), monkeypatch)
    first = loop.run(DATE)
    assert first.status == "ok"
    second = loop.run(DATE)
    assert second.status == "ok"
    assert count_orders(paths["db_path"]) == 1
    assert second.orders == []  # 已有订单不会重复 stage


# ---------------------------------------------------------------------------
# Scenario F：数据质量阻断
# ---------------------------------------------------------------------------

def test_data_quality_block(paths, monkeypatch):
    register_paper_strategy(paths["db_path"], "fake_alpha")
    cfg = make_cfg(paths)
    loop = make_loop(cfg, make_pipeline_result("fake_alpha"), monkeypatch, pit_safe=False)
    result = loop.run(DATE)

    assert result.status == "blocked"
    assert result.reason == "data_quality"
    assert result.orders == []
    assert count_orders(paths["db_path"]) == 0


def test_missing_pit_archive_blocks(paths, monkeypatch):
    """PIT 档案不存在 → fail closed（不做数据质量伪装）。"""
    register_paper_strategy(paths["db_path"], "fake_alpha")
    cfg = make_cfg(paths)
    loop = DailyExecutionLoop.from_config(
        cfg, pipeline_result=make_pipeline_result("fake_alpha"),
        quote_provider=stub_quote)
    result = loop.run(DATE)
    assert result.status == "blocked"
    assert result.reason == "data_quality"


# ---------------------------------------------------------------------------
# Scenario I：风控否决
# ---------------------------------------------------------------------------

def test_risk_veto_blocks_order(paths, monkeypatch):
    register_paper_strategy(paths["db_path"], "fake_alpha")
    cfg = make_cfg(paths)
    loop = make_loop(cfg, make_pipeline_result("fake_alpha"), monkeypatch)
    # 当日亏损 -6%（超过 -5% 硬限）
    loop.state.update_daily_pnl(DATE, -60_000.0, 0.0)
    result = loop.run(DATE)

    assert result.status == "ok"  # 循环本身完成，但一笔订单都没有
    assert result.orders == []
    assert count_orders(paths["db_path"]) == 0
    assert result.risk_decisions, "应有风险决策记录"
    assert all(rd["approved"] is False for rd in result.risk_decisions)
    assert "daily_loss_hard" in result.risk_decisions[0]["reason"]
    assert any("risk_veto" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Scenario G/H：LIVE 被 compliance 拦截
# ---------------------------------------------------------------------------

def test_live_mode_blocked_by_compliance(paths, monkeypatch):
    register_paper_strategy(paths["db_path"], "fake_alpha")
    cfg = make_cfg(paths, execution_mode="LIVE")
    loop = make_loop(cfg, make_pipeline_result("fake_alpha"), monkeypatch)
    result = loop.run(DATE)

    assert result.status == "live_blocked"
    joined = "; ".join(result.failed_checks)
    assert "compliance" in joined
    assert result.orders == []
    assert count_orders(paths["db_path"]) == 0


# ---------------------------------------------------------------------------
# kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_blocks(paths, monkeypatch):
    register_paper_strategy(paths["db_path"], "fake_alpha")
    cfg = make_cfg(paths)
    loop = make_loop(cfg, make_pipeline_result("fake_alpha"), monkeypatch)
    loop.oms.engage_kill_switch(operator="tester", reason="drill")
    result = loop.run(DATE)

    assert result.status == "blocked"
    assert result.reason == "kill_switch"
    assert count_orders(paths["db_path"]) == 0


# ---------------------------------------------------------------------------
# SHADOW：只挂单不提交
# ---------------------------------------------------------------------------

def test_shadow_mode_stages_created_only(paths, monkeypatch):
    register_paper_strategy(paths["db_path"], "fake_alpha")
    cfg = make_cfg(paths, execution_mode="SHADOW")
    loop = make_loop(cfg, make_pipeline_result("fake_alpha"), monkeypatch)
    result = loop.run(DATE)

    assert result.status == "ok"
    assert len(result.orders) == 1
    assert result.orders[0]["status"] == "CREATED"
    with sqlite3.connect(paths["paper_db_path"]) as conn:
        broker_orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert broker_orders == 0 and trades == 0  # 券商侧毫无痕迹


# ---------------------------------------------------------------------------
# disabled / 配置
# ---------------------------------------------------------------------------

def test_disabled_config_returns_disabled(paths):
    cfg = make_cfg(paths, enabled=False)
    loop = DailyExecutionLoop.from_config(cfg, pipeline_result=make_pipeline_result())
    result = loop.run(DATE)
    assert result.status == "disabled"


def test_config_defaults_are_off():
    cfg = ExecutionLoopConfig()
    assert cfg.enabled is False
    assert cfg.execution_mode == "PAPER"


# ---------------------------------------------------------------------------
# run_daily_loop 顶层入口
# ---------------------------------------------------------------------------

def test_run_daily_loop_persists_result(paths, tmp_path, monkeypatch):
    register_paper_strategy(paths["db_path"], "fake_alpha")
    cfg = make_cfg(paths)
    _FakeDQ.pit_safe = True
    monkeypatch.setattr(daily_loop_mod, "DataQualityReport", _FakeDQ)
    result = run_daily_loop(date=DATE, cfg=cfg,
                            pipeline_result=make_pipeline_result("fake_alpha"))
    out_file = Path(paths["output_dir"]) / f"daily_loop_{DATE}.json"
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["decision_date"] == DATE
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# scheduler 集成
# ---------------------------------------------------------------------------

SCHED_CFG_BASE = {
    "data": {"cache_dir": "data/cache", "snapshot_dir": "data/snapshots"},
    "output": {"report_dir": "data/reports", "db_path": "data/pangu.db", "pick_count": 5},
}


def test_scheduler_default_off_keeps_steps_unchanged(tmp_path):
    """execution 未启用：步骤列表与历史完全一致（既有测试语义保持）。"""
    scheduler = DailyScheduler(SCHED_CFG_BASE, date=DATE, dry_run=True,
                               status_dir=tmp_path / "s1")
    summary = scheduler.run()
    names = [s["name"] for s in summary["steps"]]
    assert "execution_loop" not in names
    assert names == ["rps_build", "snapshot_build", "scan", "report",
                     "recommendation_loop", "notify"]


def test_scheduler_enabled_appends_execution_step(tmp_path, monkeypatch):
    """execution.enabled=true：dry-run 中 execution_loop 以 skipped 出现在步骤里。"""
    cfg = {**SCHED_CFG_BASE, "execution": {"enabled": True}}
    scheduler = DailyScheduler(cfg, date=DATE, dry_run=True, status_dir=tmp_path / "s2")
    summary = scheduler.run()
    names = [s["name"] for s in summary["steps"]]
    assert "execution_loop" in names
    step = next(s for s in summary["steps"] if s["name"] == "execution_loop")
    assert step["status"] == "skipped"  # dry-run 跳过真实执行


def test_scheduler_step_calls_run_daily_loop(tmp_path, monkeypatch):
    """非 dry-run：_step_execution_loop 携带 cfg 与 scan 的 pipeline_result。"""
    cfg = {**SCHED_CFG_BASE, "execution": {"enabled": True}}
    captured = {}

    def fake_run_daily_loop(**kwargs):
        captured.update(kwargs)
        return LoopResult(status="ok", decision_date=DATE, execution_mode="PAPER")

    monkeypatch.setattr(daily_loop_mod, "run_daily_loop", fake_run_daily_loop)
    scheduler = DailyScheduler(cfg, date=DATE, dry_run=False,
                               status_dir=tmp_path / "s3")
    scheduler.pipeline_result = make_pipeline_result()
    step = scheduler._run_step("execution_loop", scheduler._step_execution_loop)
    assert step.status == "ok"
    assert captured["date"] == DATE
    assert captured["cfg"] is cfg
    assert captured["pipeline_result"] == make_pipeline_result()


def test_scheduler_step_skipped_when_disabled(tmp_path):
    cfg = {**SCHED_CFG_BASE, "execution": {"enabled": False}}
    scheduler = DailyScheduler(cfg, date=DATE, dry_run=False,
                               status_dir=tmp_path / "s4")
    step = scheduler._run_step("execution_loop", scheduler._step_execution_loop)
    assert step.status == "ok"
    assert step.output["status"] == "skipped"
    assert step.output["enabled"] is False
