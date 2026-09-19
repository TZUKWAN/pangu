"""Strategy registry lifecycle tests (offline, tmp SQLite db)."""
from __future__ import annotations

import pytest

from engine.contracts import StrategyStatus
from engine.strategies import (
    AUTO_SUSPEND_TRIGGERS,
    GateFailed,
    IllegalStrategyTransition,
    PromotionContext,
    StrategyManifest,
    StrategyRegistry,
    seed_registry,
)

PASSING_METRICS = {
    "oos_net_expectancy": 0.004,
    "oos_profit_factor": 1.35,
    "oos_sharpe": 1.25,
    "oos_max_drawdown": 0.08,
    "trade_count": 320,
    "trade_date_count": 185,
    "cost2x_positive": True,
    "parameter_stable": True,
    "leakage_audit_passed": True,
    "independent_backtest_consistent": True,
}


def passing_ctx(**over) -> PromotionContext:
    defaults = dict(
        validation_metrics=dict(PASSING_METRICS),
        paper_report_ref="data/research/paper_report.md",
        shadow_report_ref="data/research/shadow_report.md",
        compliance_ok=True,
        broker_ok=True,
        reconcile_ok=True,
        kill_switch_ready=True,
    )
    metrics_over = over.pop("validation_metrics", {})
    defaults.update(over)
    defaults["validation_metrics"].update(metrics_over)
    return PromotionContext(**defaults)


def make_manifest(strategy_id: str = "测试策略", status: str = "research") -> StrategyManifest:
    return StrategyManifest(
        strategy_id=strategy_id,
        version="1.0",
        code_sha="0" * 64,
        data_snapshot="test_snapshot",
        features=["f1"],
        model="test_model_v1",
        params={},
        train_range=None,
        validation_range=None,
        test_range=None,
        execution_assumptions={"decision_time": "15:05"},
        metrics={},
        approval_state=StrategyStatus(status),
        created_at="2026-09-19T00:00:00+08:00",
        updated_at="2026-09-19T00:00:00+08:00",
        evidence_refs=["evidence.json"],
        notes="test",
    )


@pytest.fixture()
def registry(tmp_path):
    return StrategyRegistry(tmp_path / "exec.db")


def walk_to(registry: StrategyRegistry, strategy_id: str, target: StrategyStatus) -> None:
    """Promote along the legal chain up to `target`, with passing gates."""
    chain = [
        StrategyStatus.RESEARCH, StrategyStatus.VALIDATED, StrategyStatus.PAPER,
        StrategyStatus.SHADOW_LIVE, StrategyStatus.LIMITED_LIVE, StrategyStatus.LIVE,
    ]
    for status in chain:
        registry.transition(strategy_id, status, operator="test", reason="walk",
                            checks=passing_ctx() if status in (
                                StrategyStatus.PAPER, StrategyStatus.SHADOW_LIVE,
                                StrategyStatus.LIMITED_LIVE, StrategyStatus.LIVE) else None)
        if status == target:
            break


# ---------------------------------------------------------------------- #
def test_register_rejects_live_initial_status(registry):
    with pytest.raises(ValueError):
        registry.register(make_manifest(status="live"))
    with pytest.raises(ValueError):
        registry.register(make_manifest(status="suspended"))
    registry.register(make_manifest(status="idea"))  # ok


def test_full_promotion_chain_idea_to_live(registry):
    m = make_manifest("链路策略", status="idea")
    registry.register(m)
    walk_to(registry, "链路策略", StrategyStatus.LIVE)
    assert registry.get("链路策略").approval_state == StrategyStatus.LIVE
    assert registry.executable_strategy_ids() == ["链路策略"]
    history = [dict(r) for r in registry.history("链路策略")]
    statuses = [h["to_status"] for h in history]
    assert statuses == ["idea", "research", "validated", "paper",
                        "shadow_live", "limited_live", "live"]


def test_skip_transition_raises(registry):
    registry.register(make_manifest("跳跃", status="validated"))
    with pytest.raises(IllegalStrategyTransition):
        registry.transition("跳跃", StrategyStatus.LIVE, "test", "skip to live",
                            checks=passing_ctx())
    with pytest.raises(IllegalStrategyTransition):
        registry.transition("跳跃", StrategyStatus.IDEA, "test", "backwards")
    assert registry.get("跳跃").approval_state == StrategyStatus.VALIDATED


def test_promotion_gate_failure_blocks_and_records(registry):
    registry.register(make_manifest("门槛", status="validated"))
    bad = passing_ctx(validation_metrics={"oos_profit_factor": 1.05})
    with pytest.raises(IllegalStrategyTransition):
        registry.transition("门槛", StrategyStatus.PAPER, "test", "weak pf", checks=bad)
    assert registry.get("门槛").approval_state == StrategyStatus.VALIDATED
    with pytest.raises(IllegalStrategyTransition):  # missing checks entirely
        registry.transition("门槛", StrategyStatus.PAPER, "test", "no evidence")
    # shadow gate must NOT demand compliance (read-only stage)
    registry.transition("门槛", StrategyStatus.PAPER, "test", "ok", checks=passing_ctx())
    shadow_ctx = passing_ctx(compliance_ok=False)
    registry.transition("门槛", StrategyStatus.SHADOW_LIVE, "test", "shadow no compliance",
                        checks=shadow_ctx)
    assert registry.get("门槛").approval_state == StrategyStatus.SHADOW_LIVE


def test_auto_suspend_all_triggers(registry):
    for trigger in sorted(AUTO_SUSPEND_TRIGGERS):
        sid = f"池-{trigger}"
        registry.register(make_manifest(sid, status="research"))
        result = registry.auto_suspend(sid, trigger, detail="测试详情")
        assert result is not None
        assert registry.get(sid).approval_state == StrategyStatus.SUSPENDED
        assert any(f"auto_suspend:{trigger}" in (r["reason"] or "")
                   for r in registry.history(sid))
        # idempotent when already suspended
        assert registry.auto_suspend(sid, trigger) is None
    with pytest.raises(ValueError):
        registry.auto_suspend("池-drawdown_breach", "unknown_trigger")


def test_suspend_resume_and_retire(registry):
    registry.register(make_manifest("全周期", status="research"))
    registry.transition("全周期", StrategyStatus.VALIDATED, "test", "v")
    registry.transition("全周期", StrategyStatus.SUSPENDED, "test", "paused")
    registry.transition("全周期", StrategyStatus.PAPER, "test", "resumed", checks=passing_ctx())
    registry.transition("全周期", StrategyStatus.SUSPENDED, "test", "paused again")
    registry.transition("全周期", StrategyStatus.RETIRED, "test", "gone")
    with pytest.raises(IllegalStrategyTransition):
        registry.transition("全周期", StrategyStatus.RESEARCH, "test", "zombie")
    with pytest.raises(IllegalStrategyTransition):
        registry.transition("全周期", StrategyStatus.SUSPENDED, "test", "even suspend")


def test_executable_strategy_ids_filters_by_status(registry):
    registry.register(make_manifest("纸面", status="research"))
    registry.register(make_manifest("研究", status="research"))
    registry.transition("纸面", StrategyStatus.VALIDATED, "test", "v")
    registry.transition("纸面", StrategyStatus.PAPER, "test", "p", checks=passing_ctx())
    assert registry.executable_strategy_ids() == ["纸面"]
    registry.auto_suspend("纸面", "drawdown_breach")
    assert registry.executable_strategy_ids() == []


# ---------------------------------------------------------------------- #
def test_seed_registry_idempotent_seven_research_manifests(tmp_path):
    db = tmp_path / "seed.db"
    r1 = StrategyRegistry(db)
    seeded = seed_registry(r1)
    assert len(seeded) == 7
    assert all(m.approval_state == StrategyStatus.RESEARCH for m in seeded)
    assert len(r1.list()) == 7
    # re-seed: idempotent, no duplicates, no exception
    seed_registry(r1)
    assert len(r1.list()) == 7
    research = r1.list(StrategyStatus.RESEARCH)
    assert len(research) == 7
    assert {m.strategy_id for m in research} == {
        "题材龙头", "连板梯队", "趋势回踩", "超跌反弹", "小盘优质", "大市值低波", "事件驱动",
    }
    for m in research:
        assert len(m.code_sha) == 64
        assert m.features == ["heuristic_pool_score"]
        assert m.model == "manual_rules_v1"
        assert m.execution_assumptions == {"decision_time": "15:05", "entry": "T+1 open zone"}
        assert m.evidence_refs, "must reference real data/research replay files"
        assert "禁止实盘" in m.notes
    # fresh registry over the same db file also stays at 7
    r2 = StrategyRegistry(db)
    seed_registry(r2)
    assert len(r2.list()) == 7
