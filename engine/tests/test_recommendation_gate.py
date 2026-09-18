"""推荐闸门单元测试：覆盖新闻证据审计、反追涨、追价买点拦截。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine.quant_guard import GuardResult
from engine.recommendation_gate import RecommendationGate
from engine.strategy_pools import StrategySignal
from engine.trend_scanner import StockCandidate


def _candidate(code: str = "000001", name: str = "测试", board: str = "机器人") -> StockCandidate:
    return StockCandidate(
        code=code, name=name, board=board, close=10.0, pct_change=2.0,
        turnover_rate=5.0, circ_mv_yi=100.0, rps=90.0, rps_mode="real",
        fund_inflow_days=3, fund_flow_status="available", score=80.0,
    )


def _signal(code: str = "000001", score: float = 80.0, strategy: str = "趋势突破") -> StrategySignal:
    return StrategySignal(
        strategy_name=strategy, code=code, name="测试", board="机器人",
        trigger_reason="突破", score=score,
    )


def _entry_exit() -> dict:
    return {
        "buy_points": [{"is_primary": True, "price": 10.0, "type": "低吸", "condition": "回踩均线"}],
        "stop_loss": {"price": 9.0, "method": "均线"},
        "take_profit": [{"price": 11.0, "method": "1:1"}],
        "exit_plan": _exit_plan(),
        "warnings": [],
    }


def _exit_plan() -> dict:
    rule_types = [
        "hard_stop", "news_invalidation", "market_retreat", "theme_invalidation",
        "trend_break", "first_target", "final_target", "trailing_stop", "time_stop",
    ]
    return {
        "entry_price": 10.0,
        "initial_stop": 9.0,
        "first_target": 11.0,
        "final_target": 12.0,
        "trailing_reference": 10.5,
        "max_holding_days": 3,
        "sentiment_exit_drop": 15.0,
        "conservative_same_day_order": "stop_first",
        "rules": [{"rule_type": name, "action": "exit_all", "condition": name} for name in rule_types],
    }


def _gate(
    candidates: list[dict[str, object]] | None = None,
    pooled: dict[str, list[StrategySignal]] | None = None,
    phase: dict[str, object] | None = None,
    recommendation_allowed: bool = True,
    cfg_overrides: dict | None = None,
) -> RecommendationGate:
    dl = MagicMock()
    guard = GuardResult(kept=[_candidate()], watch=[], rejected=[])
    return RecommendationGate(
        dl=dl,
        guard_result=guard,
        market_phase=phase or {"market_phase": "震荡", "allowed_strategies": ["趋势突破"], "forbidden_strategies": []},
        cfg=cfg_overrides or {},
        recommendation_allowed=recommendation_allowed,
    )


def test_news_evidence_bearish_rejects() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "news_evidence": {
            "sentiment_label": "bearish",
            "verdict_reason": "业绩变脸",
            "risk_events": ["业绩变脸"],
        },
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.rejected)
    assert any(log["gate"] == "news_evidence" for log in res.gate_log)


def test_news_evidence_bearish_without_risk_goes_watch() -> None:
    """一般利空（无重大风险事件）降级到观察池，而不是 rejected。"""
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "news_evidence": {
            "sentiment_label": "bearish",
            "verdict_reason": "板块短期分歧",
            "risk_events": [],
        },
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert not any(i["code"] == "000001" for i in res.rejected)
    assert any(i["code"] == "000001" for i in res.watchlist)
    assert any("无重大风险" in log.get("reason", "") for log in res.gate_log if log.get("gate") == "news_evidence")


def test_news_evidence_mixed_with_risk_watch() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "news_evidence": {
            "sentiment_label": "mixed",
            "verdict_reason": "多空交织",
            "risk_events": ["减持公告"],
        },
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.watchlist)


def test_news_evidence_bullish_passes() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "news_evidence": {
            "sentiment_label": "bullish",
            "verdict_reason": "利多",
            "support_count": 3,
        },
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.final_recommendations)


def test_missing_exit_plan_can_only_enter_watch() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试",
        "entry_exit": {"buy_points": [{"is_primary": True, "price": 10.0}]},
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
    }
    res = gate.pass_gate({"趋势突破": [_signal()]}, {"000001": _candidate()}, candidates=[item])
    assert res.final_recommendations == []
    assert any("卖出计划不完整" in candidate.get("watch_reason", "") for candidate in res.watchlist)
    assert any(log.get("gate") == "exit_plan" for log in res.gate_log)


def test_anti_chase_blocked_rejects() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "blocked", "reason": "已加速 3 天"},
        "entry_plan": {"is_chasing": False},
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.rejected)


def test_entry_plan_chasing_defaults_to_soft_note() -> None:
    """默认 soft_entry_chase：追价型买点只提示不拦截（条件单未触发即不成交）。"""
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": True, "trigger_condition": "突破追高"},
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.final_recommendations)
    assert res.final_recommendations[0].get("entry_chase_note")


def test_entry_plan_chasing_strict_mode_still_watches() -> None:
    """strict 配置（soft_entry_chase=False）下追价型买点仍降观察。"""
    gate = _gate(cfg_overrides={"strategy_framework": {"gate": {"soft_entry_chase": False}}})
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": True, "trigger_condition": "突破追高"},
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.watchlist)


def test_breakout_confirm_deviation_soft_note_by_default() -> None:
    """默认 soft_entry_chase：突破确认偏离只提示；strict 配置下降观察。"""
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_style": "breakout_confirm",
        "entry_plan": {"is_chasing": False, "trigger_price": 10.0, "current_price": 10.2},
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.final_recommendations)
    assert res.final_recommendations[0].get("entry_chase_note")

    strict_gate = _gate(cfg_overrides={"strategy_framework": {"gate": {"soft_entry_chase": False}}})
    res2 = strict_gate.pass_gate(pooled, cand_map, candidates=[dict(item)])
    assert any(
        i["code"] == "000001" and i.get("watch_reason", "").startswith("突破确认")
        for i in res2.watchlist
    )


def test_volume_audit_missing_goes_watch() -> None:
    gate = _gate()
    item = {
        "code": "000001",
        "name": "测试",
        "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "volume_audit": {
            "status": "missing",
            "price_volume_pattern": "missing",
            "reason": "量能数据缺失，不能进入正式推荐",
        },
        "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])

    assert res.final_recommendations == []
    assert any(i["code"] == "000001" and "量能审计不足" in i.get("watch_reason", "") for i in res.watchlist)


def test_fund_flow_unavailable_does_not_block_normal_strategy() -> None:
    gate = _gate()
    cand = _candidate()
    cand.fund_flow_status = "unavailable"
    item = {
        "code": "000001",
        "name": "测试",
        "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "volume_audit": {"status": "ok", "price_volume_pattern": "pullback_shrink"},
        "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
    }
    res = gate.pass_gate({"趋势突破": [_signal(strategy="趋势突破")]}, {"000001": cand}, candidates=[item])

    assert any(i["code"] == "000001" for i in res.final_recommendations)
    assert res.final_recommendations[0]["fund_flow_risk"].startswith("资金流状态 unavailable")


def test_fund_flow_unavailable_blocks_fund_flow_strategy() -> None:
    gate = _gate(phase={"market_phase": "震荡", "allowed_strategies": ["主力资金"], "forbidden_strategies": []})
    cand = _candidate()
    cand.fund_flow_status = "unavailable"
    item = {
        "code": "000001",
        "name": "测试",
        "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "volume_audit": {"status": "ok", "price_volume_pattern": "pullback_shrink"},
        "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
    }
    res = gate.pass_gate({"主力资金": [_signal(strategy="主力资金")]}, {"000001": cand}, candidates=[item])

    assert res.final_recommendations == []
    assert any("策略强依赖资金流" in i.get("watch_reason", "") for i in res.watchlist)


def test_gate_consumes_evidence_map() -> None:
    """Gate 优先从 evidence_map 读取审计字段，而不是 item 中的旧字段。"""
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        # item 里是 bullish，但 evidence_map 里是 bearish 且带风险
        "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
    }
    evidence_map = {
        "000001": {
            "data_quality": {"overall": "ok"},
            "volume_audit": {"status": "ok", "price_volume_pattern": "pullback_shrink"},
            "news_evidence": {
                "sentiment_label": "bearish",
                "verdict_reason": "业绩变脸",
                "risk_events": ["业绩变脸"],
            },
            "anti_chase": {"status": "ok"},
            "entry_plan": {"is_chasing": False},
        }
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item], evidence_map=evidence_map)
    assert any(i["code"] == "000001" for i in res.rejected)
    assert any("重大风险" in log.get("reason", "") for log in res.gate_log if log.get("gate") == "news_evidence")


def test_gate_rejects_code_missing_candidate_evidence() -> None:
    gate = _gate()
    res = gate.pass_gate(
        {"趋势突破": [_signal()]},
        {"000001": _candidate()},
        candidates=[{"code": "000001", "entry_exit": _entry_exit()}],
        evidence_map={},
    )
    assert res.final_recommendations == []
    assert any(i["code"] == "000001" for i in res.rejected)
    assert any(log.get("gate") == "candidate_evidence" for log in res.gate_log)


def test_degraded_evidence_enters_final_with_note_but_failed_vetoes() -> None:
    """全局 degraded 不再一票否决（逐候选闸门单独把关）；failed 仍然否决。"""
    gate = _gate()
    evidence_map = {
        "000001": {
            "data_quality": {"overall": "degraded"},
            "volume_audit": {"status": "ok", "price_volume_pattern": "pullback_shrink"},
            "anti_chase": {"status": "ok"},
            "entry_plan": {"is_chasing": False, "trigger_condition": "回踩确认"},
            "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
        }
    }
    res = gate.pass_gate(
        {"趋势突破": [_signal()]},
        {"000001": _candidate()},
        candidates=[{"code": "000001", "entry_exit": _entry_exit()}],
        evidence_map=evidence_map,
    )
    assert any(i["code"] == "000001" for i in res.final_recommendations)
    assert res.final_recommendations[0].get("data_quality_note")

    evidence_map_failed = {
        "000001": {
            **evidence_map["000001"],
            "data_quality": {"overall": "failed"},
        }
    }
    res2 = gate.pass_gate(
        {"趋势突破": [_signal()]},
        {"000001": _candidate()},
        candidates=[{"code": "000001", "entry_exit": _entry_exit()}],
        evidence_map=evidence_map_failed,
    )
    assert res2.final_recommendations == []
    assert any("数据链路 failed" in i.get("watch_reason", "") for i in res2.watchlist)


def test_quant_guard_watch_cannot_enter_final() -> None:
    candidate = _candidate()
    gate = RecommendationGate(
        dl=MagicMock(),
        guard_result=GuardResult(kept=[], watch=[candidate], rejected=[]),
        market_phase={"market_phase": "震荡", "allowed_strategies": ["趋势突破"], "forbidden_strategies": []},
        cfg={},
        recommendation_allowed=True,
    )
    item = {
        "code": "000001",
        "entry_exit": _entry_exit(),
        "volume_audit": {"status": "ok", "price_volume_pattern": "pullback_shrink"},
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
    }
    res = gate.pass_gate({"趋势突破": [_signal()]}, {"000001": candidate}, candidates=[item])
    assert res.final_recommendations == []
    assert any("QuantGuard" in i.get("watch_reason", "") for i in res.watchlist)


# ---------------------------------------------------------------------------
# 市场阶段 → 策略池语义映射（修复：阶段禁令此前按名字子串匹配，几乎从不生效）
# ---------------------------------------------------------------------------
from engine.strategy_pools import StrategySignal as _Sig


def _sig_only(strategy: str, role: str | None = None) -> _Sig:
    return _Sig(
        strategy_name=strategy, code="000001", name="t", board="x",
        raw_features={"role": role} if role else {},
    )


def _bare_gate(forbidden: list[str]) -> RecommendationGate:
    return RecommendationGate(
        dl=MagicMock(),
        guard_result=GuardResult(kept=[], watch=[], rejected=[]),
        market_phase={"market_phase": "x", "forbidden_strategies": forbidden},
        cfg={},
        recommendation_allowed=True,
    )


def test_phase_map_blocks_limitup_pool_in_freeze() -> None:
    gate = _bare_gate(["追涨", "高位接力", "短线进攻"])
    assert gate._phase_allowed("连板梯队", _sig_only("连板梯队")) is False
    assert gate._phase_allowed("题材龙头", _sig_only("题材龙头")) is False
    assert gate._phase_allowed("超跌反弹", _sig_only("超跌反弹")) is True
    assert gate._phase_allowed("大市值低波", _sig_only("大市值低波")) is True


def test_phase_map_rear_only_forbids_spare_core_roles() -> None:
    gate = _bare_gate(["后排追涨", "高位无承接追涨"])
    assert gate._phase_allowed("题材龙头", _sig_only("题材龙头", "龙头")) is True
    assert gate._phase_allowed("题材龙头", _sig_only("题材龙头", "补涨")) is False
    assert gate._phase_allowed("连板梯队", _sig_only("连板梯队", "中军")) is True


def test_phase_map_can_be_disabled_via_config() -> None:
    gate = RecommendationGate(
        dl=MagicMock(),
        guard_result=GuardResult(kept=[], watch=[], rejected=[]),
        market_phase={"market_phase": "冰点期", "forbidden_strategies": ["追涨"]},
        cfg={"strategy_framework": {"gate": {"enforce_phase_strategy_map": False}}},
        recommendation_allowed=True,
    )
    assert gate._phase_allowed("连板梯队", _sig_only("连板梯队")) is True
