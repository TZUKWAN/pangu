from engine.xuanwu_pool import XuanwuPoolBuilder


def _base_candidate(**overrides):
    candidate = {
        "code": "000001",
        "name": "测试股份",
        "board": "AI算力",
        "close": 12.85,
        "pct_change": 6.5,
        "turnover_rate": 8.5,
        "rps": 92,
        "recommend": {
            "recommend_score": 88,
            "score_breakdown": {
                "news_sentiment": 72,
                "news_themes": ["AI"],
                "news_explain": "与板块催化形成共振",
            },
        },
        "entry_exit": {
            "buy_points": [{"price": 12.5, "type": "回踩位", "is_primary": True}],
            "stop_loss": {"price": 11.8},
            "take_profit": [{"price": 14.2}],
            "position": {"risk_pct": 0.85},
            "risk_reward_ratio": 2.1,
        },
        "technical": {
            "volume": {"volume_ratio": 1.8},
            "hints": ["短中期均线多头排列"],
        },
        "stock_news": [{"title": "测试股份受益AI算力景气"}],
        "debate": {"verdict": "推荐", "confidence": 85, "reason": "趋势、板块、计划均成立"},
    }
    candidate.update(overrides)
    return candidate


def _context():
    return {
        "sentiment": {"temperature": 72.5},
        "boards": [{"name": "AI算力", "score": 1.8}],
        "news": {"hot_themes": [["AI", 3]]},
        "source_status": {"market_data": "ok", "structured_data": "ok"},
    }


def test_strong_candidate_enters_xuanwu_pool():
    ctx = _context()
    result = XuanwuPoolBuilder().build(
        sentiment=ctx["sentiment"],
        boards=ctx["boards"],
        candidates=[_base_candidate()],
        news=ctx["news"],
        source_status=ctx["source_status"],
    )

    decision = result["all_decisions"]["000001"]
    assert result["summary"]["xuanwu_count"] == 1
    assert decision["status"] == "xuanwu"
    assert decision["score"] >= 75
    assert decision["gates"]["board"] == "pass"
    assert decision["gates"]["debate"] == "pass"


def test_generic_board_and_missing_debate_stays_watch():
    ctx = _context()
    candidate = _base_candidate(board="全市场强势", debate={})
    result = XuanwuPoolBuilder().build(
        sentiment=ctx["sentiment"],
        boards=ctx["boards"],
        candidates=[candidate],
        news=ctx["news"],
        source_status=ctx["source_status"],
    )

    decision = result["all_decisions"]["000001"]
    assert decision["status"] == "watch"
    assert "board_mapping_missing" in decision["blockers"]
    assert "multi_agent_missing" not in decision["blockers"]
    assert decision["gates"]["debate"] == "not_scoped"
    assert result["summary"]["xuanwu_count"] == 0


def test_otherwise_strong_candidate_without_debate_is_pending():
    ctx = _context()
    candidate = _base_candidate(debate={})
    result = XuanwuPoolBuilder().build(
        sentiment=ctx["sentiment"],
        boards=ctx["boards"],
        candidates=[candidate],
        news=ctx["news"],
        source_status=ctx["source_status"],
    )

    decision = result["all_decisions"]["000001"]
    assert decision["status"] == "pending_ai"
    assert "multi_agent_missing" in decision["blockers"]
    assert result["summary"]["xuanwu_count"] == 0


def test_negative_news_rejects_candidate():
    ctx = _context()
    candidate = _base_candidate(
        recommend={
            "recommend_score": 88,
            "score_breakdown": {
                "news_sentiment": 80,
                "news_themes": ["AI"],
                "news_explain": "公司收到监管问询，存在风险提示",
            },
        },
        stock_news=[{"title": "测试股份收到问询函"}],
    )
    result = XuanwuPoolBuilder().build(
        sentiment=ctx["sentiment"],
        boards=ctx["boards"],
        candidates=[candidate],
        news=ctx["news"],
        source_status=ctx["source_status"],
    )

    decision = result["all_decisions"]["000001"]
    assert decision["status"] == "rejected"
    assert "negative_news_risk" in decision["blockers"]
    assert result["summary"]["xuanwu_count"] == 0


def test_weak_industry_mapping_does_not_enter_xuanwu_pool():
    ctx = _context()
    candidate = _base_candidate(board="行业:化工")
    result = XuanwuPoolBuilder().build(
        sentiment=ctx["sentiment"],
        boards=ctx["boards"],
        candidates=[candidate],
        news=ctx["news"],
        source_status=ctx["source_status"],
    )

    decision = result["all_decisions"]["000001"]
    assert decision["status"] == "watch"
    assert decision["gates"]["board"] == "weak_mapping"
    assert "board_mapping_weak" in decision["blockers"]
    assert result["summary"]["xuanwu_count"] == 0


def test_breakout_primary_buy_point_is_rejected_as_chasing():
    ctx = _context()
    candidate = _base_candidate(entry_exit={
        "buy_points": [{"price": 13.2, "type": "突破位", "is_primary": True}],
        "stop_loss": {"price": 11.8},
        "take_profit": [{"price": 15.2}],
        "position": {"risk_pct": 0.85},
        "risk_reward_ratio": 2.1,
    })
    result = XuanwuPoolBuilder().build(
        sentiment=ctx["sentiment"],
        boards=ctx["boards"],
        candidates=[candidate],
        news=ctx["news"],
        source_status=ctx["source_status"],
    )

    decision = result["all_decisions"]["000001"]
    assert decision["status"] == "rejected"
    assert "chasing_buy_point" in decision["blockers"]
    assert result["summary"]["xuanwu_count"] == 0


def test_rule_validation_reject_only_warns_not_hard_blocks():
    ctx = _context()
    candidate = _base_candidate(
        debate={
            "verdict": "回避",
            "confidence": 30,
            "mode": "rule_validation",
            "rule_degraded": True,
            "reason": "规则验证发现追高风险",
            "bull_points": [],
            "bear_points": ["追高风险"],
        }
    )
    result = XuanwuPoolBuilder().build(
        sentiment=ctx["sentiment"],
        boards=ctx["boards"],
        candidates=[candidate],
        news=ctx["news"],
        source_status=ctx["source_status"],
    )

    decision = result["all_decisions"]["000001"]
    assert decision["gates"]["debate"] == "rule_warn"
    assert "multi_agent_rejected" not in decision["blockers"]


def test_rule_validated_trial_can_enter_xuanwu_when_core_gates_pass():
    ctx = _context()
    candidate = _base_candidate(
        debate={
            "verdict": "回避",
            "confidence": 35,
            "mode": "rule_validation",
            "rule_degraded": True,
            "reason": "LLM 未配置，规则验证通过核心闸门但保持轻仓",
            "bull_points": ["板块、量价和交易计划均通过"],
            "bear_points": ["缺少真实 LLM 辩论"],
        }
    )
    result = XuanwuPoolBuilder().build(
        sentiment=ctx["sentiment"],
        boards=ctx["boards"],
        candidates=[candidate],
        news=ctx["news"],
        source_status=ctx["source_status"],
    )

    decision = result["all_decisions"]["000001"]
    assert result["summary"]["xuanwu_count"] == 1
    assert decision["status"] == "xuanwu"
    assert decision["gates"]["debate"] == "rule_warn"
    assert decision["evidence"]["validation_mode"] == "rule_trial"


def test_rule_validated_trial_requires_strong_board_mapping():
    ctx = _context()
    candidate = _base_candidate(
        board="行业:化工",
        debate={
            "verdict": "回避",
            "confidence": 35,
            "mode": "rule_validation",
            "rule_degraded": True,
        },
    )
    result = XuanwuPoolBuilder().build(
        sentiment=ctx["sentiment"],
        boards=ctx["boards"],
        candidates=[candidate],
        news=ctx["news"],
        source_status=ctx["source_status"],
    )

    decision = result["all_decisions"]["000001"]
    assert result["summary"]["xuanwu_count"] == 0
    assert decision["status"] == "watch"
    assert "board_mapping_weak" in decision["blockers"]
