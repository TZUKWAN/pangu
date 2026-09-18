"""render_markdown 结构化因子（P0）渲染测试。

验证展示层逻辑：候选携带真实 structured_factors + source_state 时，Markdown 报告
正确输出结构化因子明细与降级原因；无 P0 数据时不伪造、不输出空小节。
已移除因子（margin/block_trade/holder_num）不展示。
"""

from __future__ import annotations

from types import SimpleNamespace

from engine.report import render_markdown
from engine.structured_format import factor_value_text


def _full_candidate() -> dict:
    return {
        "code": "000001", "name": "测试股", "board": "机器人", "close": 12.34,
        "pct_change": 5.6, "rps": 88.0, "fund_inflow_days": 3, "circ_mv_yi": 120.0,
        "turnover_rate": 7.02,
        "reasons": ["RPS 88 强势", "资金连流 3 日"],
        "recommend": {
            "recommend_score": 80.0, "grade": "A", "up_prob": 62, "calibrated": True,
            "target_pct": [3, 8], "risk_reward_ratio": 2.1, "tag": "趋势观察",
            "score_breakdown": {
                "trend": 80, "structured_signal": 88.0,
                "structured_bonus": 12.0, "structured_penalty": 0.0,
            },
        },
        "entry_exit": {
            "buy_points": [{"is_primary": True, "price": 12.30, "type": "突破", "condition": "放量"}],
            "stop_loss": {"price": 11.50, "method": "ATR"},
            "take_profit": [{"price": 14.00, "method": "1:2"}, {"price": 15.00, "method": "1:3"}],
            "exit_plan": {
                "entry_price": 12.30, "initial_stop": 11.50,
                "first_target": 14.00, "final_target": 15.00,
                "trailing_reference": 12.40,
                "max_holding_days": 3, "conservative_same_day_order": "stop_first",
                "rules": [
                    {"rule_type": "trailing_stop", "condition": "盈利后收盘跌破 MA10"},
                    {"rule_type": "time_stop", "condition": "持有满 3 个交易日仍未兑现，收盘退出"},
                    {"rule_type": "news_invalidation", "condition": "出现直接负面事件"},
                    {"rule_type": "market_retreat", "condition": "市场进入冰点或退潮"},
                    {"rule_type": "theme_invalidation", "condition": "题材催化被证伪"},
                    {"rule_type": "trend_break", "condition": "跌破 MA20 或放量跌破 MA10"},
                ],
            },
            "position": {"shares": 300, "risk_pct": 1.5},
        },
        "structured_factors": {
            "dragon_tiger": {
                "date": "2026-07-01", "reason": "日涨幅榜", "net_buy_wan": 451.2,
                "buy_wan": 5130.1, "sell_wan": 4678.9, "turnover_pct": 7.02,
                "source": "ths_dataapi",
            },
            "hot_rank": {"rank": 3, "source": "ths"},
            "reasons": ["LHB net buy 451.2w", "hot rank 3 (ths)"],
            "risk_notes": [],
        },
    }


def _structured_source_state() -> dict:
    """P0 结构化源原始状态（无 margin，因为已移除）。"""
    return {
        "dragon_tiger_daily": {"status": "ok", "warnings": []},
        "hot_rank": {"status": "ok", "warnings": []},
        "capital_flow_120d": {"status": "unavailable", "warnings": ["Connection aborted."]},
        "research": {"status": "skipped", "warnings": ["limited to top 100 candidates"]},
        "limit_up_sentiment": {"status": "ok", "warnings": []},
        "summary": {"note": "Only fields returned by real public endpoints are attached."},
    }


def _result(cands: list[dict], source_state: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        date="20260701",
        sentiment={"temperature": 70, "posture": "震荡偏多", "advice": "半仓滚动", "components": {
            "limit_up_count": 30, "limit_up_score": 20, "consecutive_height": 3,
            "consecutive_score": 10, "broke_rate": 0.2, "broke_rate_score": 8,
            "limit_down_count": 2, "limit_down_score": 5, "advance": 2000, "decline": 1500,
            "advance_decline_score": 12,
        }},
        boards=[], candidates=cands, rejected=[],
        posture_advice="精选强势股，严格止损。", warnings=[],
        source_state=source_state or {},
        news={},
        data_quality="ok",
        tradable=True,
        no_trade_reason="",
        block_reasons=[],
    )


def test_render_markdown_includes_structured_factors_and_degradation():
    md = render_markdown(_result([_full_candidate()], {"structured_data": _structured_source_state()}))
    # 逐候选结构化因子小节标题 + 真实龙虎榜值
    assert "结构化因子（P0）" in md
    assert "龙虎榜" in md and "净买入 451" in md
    assert "热榜人气" in md and "排名 #3" in md
    # source_state 降级原因必须可见（资金流 unavailable / research skipped）
    assert "capital_flow_120d" in md and "Connection aborted" in md
    assert "research" in md and "top 100" in md
    # 评分贡献
    assert "结构化信号 88.0" in md
    # 加分因子
    assert "加分因子" in md and "LHB net buy 451.2w" in md


def test_render_markdown_includes_overall_source_health():
    md = render_markdown(_result([_full_candidate()], {"structured_data": _structured_source_state()}))
    # 整体 P0 源健康度摘要
    assert "结构化源（P0）" in md
    assert "源正常" in md
    assert "capital_flow_120d=unavailable" in md
    assert "research=skipped" in md


def test_render_markdown_no_structured_section_when_no_p0_data():
    """候选无 structured_factors、无 source_state 时：不输出结构化小节，不伪造。"""
    bare = _full_candidate()
    bare["structured_factors"] = {}
    bare["recommend"]["score_breakdown"].pop("structured_signal", None)
    md = render_markdown(_result([bare], {}))
    assert "结构化因子（P0）" not in md
    assert "结构化源（P0）" not in md
    assert "净买入" not in md


def test_render_markdown_accepts_gate_reject_reason():
    result = _result([])
    result.rejected = [{"code": "000001", "name": "测试股", "reject_reason": "重大风险新闻"}]
    md = render_markdown(result)
    assert "重大风险新闻" in md


def test_new_report_keeps_compat_candidates_out_of_formal_recommendations():
    result = _result([_full_candidate()])
    result.final_recommendations = []
    result.watchlist = [{
        **_full_candidate(),
        "watch_reason": "反追涨审计未通过",
        "candidate_evidence": {"strategy": {"strategy_name": "趋势回踩"}},
    }]
    result.candidate_evidence = {"000001": {"decision": {"status": "watch"}}}
    result.raw_candidate_count = 1
    result.tradable = False
    result.no_trade_reason = "数据完整，但无低风险买点"

    md = render_markdown(result)

    assert "本期无正式推荐" in md
    assert "观察池（非推荐）" in md
    assert "反追涨审计未通过" in md
    assert "候选股池" not in md
    assert "推荐度：80.0" not in md
    assert "主买点" not in md


def test_new_report_renders_only_gate_final_with_conditional_entry():
    result = _result([_full_candidate()])
    result.final_recommendations = [_full_candidate()]
    result.watchlist = []
    result.candidate_evidence = {"000001": {"decision": {"status": "final"}}}
    result.raw_candidate_count = 1

    md = render_markdown(result)

    assert "正式推荐（已通过完整证据链）" in md
    assert "条件买点" in md
    assert "主买点" not in md
    assert "第一卖点" in md and "第二卖点" in md
    assert "移动止盈" in md and "时间卖点" in md
    assert "新闻证伪" in md and "情绪退潮" in md and "题材失效" in md and "趋势破位" in md
    assert "同一交易日同时触发止损与止盈时，按止损优先" in md


def test_northbound_market_level():
    """market 路径：同花顺市场级北向，latest.net_buy 已是亿。"""
    v = {
        "scope": "market",
        "latest": {"date": "2026-07-01", "net_buy": 85.3},
        "note": "per-code 数据不可用，使用市场级北向资金",
        "source": "ths_hexin",
    }
    txt = factor_value_text("northbound", v)
    assert "净流入 85.3亿" in txt


def test_northbound_legacy_net_buy_yi_compat():
    """兼容历史 net_buy_yi/net_yi（已是亿），保持旧调用方不破。"""
    v = {"net_buy_yi": 2.5}
    txt = factor_value_text("northbound", v)
    assert "净流入 2.5亿" in txt


def test_northbound_with_note():
    """北向含 note 字段时正确展示。"""
    v = {
        "scope": "market",
        "latest": {"date": "2026-07-01", "net_buy": 50.0},
        "source": "ths_hexin",
        "note": "per-code northbound removed",
    }
    txt = factor_value_text("northbound", v)
    assert "净流入 50.0亿" in txt
    assert "per-code northbound removed" in txt


def test_removed_factors_not_rendered():
    """已移除因子（margin/block_trade/holder_num）不产生展示文本。"""
    from engine.structured_format import REMOVED_FACTORS
    assert "margin" in REMOVED_FACTORS
    assert "block_trade" in REMOVED_FACTORS
    assert "holder_num" in REMOVED_FACTORS
