"""Web UI 后端逻辑测试：FastAPI 路由 + 数据结构。

不依赖真实 akshare / LLM，通过注入 mock 数据验证 API 返回结构正确。
SSE 流式接口只验证协议格式（不真连 LLM）。
"""

import json

import pytest

fastapi = pytest.importorskip("fastapi")  # 没装 fastapi 则整体跳过
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------- #
# 测试用 mock 选股结果（覆盖所有 UI 渲染字段）
# ---------------------------------------------------------------------- #
MOCK_RESULT = {
    "date": "20260630",
    "sentiment": {
        "temperature": 72.5,
        "posture": "正常",
        "advice": "情绪温和，存在结构性机会。",
        "percentile_60d": 65.0,
        "percentile_250d": 48.0,
        "momentum": {"roc_5d": 8.5, "roc_10d": 3.2, "accel": 5.3},
        "signals": {"peak": False, "bottom": False, "divergence": "无"},
        "components": {
            "limit_up_count": {"raw": 65, "score": 78},
            "broke_rate": {"raw": 0.25, "score": 70},
        },
        "market_structure": {
            "indices": {"上证指数": {"close": 3000, "ma20": 2950, "deviate_pct": 1.69}},
            "margin": {"balance_yi": None, "warning": "两融数据源已停用"},
            "fund_flow": {"inflow_sectors": 12, "outflow_sectors": 8, "top3_inflow": ["AI算力"]},
        },
    },
    "boards": [
        {"name": "AI算力", "symbol": "BK0901", "pct": 4.3, "fund_net_wan": 85000, "score": 1.8},
    ],
    "candidates": [
        {
            "code": "000001", "name": "平安银行", "board": "AI算力",
            "close": 12.85, "pct_change": 6.5, "turnover_rate": 8.5,
            "circ_mv_yi": 2495, "rps": 92, "fund_inflow_days": 5, "score": 85,
            "reasons": ["均线多头排列", "突破近20日平台"],
            "recommend": {
                "recommend_score": 88, "grade": "S", "up_prob": 65,
                "target_pct": [5.2, 12.5], "tag": "趋势强+资金流入",
                "risk_reward_ratio": 2.1, "calibrated": True,
                "score_breakdown": {
                    "trend": 88, "fund": 75, "pattern": 85,
                    "risk_reward": 90, "risk_control": 70, "momentum": 80,
                    "news_sentiment": 72, "news_themes": ["AI", "资金流入"],
                    "news_explain": "来源：个股新闻/flash；情绪：positive，得分 72；与个股板块/理由共振，额外加分 +5.0",
                },
            },
            "entry_exit": {
                "buy_points": [{"price": 12.5, "type": "回踩位", "is_primary": True}],
                "stop_loss": {"price": 11.8, "method": "ATR止损"},
                "take_profit": [{"price": 14.2, "method": "盈亏比2:1"}],
                "position": {"shares": 800, "risk_pct": 0.85, "emotion_factor": 1.0},
                "risk_reward_ratio": 2.1,
            },
            "technical": {
                "ma": {"ma5": 12.5, "ma10": 12.0, "ma30": 11.5, "ma60": 10.8, "ma120": 10.1},
                "macd": {"dif": 0.12, "dea": 0.08, "hist": 0.08, "golden_cross": True, "hint": "MACD金叉，动能转强"},
                "volume": {"latest": 100000, "avg5": 80000, "volume_ratio": 1.25, "hint": "量能平稳"},
                "trend_windows": {
                    "week_1": {"days": 5, "return_pct": 3.2, "pullback_from_high_pct": -2.1, "state": "趋势内回踩"},
                    "week_2": {"days": 10, "return_pct": 6.5, "pullback_from_high_pct": -3.0, "state": "趋势内回踩"},
                    "month_1": {"days": 20, "return_pct": 12.0, "pullback_from_high_pct": -4.5, "state": "趋势内回踩"},
                },
                "kline": [{"date": "2026-06-30", "open": 12.0, "high": 13.0, "low": 11.9, "close": 12.85, "volume": 100000}],
                "hints": ["短中期均线多头排列（MA5>MA10>MA30）"],
                "warnings": [],
            },
            "debate": {
                "verdict": "观望",
                "confidence": 65,
                "reason": "技术面强势但估值偏高，建议观望。",
                "bull_points": ["均线多头排列", "放量突破平台"],
                "bear_points": ["估值偏高", "短期涨幅过大"],
                "mode": "rule",
            },
        }
    ],
    "rejected": [{"code": "300001", "name": "特锐德", "reason": "估值过高"}],
    "posture_advice": "按趋势选股，严格止损。",
    "warnings": ["采样口径说明"],
    "news": {
        "date": "20260630",
        "flashes": [{"time": "15:30", "content": "测试新闻", "important": False}],
        "stock_news": {"000001": [{"title": "平安银行业绩预增", "content": "", "source": "财联社", "time": "10:00"}]},
        "hot_themes": [["AI", 3]],
        "warnings": [],
    },
    "market_modules": {
        "short_line": {"title": "短线连板", "highest": 3, "two_plus_count": 5, "one_word_count": 1, "items": [{"label": "2板", "value": 3}]},
        "yesterday_performance": {"title": "昨日表现", "prev_trade_date": "20260629", "yesterday_limit_up_count": 20, "yesterday_two_plus_count": 4, "yesterday_one_word_count": 2, "today_avg_pct": 1.2},
    },
}


@pytest.fixture
def client(monkeypatch):
    """构造 TestClient，注入 mock 数据避免真实取数。"""
    from engine.web import server

    # 注入内存缓存，跳过真实 Pipeline
    monkeypatch.setattr(server, "_latest_result", MOCK_RESULT)
    monkeypatch.setattr(server, "_load_trade_calendar", lambda: ({"20260630", "20260701", "20260702"}, [], "akshare.tool_trade_date_hist_sina"))

    return TestClient(server.app)


# ---------------------------------------------------------------------- #
# 测试用例
# ---------------------------------------------------------------------- #
def test_index_returns_html(client):
    """根路由返回 HTML 单页。"""
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "盘古" in r.text
    assert "市场总览" in r.text  # 关键区块存在
    assert "价格" in r.text


def test_latest_returns_mock_data(client):
    """/api/latest 返回注入的 mock 结果。"""
    r = client.get("/api/latest")
    assert r.status_code == 200
    data = r.json()
    assert data["date"] == "20260630"
    assert data["sentiment"]["temperature"] == 72.5
    assert data["runtime"]["timezone"] == "Asia/Shanghai"
    assert "data_update_time" in data
    assert data["source_status"]["margin"] == "unavailable"
    assert data["daily_loop"]["target_trade_date"]
    assert data["report_status"]["date"] == "20260630"
    assert "freshness_status" in data["report_status"]
    assert data["sentiment_report"]["mode"] == "rule_multi_role_synthesis"
    assert data["sentiment_report"]["agents"]
    assert data["daily_loop"]["debate"]["with_debate"] == 1
    assert data["daily_loop"]["news"]["flash_count"] == 1
    assert data["xuanwu_pool"]["summary"]["candidate_count"] == 1
    assert data["candidates"][0]["xuanwu"]["status"] in ("xuanwu", "watch", "pending_ai", "rejected")
    assert "llm" in data["daily_loop"]
    assert "credentials" in data["daily_loop"]
    assert "scheduler" in data["daily_loop"]
    assert "notifier" in data["daily_loop"]


def test_governance_status_endpoint(client):
    """/api/governance/status exposes the daily decision-loop status."""
    r = client.get("/api/governance/status")
    assert r.status_code == 200
    data = r.json()
    loop = data["daily_loop"]
    assert loop["report"]["date"] == "20260630"
    assert loop["source"]["margin"] == "unavailable"
    assert loop["debate"]["coverage_pct"] == 100.0
    assert "backtest" in loop
    assert "portfolio" in loop
    assert "llm" in loop
    assert "credentials" in loop
    assert "scheduler" in loop
    assert "notifier" in loop
    status_text = json.dumps(loop, ensure_ascii=False)
    assert "api_key" not in status_text.lower()
    assert "webhook" in status_text.lower()
    assert "PANGU_LLM" in status_text


def test_recommendation_performance_endpoint_returns_empty_journal(client, monkeypatch, tmp_path):
    """/api/recommendations/performance returns a stable structure before any journal exists."""
    from engine.web import server

    db_path = tmp_path / "journal.db"
    monkeypatch.setattr(server, "load_config", lambda: {"output": {"db_path": str(db_path)}})
    monkeypatch.setattr(server, "build_data_loader", lambda cfg: object())

    r = client.get("/api/recommendations/performance?days=30")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["evaluation"] is None
    assert data["performance"]["total"] == 0
    assert data["performance"]["recommended"] == 0
    assert "horizons" in data["performance"]


def test_recommendation_record_latest_endpoint_writes_journal(client, monkeypatch, tmp_path):
    """/api/recommendations/record-latest records the current report for later review."""
    from engine.web import server

    db_path = tmp_path / "journal.db"
    monkeypatch.setattr(server, "load_config", lambda: {"output": {"db_path": str(db_path)}})
    monkeypatch.setattr(server, "build_data_loader", lambda cfg: object())

    r = client.post("/api/recommendations/record-latest")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["recorded"]["recorded"] == 1
    assert data["recorded"]["run_date"] == "20260630"

    perf = client.get("/api/recommendations/performance?days=365&only_recommended=false").json()["performance"]
    assert perf["total"] == 1

    recommended_perf = client.get("/api/recommendations/performance?days=365").json()["performance"]
    assert recommended_perf["only_recommended"] is True


def test_settings_endpoint_does_not_expose_api_key(client, monkeypatch):
    """/api/settings only returns provider metadata and key configured state."""
    from engine.web import server

    monkeypatch.setenv("PANGU_TEST_KEY", "super-secret-value")
    monkeypatch.setattr(
        server,
        "load_config",
        lambda: {
            "llm": {
                "providers": [
                    {
                        "name": "test",
                        "base_url": "https://llm.example/v1",
                        "model": "test-model",
                        "api_key_env": "PANGU_TEST_KEY",
                        "timeout": 30,
                        "enabled": True,
                    }
                ]
            }
        },
    )

    r = client.get("/api/settings")
    assert r.status_code == 200
    data = r.json()
    body = json.dumps(data, ensure_ascii=False)
    assert data["llm"]["providers"][0]["api_key_configured"] is True
    assert "super-secret-value" not in body


def test_settings_save_accepts_strategy_without_plain_key_persistence(client, monkeypatch):
    """/api/settings saves strategy overrides and passes only env-based LLM config."""
    from engine.web import server

    saved = {}
    monkeypatch.setattr(server, "load_config", lambda: {"llm": {"providers": []}, "xuanwu_pool": {}})
    monkeypatch.setattr(server, "save_llm_providers", lambda providers: saved.setdefault("providers", providers))
    monkeypatch.setattr(server, "save_strategy_settings", lambda strategy: saved.setdefault("strategy", strategy))

    r = client.post(
        "/api/settings",
        json={
            "llm": {
                "providers": [
                    {
                        "name": "test",
                        "base_url": "https://llm.example/v1",
                        "model": "m",
                        "api_key_env": "PANGU_TEST_KEY",
                        "api_key": "super-secret-value",
                    }
                ]
            },
            "strategy": {
                "xuanwu_pool": {
                    "min_total_score": 82,
                    "rule_min_total_score": 63,
                    "debate_top_n": 40,
                },
                "agent_prompts": {"bull": "看多 {stock_name}", "judge": "裁决 {data_json}"},
            },
        },
    )
    assert r.status_code == 200
    assert saved["providers"][0]["api_key_env"] == "PANGU_TEST_KEY"
    assert "api_key" not in saved["providers"][0]
    assert saved["strategy"]["xuanwu_pool"]["min_total_score"] == 82
    assert saved["strategy"]["xuanwu_pool"]["rule_min_total_score"] == 63
    assert saved["strategy"]["agent_prompts"]["bull"] == "看多 {stock_name}"
    assert "super-secret-value" not in json.dumps(r.json(), ensure_ascii=False)


def test_latest_sentiment_structure(client):
    """情绪数据结构完整（UI 渲染依赖的字段都在）。"""
    s = client.get("/api/latest").json()["sentiment"]
    assert 0 <= s["temperature"] <= 100
    assert s["posture"] in ("冰点", "正常", "亢奋")
    assert "components" in s
    assert "signals" in s
    assert "momentum" in s
    # components 是 {raw, score} 结构
    for k, v in s["components"].items():
        assert isinstance(v, dict)
        assert "score" in v


def test_candidate_has_all_render_fields(client):
    """候选股含 UI 需要的 recommend + entry_exit + debate 嵌套结构。"""
    c = client.get("/api/latest").json()["candidates"][0]
    # 基础字段
    for f in ("code", "name", "board", "close", "pct_change", "rps", "turnover_rate", "circ_mv_yi", "fund_inflow_days", "reasons"):
        assert f in c, f"缺少字段 {f}"
    # recommend
    rec = c["recommend"]
    for f in ("recommend_score", "grade", "up_prob", "target_pct", "tag", "calibrated", "score_breakdown"):
        assert f in rec, f"recommend 缺少 {f}"
    assert rec["grade"] in ("S", "A", "B", "C")
    # 雷达图 6 维（lhb 已移除）
    bd = rec["score_breakdown"]
    for dim in ("trend", "fund", "pattern", "risk_reward", "risk_control", "momentum", "news_sentiment"):
        assert dim in bd, f"score_breakdown 缺少 {dim}"
    # entry_exit
    ee = c["entry_exit"]
    for f in ("buy_points", "stop_loss", "take_profit", "position", "risk_reward_ratio"):
        assert f in ee, f"entry_exit 缺少 {f}"
    assert any(bp.get("is_primary") for bp in ee["buy_points"])
    tech = c["technical"]
    assert tech["ma"]["ma5"] == 12.5
    assert tech["macd"]["golden_cross"] is True
    assert tech["volume"]["volume_ratio"] == 1.25
    assert tech["trend_windows"]["week_1"]["return_pct"] == 3.2
    # debate 字段且为数组（前端 renderDebate 依赖数组）
    debate = c["debate"]
    for f in ("verdict", "confidence", "reason", "bull_points", "bear_points", "mode"):
        assert f in debate, f"debate 缺少 {f}"
    assert isinstance(debate["bull_points"], list)
    assert isinstance(debate["bear_points"], list)
    assert c["xuanwu"]["gates"]["board"] == "pass"
    assert "score_parts" in c["xuanwu"]["evidence"]


def test_latest_has_market_modules(client):
    """短线连板/昨日表现与热门板块分开展示。"""
    data = client.get("/api/latest").json()
    assert data["boards"][0]["name"] == "AI算力"
    assert data["market_modules"]["short_line"]["title"] == "短线连板"
    assert data["market_modules"]["yesterday_performance"]["yesterday_two_plus_count"] == 4


def test_news_latest_and_stream(client):
    """新闻快照与 SSE 推送可用。"""
    r = client.get("/api/news/latest")
    assert r.status_code == 200
    assert r.json()["news"]["flashes"][0]["content"] == "测试新闻"

    s = client.get("/api/news/stream?limit=1&interval_seconds=5")
    assert s.status_code == 200
    assert "event: news" in s.text
    assert "测试新闻" in s.text


def test_reports_endpoint(client):
    """/api/reports 返回列表结构。"""
    r = client.get("/api/reports")
    assert r.status_code == 200
    data = r.json()
    assert "reports" in data
    assert isinstance(data["reports"], list)


def test_scan_status_404_for_unknown_task(client):
    """查询不存在的 task_id 返回 404。"""
    r = client.get("/api/scan/nonexistent/status")
    assert r.status_code == 404


def test_get_pipeline_singleton_exists(monkeypatch):
    """/api/scan 依赖 get_pipeline；函数必须存在且复用单例。"""
    from engine.web import server

    created = []

    class DummyPipeline:
        pass

    def fake_build(cfg):
        created.append(cfg)
        return DummyPipeline()

    monkeypatch.setattr(server, "_pipeline", None)
    monkeypatch.setattr(server, "load_config", lambda: {"ok": True})
    monkeypatch.setattr(server, "_build_pipeline", fake_build)

    p1 = server.get_pipeline()
    p2 = server.get_pipeline()

    assert p1 is p2
    assert len(created) == 1


def test_llm_summary_handles_no_llm_config(monkeypatch):
    """LLM 未配置时，SSE 推送 error 事件而非崩溃。"""
    from engine.web import server

    # 清空配置让 build_llm_client 抛 ValueError
    monkeypatch.setattr(server, "_latest_result", MOCK_RESULT)
    monkeypatch.setattr(server, "load_config", lambda: {})

    with TestClient(server.app) as c:
        r = c.get("/api/llm/summary")
        assert r.status_code == 200
        body = r.text
        # 应该有 error 事件
        assert "event: error" in body or "未配置" in body or "error" in body


def test_llm_summary_handles_no_data(monkeypatch):
    """无选股数据时，SSE 推送 error 而非崩溃。"""
    from engine.web import server

    monkeypatch.setattr(server, "_latest_result", None)
    monkeypatch.setattr(server, "_find_latest_report", lambda: None)

    with TestClient(server.app) as c:
        r = c.get("/api/llm/summary")
        assert r.status_code == 200
        assert "error" in r.text or "无数据" in r.text or "暂无" in r.text


def test_empty_state(client, monkeypatch):
    """无数据时 /api/latest 返回 empty 标记。"""
    from engine.web import server
    monkeypatch.setattr(server, "_latest_result", None)
    monkeypatch.setattr(server, "_find_latest_report", lambda: None)
    r = client.get("/api/latest")
    assert r.status_code == 200
    assert r.json().get("empty") is True


# ---------------------------------------------------------------------- #
# 15:00 交易日推荐逻辑测试（monkeypatch 时间与日历，不影响生产逻辑）
# ---------------------------------------------------------------------- #
def _make_runtime(monkeypatch, dt_str: str, calendar_set: set[str], calendar_warnings: list[str] | None = None):
    """构造指定北京时间的 /api/latest 响应。"""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from engine.web import server

    dt = datetime.fromisoformat(dt_str).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(server, "_beijing_now", lambda: dt)
    monkeypatch.setattr(server, "_load_trade_calendar", lambda: (calendar_set, list(calendar_warnings or []), "akshare.tool_trade_date_hist_sina"))
    # 清空缓存避免受其他用例影响
    monkeypatch.setattr(server, "_latest_result", MOCK_RESULT)
    with TestClient(server.app) as c:
        return c.get("/api/latest").json()["runtime"]


def test_runtime_before_1500_recommends_today(monkeypatch):
    """交易日 15:00 前推荐当天。"""
    rt = _make_runtime(monkeypatch, "2026-07-01 14:30:00", {"20260701", "20260702"})
    assert rt["is_trade_day"] is True
    assert rt["market_session"] == "盘中"
    assert rt["recommended_trade_date"] == "20260701"
    assert rt["calendar_coverage"] == "含交易所节假日历"


def test_runtime_after_1500_recommends_next_trade_day(monkeypatch):
    """交易日 15:00 后推荐下一交易日。"""
    rt = _make_runtime(monkeypatch, "2026-07-01 15:30:00", {"20260701", "20260702", "20260703"})
    assert rt["is_trade_day"] is True
    assert rt["market_session"] == "盘后"
    assert rt["recommended_trade_date"] == "20260702"


def test_runtime_weekend_recommends_next_trade_day(monkeypatch):
    """周末推荐下一交易日（2026-07-05 为周日，下一交易日为 07-06）。"""
    rt = _make_runtime(monkeypatch, "2026-07-05 10:00:00", {"20260706", "20260707"})
    assert rt["is_trade_day"] is False
    assert rt["recommended_trade_date"] == "20260706"


def test_runtime_calendar_failure_warns_and_degrades(monkeypatch):
    """交易日历失败时显式 warning 并降级为仅周末判断。"""
    rt = _make_runtime(monkeypatch, "2026-07-01 14:30:00", set())
    assert rt["calendar_source"] == "weekday_fallback"
    assert rt["calendar_coverage"] == "仅周末降级，不保证节假日覆盖"
    assert any("交易日历" in w for w in rt["warnings"])


def test_runtime_non_trade_weekday_recommends_next_trade_day(monkeypatch):
    """非交易日（如国庆调休假期）按日历推荐下一交易日，避免仅看 weekday。"""
    rt = _make_runtime(monkeypatch, "2026-10-01 10:00:00", {"20260930", "20261008", "20261009"})
    assert rt["is_trade_day"] is False
    assert rt["recommended_trade_date"] == "20261008"


def test_runtime_supports_calendar_json_source(monkeypatch):
    """akshare import 失败但内置 calendar.json 可用时，应展示真实日历源而非 weekday fallback。"""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from engine.web import server

    dt = datetime.fromisoformat("2026-10-01 10:00:00").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(server, "_beijing_now", lambda: dt)
    monkeypatch.setattr(
        server,
        "_load_trade_calendar",
        lambda: ({"20260930", "20261008", "20261009"}, ["交易日历主源 akshare 不可用：boom"], "akshare.file_fold.calendar_json"),
    )
    monkeypatch.setattr(server, "_latest_result", MOCK_RESULT)
    with TestClient(server.app) as c:
        rt = c.get("/api/latest").json()["runtime"]
    assert rt["is_trade_day"] is False
    assert rt["recommended_trade_date"] == "20261008"
    assert rt["calendar_source"] == "akshare.file_fold.calendar_json"
    assert rt["calendar_coverage"] == "含 akshare 内置交易日历，覆盖到内置文件末端"


def test_load_trade_calendar_falls_back_to_akshare_calendar_json(monkeypatch, tmp_path):
    """不 import akshare 也能读取包内 file_fold/calendar.json，避开 py_mini_racer 故障。"""
    import builtins
    from types import SimpleNamespace
    from engine.web import server

    package_dir = tmp_path / "akshare"
    calendar_dir = package_dir / "file_fold"
    calendar_dir.mkdir(parents=True)
    (calendar_dir / "calendar.json").write_text(json.dumps(["20260701", "20260702"]), encoding="utf-8")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "akshare":
            raise ImportError("py_mini_racer broken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(server, "_trade_calendar_cache", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        server.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(package_dir)]) if name == "akshare" else None,
    )

    dates, warnings, source = server._load_trade_calendar()
    assert dates == {"20260701", "20260702"}
    assert source == "akshare.file_fold.calendar_json"
    assert any("akshare 不可用" in w for w in warnings)



def test_latest_candidate_news_sentiment_has_explain_and_sources(client):
    """候选股 recommend.score_breakdown 必须含 news_sentiment/news_explain/news_themes。"""
    c = client.get("/api/latest").json()["candidates"][0]
    rec = c["recommend"]
    bd = rec["score_breakdown"]
    assert "news_sentiment" in bd
    assert "news_themes" in bd
    # explain 字段由后端注入；mock 数据未带，因此只断言字段存在即代表契约支持
    assert "news_explain" in bd


def test_latest_top_level_news_has_stock_news_field(client):
    """/api/latest 顶层 news 必须含 stock_news 字段（可为空 dict）。"""
    data = client.get("/api/latest").json()
    assert "news" in data
    assert "stock_news" in data["news"]
    assert "hot_themes" in data["news"]
    assert "flashes" in data["news"]


def test_enrich_backfills_trend_windows_for_old_reports():
    """旧报告只有 K 线时，也应补出 1周/2周/1月窗口。"""
    from engine.web import server

    rows = [
        {"date": f"2026-06-{i:02d}", "high": 10 + i * 0.2, "low": 9 + i * 0.1, "close": 10 + i * 0.15}
        for i in range(1, 31)
    ]
    data = server._enrich_response({
        "date": "20260703",
        "sentiment": {"temperature": 60, "posture": "正常"},
        "boards": [],
        "candidates": [{"code": "000001", "name": "测试股", "technical": {"kline": rows}}],
        "news": {"flashes": [], "hot_themes": [], "stock_news": {}},
    })

    tw = data["candidates"][0]["technical"]["trend_windows"]
    assert set(tw) >= {"week_1", "week_2", "month_1"}
    assert "return_pct" in tw["month_1"]


def test_enrich_backfills_missing_debate_for_candidates():
    """候选缺少辩论时，后端应补出明确的规则验证结构。"""
    from engine.web import server

    data = server._enrich_response({
        "date": "20260703",
        "sentiment": {"temperature": 60, "posture": "正常"},
        "boards": [],
        "candidates": [{
            "code": "000001",
            "name": "测试股",
            "close": 10,
            "pct_change": 2,
            "turnover_rate": 5,
            "rps": 60,
            "recommend": {"recommend_score": 50},
            "entry_exit": {"buy_points": [], "risk_reward_ratio": 0},
            "technical": {"kline": []},
        }],
        "news": {"flashes": [], "hot_themes": [], "stock_news": {}},
    })

    debate = data["candidates"][0]["debate"]
    assert debate["verdict"] in {"推荐", "观望", "回避"}
    assert debate["mode"] == "rule_validation"
    assert debate["rule_degraded"] is True
