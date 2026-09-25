"""Phase 7/12 服务层测试：无 LLM、无网络的完整链路 + 运行历史。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_p3_ranker import (FakeStore, LAST_DAY, make_panel, make_event,
                            write_fake_research)

from engine.decision.asof import TradingCalendar
from engine.decision.clock import FrozenClock
from engine.decision.contracts import DecisionAction, DecisionRequest
from engine.decision.freshness import MarketContextRefresher
from engine.decision.runstore import DecisionRunStore
from engine.decision.service import PanguDecisionService


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    panel, idx, names = make_panel()
    store = FakeStore(panel, idx, names)
    ens = write_fake_research(tmp_path)
    clock = FrozenClock("2026-08-13T15:05:00+08:00")
    cal = TradingCalendar(known_days=["20260813", "20260814"], allow_online=False)

    class _Clock:
        def now(self):
            return clock.now()

    refresher = MarketContextRefresher(
        refreshers={
            "realtime_quote": lambda a: {"fetched_at": a, "quality": "ok"},
            "news": lambda a: {"fetched_at": a, "quality": "ok"},
            "announcements": lambda a: {"fetched_at": "2026-08-13T08:00:00+08:00",
                                        "quality": "ok"},
            "daily_kline": lambda a: {"fetched_at": "2026-08-13T15:05:00+08:00",
                                      "quality": "ok"},
            "industry_concept": lambda a: {"fetched_at": "2026-08-10T08:00:00+08:00",
                                           "quality": "ok"},
            "financials": lambda a: {"fetched_at": "2026-07-30T08:00:00+08:00",
                                     "quality": "ok"},
        },
        clock=_Clock())
    from engine.decision.ranker import Top20Ranker
    return PanguDecisionService(
        store=store, refresher=refresher, calendar=cal,
        runstore=DecisionRunStore(runs_dir=tmp_path / "runs"),
        ranker=Top20Ranker(store, ensemble=ens), news_days=3,
        clock=_Clock())


class TestServiceNoLLM:
    def test_module_imports_without_llm_stack(self):
        """核心服务不得 import 任何 LLM/agent 模块。"""
        import engine.decision.service as svc_mod
        src = open(svc_mod.__file__, encoding="utf-8").read()
        for banned in ("from engine.agent", "import engine.agent",
                       "openai", "anthropic", "llm", "LLMClient"):
            assert banned not in src, f"service imports banned module: {banned}"

    def test_recommend_works_without_api_key(self, svc, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        run = svc.recommend_next_session(limit=10)
        assert run.model_version == "pangu_ranker_v1"
        assert run.execution_date == "2026-08-14"
        assert len(run.recommendations.decisions) == 10

    def test_run_persisted_and_rebuildable(self, svc, tmp_path):
        run = svc.recommend_next_session(limit=5)
        loaded = svc.runstore.load(run.run_id)
        assert loaded is not None
        assert loaded.to_dict() == run.to_dict()
        assert svc.runstore.latest("2026-08-14").run_id == run.run_id

    def test_explain_reconstructs_decision(self, svc):
        run = svc.recommend_next_session(limit=5)
        code = run.recommendations.decisions[0].code
        out = svc.explain(run.run_id, code)
        assert out is not None
        assert out["decision"]["code"] == code
        assert "score_breakdown" in out["decision"]

    def test_status_reports_freshness(self, svc):
        st = svc.status()
        assert st["execution_date"] == "2026-08-14"
        assert st["data_status"] in ("ok", "degraded", "failed")
        assert "realtime_quote" in st["sources"]
        assert st["mcp"]["available"] is True

    def test_stale_critical_source_degrades(self, svc):
        # 关键源缓存 10 分钟前 → 超过 180s SLA → 刷新器会拉新（这里拉新返回 ok）
        # 改造：让实时行情刷新失败 → data_status=failed → 无 BUY
        def broken(asof):
            raise ConnectionError("down")
        svc.refresher._refreshers["realtime_quote"] = broken
        run = svc.recommend_next_session(limit=10)
        assert run.recommendations.data_status == "failed"
        assert all(d.decision != DecisionAction.BUY
                   for d in run.recommendations.decisions)

    def test_force_refresh_clears_cache(self, svc):
        svc.refresher._last_fetch["news"] = "2026-08-13T14:00:00+08:00"
        calls = []
        orig = svc.refresher._refreshers["news"]
        svc.refresher._refreshers["news"] = lambda a: (calls.append(a) or orig(a))
        svc.recommend_next_session(limit=3, force_refresh=True)
        assert calls, "force_refresh 必须触发增量刷新"

    def test_analyze_stock_single(self, svc):
        run = svc.analyze_stock("600001")
        assert len(run.recommendations.decisions) == 1
        assert run.recommendations.decisions[0].code == "600001"


class TestNoArchiveCleanRoom:
    """Round 5 发现：PIT 档案缺失（clean checkout）时服务不得崩溃。"""

    def test_status_without_archive_is_honest_failed(self, tmp_path, monkeypatch):
        import engine.data.pit_store as ps
        import engine.decision.service as svc_mod

        class BrokenStore:
            def __init__(self, *a, **k):
                raise FileNotFoundError("PIT 档案不存在")

        monkeypatch.setattr(ps, "PITStore", BrokenStore)
        svc = svc_mod.PanguDecisionService(
            refresher=MarketContextRefresher(refreshers={}),
            calendar=TradingCalendar(known_days=["20260813"], allow_online=False),
            runstore=DecisionRunStore(runs_dir=tmp_path / "runs"))
        st = svc.status()
        assert st["data_status"] == "failed"
        assert st["pit_archive"]["max_date"] is None
        assert "PIT" in st.get("pit_error", "")

    def test_recommend_without_archive_fails_closed(self, tmp_path, monkeypatch):
        import engine.data.pit_store as ps
        import engine.decision.service as svc_mod

        class BrokenStore:
            def __init__(self, *a, **k):
                raise FileNotFoundError("PIT 档案不存在")

        monkeypatch.setattr(ps, "PITStore", BrokenStore)
        svc = svc_mod.PanguDecisionService(
            refresher=MarketContextRefresher(refreshers={}),
            calendar=TradingCalendar(known_days=["20260813"], allow_online=False),
            runstore=DecisionRunStore(runs_dir=tmp_path / "runs"))
        with pytest.raises(RuntimeError, match="fail-closed"):
            svc.recommend_next_session(limit=5)
