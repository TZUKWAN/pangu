"""Phase 2 时间/新鲜度测试：任务书要求的 10 个场景 + 泄漏守卫。"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from engine.decision.asof import (AsOfContext, TradingCalendar,
                                  build_asof_context, market_status_at)
from engine.decision.clock import FrozenClock, SHANGHAI
from engine.decision.contracts import MarketStatus
from engine.decision.freshness import (CRITICAL_SOURCES, SOURCE_SLA,
                                       MarketContextRefresher, SourceFreshness)

# 2026-09 实际日历：9-26 是周六；9-28~9-30 交易日；10-1~10-8 国庆假期（简化）
KNOWN = [d for d in [
    "20260921", "20260922", "20260923", "20260924", "20260925",
    "20260928", "20260929", "20260930",
    "20261009", "20261012", "20261013",
] if d]
CAL = TradingCalendar(known_days=KNOWN, allow_online=False)


def _clock(s: str) -> FrozenClock:
    return FrozenClock(s)


class TestCalendarScenarios:
    def test_1_weekday_morning(self):
        ctx = build_asof_context(clock=_clock("2026-09-28T10:30:00+08:00"), cal=CAL)
        assert ctx.market_status == MarketStatus.OPEN
        assert ctx.decision_date == "2026-09-28"
        assert ctx.execution_date == "2026-09-29"

    def test_2_weekday_afternoon(self):
        ctx = build_asof_context(clock=_clock("2026-09-28T14:00:00+08:00"), cal=CAL)
        assert ctx.market_status == MarketStatus.OPEN
        assert ctx.execution_date == "2026-09-29"

    def test_3_after_close(self):
        ctx = build_asof_context(clock=_clock("2026-09-28T15:05:00+08:00"), cal=CAL)
        assert ctx.market_status == MarketStatus.CLOSED_AFTER
        assert ctx.decision_date == "2026-09-28"
        assert ctx.execution_date == "2026-09-29"

    def test_4_weekend_snaps_back_and_next_is_monday(self):
        ctx = build_asof_context(clock=_clock("2026-09-26T12:00:00+08:00"), cal=CAL)
        assert ctx.market_status == MarketStatus.WEEKEND
        assert ctx.decision_date == "2026-09-25"      # 吸附周五收盘信息
        assert ctx.execution_date == "2026-09-28"     # 下周一
        assert any("非交易日" in w for w in ctx.warnings)

    def test_5_holiday_before_and_after(self):
        # 节前最后交易日 09-30 → 执行日跳过整个国庆
        ctx = build_asof_context(clock=_clock("2026-09-30T15:05:00+08:00"), cal=CAL)
        assert ctx.execution_date == "2026-10-09"
        # 节假日期间询问 → 决策日吸附 09-30，执行日 10-09
        ctx2 = build_asof_context(clock=_clock("2026-10-03T12:00:00+08:00"), cal=CAL)
        assert ctx2.market_status == MarketStatus.HOLIDAY
        assert ctx2.decision_date == "2026-09-30"
        assert ctx2.execution_date == "2026-10-09"

    def test_9_midnight_crossover(self):
        clk = _clock("2026-09-28T23:59:00+08:00")
        ctx1 = build_asof_context(clock=clk, cal=CAL)
        clk.advance(minutes=2)                        # 跨日 → 09-29 00:01
        ctx2 = build_asof_context(clock=clk, cal=CAL)
        assert ctx1.decision_date == "2026-09-28"
        assert ctx2.decision_date == "2026-09-29"
        assert ctx2.execution_date == "2026-09-30"

    def test_10_timezone_conversion(self):
        # UTC 输入 07:30Z = 北京 15:30
        ctx = build_asof_context(clock=_clock("2026-09-28T07:30:00+00:00"), cal=CAL)
        assert ctx.market_status == MarketStatus.CLOSED_AFTER
        assert ctx.asof_timestamp.startswith("2026-09-28T15:30")

    def test_estimated_calendar_warns(self):
        # 超出已知范围的日期 → 工作日规则估算 + 显式 warning
        cal = TradingCalendar(known_days=["20260925"], allow_online=False)
        ctx = build_asof_context(clock=_clock("2026-12-28T15:05:00+08:00"), cal=cal)
        assert ctx.calendar_estimated is True
        assert any("估算" in w for w in ctx.warnings)

    def test_no_lookahead_execution_always_future(self):
        ctx = build_asof_context(clock=_clock("2026-09-26T12:00:00+08:00"), cal=CAL)
        assert ctx.execution_date > ctx.decision_date

    def test_lunch_break(self):
        ctx = build_asof_context(clock=_clock("2026-09-28T12:30:00+08:00"), cal=CAL)
        assert ctx.market_status == MarketStatus.LUNCH_BREAK


class TestFreshness:
    def _ctx(self, asof="2026-09-28T15:05:00+08:00"):
        return AsOfContext(query_timestamp=asof, asof_timestamp=asof,
                           decision_date="2026-09-28", execution_date="2026-09-29",
                           market_status=MarketStatus.CLOSED_AFTER)

    def test_7_cache_within_sla_no_refresh(self):
        calls = []
        r = MarketContextRefresher(
            refreshers={"news": lambda a: calls.append(a) or {"fetched_at": a}},
            sla_overrides={"news": 1800.0},                      # 30 分钟 SLA
            last_fetch={"news": "2026-09-28T14:36:00+08:00"})   # 29 分钟前
        items = r.refresh_context(self._ctx())
        news = next(i for i in items if i.source == "news")
        assert news.age_seconds == 29 * 60.0
        assert news.stale is False and news.fallback_used is False
        assert calls == []                                       # 未触发刷新

    def test_8_cache_beyond_sla_triggers_refresh(self):
        calls = []
        r = MarketContextRefresher(
            refreshers={"news": lambda a: calls.append(a) or {"fetched_at": a}},
            sla_overrides={"news": 1800.0},
            last_fetch={"news": "2026-09-28T14:34:00+08:00"})   # 31 分钟前
        items = r.refresh_context(self._ctx())
        news = next(i for i in items if i.source == "news")
        assert calls == ["2026-09-28T15:05:00+08:00"]           # 增量刷新被触发
        assert news.stale is False and news.fetched_at is not None

    def test_6_source_down_marks_failed_and_critical(self):
        def boom(asof):
            raise ConnectionError("DNS/timeout")
        r = MarketContextRefresher(refreshers={
            "realtime_quote": boom, "news": boom})
        s = MarketContextRefresher.summarize(r.refresh_context(self._ctx()))
        assert s["data_status"] == "failed"
        assert "realtime_quote" in s["critical_stale_or_failed"]
        rt = next(i for i in s["sources"] if i["source"] == "realtime_quote")
        assert rt["quality"] == "failed" and rt["fallback_used"] is True
        assert rt["stale"] is True

    def test_4_freshness_metadata_complete(self):
        r = MarketContextRefresher(last_fetch={}, refreshers={
            "realtime_quote": lambda a: {"fetched_at": a, "published_at": a,
                                         "effective_at": a, "quality": "ok"}})
        s = MarketContextRefresher.summarize(r.refresh_context(self._ctx()))
        rt = next(i for i in s["sources"] if i["source"] == "realtime_quote")
        for key in ("source", "fetched_at", "published_at", "effective_at",
                    "asof", "age_seconds", "stale", "quality", "fallback_used"):
            assert key in rt, f"freshness metadata missing {key}"

    def test_critical_sources_constant_matches_sla(self):
        for s in CRITICAL_SOURCES:
            assert s in SOURCE_SLA

    def test_no_refresh_rebuilds_full_archive(self):
        """增量语义：刷新器只被调用一次且不带全量参数（结构性约束由契约保证）。"""
        calls = []
        r = MarketContextRefresher(refreshers={
            "announcements": lambda a: calls.append(a) or {"fetched_at": a}},
            last_fetch={})
        r.refresh_context(self._ctx())
        assert len(calls) == 1
