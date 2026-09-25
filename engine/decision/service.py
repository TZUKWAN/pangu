"""Pangu 3.0 服务层（Phase 7 / Task 7.1）。

`recommend_next_session()` 是核心决策入口：
- 不依赖任何 LLM API key / LLM 模块（无网络时诚实降级）；
- 每次调用都检查数据源新鲜度并做增量刷新（Phase 2）；
- 输出完整可审计 DecisionRun（Phase 12 持久化）。

LLM 只在宿主侧负责理解问题和把结构化结果翻译成人话。
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.decision.asof import AsOfContext, TradingCalendar, build_asof_context
from engine.decision.clock import now_shanghai
from engine.decision.contracts import DecisionRequest, DecisionRun
from engine.decision.freshness import MarketContextRefresher
from engine.decision.ranker import Top20Ranker
from engine.decision.runstore import DecisionRunStore
from engine.evidence.engine import (EntityLinker, build_news_evidence,
                                    cluster_events)
from engine.evidence.model import EvidenceItem

ANNOUNCEMENT_DIR = Path("data/announcement_archive")
WSCN_DIR = Path("data/wscn_news_archive")


class PanguDecisionService:
    """组装数据面 → 证据面 → 排序器。所有依赖可注入以便测试。"""

    def __init__(self, store=None, refresher: Optional[MarketContextRefresher] = None,
                 calendar: Optional[TradingCalendar] = None,
                 runstore: Optional[DecisionRunStore] = None,
                 ranker: Optional[Top20Ranker] = None,
                 news_days: int = 3,
                 clock=None):
        self.store = store or self._build_store()
        self.clock = clock
        self.calendar = calendar or TradingCalendar()
        self.refresher = refresher or self._build_refresher()
        self.runstore = runstore or DecisionRunStore()
        self.ranker = ranker or Top20Ranker(self.store)
        self.news_days = news_days

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #
    def recommend_next_session(self, request: Optional[DecisionRequest] = None,
                               asof: Optional[str] = None,
                               limit: int = 20,
                               force_refresh: bool = False,
                               codes: Optional[List[str]] = None,
                               persist: bool = True) -> DecisionRun:
        """/pangu 主入口：生成下一交易日 Top20 决策候选（无 LLM 参与）。"""
        request = request or DecisionRequest(
            limit=limit, force_refresh=force_refresh, codes=codes)
        ctx = build_asof_context(clock=self.clock, asof=request.asof or asof,
                                 cal=self.calendar)
        if request.query_timestamp:
            ctx.query_timestamp = request.query_timestamp
        if request.requested_execution_date:
            ctx.execution_date = request.requested_execution_date

        # 查询时新鲜度检查 + 增量刷新
        if request.force_refresh and self.refresher._last_fetch:
            self.refresher._last_fetch.clear()
        self.refresher._force = request.force_refresh
        fresh_items = self.refresher.refresh_context(ctx)
        summary = MarketContextRefresher.summarize(fresh_items)
        ctx.warnings.extend(
            f"数据源 {s} 陈旧/失败" for s in summary["failed_sources"])

        # 证据面：近 N 天新闻 + 公告（asof 纪律由 build_news_evidence 强制）
        evidence = self._collect_evidence(ctx)
        names = self._universe_names(ctx.decision_date.replace("-", ""))
        linker = EntityLinker(names)
        raw_items = evidence["items"]
        links = {str(i): linker.link((it.get("title") or "") + " " +
                                     (it.get("summary") or ""))
                 for i, it in enumerate(raw_items)}
        ev_items = cluster_events(build_news_evidence(raw_items, ctx.asof_timestamp,
                                                      entity_links=links))

        run = self.ranker.rank(request, ctx, evidence=ev_items,
                               data_status=summary["data_status"],
                               source_health={i.source: i.quality
                                              for i in fresh_items},
                               data_freshness={"sources": [i.to_dict()
                                                           for i in fresh_items],
                                               "data_status": summary["data_status"]})
        run.evidence_version = f"evidence.v1:{len(ev_items)}"
        if persist:
            self.runstore.save(run)
        return run

    def status(self) -> Dict[str, Any]:
        """/pangu status：行情/新闻/公告/PIT 更新时间与 MCP 状态。"""
        ctx = build_asof_context(clock=self.clock, cal=self.calendar)
        items = self.refresher.refresh_context(ctx)
        s = MarketContextRefresher.summarize(items)
        return {
            "asof": ctx.asof_timestamp,
            "decision_date": ctx.decision_date,
            "execution_date": ctx.execution_date,
            "market_status": ctx.market_status.value,
            "data_status": s["data_status"],
            "sources": {i["source"]: {"fetched_at": i["fetched_at"],
                                      "age_seconds": i["age_seconds"],
                                      "stale": i["stale"], "quality": i["quality"]}
                        for i in s["sources"]},
            "pit_archive": {"max_date": self._pit_max_date()},
            "mcp": {"available": True, "transport": "stdio"},
        }

    def analyze_stock(self, code: str, asof: Optional[str] = None) -> DecisionRun:
        """/pangu 代码：单票完整分析。"""
        code6 = str(code).split(".")[-1].zfill(6)
        return self.recommend_next_session(
            DecisionRequest(limit=100, codes=[code6]), asof=asof)

    def explain(self, run_id: str, code: str) -> Optional[Dict[str, Any]]:
        """/pangu why <run_id> <code>：完整证据链重建。"""
        run = self.runstore.load(run_id)
        if not run or not run.recommendations:
            return None
        for d in run.recommendations.decisions:
            if d.code == str(code).split(".")[-1].zfill(6):
                return {"run": run.to_dict(), "decision": d.to_dict()}
        return None

    # ------------------------------------------------------------------ #
    def _collect_evidence(self, ctx: AsOfContext) -> Dict[str, Any]:
        days = self.news_days
        end = ctx.decision_date.replace("-", "")
        start = (_dt.datetime.strptime(end, "%Y%m%d")
                 - _dt.timedelta(days=days + 2)).strftime("%Y%m%d")
        items: List[dict] = []
        # WSCN 新闻档案（精确时间戳）
        d0 = _dt.datetime.strptime(start, "%Y%m%d")
        d1 = _dt.datetime.strptime(end, "%Y%m%d")
        cur = d0
        while cur <= d1:
            p = WSCN_DIR / f"{cur.strftime('%Y%m%d')}.json"
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    rows = data.get("items", data) if isinstance(data, dict) else data
                    for it in rows or []:
                        ts = it.get("published_at") or it.get("display_time")
                        if isinstance(ts, (int, float)):
                            ts = _dt.datetime.fromtimestamp(
                                ts / 1000 if ts > 1e11 else ts,
                                tz=_dt.timezone.utc).astimezone().isoformat()
                        items.append({"title": it.get("title", ""),
                                      "summary": it.get("summary", ""),
                                      "source": it.get("source", "wscn"),
                                      "published_at": ts,
                                      "url": it.get("url", "")})
                except (json.JSONDecodeError, OSError):
                    pass
            cur += _dt.timedelta(days=1)
        # 公告档案（Tier A）
        cur = d0
        while cur <= d1:
            p = ANNOUNCEMENT_DIR / f"{cur.strftime('%Y%m%d')}.json"
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    rows = data.get("events", data) if isinstance(data, dict) else data
                    for it in rows or []:
                        items.append({"title": it.get("title", ""),
                                      "summary": "",
                                      "source": "cninfo",
                                      "published_at": it.get("published_at"),
                                      "url": it.get("adjunctUrl", "")})
                except (json.JSONDecodeError, OSError):
                    pass
            cur += _dt.timedelta(days=1)
        return {"items": items}

    def _universe_names(self, day_c: str) -> Dict[str, str]:
        iso = f"{day_c[:4]}-{day_c[4:6]}-{day_c[6:]}"
        try:
            uni = self.store.universe(iso)
            return {str(c): str(n) for c, n in zip(uni.index, uni.get("name", uni.index))}
        except Exception:  # noqa: BLE001
            return {}

    def _pit_max_date(self) -> Optional[str]:
        try:
            d = self.store.trading_days("1990-01-01", "2099-12-31")
            return d[-1] if d else None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _build_store():
        from engine.data.pit_store import PITStore
        return PITStore()

    @staticmethod
    def _build_refresher() -> MarketContextRefresher:
        def _quote(asof_iso):
            from engine.tdx_source import ths_all_spot
            df = ths_all_spot()
            if df is None or df.empty:
                return {"quality": "failed", "fetched_at": asof_iso}
            return {"fetched_at": now_shanghai().isoformat(timespec="seconds")}

        def _news(asof_iso):
            from engine.news_fetcher import fetch_news
            rows = fetch_news()
            ok = bool(rows)
            return {"fetched_at": now_shanghai().isoformat(timespec="seconds"),
                    "quality": "ok" if ok else "degraded",
                    "detail": f"{len(rows)} items" if ok else "empty fetch"}

        def _static_source(kind: str):
            def _fn(asof_iso):
                base = {"announcements": ANNOUNCEMENT_DIR,
                        "wscn": WSCN_DIR}[kind]
                files = sorted(base.glob("*.json")) if base.exists() else []
                if not files:
                    return {"quality": "degraded", "detail": "archive empty"}
                latest = files[-1].stem           # YYYYMMDD
                return {"fetched_at": f"{latest[:4]}-{latest[4:6]}-{latest[6:8]}"
                                      "T08:00:00+08:00",
                        "quality": "ok"}
            return _fn

        def _daily_kline(asof_iso):
            store = None
            try:
                from engine.data.pit_store import PITStore
                store = PITStore()
                days = store.trading_days("1990-01-01", "2099-12-31")
                return {"fetched_at": days[-1] + "T15:05:00+08:00" if days else None,
                        "quality": "ok" if days else "failed",
                        "effective_at": days[-1] + "T15:05:00+08:00" if days else None}
            except Exception as e:  # noqa: BLE001
                return {"quality": "failed", "detail": repr(e)[:120]}

        return MarketContextRefresher(refreshers={
            "realtime_quote": _quote,
            "news": _news,
            "announcements": _static_source("announcements"),
            "daily_kline": _daily_kline,
        })


def recommend_next_session(asof: Optional[str] = None, limit: int = 20,
                           force_refresh: bool = False,
                           codes: Optional[List[str]] = None,
                           service: Optional[PanguDecisionService] = None,
                           persist: bool = True) -> DecisionRun:
    """模块级便捷入口（Phase 7 验收函数）。"""
    svc = service or PanguDecisionService()
    return svc.recommend_next_session(asof=asof, limit=limit,
                                      force_refresh=force_refresh,
                                      codes=codes, persist=persist)
