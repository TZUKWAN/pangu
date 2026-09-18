"""Research-only replay of official announcement catalysts plus technical confirmation.

This module helps measure whether a news-event family has signal before it is
promoted into the full Agent.  It is intentionally labelled research-only:
historical market-emotion and theme-strength archives are not yet complete, so
its result can reject a weak idea but cannot validate the full 85% objective.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .announcement_archive import CninfoAnnouncementArchive
from .announcement_detail import AnnouncementDetail, AnnouncementDetailAnalyzer
from .frame_utils import safe_float
from .industry_trend_archive import IndustryTrendReader
from .market_breadth_archive import ArchivedBreadthKlineLoader
from .entry_exit import EntryExitEngine
from .short_term_replay import ReplayOutcome, ShortTermReplayConfig, ShortTermReplayEngine


_CORE_RESEARCH_EVENT_TYPES = {
    "performance_growth", "major_contract", "project_win", "shareholder_increase",
}


@dataclass(frozen=True)
class AnnouncementSignal:
    signal_date: str
    code: str
    name: str
    event_type: str
    title: str
    announcement_id: str
    adjunct_url: str


class _HistoricalSliceLoader:
    def __init__(self, frames: Mapping[str, pd.DataFrame]) -> None:
        self.frames = frames

    def daily_kline(self, code: str, days: int = 120, date: str | None = None, **_: Any) -> pd.DataFrame:
        frame = self.frames.get(str(code).zfill(6), pd.DataFrame()).copy()
        if frame.empty:
            return frame
        date_col = _column(frame, "日期", "date", "trade_date")
        if date_col and date:
            normalized = frame[date_col].astype(str).str.replace("-", "", regex=False).str[:8]
            frame = frame[normalized <= str(date)]
        return frame.tail(days).reset_index(drop=True)


class AnnouncementReplayResearch:
    def __init__(
        self,
        data_loader: Any,
        *,
        announcement_root: str | Path = "data/announcement_archive",
        workers: int = 8,
        replay_cfg: Mapping[str, Any] | None = None,
        detail_analyzer: AnnouncementDetailAnalyzer | None = None,
        market_breadth_root: str | Path = "data/market_breadth",
        industry_trend_root: str | Path = "data/industry_trend",
        industry_breadth_db: str | Path = "data/market_breadth/raw.sqlite3",
        industry_reader: IndustryTrendReader | None = None,
        event_entry_enabled: bool = False,
        include_buyback: bool = False,
    ) -> None:
        self.dl = data_loader
        self.archive = CninfoAnnouncementArchive(announcement_root)
        self.workers = max(1, int(workers))
        self.replay_cfg = ShortTermReplayConfig.from_dict(replay_cfg)
        self.detail_analyzer = detail_analyzer or AnnouncementDetailAnalyzer(
            Path(announcement_root).parent / "announcement_pdf"
        )
        self.market_breadth_root = Path(market_breadth_root)
        self._market_context_cache: dict[str, dict[str, Any]] = {}
        self.industry_reader = industry_reader or IndustryTrendReader(
            industry_trend_root, breadth_db=industry_breadth_db
        )
        self.event_entry_enabled = bool(event_entry_enabled)
        self.include_buyback = bool(include_buyback)

    def run(self, start: str, end: str, *, as_of: str) -> dict[str, Any]:
        signals = self._load_signals(start, end)
        frames, source_failures = self._load_klines({signal.code for signal in signals}, as_of)
        entry_engine = EntryExitEngine(
            _HistoricalSliceLoader(frames),
            {"horizon_days": 3, "sentiment_exit_drop": 15.0},
        )
        replay_engine = ShortTermReplayEngine(self.replay_cfg)
        outcomes: list[ReplayOutcome] = []
        selected_signals: list[AnnouncementSignal] = []
        technical_candidates: list[tuple[AnnouncementSignal, dict[str, float]]] = []
        sentiment_candidates: list[tuple[AnnouncementSignal, dict[str, float]]] = []
        industry_candidates: list[tuple[AnnouncementSignal, dict[str, float]]] = []
        signal_market_contexts: list[dict[str, Any]] = []
        signal_industry_contexts: list[dict[str, Any]] = []
        rejection_reasons: Counter[str] = Counter()

        for signal in signals:
            frame = frames.get(signal.code)
            snapshot, reason = self._technical_snapshot(frame, signal.signal_date)
            if reason:
                rejection_reasons[reason] += 1
                continue
            technical_candidates.append((signal, snapshot))

            market_context = self._market_context(signal.signal_date, allow_previous=True)
            signal_market_contexts.append({
                "signal_date": signal.signal_date,
                "code": signal.code,
                **market_context,
            })
            if not market_context.get("market_context_exact"):
                rejection_reasons["market_context_missing_or_incomplete"] += 1
                continue
            temperature = safe_float(market_context.get("current_temperature"), 0.0)
            if temperature < 40.0:
                rejection_reasons["market_sentiment_cold"] += 1
                continue
            if temperature > 85.0:
                rejection_reasons["market_sentiment_overheated"] += 1
                continue
            sentiment_candidates.append((signal, snapshot))

            industry_context = self.industry_reader.context(
                signal.code, signal.signal_date, allow_previous=True
            )
            signal_industry_contexts.append({
                "signal_date": signal.signal_date,
                "code": signal.code,
                **industry_context,
            })
            if not industry_context.get("theme_status_known"):
                rejection_reasons["industry_context_missing_or_incomplete"] += 1
                continue
            if not industry_context.get("theme_trend_strong"):
                status = str((industry_context.get("industry_trend") or {}).get("trend_status") or "unknown")
                rejection_reasons[f"industry_trend_{status}"] += 1
                continue
            industry_candidates.append((signal, snapshot))

        details = self._analyze_details([signal for signal, _ in industry_candidates])
        detail_records: list[dict[str, Any]] = []
        substantive_candidates: list[
            tuple[AnnouncementSignal, dict[str, float], AnnouncementDetail]
        ] = []
        for signal, snapshot in industry_candidates:
            detail = details[signal.announcement_id]
            detail_records.append({
                "signal_date": signal.signal_date,
                "code": signal.code,
                "name": signal.name,
                "title": signal.title,
                "adjunct_url": signal.adjunct_url,
                **detail.to_dict(),
            })
            if not detail.tradable_catalyst:
                rejection_reasons[self._detail_rejection_reason(detail)] += 1
                continue
            substantive_candidates.append((signal, snapshot, detail))

        # A stock can publish several same-day announcements. Keep only the
        # strongest catalyst known at signal time so a single trade is not
        # counted repeatedly.
        strongest: dict[
            tuple[str, str], tuple[AnnouncementSignal, dict[str, float], AnnouncementDetail]
        ] = {}
        for item in substantive_candidates:
            signal, _, detail = item
            key = (signal.signal_date, signal.code)
            incumbent = strongest.get(key)
            if incumbent is None or detail.catalyst_strength > incumbent[2].catalyst_strength:
                if incumbent is not None:
                    rejection_reasons["duplicate_lower_strength_same_stock_day"] += 1
                strongest[key] = item
            else:
                rejection_reasons["duplicate_lower_strength_same_stock_day"] += 1

        for signal, snapshot, _detail in sorted(
            strongest.values(), key=lambda item: (item[0].signal_date, item[0].code)
        ):
            frame = frames.get(signal.code)
            candidate = {
                "code": signal.code,
                "name": signal.name,
                "close": snapshot["close"],
            }
            event_entry = None
            if self.event_entry_enabled:
                reference_close = float(snapshot["close"])
                event_entry = {
                    "entry_style": "breakout_confirm",
                    "type": "公告次日确认买点",
                    "trigger_price": round(reference_close * 1.005, 2),
                    "trigger_condition": "公告后下一交易日价格上穿昨收约0.5%，且开盘未高于昨收3%",
                    "ideal_entry_zone": [
                        round(reference_close * 0.995, 2),
                        round(reference_close * 1.03, 2),
                    ],
                    "invalid_condition": "下一交易日高开超过昨收3%不追，或未触发确认价则放弃",
                }
            plan = entry_engine.compute(
                candidate,
                temperature=60,
                date=signal.signal_date,
                entry_override=event_entry,
            ).to_dict()
            if plan.get("warnings") or not plan.get("exit_plan"):
                rejection_reasons["entry_exit_unavailable"] += 1
                continue
            recommendation = {
                "code": signal.code,
                "entry_plan": plan.get("entry_plan") or {},
                "entry_exit": plan,
            }
            contexts = self._announcement_context(signal.code, frame, signal.signal_date)
            outcomes.append(replay_engine.replay(
                recommendation,
                frame,
                signal_date=signal.signal_date,
                daily_context=contexts,
                causal_signal_evidence=True,
            ))
            selected_signals.append(signal)

        acceptance = replay_engine.acceptance(outcomes)
        groups: dict[str, list[ReplayOutcome]] = defaultdict(list)
        signal_by_key = {
            (signal.signal_date, signal.code): signal for signal in selected_signals
        }
        for outcome in outcomes:
            signal = signal_by_key.get((outcome.signal_date, outcome.code))
            if signal:
                groups[signal.event_type].append(outcome)
        by_event = {
            event_type: self._group_metrics(items)
            for event_type, items in sorted(groups.items())
        }
        return {
            "research_only": True,
            "full_agent_verified": False,
            "entry_variant": "announcement_confirmation" if self.event_entry_enabled else "strict_technical",
            "event_scope": "core_plus_buyback" if self.include_buyback else "core_immediate_catalysts",
            "cannot_validate_reason": (
                "历史市场情绪与题材强度精确日期上下文未完整，"
                "本报告只能评估官方公告事件+技术确认子策略"
            ),
            "start": start,
            "end": end,
            "as_of": as_of,
            "raw_positive_event_signals": len(signals),
            "technical_confirmed_signals": len(technical_candidates),
            "sentiment_confirmed_signals": len(sentiment_candidates),
            "industry_trend_confirmed_signals": len(industry_candidates),
            "substance_confirmed_signals": len(strongest),
            "source_failures": source_failures,
            "rejection_reasons": dict(rejection_reasons),
            "announcement_details": detail_records,
            "signal_market_contexts": signal_market_contexts,
            "signal_industry_contexts": signal_industry_contexts,
            "acceptance": acceptance,
            "by_event_type": by_event,
            "outcomes": [outcome.to_dict() for outcome in outcomes],
        }

    def save_report(self, report: Mapping[str, Any], path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)
        return target

    def _load_signals(self, start: str, end: str) -> list[AnnouncementSignal]:
        seen: set[str] = set()
        output: list[AnnouncementSignal] = []
        current = datetime.strptime(start, "%Y%m%d").date()
        end_date = datetime.strptime(end, "%Y%m%d").date()
        while current <= end_date:
            date = current.strftime("%Y%m%d")
            archived = self.archive.load(date)
            if not archived or not archived.get("complete"):
                current += timedelta(days=1)
                continue
            for event in archived.get("events") or []:
                code = str(event.get("code") or "").zfill(6)
                event_type = str(event.get("event_type") or "")
                announcement_id = str(event.get("announcement_id") or "")
                allowed_types = set(_CORE_RESEARCH_EVENT_TYPES)
                if self.include_buyback:
                    allowed_types.add("share_buyback")
                if (
                    event.get("polarity") != "positive"
                    or event_type not in allowed_types
                    or not code.startswith(("00", "30", "60", "68"))
                    or not announcement_id
                    or announcement_id in seen
                ):
                    continue
                seen.add(announcement_id)
                output.append(AnnouncementSignal(
                    signal_date=date,
                    code=code,
                    name=str(event.get("name") or code),
                    event_type=event_type,
                    title=str(event.get("title") or ""),
                    announcement_id=announcement_id,
                    adjunct_url=str(event.get("adjunct_url") or ""),
                ))
            current += timedelta(days=1)
        return sorted(output, key=lambda signal: (signal.signal_date, signal.code, signal.event_type))

    def _analyze_details(self, signals: list[AnnouncementSignal]) -> dict[str, AnnouncementDetail]:
        details: dict[str, AnnouncementDetail] = {}

        def analyze(signal: AnnouncementSignal) -> tuple[str, AnnouncementDetail]:
            event = {
                "announcement_id": signal.announcement_id,
                "adjunct_url": signal.adjunct_url,
                "code": signal.code,
                "name": signal.name,
                "event_type": signal.event_type,
                "title": signal.title,
            }
            return signal.announcement_id, self.detail_analyzer.analyze(event)

        with ThreadPoolExecutor(max_workers=min(self.workers, max(1, len(signals)))) as executor:
            futures = [executor.submit(analyze, signal) for signal in signals]
            for future in as_completed(futures):
                announcement_id, detail = future.result()
                details[announcement_id] = detail
        return details

    @staticmethod
    def _detail_rejection_reason(detail: AnnouncementDetail) -> str:
        if detail.extraction_status != "ok":
            return f"announcement_detail_{detail.extraction_status}"
        if detail.contract_direction == "purchase":
            return "purchase_contract_not_revenue"
        if detail.event_type == "performance_growth" and detail.growth_lower_pct is None:
            return "performance_growth_without_quantified_lower_bound"
        if any("ST" in flag for flag in detail.risk_flags):
            return "st_or_delisting_risk"
        return "announcement_substance_not_tradable"

    def _load_klines(self, codes: set[str], as_of: str) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
        frames: dict[str, pd.DataFrame] = {}
        failures: list[dict[str, str]] = []

        def fetch(code: str) -> tuple[str, pd.DataFrame]:
            return code, self.dl.daily_kline(code, days=180, date=as_of)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(fetch, code): code for code in sorted(codes)}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    fetched_code, frame = future.result()
                    if frame is None or len(frame) == 0:
                        failures.append({"code": code, "reason": "empty_kline"})
                    else:
                        frames[fetched_code] = frame
                except Exception as exc:  # noqa: BLE001
                    failures.append({"code": code, "reason": str(exc)})
        return frames, failures

    @staticmethod
    def _technical_snapshot(frame: pd.DataFrame | None, signal_date: str) -> tuple[dict[str, float], str]:
        if frame is None or len(frame) == 0:
            return {}, "kline_missing"
        date_col = _column(frame, "日期", "date", "trade_date")
        close_col = _column(frame, "收盘", "close")
        volume_col = _column(frame, "成交量", "volume", "vol")
        if not date_col or not close_col:
            return {}, "kline_columns_missing"
        normalized = frame[date_col].astype(str).str.replace("-", "", regex=False).str[:8]
        history = frame[normalized <= signal_date].copy()
        if len(history) < 25:
            return {}, "history_lt_25"
        closes = pd.to_numeric(history[close_col], errors="coerce").dropna()
        if len(closes) < 25:
            return {}, "valid_close_lt_25"
        close = float(closes.iloc[-1])
        ma5 = float(closes.tail(5).mean())
        ma10 = float(closes.tail(10).mean())
        ma20 = float(closes.tail(20).mean())
        return_5d = close / float(closes.iloc[-6]) - 1.0
        if close < ma20:
            return {}, "below_ma20"
        if ma5 < ma10 * 0.995:
            return {}, "ma5_below_ma10"
        if return_5d > 0.08:
            return {}, "five_day_chasing"
        if return_5d < -0.05:
            return {}, "five_day_weakness"
        volume_ratio = 1.0
        if volume_col:
            volumes = pd.to_numeric(history[volume_col], errors="coerce").dropna()
            if len(volumes) >= 5 and float(volumes.tail(5).mean()) > 0:
                volume_ratio = float(volumes.iloc[-1]) / float(volumes.tail(5).mean())
        if volume_ratio < 0.70:
            return {}, "volume_too_weak"
        return {
            "close": close,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "return_5d": return_5d,
            "volume_ratio": volume_ratio,
        }, ""

    def _announcement_context(
        self,
        code: str,
        frame: pd.DataFrame,
        signal_date: str,
    ) -> dict[str, dict[str, Any]]:
        date_col = _column(frame, "日期", "date", "trade_date")
        if not date_col:
            return {}
        dates = sorted({
            value for value in frame[date_col].astype(str).str.replace("-", "", regex=False).str[:8]
            if value > signal_date
        })[:3]
        contexts: dict[str, dict[str, Any]] = {}
        for date in dates:
            archived = self.archive.load(date)
            if not archived or not archived.get("complete"):
                continue
            events = [
                event for event in (archived.get("events") or [])
                if str(event.get("code") or "").zfill(6) == code
            ]
            risks = [event.get("title") for event in events if event.get("polarity") == "negative"]
            market = self._market_context(date, allow_previous=False)
            theme = self.industry_reader.context(code, date, allow_previous=False)
            contexts[date] = {
                "news_evidence": {
                    "sentiment_label": "bearish" if risks else "neutral",
                    "risk_events": risks,
                },
                "news_context_exact": True,
                "market_phase": market.get("market_phase", "unknown"),
                "current_temperature": market.get("current_temperature", 0),
                "entry_temperature": self._market_context(
                    signal_date, allow_previous=True
                ).get("current_temperature", 0),
                "market_context_exact": bool(market.get("market_context_exact")),
                **theme,
            }
        return contexts

    def _market_context(self, date: str, *, allow_previous: bool) -> dict[str, Any]:
        cache_key = f"{date}:{int(allow_previous)}"
        cached = self._market_context_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        source_date = date
        path = self.market_breadth_root / f"{date}.json"
        if not path.exists() and allow_previous:
            available = sorted(path.stem for path in self.market_breadth_root.glob("20*.json"))
            position = bisect.bisect_right(available, date) - 1
            if position >= 0:
                candidate = available[position]
                gap = (
                    datetime.strptime(date, "%Y%m%d").date()
                    - datetime.strptime(candidate, "%Y%m%d").date()
                ).days
                if gap <= 3:
                    source_date = candidate
                    path = self.market_breadth_root / f"{candidate}.json"
        result: dict[str, Any] = {
            "market_context_exact": False,
            "market_context_date": "",
            "market_phase": "unknown",
            "current_temperature": 0.0,
        }
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                quality = payload.get("data_quality") or {}
                result = {
                    "market_context_exact": bool(quality.get("market_context_exact")),
                    "market_context_date": source_date,
                    "market_phase": str(payload.get("posture") or "unknown"),
                    "current_temperature": safe_float(payload.get("temperature"), 0.0),
                    "market_breadth": payload.get("breadth") or {},
                }
            except (OSError, ValueError, TypeError):
                pass
        self._market_context_cache[cache_key] = dict(result)
        return result

    @staticmethod
    def _group_metrics(outcomes: list[ReplayOutcome]) -> dict[str, Any]:
        closed = [outcome for outcome in outcomes if outcome.status == "closed" and outcome.net_return is not None]
        wins = [outcome for outcome in closed if outcome.win]
        returns = [float(outcome.net_return or 0.0) for outcome in closed]
        return {
            "signals": len(outcomes),
            "closed": len(closed),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(closed), 6) if closed else 0.0,
            "average_net_return": round(sum(returns) / len(returns), 6) if returns else 0.0,
            "no_entry": sum(1 for outcome in outcomes if outcome.status == "no_entry"),
            "pending": sum(1 for outcome in outcomes if outcome.status == "pending"),
            "invalid": sum(1 for outcome in outcomes if outcome.status == "invalid"),
        }


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    return next((lowered[name.lower()] for name in names if name.lower() in lowered), None)


def main() -> int:
    parser = argparse.ArgumentParser(description="回放巨潮公告正文催化与1-3日买卖闭环（研究用途）")
    parser.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="信号结束日期 YYYYMMDD")
    parser.add_argument("--as-of", required=True, help="行情可用截止日期 YYYYMMDD")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--kline-db",
        default="data/market_breadth/raw.sqlite3",
        help="优先使用已审计的精确历史OHLCV SQLite；不存在时才走配置行情源",
    )
    parser.add_argument(
        "--include-buyback",
        action="store_true",
        help="研究性纳入新回购方案；默认关闭，当前真实1-3日样本未显示优势",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--event-entry",
        action="store_true",
        help="研究性宽公告次日买点；默认关闭，因当前真实样本表现显著更差",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for label, value in (("start", args.start), ("end", args.end), ("as-of", args.as_of)):
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as exc:
            parser.error(f"{label} 必须是有效 YYYYMMDD 日期: {exc}")
    if args.start > args.end or args.end > args.as_of:
        parser.error("日期必须满足 start <= end <= as-of")

    from .config import load_config

    cfg = load_config(args.config)
    kline_db = Path(args.kline_db)
    if kline_db.exists():
        data_loader: Any = ArchivedBreadthKlineLoader(kline_db)
    else:
        from .config import build_data_loader
        data_loader = build_data_loader(cfg)
    research = AnnouncementReplayResearch(
        data_loader,
        workers=args.workers,
        replay_cfg=cfg.get("short_term_replay"),
        industry_breadth_db=kline_db,
        event_entry_enabled=args.event_entry,
        include_buyback=args.include_buyback,
    )
    report = research.run(args.start, args.end, as_of=args.as_of)
    target = research.save_report(report, args.output)
    acceptance = report["acceptance"]
    print(json.dumps({
        "report": str(target),
        "raw_positive_event_signals": report["raw_positive_event_signals"],
        "technical_confirmed_signals": report["technical_confirmed_signals"],
        "sentiment_confirmed_signals": report["sentiment_confirmed_signals"],
        "industry_trend_confirmed_signals": report["industry_trend_confirmed_signals"],
        "substance_confirmed_signals": report["substance_confirmed_signals"],
        "executed_count": acceptance["executed_count"],
        "win_rate": acceptance["observed_win_rate"],
        "wilson_lower_bound": acceptance["wilson_95_lower"],
        "status": acceptance["verification_status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
