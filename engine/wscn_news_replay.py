"""Research replay for exact-time A-share 7x24 news, sentiment, industry trend, and exits."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .announcement_replay import (
    AnnouncementReplayResearch,
    AnnouncementSignal,
    _HistoricalSliceLoader,
    _column,
)
from .config import load_config
from .entry_exit import EntryExitEngine
from .market_breadth_archive import ArchivedBreadthKlineLoader
from .news_fetcher import NewsFlash, NewsResult
from .news_corroboration import OfficialCorroboration, OfficialNewsCorroborator
from .news_opportunity import NewsOpportunityScanner
from .short_term_replay import ReplayOutcome, ShortTermReplayConfig, ShortTermReplayEngine


_ALLOWED_EVENTS = {
    "earnings_positive",
    "major_contract",
    "product_breakthrough",
    "restructuring",
    "policy_support",
    "capacity_growth",
}
_TAG_RE = re.compile(r"<[^>]+>")
_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


@dataclass(frozen=True)
class WscnSignal:
    signal_date: str
    code: str
    name: str
    event_type: str
    theme: str
    score: float
    confidence: float
    published_at: str
    item_id: str
    evidence: str


class WscnNewsReplayResearch(AnnouncementReplayResearch):
    def __init__(
        self,
        data_loader: Any,
        *,
        news_root: str | Path = "data/wscn_news_archive",
        announcement_root: str | Path = "data/announcement_archive",
        breadth_db: str | Path = "data/market_breadth/raw.sqlite3",
        workers: int = 8,
        replay_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            data_loader,
            announcement_root=announcement_root,
            workers=workers,
            replay_cfg=replay_cfg,
            industry_breadth_db=breadth_db,
        )
        self.news_root = Path(news_root)
        self.breadth_db = Path(breadth_db)
        self.corroborator = OfficialNewsCorroborator(announcement_root)
        self.scanner = NewsOpportunityScanner({
            "short_term_agent": {
                "news_discovery_min_score": 76,
                "news_discovery_max_candidates": 500,
            }
        })
        self._universe_dates = self._available_universe_dates()
        self._universe_cache: dict[str, pd.DataFrame] = {}
        self._name_index_cache: dict[
            str, tuple[dict[str, tuple[str, str]], re.Pattern[str]]
        ] = {}

    def run(self, start: str, end: str, *, as_of: str) -> dict[str, Any]:
        signals, archive_failures, raw_items = self._load_news_signals(start, end)
        rejection_reasons: Counter[str] = Counter()
        corroborated: list[WscnSignal] = []
        corroboration_records: list[dict[str, Any]] = []
        corroboration_by_key: dict[tuple[str, str], OfficialCorroboration] = {}
        for signal in signals:
            official = self.corroborator.corroborate(
                code=signal.code,
                signal_date=signal.signal_date,
                signal_event_type=signal.event_type,
                signal_published_at=signal.published_at,
            )
            corroboration_records.append({
                "signal_date": signal.signal_date,
                "code": signal.code,
                "name": signal.name,
                "event_type": signal.event_type,
                "published_at": signal.published_at,
                "item_id": signal.item_id,
                **official.to_dict(),
            })
            if not official.confirmed:
                rejection_reasons[f"official_{official.status}"] += 1
                continue
            corroborated.append(signal)
            corroboration_by_key[(signal.signal_date, signal.code)] = official

        announcement_signals = [
            AnnouncementSignal(
                signal_date=signal.signal_date,
                code=signal.code,
                name=signal.name,
                event_type=corroboration_by_key[
                    (signal.signal_date, signal.code)
                ].official_event_type,
                title=corroboration_by_key[(signal.signal_date, signal.code)].title,
                announcement_id=corroboration_by_key[
                    (signal.signal_date, signal.code)
                ].announcement_id,
                adjunct_url=corroboration_by_key[
                    (signal.signal_date, signal.code)
                ].adjunct_url,
            )
            for signal in corroborated
        ]
        details = self._analyze_details(announcement_signals)
        detail_records: list[dict[str, Any]] = []
        identity_records: list[dict[str, Any]] = []
        detail_by_key: dict[tuple[str, str], Any] = {}
        identity_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        evidence_confirmed: list[WscnSignal] = []
        for signal in corroborated:
            key = (signal.signal_date, signal.code)
            official = corroboration_by_key[key]
            detail = details.get(official.announcement_id)
            if detail is None:
                rejection_reasons["announcement_detail_missing"] += 1
                continue
            detail_payload = detail.to_dict()
            detail_records.append({
                "signal_date": signal.signal_date,
                "code": signal.code,
                "name": signal.name,
                "news_event_type": signal.event_type,
                "official_corroboration": official.to_dict(),
                **detail_payload,
            })
            if not detail.tradable_catalyst:
                rejection_reasons[self._detail_rejection_reason(detail)] += 1
                continue
            identity = self.corroborator.audit_event_identity(
                news_evidence=signal.evidence,
                signal_event_type=signal.event_type,
                announcement_detail=detail_payload,
            )
            identity_payload = identity.to_dict()
            identity_records.append({
                "signal_date": signal.signal_date,
                "code": signal.code,
                "name": signal.name,
                "event_type": signal.event_type,
                "item_id": signal.item_id,
                "official_announcement_id": official.announcement_id,
                **identity_payload,
            })
            if not identity.matched:
                rejection_reasons[f"event_identity_{identity.status}"] += 1
                continue
            evidence_confirmed.append(signal)
            detail_by_key[key] = detail
            identity_by_key[key] = identity_payload

        frames, source_failures = self._load_klines(
            {signal.code for signal in evidence_confirmed}, as_of
        )
        entry_engine = EntryExitEngine(
            _HistoricalSliceLoader(frames),
            {"horizon_days": 3, "sentiment_exit_drop": 15.0},
        )
        replay_engine = ShortTermReplayEngine(self.replay_cfg)
        technical_count = 0
        sentiment_count = 0
        industry_count = 0
        substantive_count = len(evidence_confirmed)
        selected: list[WscnSignal] = []
        selected_records: list[dict[str, Any]] = []
        outcomes: list[ReplayOutcome] = []
        signal_contexts: list[dict[str, Any]] = []
        industry_candidates: list[
            tuple[WscnSignal, dict[str, float], float, pd.DataFrame, OfficialCorroboration]
        ] = []

        for signal in evidence_confirmed:
            key = (signal.signal_date, signal.code)
            official = corroboration_by_key[key]
            detail = detail_by_key[key]
            identity = identity_by_key[key]
            frame = frames.get(signal.code)
            snapshot, reason = self._technical_snapshot(frame, signal.signal_date)
            if reason:
                rejection_reasons[reason] += 1
                continue
            technical_count += 1
            market = self._market_context(signal.signal_date, allow_previous=True)
            if not market.get("market_context_exact"):
                rejection_reasons["market_context_missing_or_incomplete"] += 1
                continue
            temperature = float(market.get("current_temperature") or 0)
            if temperature < 40:
                rejection_reasons["market_sentiment_cold"] += 1
                continue
            if temperature > 85:
                rejection_reasons["market_sentiment_overheated"] += 1
                continue
            sentiment_count += 1
            industry = self.industry_reader.context(
                signal.code, signal.signal_date, allow_previous=True
            )
            signal_contexts.append({
                "signal_date": signal.signal_date,
                "code": signal.code,
                "event_type": signal.event_type,
                "theme": signal.theme,
                "score": signal.score,
                "confidence": signal.confidence,
                "official_corroboration": official.to_dict(),
                "event_identity": identity,
                "announcement_detail": detail.to_dict(),
                "market": market,
                "industry": industry,
            })
            if not industry.get("theme_status_known"):
                rejection_reasons["industry_context_missing_or_incomplete"] += 1
                continue
            if not industry.get("theme_trend_strong"):
                status = str((industry.get("industry_trend") or {}).get("trend_status") or "unknown")
                rejection_reasons[f"industry_trend_{status}"] += 1
                continue
            industry_count += 1
            industry_candidates.append((signal, snapshot, temperature, frame, official))

        for signal, snapshot, temperature, frame, official in industry_candidates:
            key = (signal.signal_date, signal.code)
            detail = detail_by_key[key]
            identity = identity_by_key[key]
            candidate = {"code": signal.code, "name": signal.name, "close": snapshot["close"]}
            plan = entry_engine.compute(
                candidate, temperature=temperature, date=signal.signal_date
            ).to_dict()
            if plan.get("warnings") or not plan.get("exit_plan"):
                rejection_reasons["entry_exit_unavailable"] += 1
                continue
            recommendation = {
                "code": signal.code,
                "entry_plan": plan.get("entry_plan") or {},
                "entry_exit": plan,
            }
            outcome = replay_engine.replay(
                recommendation,
                frame,
                signal_date=signal.signal_date,
                daily_context=self._daily_context(signal, frame),
                causal_signal_evidence=True,
            )
            outcomes.append(outcome)
            selected.append(signal)
            selected_records.append({
                **signal.__dict__,
                "official_corroboration": official.to_dict(),
                "event_identity": identity,
                "announcement_detail": detail.to_dict(),
            })

        acceptance = replay_engine.acceptance(outcomes)
        groups: dict[str, list[ReplayOutcome]] = defaultdict(list)
        signal_by_key = {(item.signal_date, item.code): item for item in selected}
        for outcome in outcomes:
            signal = signal_by_key.get((outcome.signal_date, outcome.code))
            if signal:
                groups[signal.event_type].append(outcome)
        return {
            "research_only": True,
            "full_agent_verified": False,
            "source": "wallstreetcn_a_stock_7x24_exact_cursor_archive",
            "start": start,
            "end": end,
            "as_of": as_of,
            "raw_news_items": raw_items,
            "news_signals": len(signals),
            "official_corroborated_signals": len(corroborated),
            "technical_confirmed_signals": technical_count,
            "sentiment_confirmed_signals": sentiment_count,
            "industry_trend_confirmed_signals": industry_count,
            "substantive_catalyst_signals": substantive_count,
            "archive_failures": archive_failures,
            "source_failures": source_failures,
            "rejection_reasons": dict(rejection_reasons),
            "causal_signal_basis": (
                "WSCN exact published time provides signal availability; CNINFO confirms fact and "
                "beneficiary role, but same-day historical CNINFO timestamps may have date-only precision"
            ),
            "official_corroboration": corroboration_records,
            "announcement_details": detail_records,
            "event_identity_audits": identity_records,
            "signal_contexts": signal_contexts,
            "signals": selected_records,
            "acceptance": acceptance,
            "by_event_type": {
                key: self._group_metrics(items) for key, items in sorted(groups.items())
            },
            "outcomes": [item.to_dict() for item in outcomes],
        }

    def _load_news_signals(
        self, start: str, end: str
    ) -> tuple[list[WscnSignal], list[str], int]:
        current = datetime.strptime(start, "%Y%m%d").date()
        end_date = datetime.strptime(end, "%Y%m%d").date()
        failures: list[str] = []
        raw_items = 0
        output: dict[tuple[str, str], WscnSignal] = {}
        while current <= end_date:
            date = current.strftime("%Y%m%d")
            payload = self._news_payload(date)
            if not payload or not payload.get("complete"):
                failures.append(date)
                current += timedelta(days=1)
                continue
            items = list(payload.get("items") or [])
            raw_items += len(items)
            spot = self._spot_for_date(date)
            news = NewsResult(date=date)
            news.flashes = [
                flash
                for item in items
                for flash in self._flashes(item, spot)
            ]
            scanned = self.scanner.scan(news, spot)
            for opportunity in scanned.opportunities:
                if (
                    opportunity.event_type not in _ALLOWED_EVENTS
                    or opportunity.score < 84.0
                    or opportunity.confidence < 80.0
                ):
                    continue
                evidence = opportunity.evidence[0] if opportunity.evidence else ""
                published_at = opportunity.published_at
                item_id = opportunity.event_fingerprints[0] if opportunity.event_fingerprints else ""
                signal = WscnSignal(
                    signal_date=date,
                    code=opportunity.code,
                    name=opportunity.name,
                    event_type=opportunity.event_type,
                    theme=opportunity.theme,
                    score=opportunity.score,
                    confidence=opportunity.confidence,
                    published_at=published_at,
                    item_id=item_id,
                    evidence=evidence,
                )
                key = (date, signal.code)
                incumbent = output.get(key)
                if incumbent is None or (signal.score, signal.confidence) > (
                    incumbent.score, incumbent.confidence
                ):
                    output[key] = signal
            current += timedelta(days=1)
        return sorted(output.values(), key=lambda item: (item.signal_date, item.code)), failures, raw_items

    def _flashes(
        self, item: Mapping[str, Any], spot: pd.DataFrame
    ) -> list[NewsFlash]:
        content = str(item.get("content") or item.get("title") or "")
        subjects = [
            str(theme.get("title") or "")
            for theme in (item.get("themes") or [])
            if isinstance(theme, Mapping) and theme.get("title")
        ]
        published = str(item.get("published_at") or "")
        time_text = published[11:16] if len(published) >= 16 else ""
        paragraphs = [
            segment.strip()
            for segment in re.split(r"\n\s*\n+", content)
            if segment.strip()
        ] or [content.strip()]
        segments: list[str] = []
        for paragraph in paragraphs:
            if len(self._direct_stocks(paragraph, spot)) <= 1:
                segments.append(paragraph)
                continue
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[。！？；])", paragraph)
                if sentence.strip()
            ]
            segments.extend(sentences or [paragraph])
        identifier = str(item.get("id") or "")
        return [
            NewsFlash(
                time=time_text,
                content=segment,
                important=bool(item.get("important")),
                subjects=subjects,
                stocks=self._direct_stocks(segment, spot),
                source="wscn_archive",
                content_hash=f"{identifier}:{index}",
            )
            for index, segment in enumerate(segments)
            if segment
        ]

    def _direct_stocks(self, content: str, spot: pd.DataFrame) -> list[dict[str, str]]:
        if spot.empty:
            return []
        names, name_pattern = self._name_index(spot)
        linked: dict[str, dict[str, str]] = {}
        for raw in _CODE_RE.findall(content):
            if raw.startswith(("00", "30", "60", "68")):
                linked[raw] = {"code": raw, "name": ""}
        parts = re.split(r"[：:]", content, maxsplit=1)
        prefix = parts[0] if len(parts) > 1 else ""
        normalized_prefix = self._normalize_name(prefix[-24:])
        if normalized_prefix in names:
            code, name = names[normalized_prefix]
            linked[code] = {"code": code, "name": name}
        action_markers = (
            "中标", "入围", "获批", "批准上市", "取得注册证", "签订", "获得订单",
            "收到通知书", "扭亏", "预增", "净利润增长", "投产", "量产", "技术突破",
            "拟收购", "资产注入", "控制权变更",
        )
        compact = re.sub(r"\s+", "", content)
        normalized_content = self._normalize_name(compact)
        matched_names = {
            match.group(0) for match in name_pattern.finditer(normalized_content)
        }
        for name_key in matched_names:
            code, name = names[name_key]
            if code in linked:
                continue
            for match in re.finditer(re.escape(name_key), normalized_content):
                following = normalized_content[match.end():match.end() + 18]
                for marker in action_markers:
                    position = following.find(marker)
                    if position < 0:
                        continue
                    bridge = following[:position]
                    ownership_bridge = any(value in bridge for value in (
                        "子公司", "孙公司", "旗下", "控股", "全资",
                    )) and len(bridge) <= 10
                    if position <= 6 or ownership_bridge:
                        linked[code] = {"code": code, "name": name}
                        break
                if code in linked:
                    break
        return list(linked.values())

    def _name_index(
        self, spot: pd.DataFrame
    ) -> tuple[dict[str, tuple[str, str]], re.Pattern[str]]:
        cache = getattr(self, "_name_index_cache", None)
        if cache is None:
            cache = {}
            self._name_index_cache = cache
        cache_key = str(spot.attrs.get("trade_date") or f"frame:{id(spot)}")
        if cache_key in cache:
            return cache[cache_key]
        names = {
            self._normalize_name(name): (str(code).zfill(6), str(name))
            for code, name in zip(spot["代码"], spot["名称"])
            if len(self._normalize_name(name)) >= 3
        }
        alternatives = sorted(names, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(name) for name in alternatives))
        cache[cache_key] = (names, pattern)
        return cache[cache_key]

    @staticmethod
    def _normalize_name(value: Any) -> str:
        text = _TAG_RE.sub("", str(value or ""))
        return re.sub(r"[\s*]+", "", text).strip()

    def _available_universe_dates(self) -> list[str]:
        with sqlite3.connect(f"file:{self.breadth_db.as_posix()}?mode=ro", uri=True) as connection:
            return [row[0] for row in connection.execute(
                "SELECT DISTINCT date FROM breadth_universe ORDER BY date"
            ).fetchall()]

    def _spot_for_date(self, date: str) -> pd.DataFrame:
        trade_date = self._previous_trade_date(date)
        if trade_date in self._universe_cache:
            return self._universe_cache[trade_date].copy()
        if not trade_date:
            return pd.DataFrame(columns=["代码", "名称"])
        with sqlite3.connect(f"file:{self.breadth_db.as_posix()}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT code,name FROM breadth_universe WHERE date=?", (trade_date,)
            ).fetchall()
        frame = pd.DataFrame([
            {"代码": code[3:], "名称": name} for code, name in rows
        ])
        frame.attrs["trade_date"] = trade_date
        self._universe_cache[trade_date] = frame
        return frame

    def _previous_trade_date(self, date: str) -> str:
        import bisect
        position = bisect.bisect_right(self._universe_dates, date) - 1
        return self._universe_dates[position] if position >= 0 else ""

    def _news_payload(self, date: str) -> dict[str, Any]:
        path = self.news_root / f"{date}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    def _daily_context(
        self, signal: WscnSignal, frame: pd.DataFrame
    ) -> dict[str, dict[str, Any]]:
        date_col = _column(frame, "日期", "date", "trade_date")
        if not date_col:
            return {}
        dates = sorted({
            value for value in frame[date_col].astype(str).str.replace("-", "", regex=False).str[:8]
            if value > signal.signal_date
        })[:3]
        entry_market = self._market_context(signal.signal_date, allow_previous=True)
        contexts: dict[str, dict[str, Any]] = {}
        for date in dates:
            payload = self._news_payload(date)
            exact_news = bool(payload.get("complete"))
            risks = self._risk_events(signal.code, date, payload)
            market = self._market_context(date, allow_previous=False)
            theme = self.industry_reader.context(signal.code, date, allow_previous=False)
            contexts[date] = {
                "news_evidence": {
                    "sentiment_label": "bearish" if risks else "neutral",
                    "risk_events": risks,
                },
                "news_context_exact": exact_news,
                "market_phase": market.get("market_phase", "unknown"),
                "current_temperature": market.get("current_temperature", 0),
                "entry_temperature": entry_market.get("current_temperature", 0),
                "market_context_exact": bool(market.get("market_context_exact")),
                **theme,
            }
        return contexts

    def _risk_events(self, code: str, date: str, payload: Mapping[str, Any]) -> list[str]:
        if not payload.get("complete"):
            return []
        spot = self._spot_for_date(date)
        news = NewsResult(date=date)
        news.flashes = [
            flash
            for item in (payload.get("items") or [])
            for flash in self._flashes(item, spot)
        ]
        scan = self.scanner.scan(news, spot)
        return [
            evidence
            for item in scan.risk_alerts if item.code == code
            for evidence in item.evidence[:3]
        ]

    @staticmethod
    def save_report(report: Mapping[str, Any], path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)
        return target


def main() -> int:
    parser = argparse.ArgumentParser(description="回放精确历史A股快讯+情绪+行业趋势+完整卖点")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--news-root", default="data/wscn_news_archive")
    parser.add_argument("--announcement-root", default="data/announcement_archive")
    parser.add_argument("--kline-db", default="data/market_breadth/raw.sqlite3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    research = WscnNewsReplayResearch(
        ArchivedBreadthKlineLoader(args.kline_db),
        news_root=args.news_root,
        announcement_root=args.announcement_root,
        breadth_db=args.kline_db,
        workers=args.workers,
        replay_cfg=cfg.get("short_term_replay"),
    )
    report = research.run(args.start, args.end, as_of=args.as_of)
    target = research.save_report(report, args.output)
    acceptance = report["acceptance"]
    print(json.dumps({
        "report": str(target),
        "raw_news_items": report["raw_news_items"],
        "news_signals": report["news_signals"],
        "official_corroborated_signals": report["official_corroborated_signals"],
        "technical_confirmed_signals": report["technical_confirmed_signals"],
        "industry_trend_confirmed_signals": report["industry_trend_confirmed_signals"],
        "substantive_catalyst_signals": report["substantive_catalyst_signals"],
        "executed_count": acceptance["executed_count"],
        "win_rate": acceptance["observed_win_rate"],
        "wilson_lower_bound": acceptance["wilson_95_lower"],
        "status": acceptance["verification_status"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
