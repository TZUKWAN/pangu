"""Structured P0 market factors from non-Eastmoney public data sources.

All Eastmoney (东方财富) endpoints (datacenter-web, reportapi, push2ex, push2his,
emappdata) have been removed per policy. Replaced with 同花顺/巨潮/新浪/腾讯/百度.

Data exclusively available only from Eastmoney (no free alternative):
  - 龙虎榜 (dragon_tiger) → removed_per_policy
  - 融资融券 (margin) → removed_per_policy
  - 大宗交易 (block_trade) → removed_per_policy
  - 股东户数 (holder_num) → removed_per_policy

The collector is deliberately defensive: every external source reports its own
state and failures never block the main scan. Candidate scores may only consume
fields that were really returned by an upstream endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import requests

logger = logging.getLogger("pangu.p0_factors")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# ── Removed Eastmoney URLs ──
# DATACENTER_URL / REPORT_API / push2ex / push2his / emappdata all deleted.
_REMOVED_EASTMONEY_NOTE = (
    "removed_per_policy: 东方财富数据源已全部移除，该因子无免费替代源"
)


def _num(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "-"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _code(value: Any) -> str:
    return str(value or "").strip().zfill(6)[-6:]


def _date_iso(date: str) -> str:
    s = str(date or "")
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s[:10]


def _date_yyyymmdd(date: str) -> str:
    return _date_iso(date).replace("-", "")


def _cache_key(parts: list[str]) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


@dataclass
class SourceRecorder:
    name: str
    target_count: int = 0
    item_count: int = 0
    success_count: int = 0
    skipped_count: int = 0
    warnings: list[str] = field(default_factory=list)
    status: str = "empty"

    def ok(self, item_count: int = 0) -> None:
        self.item_count += int(item_count or 0)
        self.success_count += 1
        if self.status != "degraded":
            self.status = "ok" if self.item_count else "empty"

    def degraded_ok(self, item_count: int = 0) -> None:
        self.item_count += int(item_count or 0)
        self.success_count += 1
        self.status = "degraded"

    def warn(self, message: str) -> None:
        if message and message not in self.warnings:
            self.warnings.append(message)
        if self.status not in ("ok", "degraded", "removed"):
            self.status = "degraded"

    def unavailable(self, message: str) -> None:
        self.warn(message)
        self.status = "unavailable"

    def removed(self, message: str = "") -> None:
        if message and message not in self.warnings:
            self.warnings.append(message)
        self.status = "removed"

    def skipped(self, message: str) -> None:
        self.skipped_count += 1
        self.warn(message)
        if self.status == "empty":
            self.status = "skipped"

    def to_dict(self) -> dict[str, Any]:
        status = self.status
        if self.target_count and self.success_count and self.success_count < self.target_count:
            status = "degraded"
        return {
            "status": status,
            "target_count": self.target_count,
            "success_count": self.success_count,
            "skipped_count": self.skipped_count,
            "item_count": self.item_count,
            "coverage_pct": round(self.success_count / self.target_count * 100, 2) if self.target_count else None,
            "warnings": self.warnings[:8],
        }


class P0FactorCollector:
    """Collect high-value structured factors for candidates (non-Eastmoney only)."""

    def __init__(self, cfg: dict[str, Any] | None = None, dl: Any = None) -> None:
        self.cfg = (cfg or {}).get("structured_data", {}) if cfg else {}
        self.enabled = self.cfg.get("enabled", True)
        self.deep_max_candidates = int(self.cfg.get("deep_max_candidates", 100))
        self.fund_flow_max_candidates = int(self.cfg.get("fund_flow_max_candidates", 100))
        self.irm_max_candidates = int(self.cfg.get("irm_max_candidates", 10))
        self.workers = max(1, int(self.cfg.get("workers", 4)))
        self.timeout = float(self.cfg.get("timeout", 15.0))
        self.total_budget_seconds = float(self.cfg.get("total_budget_seconds", 300))
        self.min_interval = float(self.cfg.get("min_interval", 0.5))
        self.dl = dl
        self.source_limits = {
            "capital_flow_120d": int(self.cfg.get("capital_flow_120d_max_candidates", self.fund_flow_max_candidates)),
            "margin": int(self.cfg.get("margin_max_candidates", 0)),
            "lockup": int(self.cfg.get("lockup_max_candidates", self.deep_max_candidates)),
            "research": int(self.cfg.get("research_max_candidates", 50)),
            "announcements": int(self.cfg.get("announcements_max_candidates", 100)),
            "irm": int(self.cfg.get("irm_max_candidates", self.irm_max_candidates)),
            "northbound": int(self.cfg.get("northbound_max_candidates", 50)),
            "block_trade": int(self.cfg.get("block_trade_max_candidates", 0)),
            "holder_num": int(self.cfg.get("holder_num_max_candidates", 0)),
            "dividend": int(self.cfg.get("dividend_max_candidates", 50)),
        }
        self.source_enabled = {
            "dragon_tiger_daily": self.cfg.get("dragon_tiger_daily_enabled", True),
            "hot_rank": self.cfg.get("hot_rank_enabled", True),
            "limit_up_sentiment": self.cfg.get("limit_up_sentiment_enabled", True),
            "capital_flow_120d": self.cfg.get("capital_flow_120d_enabled", True),
            "margin": self.cfg.get("margin_enabled", False),
            "lockup": self.cfg.get("lockup_enabled", True),
            "research": self.cfg.get("research_enabled", True),
            "announcements": self.cfg.get("announcements_enabled", True),
            "irm": self.cfg.get("irm_enabled", False),
            "northbound": self.cfg.get("northbound_enabled", True),
            "block_trade": self.cfg.get("block_trade_enabled", False),
            "holder_num": self.cfg.get("holder_num_enabled", False),
            "dividend": self.cfg.get("dividend_enabled", True),
        }
        self.cache_ttl_hours = float(self.cfg.get("cache_ttl_hours", 24))
        self.cache_dir = Path(self.cfg.get("cache_dir", "data/cache/p0_factors"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._cninfo_orgid_map: dict[str, str] = {}
        self._deadline = 0.0

    def collect(
        self,
        date: str,
        candidates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Mutate candidates with ``structured_factors`` and return source state."""
        if not self.enabled or not candidates:
            return {}, {}
        self._deadline = time.monotonic() + max(1.0, self.total_budget_seconds)

        codes = [_code(c.get("code")) for c in candidates if c.get("code")]
        factors: dict[str, dict[str, Any]] = {
            code: {"source_coverage": {}} for code in codes
        }
        state: dict[str, SourceRecorder] = {}

        def rec(name: str, target_count: int = 0) -> SourceRecorder:
            if name not in state:
                state[name] = SourceRecorder(name=name, target_count=target_count)
            elif target_count:
                state[name].target_count = max(state[name].target_count, target_count)
            return state[name]

        iso_date = _date_iso(date)
        ymd = _date_yyyymmdd(date)

        # ── Market-level sources ──
        if self._source_allowed("dragon_tiger_daily", rec("dragon_tiger_daily")):
            self._attach_daily_dragon_tiger(iso_date, factors, rec("dragon_tiger_daily"))
        if self._source_allowed("hot_rank", rec("hot_rank")):
            self._attach_hot_rank(factors, rec("hot_rank"))
        market_extra = {}
        if self._source_allowed("limit_up_sentiment", rec("limit_up_sentiment")):
            market_extra = self._collect_limit_sentiment(ymd, rec("limit_up_sentiment"))

        # ── Per-code sources ──
        self._collect_limited_source("capital_flow_120d", codes,
                                     self._stock_fund_flow_120d_summary, factors, rec)
        self._collect_limited_source("margin", codes,
                                     self._margin_trading_summary, factors, rec)
        self._collect_limited_source("lockup", codes,
                                     lambda code: self._lockup_summary(code, iso_date), factors, rec)
        self._collect_limited_source("research", codes,
                                     self._research_summary, factors, rec)
        self._collect_limited_source("announcements", codes,
                                     self._announcement_summary, factors, rec)
        self._collect_limited_source("irm", codes,
                                     self._irm_summary, factors, rec)
        self._collect_limited_source("northbound", codes,
                                     self._northbound_summary, factors, rec)
        self._collect_limited_source("block_trade", codes,
                                     self._block_trade_summary, factors, rec)
        self._collect_limited_source("holder_num", codes,
                                     self._holder_num_summary, factors, rec)
        self._collect_limited_source("dividend", codes,
                                     self._dividend_summary, factors, rec)

        for c in candidates:
            code = _code(c.get("code"))
            sf = factors.get(code) or {"source_coverage": {}}
            self._derive_notes(sf)
            c["structured_factors"] = sf
            for reason in sf.get("reasons") or []:
                reasons = c.setdefault("reasons", [])
                if reason not in reasons:
                    reasons.append(reason)

        source_state = {name: recorder.to_dict() for name, recorder in state.items()}
        source_state["summary"] = {
            "status": "ok" if any(s.get("status") == "ok" for s in source_state.values() if isinstance(s, dict)) else "degraded",
            "candidate_count": len(candidates),
            "deep_max_candidates": self.deep_max_candidates,
            "total_budget_seconds": self.total_budget_seconds,
            "remaining_budget_seconds": max(0.0, round(self._deadline - time.monotonic(), 2)),
            "note": "All Eastmoney sources removed per policy. Only non-EM public endpoints used.",
        }
        return source_state, market_extra

    # ── Time / budget ────────────────────────────────────────────
    def _time_left(self) -> float:
        if not self._deadline:
            return self.total_budget_seconds
        return self._deadline - time.monotonic()

    def _source_allowed(self, name: str, recorder: SourceRecorder) -> bool:
        if not self.source_enabled.get(name, True):
            recorder.skipped("disabled by structured_data config")
            return False
        if self._time_left() <= 1.0:
            recorder.skipped("skipped because structured_data total budget was exhausted")
            return False
        return True

    def _collect_limited_source(
        self, name: str, all_codes: list[str], fn: Callable[[str], dict[str, Any]],
        factors: dict[str, dict[str, Any]], rec: Callable[[str, int], SourceRecorder],
    ) -> None:
        limit = max(0, self.source_limits.get(name, self.deep_max_candidates))
        recorder = rec(name, min(len(all_codes), limit))
        if not self._source_allowed(name, recorder):
            return
        if limit <= 0:
            recorder.removed(_REMOVED_EASTMONEY_NOTE)
            return
        codes = all_codes[:limit]
        omitted = max(0, len(all_codes) - len(codes))
        if omitted:
            recorder.warn(f"limited to top {len(codes)} candidates; {omitted} candidates omitted for runtime control")
        self._collect_per_code(name, codes, fn, factors, recorder)

    def _collect_per_code(
        self, name: str, codes: list[str], fn: Callable[[str], dict[str, Any]],
        factors: dict[str, dict[str, Any]], recorder: SourceRecorder,
    ) -> None:
        if not codes:
            return
        if self._time_left() <= 1.0:
            recorder.skipped("skipped because structured_data total budget was exhausted")
            return
        max_workers = max(1, min(self.workers, len(codes)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fn, code): code for code in codes}
            try:
                for future in as_completed(futures, timeout=max(1.0, self._time_left())):
                    if self._time_left() <= 0:
                        recorder.skipped("stopped early because structured_data total budget was exhausted")
                        for pending in futures:
                            pending.cancel()
                        break
                    code = futures[future]
                    try:
                        value = future.result()
                    except Exception as e:  # noqa: BLE001
                        # Retry once on failure
                        try:
                            value = fn(code)
                        except Exception:  # noqa: BLE001
                            recorder.warn(f"{code}: {e}")
                            factors[code]["source_coverage"][name] = "error"
                            continue
                    if value:
                        factors[code][name] = value
                        src = value.get("source", "")
                        if isinstance(src, str) and ("fallback" in src or "degraded" in src):
                            factors[code]["source_coverage"][name] = "degraded"
                            recorder.degraded_ok(1)
                        else:
                            factors[code]["source_coverage"][name] = "ok"
                            recorder.ok(1)
                    else:
                        factors[code]["source_coverage"][name] = "empty"
            except FuturesTimeoutError:
                recorder.skipped("stopped early because structured_data total budget was exhausted")
                for future, code in futures.items():
                    if not future.done():
                        future.cancel()
                        factors[code]["source_coverage"][name] = "skipped"

    # ── HTTP helpers ─────────────────────────────────────────────
    def _get_json(
        self, method: str, url: str, *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        throttle: bool = False,
    ) -> Any:
        """HTTP GET/POST with optional throttling."""
        if throttle:
            with self._lock:
                wait = self.min_interval - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.time()
        hdr = {"User-Agent": UA, **(headers or {})}
        if method.upper() == "POST":
            resp = self.session.post(url, params=params, data=data, json=json_body,
                                     headers=hdr, timeout=self.timeout)
        else:
            resp = self.session.get(url, params=params, headers=hdr, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _cached(self, name: str, parts: list[str], fn: Callable[[], Any]) -> Any:
        key = _cache_key([name, *parts])
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age <= self.cache_ttl_hours * 3600:
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    pass
        value = fn()
        try:
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        except Exception:  # noqa: BLE001
            logger.debug("failed writing p0 cache %s", path)
        return value

    # ── THS dataapi helper ───────────────────────────────────────
    def _ths_api_get(self, path: str, params: dict | None = None) -> Any:
        """同花顺 dataapi HTTP GET (no IP blocking). Returns None on failure."""
        import urllib.parse
        import urllib.request
        base = "https://data.10jqka.com.cn/dataapi" + path
        url = base
        if params:
            url = base + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA,
                "Referer": "https://data.10jqka.com.cn/",
            })
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.debug("THS dataapi %s failed: %s", path, e)
            return None

    # ═══════════════════════════════════════════════════════════════
    # Dragon Tiger (龙虎榜)
    # ═══════════════════════════════════════════════════════════════
    def _attach_daily_dragon_tiger(
        self, trade_date: str, factors: dict[str, dict[str, Any]],
        recorder: SourceRecorder,
    ) -> None:
        """Try THS dataapi for dragon tiger; fallback to removed_per_policy.

        Eastmoney datacenter RPT_DAILYBILLBOARD_DETAILSNEW was the primary
        source. THS may provide limited dragon tiger data via dataapi.
        """
        try:
            data = self._cached("daily_lhb_ths", [trade_date],
                                lambda: self._ths_api_get("/limit_up/dragon_tiger_pool", {
                                    "date": trade_date,
                                    "field": "code,name,reason,net_buy,buy_amt,sell_amt,turnover_rate",
                                }))
        except Exception as e:  # noqa: BLE001
            recorder.removed(f"THS dragon tiger not available: {e}. {_REMOVED_EASTMONEY_NOTE}")
            return

        if not data or not isinstance(data, dict):
            recorder.removed(f"THS dragon tiger API returned no data. {_REMOVED_EASTMONEY_NOTE}")
            return

        rows = data.get("data") or []
        if not rows:
            recorder.warn(f"no dragon tiger rows for {trade_date}")
            return

        matched = 0
        for row in rows:
            code = _code(row.get("code"))
            if code not in factors:
                continue
            value = {
                "date": trade_date,
                "reason": str(row.get("reason", "")),
                "net_buy_wan": round(_num(row.get("net_buy")) / 10000, 1),
                "buy_wan": round(_num(row.get("buy_amt")) / 10000, 1),
                "sell_wan": round(_num(row.get("sell_amt")) / 10000, 1),
                "turnover_pct": round(_num(row.get("turnover_rate")), 2),
                "source": "ths_dataapi",
            }
            factors[code]["dragon_tiger"] = value
            factors[code]["source_coverage"]["dragon_tiger_daily"] = "ok"
            matched += 1
        if matched:
            recorder.ok(matched)
        else:
            recorder.warn("THS dragon tiger: no candidate matched")

    # ═══════════════════════════════════════════════════════════════
    # Hot Rank (人气榜) — THS only
    # ═══════════════════════════════════════════════════════════════
    def _attach_hot_rank(self, factors: dict[str, dict[str, Any]],
                         recorder: SourceRecorder) -> None:
        rows: list[dict[str, Any]] = []
        try:
            loaded = self._cached("hot_ths", [datetime.now().strftime("%Y%m%d%H")],
                                  self._ths_hot_list)
            for item in loaded:
                item["source"] = "ths"
            rows.extend(loaded)
        except Exception as e:  # noqa: BLE001
            recorder.warn(f"THS hot rank: {e}")

        for row in rows:
            code = _code(row.get("code"))
            if code not in factors:
                continue
            current = factors[code].get("hot_rank")
            if not current or _num(row.get("rank"), 99999) < _num(current.get("rank"), 99999):
                factors[code]["hot_rank"] = row
                factors[code]["source_coverage"]["hot_rank"] = "ok"

        hits = sum(1 for f in factors.values() if f.get("hot_rank"))
        if hits:
            recorder.success_count = hits
            recorder.item_count = len(rows)
            recorder.status = "ok"
        else:
            recorder.warn("no candidate matched hot ranks")

    # ═══════════════════════════════════════════════════════════════
    # Limit Up Sentiment — THS limit up/down pools
    # ═══════════════════════════════════════════════════════════════
    def _collect_limit_sentiment(self, date: str,
                                 recorder: SourceRecorder) -> dict[str, Any]:
        """Use THS limit up/down pools (data.10jqka.com.cn) for sentiment."""
        try:
            from .data_loader import _ths_limit_up_pool, _ths_limit_pool
            zt = _ths_limit_up_pool(date)
            dt = _ths_limit_pool(date, pool_type="down")
        except Exception as e:  # noqa: BLE001
            # Retry by importing again
            try:
                from .data_loader import _ths_limit_up_pool, _ths_limit_pool
                zt = _ths_limit_up_pool(date)
                dt = _ths_limit_pool(date, pool_type="down")
            except Exception as e2:  # noqa: BLE001
                recorder.unavailable(f"THS limit pools: {e2}")
                return {}

        zt_list = zt.to_dict(orient="records") if hasattr(zt, "to_dict") and len(zt) > 0 else []
        dt_list = dt.to_dict(orient="records") if hasattr(dt, "to_dict") and len(dt) > 0 else []

        # Compute ladder from 连板数 field
        ladder: dict[int, int] = {}
        for s in zt_list:
            level = int(_num(s.get("连板数", s.get("high_days")), 1))
            ladder[level] = ladder.get(level, 0) + 1

        data = {
            "date": date,
            "zt_count": len(zt_list),
            "zb_count": 0,  # THS doesn't provide broken board pool; seal rate used as proxy
            "dt_count": len(dt_list),
            "break_rate": 0.0,
            "max_height": max(ladder.keys(), default=0),
            "ladder": dict(sorted(ladder.items())),
            "sample_limit_up": zt_list[:10],
            "source": "ths_dataapi",
            "note": "THS dataapi limit up/down pools. Break board pool (炸板池) unavailable without Eastmoney.",
        }
        recorder.ok(len(zt_list) + len(dt_list))
        return {"limit_up_sentiment": data}

    # ═══════════════════════════════════════════════════════════════
    # Capital Flow (资金流) — THS instant fund flow
    # ═══════════════════════════════════════════════════════════════
    def _stock_fund_flow_120d_summary(self, code: str) -> dict[str, Any]:
        """Use THS instant fund flow (同花顺个股资金流).

        Historical 120d fund flow was Eastmoney push2his exclusive.
        Now uses THS real-time individual fund flow as primary source.
        """
        return self._cached("fund_flow_ths", [code],
                            lambda: self._ths_instant_fund_flow(code))

    def _ths_instant_fund_flow(self, code: str) -> dict[str, Any]:
        """同花顺个股即时资金流 via data_loader."""
        if self.dl is None:
            return {}
        try:
            df = self.dl.individual_fund_flow(code, fast=True)
        except Exception as e:  # noqa: BLE001
            logger.debug("THS fund flow %s failed: %s", code, e)
            return {}
        if df is None or len(df) == 0:
            return {}
        from .data_loader import find_col
        net_col = find_col(df, ["主力净流入-净额", "净额"])
        pct_col = find_col(df, ["主力净流入-净占比", "净占比"])
        if net_col is None:
            return {}
        net_raw = str(df[net_col].iloc[0])
        net = self._parse_chinese_amount(net_raw)
        pct_raw = str(df[pct_col].iloc[0]) if pct_col else "0"
        pct = _num(pct_raw.replace("%", ""))
        return {
            "days": 1,
            "latest": {"date": str(datetime.now().date()), "main_net": net, "source": "ths_instant"},
            "sum_20d_main_net": net,
            "positive_days_20": 1 if net > 0 else 0,
            "recent": [{"date": str(datetime.now().date()), "main_net": net}],
            "source": "ths_instant",
            "note": "120日历史资金流(东财push2his)已移除，使用同花顺当日即时主力净流入",
        }

    # ═══════════════════════════════════════════════════════════════
    # Margin Trading (融资融券) — removed (Eastmoney exclusive)
    # ═══════════════════════════════════════════════════════════════
    def _margin_trading_summary(self, code: str) -> dict[str, Any]:
        return {}

    # ═══════════════════════════════════════════════════════════════
    # Lockup (限售解禁) — removed (Eastmoney exclusive)
    # ═══════════════════════════════════════════════════════════════
    def _lockup_summary(self, code: str, trade_date: str) -> dict[str, Any]:
        return {}

    # ═══════════════════════════════════════════════════════════════
    # Research (研报) — THS EPS consensus
    # ═══════════════════════════════════════════════════════════════
    def _research_summary(self, code: str) -> dict[str, Any]:
        """THS EPS consensus forecast (同花顺一致预期).

        Replaces Eastmoney reportapi which provided full research report listings.
        """
        return self._cached("research_ths_eps", [code],
                            lambda: self._ths_eps_forecast(code))

    def _ths_eps_forecast(self, code: str) -> dict[str, Any]:
        """同花顺 EPS 一致预期 (basic.10jqka.com.cn)."""
        try:
            url = f"https://basic.10jqka.com.cn/{code}/worth.html"
            resp = self.session.get(url, headers={
                "User-Agent": UA,
                "Referer": "https://basic.10jqka.com.cn/",
            }, timeout=self.timeout)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:  # noqa: BLE001
            logger.debug("THS EPS forecast %s failed: %s", code, e)
            return {}

        # Simple extraction of EPS forecast data from THS page
        result: dict[str, Any] = {"source": "ths_basic"}
        import re

        # Try to find consensus EPS
        eps_match = re.search(r'一致预期每股收益[：:]\s*[\D]*?([\d.]+)', html)
        if eps_match:
            result["eps_consensus"] = _num(eps_match.group(1))

        # Try to find buy/hold/sell count or rating
        rating_match = re.search(r'综合评级[：:]\s*([一-龥]+)', html)
        if rating_match:
            result["rating"] = rating_match.group(1)

        # Try to find number of analysts
        analyst_match = re.search(r'(\d+)位分析师', html)
        if analyst_match:
            result["analyst_count"] = int(analyst_match.group(1))

        if not result or len(result) <= 1:  # only source key
            return {}

        result["note"] = "THS basic EPS consensus (东财reportapi已移除)"
        return result

    # ═══════════════════════════════════════════════════════════════
    # Announcements (公告) — cninfo (non-EM, kept)
    # ═══════════════════════════════════════════════════════════════
    def _announcement_summary(self, code: str) -> dict[str, Any]:
        rows = self._cached("announcements", [code],
                            lambda: self._cninfo_announcements(code, page_size=10))
        if not rows:
            return {}
        risk_words = ("减持", "处罚", "诉讼", "亏损", "退市", "风险", "问询", "立案")
        return {
            "count": len(rows),
            "latest": rows[0],
            "risk_count": sum(1 for r in rows
                              if any(w in str(r.get("title", "")) for w in risk_words)),
        }

    def _cninfo_announcements(self, code: str, page_size: int = 10) -> list[dict[str, Any]]:
        org_id = self._cninfo_orgid(code)
        payload = {
            "stock": f"{code},{org_id}", "tabName": "fulltext",
            "pageSize": str(page_size), "pageNum": "1",
            "column": "", "category": "", "plate": "", "seDate": "",
            "searchkey": "", "secid": "", "sortName": "", "sortType": "",
            "isHLtitle": "true",
        }
        data = self._get_json(
            "POST", "https://www.cninfo.com.cn/new/hisAnnouncement/query",
            data=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": "https://www.cninfo.com.cn/new/disclosure",
                "Origin": "https://www.cninfo.com.cn",
            },
        )
        rows = []
        for item in data.get("announcements") or []:
            rows.append({
                "title": item.get("announcementTitle", ""),
                "type": item.get("announcementTypeName", ""),
                "date": self._cninfo_ts_to_date(item.get("announcementTime")),
                "url": f"https://www.cninfo.com.cn/new/disclosure/detail?annoId={item.get('announcementId', '')}",
            })
        return rows

    def _cninfo_orgid(self, code: str) -> str:
        if not self._cninfo_orgid_map:
            try:
                data = self._get_json("GET",
                                      "http://www.cninfo.com.cn/new/data/szse_stock.json")
                self._cninfo_orgid_map = {
                    str(s.get("code")): str(s.get("orgId"))
                    for s in data.get("stockList", [])
                    if s.get("code") and s.get("orgId")
                }
            except Exception as e:  # noqa: BLE001
                logger.debug("cninfo org id map failed: %s", e)
        org = self._cninfo_orgid_map.get(code)
        if org:
            return org
        if code.startswith("6"):
            return f"gssh0{code}"
        if code.startswith(("8", "4")):
            return f"gsbj0{code}"
        return f"gssz0{code}"

    @staticmethod
    def _cninfo_ts_to_date(value: Any) -> str:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000).strftime("%Y-%m-%d")
        return str(value or "")[:10]

    # ═══════════════════════════════════════════════════════════════
    # IRM (互动易) — cninfo (non-EM, kept)
    # ═══════════════════════════════════════════════════════════════
    def _irm_summary(self, code: str) -> dict[str, Any]:
        rows = self._cached("irm", [code],
                            lambda: self._cninfo_irm(code, page_size=10))
        answered = [r for r in rows if r.get("answer")]
        return {"count": len(rows), "answered_count": len(answered),
                "latest": rows[0]} if rows else {}

    def _cninfo_irm(self, code: str, page_size: int = 10) -> list[dict[str, Any]]:
        data1 = self._get_json("POST",
                               "https://irm.cninfo.com.cn/newircs/index/queryKeyboardInfo",
                               data={"keyWord": code})
        found = data1.get("data") or []
        if not found:
            return []
        org_id = found[0].get("secid")
        params = {"_t": 1, "stockcode": code, "orgId": org_id,
                  "pageSize": page_size, "pageNum": 1,
                  "keyWord": "", "startDay": "", "endDay": ""}
        data2 = self._get_json("POST",
                               "https://irm.cninfo.com.cn/newircs/company/question",
                               params=params)
        rows = []
        for item in data2.get("rows") or []:
            pub = item.get("pubDate")
            rows.append({
                "code": item.get("stockCode"),
                "company": item.get("companyShortName"),
                "question": item.get("mainContent"),
                "answer": item.get("attachedContent"),
                "answerer": item.get("attachedAuthor"),
                "ask_time": datetime.fromtimestamp(pub / 1000).strftime("%Y-%m-%d %H:%M") if pub else "",
            })
        return rows

    # ═══════════════════════════════════════════════════════════════
    # Northbound (北向资金) — THS market-level only
    # ═══════════════════════════════════════════════════════════════
    def _northbound_summary(self, code: str) -> dict[str, Any]:
        """北向资金：仅使用同花顺市场级数据。

        Per-code northbound was Eastmoney datacenter RPT_MUTUALSTOCK_NORTHSTREAM
        exclusive. Now returns market-level aggregate only.
        """
        try:
            market = self._cached("northbound_market",
                                  [datetime.now().strftime("%Y%m%d")],
                                  self._ths_northbound_market)
        except Exception as e:  # noqa: BLE001
            logger.debug("northbound market fallback failed: %s", e)
            return {}
        if not market:
            return {}
        return {
            "scope": "market",
            "latest": market.get("latest"),
            "sum_5d_net_buy": market.get("sum_5d_net_buy"),
            "sum_20d_net_buy": market.get("sum_20d_net_buy"),
            "positive_days_5": market.get("positive_days_5"),
            "source": "ths_hexin",
            "note": "个股北向(东财datacenter)已移除，使用同花顺市场级北向资金",
        }

    def _ths_northbound_market(self) -> dict[str, Any]:
        """同花顺市场级北向资金 dayChart."""
        url = "https://data.hexin.cn/market/hsgtApi/method/dayChart/"
        data = self._get_json(
            "GET", url,
            headers={"Host": "data.hexin.cn", "Referer": "https://data.hexin.cn/"},
        )
        times = data.get("time") or []
        hgt = data.get("hgt") or []
        sgt = data.get("sgt") or []
        if not times or not hgt or len(times) != len(hgt):
            return {}
        latest_total = _num(hgt[-1]) + _num((sgt or [0])[-1])
        return {
            "latest": {"date": str(datetime.now().date()), "net_buy": latest_total},
            "sum_5d_net_buy": None,
            "sum_20d_net_buy": None,
            "positive_days_5": None,
            "point_count": len(times),
        }

    # ═══════════════════════════════════════════════════════════════
    # Block Trade (大宗交易) — removed (Eastmoney exclusive)
    # ═══════════════════════════════════════════════════════════════
    def _block_trade_summary(self, code: str) -> dict[str, Any]:
        return {}

    # ═══════════════════════════════════════════════════════════════
    # Holder Num (股东户数) — removed (Eastmoney exclusive)
    # ═══════════════════════════════════════════════════════════════
    def _holder_num_summary(self, code: str) -> dict[str, Any]:
        return {}

    # ═══════════════════════════════════════════════════════════════
    # Dividend (分红送转) — Baidu gushitong API
    # ═══════════════════════════════════════════════════════════════
    def _dividend_summary(self, code: str) -> dict[str, Any]:
        """分红送转 via Baidu gushitong (百度股市通).

        Replaces Eastmoney datacenter RPT_SHAREBONUS_DET.
        """
        return self._cached("dividend_baidu", [code],
                            lambda: self._baidu_dividend(code))

    def _baidu_dividend(self, code: str) -> dict[str, Any]:
        """百度股市通分红数据."""
        market = "sh" if code.startswith("6") else "sz"
        try:
            url = f"https://gushitong.baidu.com/stock/{market.upper()}-{code}"
            resp = self.session.get(url, headers={
                "User-Agent": UA,
                "Referer": "https://gushitong.baidu.com/",
            }, timeout=self.timeout)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:  # noqa: BLE001
            logger.debug("Baidu dividend %s failed: %s", code, e)
            return {}

        import re
        result: dict[str, Any] = {"source": "baidu_gushitong",
                                   "note": "百度股市通分红数据(东财datacenter已移除)"}

        # Parse dividend info from Baidu page
        div_match = re.search(r'每股派息[：:\s]*([\d.]+)\s*元', html)
        if div_match:
            result["dividend_per_share"] = _num(div_match.group(1))

        ex_date_match = re.search(r'除权除息日[：:\s]*(\d{4}-\d{2}-\d{2})', html)
        if ex_date_match:
            result["latest_date"] = ex_date_match.group(1)

        bonus_match = re.search(r'送股[：:\s]*每10股送([\d.]+)股', html)
        if bonus_match:
            result["bonus_share_ratio"] = _num(bonus_match.group(1))

        transfer_match = re.search(r'转增[：:\s]*每10股转增([\d.]+)股', html)
        if transfer_match:
            result["transfer_ratio"] = _num(transfer_match.group(1))

        if len(result) <= 2:  # only source + note
            return {}
        return result

    # ═══════════════════════════════════════════════════════════════
    # THS Hot List (同花顺人气榜)
    # ═══════════════════════════════════════════════════════════════
    def _ths_hot_list(self) -> list[dict[str, Any]]:
        data = self._get_json(
            "GET",
            "https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/stock",
            params={"stock_type": "a", "type": "hour", "list_type": "normal"},
        )
        rows = []
        for item in ((data.get("data") or {}).get("stock_list") or []):
            tag = item.get("tag") or {}
            rows.append({
                "rank": item.get("order"),
                "code": _code(item.get("code")),
                "name": item.get("name"),
                "heat": item.get("rate"),
                "pct": item.get("rise_and_fall"),
                "rank_chg": item.get("hot_rank_chg"),
                "concepts": tag.get("concept_tag") or [],
                "tag": tag.get("popularity_tag", ""),
            })
        return rows

    # ═══════════════════════════════════════════════════════════════
    # Amount parsing
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def _parse_chinese_amount(value: str) -> float:
        """Parse THS fund flow strings (e.g. -6283.93万 / 4.64亿) to yuan."""
        s = str(value or "").strip().replace(",", "")
        if not s:
            return 0.0
        sign = -1.0 if s.startswith("-") else 1.0
        s = s.lstrip("-+").replace("%", "")
        unit = 1.0
        if s.endswith("万"):
            unit = 1e4
            s = s[:-1]
        elif s.endswith("亿"):
            unit = 1e8
            s = s[:-1]
        try:
            return sign * float(s) * unit
        except (TypeError, ValueError):
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    # Derive notes (reasons + risk_notes)
    # ═══════════════════════════════════════════════════════════════
    def _derive_notes(self, sf: dict[str, Any]) -> None:
        reasons: list[str] = []
        risks: list[str] = []

        lhb = sf.get("dragon_tiger") or {}
        if _num(lhb.get("net_buy_wan")) > 0:
            reasons.append(f"LHB net buy {lhb.get('net_buy_wan')}w")

        hot = sf.get("hot_rank") or {}
        if hot.get("rank"):
            reasons.append(f"hot rank {hot.get('rank')}")

        flow = sf.get("capital_flow_120d") or {}
        if _num(flow.get("sum_20d_main_net")) > 0:
            reasons.append(
                f"20d main inflow {round(_num(flow.get('sum_20d_main_net')) / 1e8, 2)}e")
        elif _num(flow.get("sum_20d_main_net")) < 0:
            risks.append(
                f"20d main outflow {round(abs(_num(flow.get('sum_20d_main_net'))) / 1e8, 2)}e")

        north = sf.get("northbound") or {}
        if _num(north.get("sum_5d_net_buy")) and _num(north.get("sum_5d_net_buy")) > 0:
            reasons.append(
                f"north 5d inflow {round(_num(north.get('sum_5d_net_buy')) / 1e8, 2)}e")
        elif _num(north.get("sum_5d_net_buy")) and _num(north.get("sum_5d_net_buy")) < 0:
            risks.append(
                f"north 5d outflow {round(abs(_num(north.get('sum_5d_net_buy'))) / 1e8, 2)}e")

        anns = sf.get("announcements") or {}
        if int(_num(anns.get("risk_count"))) > 0:
            risks.append(f"risk announcement {int(_num(anns.get('risk_count')))}")

        sf["reasons"] = reasons[:4]
        sf["risk_notes"] = risks[:4]
