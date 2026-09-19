"""PIT（Point-In-Time）数据底座：全市场日线档案的因果读取层。

数据源：SQLite 档案 ``data/market_breadth/raw.sqlite3``（可能有并发写进程，
所有连接带 busy_timeout），表结构：

- ``breadth_raw(date, code, open, high, low, close, preclose, volume, amount,
  pct_change, turnover, is_st, PK(date, code))`` — date 为紧凑 'YYYYMMDD'。
  ``pct_change`` 是分红/拆股调整后的当日收益（百分数），收益计算必须用它。
- ``breadth_universe(date, code, name)`` — PIT 每日宇宙成员。
- ``industry_membership(date, code, industry, ...)`` — 仅 2026-01-28 起有数据。
- ``index_daily(date, code, open, ..., pct_change)``。

PIT 铁律（见 lookahead.py）：
- 任何返回数据的日期严格 <= 请求 asof；内部 ``_guard_asof`` 兜底抛错。
- 静态前复权/后复权整列重算被禁止（会改写历史）；收益一律用 pct_change。
- API 边界同时接受 'YYYY-MM-DD' 与 'YYYYMMDD'；内部统一紧凑格式，
  对外（trading_days / next / prev / daily_panel 索引）统一返回 ISO。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .lookahead import assert_no_future_rows, to_compact, to_iso

logger = logging.getLogger("pangu.pit_store")

DEFAULT_DB = "data/market_breadth/raw.sqlite3"
DEFAULT_ANNOUNCEMENT_DIR = "data/announcement_archive"
DEFAULT_NEWS_DIR = "data/wscn_news_archive"
DEFAULT_PIT_DIR = "data/pit"
REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"

# 并发写进程存在：读连接必须带 busy_timeout，避免 "database is locked"
BUSY_TIMEOUT_MS = 5000
# 行业成员 asof 回退窗口（与 replay_loader 一致，最多向前找 7 个交易日）
INDUSTRY_LOOKBACK_DAYS = 7

BAR_COLUMNS = [
    "open", "high", "low", "close", "preclose",
    "volume", "amount", "pct_change", "turnover", "is_st",
]
NUMERIC_BAR_COLUMNS = [
    "open", "high", "low", "close", "preclose",
    "volume", "amount", "pct_change", "turnover",
]
MANIFEST_TABLES = ("breadth_raw", "breadth_universe", "industry_membership", "index_daily")

BAR_SELECT = (
    "SELECT date, code, open, high, low, close, preclose, volume, amount, "
    "pct_change, turnover, is_st FROM breadth_raw"
)


def normalize_code(value: Any) -> str:
    """'sh.600000' / 'sz.000001' / 600000 → '600000'。"""
    s = str(value).strip()
    if "." in s:
        s = s.split(".", 1)[1]
    return s.zfill(6)


def code_variants(code: Any) -> List[str]:
    """生成同一只票在档案中可能出现的写法（带/不带交易所前缀）。"""
    s = str(code).strip()
    bare = normalize_code(s)
    variants = {s, bare}
    if len(bare) == 6:
        prefix = "sh" if bare.startswith(("6", "9", "5")) else ("bj" if bare.startswith(("8", "4")) else "sz")
        variants.add(f"{prefix}.{bare}")
    return sorted(variants)


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    return conn


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=str(REPO_ROOT),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


class PITStore:
    """因果（point-in-time）读取全市场日线/宇宙/行业/新闻公告档案。

    用法::

        store = PITStore("data/market_breadth/raw.sqlite3")
        panel = store.daily_panel("2026-01-05", "2026-01-28")
        uni = store.universe("2026-01-28")
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        preload: bool = False,
        announcement_dir: str | Path = DEFAULT_ANNOUNCEMENT_DIR,
        news_dir: str | Path = DEFAULT_NEWS_DIR,
        pit_dir: str | Path = DEFAULT_PIT_DIR,
        settings_path: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"PIT 档案不存在: {self.db_path}")
        self.announcement_dir = Path(announcement_dir)
        self.news_dir = Path(news_dir)
        self.pit_dir = Path(pit_dir)
        self.settings_path = Path(settings_path) if settings_path else SETTINGS_PATH
        self._trade_dates: List[str] = []          # 紧凑格式，升序
        self._preload = bool(preload)
        self._all_bars: Optional[pd.DataFrame] = None
        self._probe()
        if self._preload:
            self._preload_bars()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def _probe(self) -> None:
        with _connect_ro(self.db_path) as conn:
            dates = [
                to_compact(r[0]) for r in conn.execute(
                    "SELECT DISTINCT date FROM breadth_raw ORDER BY date"
                ).fetchall()
            ]
        if not dates:
            raise RuntimeError("PIT 档案为空：breadth_raw 无任何日期")
        self._trade_dates = dates

    def _preload_bars(self) -> None:
        """preload=True 时整表载入内存（研究用；默认走 SQL 按需查询）。"""
        with _connect_ro(self.db_path) as conn:
            df = pd.read_sql_query(f"{BAR_SELECT} ORDER BY code, date", conn)
        self._all_bars = self._normalize_bars(df)
        logger.info("PIT 全档案日线已载入内存：%d 行", len(self._all_bars))

    @staticmethod
    def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["code"] = df["code"].map(normalize_code)
        df["date"] = df["date"].map(to_compact)
        for col in NUMERIC_BAR_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["is_st"] = df["is_st"].astype(bool)
        return df

    # ------------------------------------------------------------------ #
    # 日期工具（对外一律 ISO）
    # ------------------------------------------------------------------ #
    @property
    def trade_dates_iso(self) -> List[str]:
        return [to_iso(d) for d in self._trade_dates]

    def trading_days(self, start: Any, end: Any) -> List[str]:
        """[start, end] 内的交易日（ISO 升序）。"""
        s, e = to_compact(start), to_compact(end)
        return [to_iso(d) for d in self._trade_dates if s <= d <= e]

    def next_trading_day(self, d: Any) -> Optional[str]:
        """严格晚于 d 的第一个交易日（ISO），无则 None。"""
        c = to_compact(d)
        later = [t for t in self._trade_dates if t > c]
        return to_iso(later[0]) if later else None

    def prev_trading_day(self, d: Any) -> Optional[str]:
        """严格早于 d 的最近交易日（ISO），无则 None。"""
        c = to_compact(d)
        prior = [t for t in self._trade_dates if t < c]
        return to_iso(prior[-1]) if prior else None

    # ------------------------------------------------------------------ #
    # PIT 守卫
    # ------------------------------------------------------------------ #
    def _guard_asof(self, df: pd.DataFrame, asof: Any, context: str = "") -> None:
        """兜底防线：返回结果中任何 date 行晚于 asof 都直接抛 ValueError。"""
        try:
            assert_no_future_rows(df, asof)
        except ValueError as e:
            raise ValueError(f"{context}: {e}") if context else e

    # ------------------------------------------------------------------ #
    # 日线面板
    # ------------------------------------------------------------------ #
    def _load_bars(self, start: str, end: str, symbols: Optional[Sequence[Any]]) -> pd.DataFrame:
        if self._all_bars is not None:
            df = self._all_bars[(self._all_bars["date"] >= start) & (self._all_bars["date"] <= end)]
            if symbols is not None:
                wanted = {normalize_code(s) for s in symbols}
                df = df[df["code"].isin(wanted)]
            return df.reset_index(drop=True)
        params: List[Any] = [end, start]
        where = "date <= ? AND date >= ?"
        if symbols is not None:
            variants: List[str] = []
            for s in symbols:
                variants.extend(code_variants(s))
            where += f" AND code IN ({','.join('?' * len(variants))})"
            params.extend(variants)
        with _connect_ro(self.db_path) as conn:
            df = pd.read_sql_query(f"{BAR_SELECT} WHERE {where}", conn, params=params)
        return self._normalize_bars(df)

    def daily_panel(
        self, start: Any, end: Any, symbols: Optional[Sequence[Any]] = None
    ) -> pd.DataFrame:
        """[start, end] 日线面板，MultiIndex(date, code)，date 为 ISO 字符串。

        严格 date <= end（SQL 过滤 + _guard_asof 双保险）；收益列 pct_change
        为分红/拆股调整后的真实收益（百分数）。
        """
        s, e = to_compact(start), to_compact(end)
        df = self._load_bars(s, e, symbols)
        self._guard_asof(df, e, context=f"daily_panel(asof={to_iso(e)})")
        keep_cols = ["date", "code", *BAR_COLUMNS]
        if df.empty:
            idx = pd.MultiIndex.from_arrays([[], []], names=["date", "code"])
            return pd.DataFrame(columns=BAR_COLUMNS, index=idx)
        df = df[keep_cols].copy()
        df["date"] = df["date"].map(to_iso)
        df = df.set_index(["date", "code"]).sort_index()
        return df

    def _bars_for_date(self, d: str) -> Dict[str, Dict[str, Any]]:
        if self._all_bars is not None:
            day = self._all_bars[self._all_bars["date"] == d]
        else:
            with _connect_ro(self.db_path) as conn:
                day = pd.read_sql_query(f"{BAR_SELECT} WHERE date = ?", conn, params=(d,))
            day = self._normalize_bars(day)
        return {str(row["code"]): row for _, row in day.iterrows()}

    # ------------------------------------------------------------------ #
    # 宇宙 / 行业
    # ------------------------------------------------------------------ #
    def _members_asof(self, d: str) -> Dict[str, str]:
        """PIT 宇宙成员：精确日优先，否则回退到 <= d 的最近成员表快照。"""
        with _connect_ro(self.db_path) as conn:
            rows = conn.execute(
                "SELECT code, name FROM breadth_universe WHERE date = ?", (d,)
            ).fetchall()
            if not rows:
                snap = conn.execute(
                    "SELECT MAX(date) FROM breadth_universe WHERE date <= ?", (d,)
                ).fetchone()
                if snap and snap[0]:
                    rows = conn.execute(
                        "SELECT code, name FROM breadth_universe WHERE date = ?", (snap[0],)
                    ).fetchall()
        return {normalize_code(c): (str(n) if n is not None else normalize_code(c)) for c, n in rows}

    def _industry_asof(self, d: str) -> Dict[str, str]:
        """行业成员：精确日优先，向前回退最多 INDUSTRY_LOOKBACK_DAYS 个交易日。"""
        dates = [t for t in self._trade_dates if t <= d][-INDUSTRY_LOOKBACK_DAYS:]
        with _connect_ro(self.db_path) as conn:
            for t in reversed(dates):
                rows = conn.execute(
                    "SELECT code, industry FROM industry_membership WHERE date = ?", (t,)
                ).fetchall()
                if rows:
                    return {normalize_code(c): str(i) for c, i in rows}
        return {}

    def _first_seen(self) -> Dict[str, str]:
        """每只票在日线档案中的首见日期（上市即已知，PIT 安全）。"""
        if self._all_bars is not None:
            grouped = self._all_bars.groupby("code")["date"].min()
            return {str(k): str(v) for k, v in grouped.items()}
        with _connect_ro(self.db_path) as conn:
            rows = conn.execute(
                "SELECT code, MIN(date) FROM breadth_raw GROUP BY code"
            ).fetchall()
        return {normalize_code(c): to_compact(m) for c, m in rows}

    def _last_known_st(self, code: str, d: str) -> bool:
        """停牌日无 bar，is_st 取 <= d 最近一条已知值（PIT 安全）。"""
        variants = code_variants(code)
        if self._all_bars is not None:
            sub = self._all_bars[
                self._all_bars["code"].isin({normalize_code(c) for c in variants})
                & (self._all_bars["date"] <= d)
            ]
            if sub.empty:
                return False
            return bool(sub.sort_values("date")["is_st"].iloc[-1])
        with _connect_ro(self.db_path) as conn:
            row = conn.execute(
                f"{BAR_SELECT} WHERE code IN ({','.join('?' * len(variants))}) "
                f"AND date <= ? ORDER BY date DESC LIMIT 1",
                [*variants, d],
            ).fetchone()
        return bool(row[11]) if row else False

    def universe(self, date: Any) -> pd.DataFrame:
        """asof 宇宙快照（index=code）。

        列：
        - name: 成员名称
        - is_st(bool): 当日 bar 的 ST 标记；停牌取最近已知值
        - suspended(bool): 是成员但当日无 bar
        - listed_days(int): 档案内首见至今的交易日数（含当日）
        - tradable(bool): 未停牌且有正收盘价
        - industry: asof 行业（无则 NaN）
        - availability: 'ok' | 'degraded'（industry 缺失即 degraded）
        """
        d = to_compact(date)
        members = self._members_asof(d)
        bars = self._bars_for_date(d)
        first_seen = self._first_seen()
        industry = self._industry_asof(d)
        rows: List[Dict[str, Any]] = []
        for code, name in members.items():
            bar = bars.get(code)
            suspended = bar is None
            is_st = bool(bar["is_st"]) if bar is not None else self._last_known_st(code, d)
            fs = first_seen.get(code)
            listed_days = (
                len([t for t in self._trade_dates if fs <= t <= d]) if fs else 0
            )
            close = float(bar["close"]) if bar is not None and bar["close"] == bar["close"] else 0.0
            ind = industry.get(code)
            rows.append({
                "name": name,
                "is_st": is_st,
                "suspended": suspended,
                "listed_days": int(listed_days),
                "tradable": (not suspended) and close > 0,
                "industry": ind if ind is not None else float("nan"),
                "availability": "ok" if ind is not None else "degraded",
            })
        columns = ["name", "is_st", "suspended", "listed_days", "tradable", "industry", "availability"]
        if not rows:
            return pd.DataFrame(columns=columns, index=pd.Index([], name="code"))
        df = pd.DataFrame(rows, index=pd.Index(list(members.keys()), name="code"))
        return df.sort_index()

    # ------------------------------------------------------------------ #
    # 指数
    # ------------------------------------------------------------------ #
    def index_daily(self, code: Any, start: Any, end: Any) -> pd.DataFrame:
        """指数日线（index=ISO date），严格 date <= end。"""
        s, e = to_compact(start), to_compact(end)
        variants = code_variants(code)
        with _connect_ro(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT date, code, open, high, low, close, preclose, volume, "
                f"amount, pct_change FROM index_daily WHERE code IN ({','.join('?' * len(variants))}) "
                "AND date >= ? AND date <= ? ORDER BY date",
                conn, params=[*variants, s, e],
            )
        self._guard_asof(df, e, context=f"index_daily({code}, asof={to_iso(e)})")
        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "preclose", "volume", "amount", "pct_change"],
                index=pd.Index([], name="date"),
            )
        df["date"] = df["date"].map(to_iso)
        df = df.drop(columns=["code"]).set_index("date")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ------------------------------------------------------------------ #
    # 行数统计（manifest / 质量报告用）
    # ------------------------------------------------------------------ #
    def row_count(self, table: str, start: Any | None = None, end: Any | None = None) -> int:
        """表行数（可选 [start, end] 日期窗口）；表不存在返回 0。"""
        try:
            with _connect_ro(self.db_path) as conn:
                if start is not None and end is not None:
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE date >= ? AND date <= ?",
                        (to_compact(start), to_compact(end)),
                    ).fetchone()
                else:
                    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------ #
    # 公告 / 新闻（事件面，含保守 availability 语义）
    # ------------------------------------------------------------------ #
    def announcements(
        self, date_range: Sequence[Any], codes: Optional[Sequence[Any]] = None
    ) -> pd.DataFrame:
        """读取公告档案（data/announcement_archive/YYYYMMDD.json）。

        可用性语义（保守，文档化）：公告的 published_at 只有日期精度（不知道
        盘中还是盘后发布），因此统一视为「发布次日（下一个交易日）才可用于
        决策」→ available_at_decision = next_trading_day(published_date)。
        调用方必须用 available_at_decision <= 决策日 过滤。
        """
        columns = ["date", "code", "name", "title", "published_at", "available_at_decision"]
        s, e = to_compact(date_range[0]), to_compact(date_range[-1])
        if not self.announcement_dir.exists():
            return pd.DataFrame(columns=columns)
        wanted = {normalize_code(c) for c in codes} if codes is not None else None
        rows: List[Dict[str, Any]] = []
        for d in self._trade_dates:
            if d < s or d > e:
                continue
            path = self.announcement_dir / f"{d}.json"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for ev in data.get("events") or []:
                code = normalize_code(ev.get("code") or "")
                if wanted is not None and code not in wanted:
                    continue
                pub_raw = str(ev.get("published_at") or d)[:10]
                pub_compact = to_compact(pub_raw)
                # 保守：次日才可用；超出档案末尾时退化为发布日本身并打日志
                avail = self.next_trading_day(pub_compact) or to_iso(pub_compact)
                rows.append({
                    "date": to_iso(d),
                    "code": code,
                    "name": ev.get("name") or "",
                    "title": ev.get("title") or "",
                    "published_at": to_iso(pub_compact),
                    "available_at_decision": avail,
                })
        return pd.DataFrame(rows, columns=columns)

    def news(
        self, date_range: Sequence[Any], codes: Optional[Sequence[Any]] = None
    ) -> pd.DataFrame:
        """读取快讯档案（data/wscn_news_archive/YYYYMMDD.json）。

        可用性语义：新闻有精确 published_at epoch。对档案日 D 的决策时刻
        （当日 15:05 收盘决策）——published_at <= D 15:05 的新闻当日即可用，
        更晚发布的保守顺延到下一交易日（available_at_decision 列）。
        """
        columns = ["date", "id", "title", "published_at", "available_at_decision", "codes"]
        s, e = to_compact(date_range[0]), to_compact(date_range[-1])
        if not self.news_dir.exists():
            return pd.DataFrame(columns=columns)
        wanted = {normalize_code(c) for c in codes} if codes is not None else None
        rows: List[Dict[str, Any]] = []
        for d in self._trade_dates:
            if d < s or d > e:
                continue
            path = self.news_dir / f"{d}.json"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            cutoff = datetime.strptime(d, "%Y%m%d").replace(hour=15, minute=5)
            for item in data.get("items") or []:
                pub_dt = self._parse_news_time(item)
                if pub_dt is None:
                    # 无时间戳 → 无法证明当日可用，保守顺延
                    avail_iso = self.next_trading_day(d) or to_iso(d)
                elif pub_dt <= cutoff:
                    avail_iso = to_iso(d)
                else:
                    avail_iso = self.next_trading_day(d) or to_iso(d)
                item_codes = self._item_codes(item)
                if wanted is not None and item_codes and not (item_codes & wanted):
                    continue
                rows.append({
                    "date": to_iso(d),
                    "id": str(item.get("id") or ""),
                    "title": str(item.get("title") or ""),
                    "published_at": pub_dt.isoformat() if pub_dt else "",
                    "available_at_decision": avail_iso,
                    "codes": sorted(item_codes),
                })
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _parse_news_time(item: Dict[str, Any]) -> Optional[datetime]:
        """新闻时间：优先 display_timestamp epoch，其次解析 published_at ISO。"""
        ts = item.get("display_timestamp")
        try:
            if ts is not None:
                return datetime.fromtimestamp(float(ts))
        except Exception:  # noqa: BLE001
            pass
        pub = item.get("published_at")
        if pub:
            try:
                return datetime.fromisoformat(str(pub).replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:  # noqa: BLE001
                return None
        return None

    @staticmethod
    def _item_codes(item: Dict[str, Any]) -> set:
        codes: set = set()
        for key in ("code", "codes", "stock_codes", "related_codes", "symbols"):
            v = item.get(key)
            if isinstance(v, str) and v:
                codes.add(normalize_code(v))
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, str) and x:
                        codes.add(normalize_code(x))
                    elif isinstance(x, dict) and x.get("code"):
                        codes.add(normalize_code(x["code"]))
        return codes

    # ------------------------------------------------------------------ #
    # 快照 manifest
    # ------------------------------------------------------------------ #
    def snapshot_manifest(self, date: Any, out_dir: str | Path | None = None) -> Dict[str, Any]:
        """写入可复现性 manifest：settings sha256 + git HEAD + 各表行数。

        输出 <pit_dir>/manifests/<YYYYMMDD>.json（out_dir 可覆盖，测试用）。
        """
        d = to_compact(date)
        settings_sha = _sha256_file(self.settings_path) if self.settings_path.exists() else None
        counts = {t: self.row_count(t) for t in MANIFEST_TABLES}
        payload = {
            "date": to_iso(d),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "settings_path": str(self.settings_path),
            "settings_sha256": settings_sha,
            "git_commit": _git_head(),
            "row_counts": counts,
        }
        out = Path(out_dir) if out_dir else (self.pit_dir / "manifests")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload
