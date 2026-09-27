"""AsOf 上下文与 A 股交易日历（Phase 2 / Task 2.1-2.2）。

- TradingCalendar：下一交易日计算。数据来源优先级：
  1) 本地日历缓存 data/pit/calendar.sqlite（baostock 交易日历，含未来）；
  2) baostock 在线查询（写缓存）；
  3) 工作日规则回退（显式标 estimated=True，附 warning）。
  禁止简单 +1 day。
- AsOfContext：query_timestamp → decision_date / execution_date / market_status。
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from engine.decision.clock import SHANGHAI, Clock, now_shanghai
from engine.decision.contracts import MarketStatus

CALENDAR_DB = Path("data/pit/calendar.sqlite")


class TradingCalendar:
    """A 股交易日历：优先缓存/在线，缺失时回退工作日规则并显式标注。"""

    def __init__(self, known_days: Optional[List[str]] = None,
                 calendar_db: Optional[Path] = None,
                 allow_online: bool = True):
        self._days: set[str] = {d.replace("-", "") for d in (known_days or [])}
        self._db = Path(calendar_db) if calendar_db else CALENDAR_DB
        self._extra_loaded = False
        self._allow_online = allow_online

    # -- 数据面 ---------------------------------------------------------- #
    def _load_extra_days(self) -> None:
        """从缓存表加载（含未来）交易日。"""
        if self._extra_loaded:
            return
        self._extra_loaded = True
        try:
            if self._db.exists():
                conn = sqlite3.connect(f"file:{self._db}?mode=ro", uri=True)
                try:
                    rows = conn.execute("SELECT date FROM trade_dates").fetchall()
                    self._days |= {r[0] for r in rows}
                finally:
                    conn.close()
        except sqlite3.Error:
            pass

    def _fetch_online(self, start: str, end: str) -> bool:
        """baostock 在线交易日历（成功写缓存）。失败静默返回 False。"""
        if not self._allow_online:
            return False
        try:
            import baostock as bs
            bs.login()
            try:
                rs = bs.query_trade_dates(start_date=start, end_date=end)
                rows = []
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    if len(row) >= 2 and row[1] == "1":
                        rows.append(row[0].replace("-", ""))
                if rows:
                    self._days |= set(rows)
                    self._persist(rows)
                    return True
            finally:
                bs.logout()
        except Exception:  # noqa: BLE001 — 网络/依赖不可用
            return False
        return False

    def _persist(self, dates: List[str]) -> None:
        try:
            self._db.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db)
            try:
                conn.execute("CREATE TABLE IF NOT EXISTS trade_dates "
                             "(date TEXT PRIMARY KEY)")
                conn.executemany("INSERT OR IGNORE INTO trade_dates VALUES (?)",
                                 [(d,) for d in dates])
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error:
            pass

    def _ensure_range(self, day_compact: str) -> None:
        """若目标日期超出已知范围，尝试在线补日历（一次）。"""
        self._load_extra_days()
        if self._days and day_compact <= max(self._days):
            return
        end = (max(self._days) if self._days else "20250101")
        self._fetch_online(end, (day_compact[:4] + "1231"))

    # -- 查询 ------------------------------------------------------------ #
    def is_trading_day(self, day: str, estimate: bool = True) -> Tuple[bool, bool]:
        """返回 (是否交易日, 是否估算)。估算 = 日历未覆盖、按工作日规则推断。"""
        c = day.replace("-", "")
        self._ensure_range(c)
        if c in self._days:
            return True, False
        if self._days and c <= max(self._days):
            return False, False          # 已知非交易日（节假日）
        wd = _dt.datetime.strptime(c, "%Y%m%d").weekday()
        return (wd < 5, True) if estimate else (False, True)

    def next_trading_day(self, day: str) -> Tuple[Optional[str], bool]:
        """严格次一交易日（不含当天）。返回 (YYYYMMDD|None, estimated)。"""
        c = day.replace("-", "")
        cur = _dt.datetime.strptime(c, "%Y%m%d")
        for _ in range(40):               # 最长跨 40 天（春节等长假安全）
            cur += _dt.timedelta(days=1)
            ok, est = self.is_trading_day(cur.strftime("%Y%m%d"))
            if ok:
                return cur.strftime("%Y%m%d"), est
        return None, True

    def prev_trading_day(self, day: str, include_self: bool = True) -> Tuple[Optional[str], bool]:
        c = day.replace("-", "")
        cur = _dt.datetime.strptime(c, "%Y%m%d")
        if include_self:
            ok, est = self.is_trading_day(c)
            if ok:
                return c, est
        for _ in range(40):
            cur -= _dt.timedelta(days=1)
            ok, est = self.is_trading_day(cur.strftime("%Y%m%d"))
            if ok:
                return cur.strftime("%Y%m%d"), est
        return None, True


def market_status_at(ts: _dt.datetime, cal: TradingCalendar) -> MarketStatus:
    """给定上海时刻的市场状态（用日历判定周末/节假日）。"""
    wd = ts.weekday()
    is_day, _ = cal.is_trading_day(ts.strftime("%Y%m%d"))
    if not is_day:
        if wd >= 5:
            # 周末若处于节假日簇内（最近一个工作日也不交易）→ HOLIDAY
            friday = ts - _dt.timedelta(days=(wd - 4) if wd == 6 else (wd - 4))
            # wd==5(周六)→回退1天到周五; wd==6(周日)→回退2天到周五
            f_ok, f_est = cal.is_trading_day(friday.strftime("%Y%m%d"))
            if not (f_ok or f_est):
                return MarketStatus.HOLIDAY
            return MarketStatus.WEEKEND
        return MarketStatus.HOLIDAY
    hm = ts.hour * 100 + ts.minute
    if hm < 915:
        return MarketStatus.CLOSED_PRE
    if hm < 1130:
        return MarketStatus.OPEN
    if hm < 1300:
        return MarketStatus.LUNCH_BREAK
    if hm < 1500:
        return MarketStatus.OPEN
    return MarketStatus.CLOSED_AFTER


@dataclass
class AsOfContext:
    """一次决策的时间上下文（全部上海时区）。"""
    query_timestamp: str
    asof_timestamp: str
    decision_date: str            # YYYY-MM-DD：asof 所属/最近交易日
    execution_date: str           # YYYY-MM-DD：下一交易日
    market_status: MarketStatus
    calendar_estimated: bool = False
    warnings: List[str] = field(default_factory=list)
    entry_style: str = "next_open"    # next_open | tail_close（service 填写）

    def to_dict(self) -> dict:
        return {
            "query_timestamp": self.query_timestamp,
            "asof_timestamp": self.asof_timestamp,
            "decision_date": self.decision_date,
            "execution_date": self.execution_date,
            "market_status": self.market_status.value,
            "calendar_estimated": self.calendar_estimated,
            "warnings": list(self.warnings),
        }


def build_asof_context(clock: Optional[Clock] = None,
                       asof: Optional[str] = None,
                       cal: Optional[TradingCalendar] = None) -> AsOfContext:
    """从请求时刻构建 AsOfContext（Task 2.1/2.2）。

    asof 为空 = clock.now()。decision_date = asof 所在交易日（非交易日向前吸附，
    周末/节假日用户问的是"以最近收盘信息分析"）；execution_date = 严格下一交易日。
    """
    clk = clock
    if clk is None:
        class _M:
            def now(self_inner):
                return now_shanghai()
        clk = _M()
    base = clk.now() if asof is None else _parse(asof)
    cal = cal or TradingCalendar()
    warnings: List[str] = []

    ms = market_status_at(base, cal)
    day_compact = base.strftime("%Y%m%d")
    is_day, est1 = cal.is_trading_day(day_compact)
    if est1:
        warnings.append("交易日历为估算（日历未覆盖该日期，按工作日规则推断）")
    if is_day:
        decision_c, est2 = day_compact, est1
    else:
        decision_c, est2 = cal.prev_trading_day(day_compact, include_self=True)
        warnings.append(f"{base.strftime('%Y-%m-%d')} 非交易日，决策日向前吸附至 "
                        f"{_iso(decision_c)}（以最近收盘信息分析）")
    next_c, est3 = cal.next_trading_day(decision_c)
    if next_c is None:
        raise ValueError("无法计算下一交易日（日历不可用且超出回退范围）")
    if est3:
        warnings.append("执行日为估算（未来交易日历不可用，按工作日规则推断，"
                        "可能受节假日影响）")
    return AsOfContext(
        query_timestamp=_ts(clk.now()),
        asof_timestamp=_ts(base),
        decision_date=_iso(decision_c),
        execution_date=_iso(next_c),
        market_status=ms,
        calendar_estimated=bool(est1 or est2 or est3),
        warnings=warnings,
    )


def _parse(s: str) -> _dt.datetime:
    dt = _dt.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SHANGHAI)
    return dt.astimezone(SHANGHAI)


def _iso(c: Optional[str]) -> str:
    if not c:
        return ""
    c = str(c).replace("-", "")
    return f"{c[:4]}-{c[4:6]}-{c[6:8]}"


def _ts(dt: _dt.datetime) -> str:
    """datetime → ISO 字符串（保留上海时区）。"""
    return dt.isoformat(timespec="seconds")
