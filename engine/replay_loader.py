"""PIT-safe 历史回放数据加载器。

基于本地全市场日线档案 ``data/market_breadth/raw.sqlite3``（BaoStock 不复权日线，
含每日宇宙与行业成员），为任意历史交易日合成完整 pipeline 所需的数据面：

- ``all_spot(date)``：当日全市场快照（收盘价即"现价"）
- ``daily_kline(code, days, date)``：PIT 截断日 K
- ``limit_up_pool / limit_down_pool(date)``：按板块涨幅限价精确合成（含连板数回溯）
- ``concept_boards / concept_constituents``：行业成员聚合
- ``announcement_events(code, date)``：本地巨潮公告档案事件

涨停判定用「四舍五入后的涨停价」而不是粗阈值：
``limit_price = round(preclose * (1 + pct_limit), 2)``，``close >= limit_price`` 才算涨停，
其中 pct_limit 按板块取 10%/20%/30%、ST 取 5%。这保证了与交易所规则一致的 PIT 判定。

所有查询只读档案，不做任何未来数据泄露（date 严格 <= 回放日）。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("pangu.replay_loader")

DEFAULT_DB = "data/market_breadth/raw.sqlite3"
DEFAULT_ANNOUNCEMENT_DIR = "data/announcement_archive"


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _price_limit_pct(code: str, is_st: bool) -> float:
    """A 股板块涨跌幅限制（小数）。北交所 30%、科创/创业 20%、ST 5%、其余 10%。"""
    if code.startswith(("8", "4", "9")):
        return 0.30
    if code.startswith(("68", "30")):
        return 0.20
    if is_st:
        return 0.05
    return 0.10


def _normalize_code(value: Any) -> str:
    """'sh.600000'/'sz.000001'/'600000' → '600000'。"""
    s = str(value).strip()
    if "." in s:
        s = s.split(".", 1)[1]
    return s.zfill(6)


def _limit_price(preclose: float, pct: float) -> float:
    return round(preclose * (1 + pct), 2)


@dataclass
class ReplayContext:
    """回放期元信息。"""

    db_path: Path
    min_date: str
    max_date: str
    trade_dates: list[str]


class ReplayDataLoader:
    """把全市场日线档案包装成与 ``DataLoader`` 鸭子类型兼容的回放数据源。

    用法::

        dl = ReplayDataLoader("data/market_breadth/raw.sqlite3")
        dl.set_date("20260618")     # 当前回放日
        spot = dl.all_spot()
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        announcement_dir: str | Path = DEFAULT_ANNOUNCEMENT_DIR,
        preload: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"回放档案不存在: {self.db_path}（先用 market_breadth_archive 回填）"
            )
        self.announcement_dir = Path(announcement_dir)
        self._current_date: Optional[str] = None
        self._trade_dates: list[str] = []
        self._universe: dict[str, str] = {}
        self._industry: dict[str, str] = {}
        self._industry_date: Optional[str] = None
        self._industry_loaded_dates: set[str] = set()
        self._kline_cache: dict[str, pd.DataFrame] = {}
        self._all_bars: Optional[pd.DataFrame] = None
        self._bars_by_date: dict[str, pd.DataFrame] = {}
        self._spot_cache: dict[str, pd.DataFrame] = {}
        self._regime_cache: dict[str, dict[str, Any]] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._probe()
        if preload:
            self._preload_bars()

    # ------------------------------------------------------------------ #
    # 初始化探测
    # ------------------------------------------------------------------ #
    def _probe(self) -> None:
        with _connect_ro(self.db_path) as conn:
            dates = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT date FROM breadth_raw ORDER BY date"
                ).fetchall()
            ]
        if not dates:
            raise RuntimeError("回放档案为空")
        self._trade_dates = [str(d) for d in dates]
        logger.info(
            "回放档案就绪：%s ~ %s 共 %d 个交易日",
            dates[0], dates[-1], len(dates),
        )

    def _preload_bars(self) -> None:
        """一次性把全档案日线读进内存并按代码建索引（避免无索引查询反复全表扫描）。"""
        with _connect_ro(self.db_path) as conn:
            df = pd.read_sql_query(
                """SELECT date,code,open,high,low,close,preclose,volume,amount,
                          pct_change,turnover,is_st
                   FROM breadth_raw ORDER BY code, date""",
                conn,
            )
        if df.empty:
            raise RuntimeError("回放档案为空")
        df["code"] = df["code"].map(_normalize_code)
        df["date"] = df["date"].astype(str)
        for col in ("open", "high", "low", "close", "preclose", "volume",
                    "amount", "pct_change", "turnover"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["is_st"] = df["is_st"].astype(bool)
        self._all_bars = df
        self._bars_by_date = {d: g for d, g in df.groupby("date")}
        logger.info("全档案日线已载入内存：%d 行", len(df))

    @property
    def context(self) -> ReplayContext:
        return ReplayContext(
            db_path=self.db_path,
            min_date=self._trade_dates[0],
            max_date=self._trade_dates[-1],
            trade_dates=list(self._trade_dates),
        )

    # ------------------------------------------------------------------ #
    # 日期控制
    # ------------------------------------------------------------------ #
    def set_date(self, date: str) -> None:
        """设置当前回放交易日（YYYYMMDD）。"""
        d = str(date)
        if d not in self._trade_dates:
            raise ValueError(f"{d} 不在回放档案交易日中")
        if self._current_date != d:
            self._current_date = d
            self._industry = {}
            self._industry_date = None

    @property
    def current_date(self) -> Optional[str]:
        return self._current_date

    def trading_days(self, start: str, end: str) -> list[str]:
        return [d for d in self._trade_dates if start <= d <= end]

    def prev_trading_day(self, date: str, steps: int = 1) -> Optional[str]:
        prior = [d for d in self._trade_dates if d < date]
        if len(prior) < steps:
            return None
        return prior[-steps]

    def _conn_rw(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    # ------------------------------------------------------------------ #
    # 基础数据
    # ------------------------------------------------------------------ #
    def _day_bars(self, date: Optional[str] = None) -> pd.DataFrame:
        """某日全市场日线行（preload=True 走内存索引；否则回退按日 SQL 查询）。"""
        d = date or self._current_date
        if d is None:
            return pd.DataFrame()
        if self._bars_by_date:
            return self._bars_by_date.get(d, pd.DataFrame())
        return self._load_day_bars_sql(str(d))

    def _load_day_bars_sql(self, d: str) -> pd.DataFrame:
        """preload=False 时的按日查询（与内存路径同样的列/类型归一）。"""
        with _connect_ro(self.db_path) as conn:
            df = pd.read_sql_query(
                """SELECT date,code,open,high,low,close,preclose,volume,amount,
                          pct_change,turnover,is_st
                   FROM breadth_raw WHERE date=?""",
                conn,
                params=(d,),
            )
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["code"] = df["code"].map(_normalize_code)
        df["date"] = df["date"].astype(str)
        for col in ("open", "high", "low", "close", "preclose", "volume",
                    "amount", "pct_change", "turnover"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["is_st"] = df["is_st"].astype(bool)
        return df

    def _universe_names(self) -> dict[str, str]:
        if self._universe:
            return self._universe
        with _connect_ro(self.db_path) as conn:
            rows = conn.execute(
                "SELECT code, name FROM breadth_universe ORDER BY date DESC"
            ).fetchall()
        names: dict[str, str] = {}
        for code, name in rows:
            code = _normalize_code(code)
            if code not in names:
                names[code] = str(name or code)
        self._universe = names
        return names

    def _industry_of(self, date: str) -> dict[str, str]:
        """某日行业成员（精确日优先，向前回退最多 7 个交易日，按日缓存）。"""
        if self._industry and self._industry_date == date:
            return self._industry
        dates = [d for d in self._trade_dates if d <= date][-7:]
        with _connect_ro(self.db_path) as conn:
            for d in reversed(dates):
                if d in self._industry_loaded_dates:
                    continue
                rows = conn.execute(
                    "SELECT code, industry FROM industry_membership WHERE date=?",
                    (d,),
                ).fetchall()
                self._industry_loaded_dates.add(d)
                if rows:
                    self._industry = {
                        _normalize_code(c): str(i) for c, i in rows
                    }
                    self._industry_date = date
                    return self._industry
        self._industry = {}
        self._industry_date = date
        return self._industry

    # ------------------------------------------------------------------ #
    # all_spot
    # ------------------------------------------------------------------ #
    def all_spot(self, date: Optional[str] = None) -> pd.DataFrame:
        d = date or self._current_date
        if d is None:
            return pd.DataFrame()
        cached = self._spot_cache.get(d)
        if cached is not None:
            return cached.copy()
        bars = self._day_bars(d)
        if bars.empty:
            return pd.DataFrame()
        names = self._universe_names()
        spot = pd.DataFrame({
            "代码": bars["code"],
            "名称": bars["code"].map(names).fillna(bars["code"]),
            "最新价": bars["close"],
            "涨跌幅": bars["pct_change"],
            "换手率": bars["turnover"],
            "成交量": bars["volume"],
            "成交额": bars["amount"],
            "今开": bars["open"],
            "最高": bars["high"],
            "最低": bars["low"],
            "昨收": bars["preclose"],
        })
        # 回放档案不含估值/市值，显式置 NaN（护栏按配置跳过并标注）
        spot["流通市值"] = float("nan")
        spot["总市值"] = float("nan")
        spot["市盈率-动态"] = float("nan")
        spot["市净率"] = float("nan")
        spot.attrs["source_quality"] = {
            "status": "ok",
            "source": "replay_archive",
            "warnings": ["回放模式：市值/估值字段不可用"],
        }
        spot.attrs["source_chain"] = [{"source": "replay_archive", "ok": True,
                                       "status": "ok", "row_count": len(spot)}]
        self._spot_cache[d] = spot
        return spot.copy()

    # ------------------------------------------------------------------ #
    # daily_kline（PIT 截断，中文列，与在线源列名兼容）
    # ------------------------------------------------------------------ #
    def daily_kline(
        self,
        symbol: str,
        days: int = 60,
        adjust: str = "qfq",
        date: Optional[str] = None,
        **_: Any,
    ) -> pd.DataFrame:
        code = _normalize_code(symbol)
        end = date or self._current_date
        if end is None:
            return pd.DataFrame()
        cached = self._kline_cache.get(code)
        if cached is None:
            if self._all_bars is not None:
                df = self._all_bars[self._all_bars["code"] == code]
            else:
                query = """
                    SELECT date,open,high,low,close,preclose,volume,amount,
                           pct_change,turnover
                    FROM breadth_raw WHERE code=? ORDER BY date
                """
                provider_code = f"{'sh' if code.startswith('6') else 'sz'}.{code}"
                with _connect_ro(self.db_path) as conn:
                    df = pd.read_sql_query(query, conn, params=(provider_code,))
                if df.empty:
                    with _connect_ro(self.db_path) as conn:
                        df = pd.read_sql_query(query, conn, params=(code,))
            if df is None or len(df) == 0:
                self._kline_cache[code] = pd.DataFrame()
                return pd.DataFrame()
            df = df.copy()
            df["日期"] = df["date"]
            df["开盘"] = df["open"]
            df["最高"] = df["high"]
            df["最低"] = df["low"]
            df["收盘"] = df["close"]
            df["成交量"] = df["volume"]
            df["成交额"] = df["amount"]
            df["涨跌幅"] = df["pct_change"]
            df["换手率"] = pd.to_numeric(df["turnover"], errors="coerce").fillna(0.0)
            df["涨跌额"] = df["close"] - df["preclose"]
            prev_close = df["close"].shift(1)
            df["振幅"] = (df["high"] - df["low"]) / prev_close * 100
            self._kline_cache[code] = df
        cached = self._kline_cache[code]
        if cached.empty:
            return pd.DataFrame()
        view = cached[cached["date"] <= str(end)]
        view = view.tail(max(1, int(days))).reset_index(drop=True)
        keep = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
                "振幅", "涨跌幅", "涨跌额", "换手率", "date"]
        return view[keep].copy()

    # ------------------------------------------------------------------ #
    # 涨停/跌停池合成
    # ------------------------------------------------------------------ #
    def _consecutive_limit_up(self, code: str, date: str, pct: float) -> int:
        """回溯连板数（含当日）。档案窗口有限，最多回溯 30 日。"""
        k = self.daily_kline(code, days=40, date=date)
        if k.empty:
            return 0
        n = 0
        for _, row in k.iloc[::-1].iterrows():
            close = float(row["收盘"])
            chg = float(row["涨跌幅"]) if row["涨跌幅"] == row["涨跌幅"] else 0.0
            denom = 1 + chg / 100
            preclose = close / denom if denom else close
            limit = _limit_price(round(preclose, 2), pct)
            if close >= limit - 1e-6:
                n += 1
            else:
                break
        return n

    def limit_up_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        d = date or self._current_date
        if d is None:
            return pd.DataFrame()
        bars = self._day_bars(d)
        if bars.empty:
            return pd.DataFrame()
        names = self._universe_names()
        industry = self._industry_of(d)
        rows: list[dict[str, Any]] = []
        for _, bar in bars.iterrows():
            code = str(bar["code"])
            preclose = float(bar["preclose"]) if bar["preclose"] == bar["preclose"] else 0.0
            close = float(bar["close"]) if bar["close"] == bar["close"] else 0.0
            if preclose <= 0 or close <= 0:
                continue
            pct = _price_limit_pct(code, bool(bar["is_st"]))
            limit = _limit_price(preclose, pct)
            if close < limit - 1e-6:
                continue
            consec = self._consecutive_limit_up(code, d, pct)
            rows.append({
                "代码": code,
                "名称": names.get(code, code),
                "最新价": close,
                "涨跌幅": float(bar["pct_change"]),
                "成交额": float(bar["amount"]) if bar["amount"] == bar["amount"] else 0.0,
                "换手率": float(bar["turnover"]) if bar["turnover"] == bar["turnover"] else 0.0,
                "连板数": consec,
                "涨停统计": f"{consec}天{consec}板" if consec else "1天1板",
                "所属行业": industry.get(code, ""),
                "炸板次数": 0,       # 档案无法回放盘中炸板
                "首次封板时间": "",   # 同上
                "封单金额": None,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["连板数", "成交额"], ascending=False).reset_index(drop=True)
        return df

    def limit_down_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        d = date or self._current_date
        if d is None:
            return pd.DataFrame()
        bars = self._day_bars(d)
        if bars.empty:
            return pd.DataFrame()
        names = self._universe_names()
        rows: list[dict[str, Any]] = []
        for _, bar in bars.iterrows():
            code = str(bar["code"])
            preclose = float(bar["preclose"]) if bar["preclose"] == bar["preclose"] else 0.0
            close = float(bar["close"]) if bar["close"] == bar["close"] else 0.0
            if preclose <= 0 or close <= 0:
                continue
            pct = _price_limit_pct(code, bool(bar["is_st"]))
            limit = round(preclose * (1 - pct), 2)
            if close > limit + 1e-6:
                continue
            rows.append({
                "代码": code,
                "名称": names.get(code, code),
                "最新价": close,
                "涨跌幅": float(bar["pct_change"]),
                "成交额": float(bar["amount"]) if bar["amount"] == bar["amount"] else 0.0,
            })
        return pd.DataFrame(rows)

    def broke_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        """档案无盘中炸板数据，与在线行为一致返回空。"""
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 板块/行业
    # ------------------------------------------------------------------ #
    def concept_boards(self, date: Optional[str] = None) -> pd.DataFrame:
        d = date or self._current_date
        if d is None:
            return pd.DataFrame()
        bars = self._day_bars(d)
        if bars.empty:
            return pd.DataFrame()
        industry = self._industry_of(d)
        bars = bars.copy()
        bars["_ind"] = bars["code"].map(industry).fillna("")
        bars = bars[bars["_ind"] != ""]
        if bars.empty:
            return pd.DataFrame()
        g = bars.groupby("_ind").agg(
            上涨家数=("pct_change", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            下跌家数=("pct_change", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
            平均涨幅=("pct_change", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            总成交额=("amount", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
        ).reset_index()
        g = g.sort_values("平均涨幅", ascending=False).reset_index(drop=True)
        g["排名"] = g.index + 1
        g["板块名称"] = g["_ind"]
        g["板块代码"] = "BKIND" + g["排名"].astype(str).str.zfill(3)
        g["涨跌幅"] = g["平均涨幅"]
        g["领涨股票"] = ""
        return g[["排名", "板块名称", "板块代码", "涨跌幅", "上涨家数", "下跌家数", "总成交额", "领涨股票"]]

    def concept_constituents(
        self,
        board_symbol: str,
        board_name: str | None = None,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        d = date or self._current_date
        if d is None:
            return pd.DataFrame()
        bars = self._day_bars(d)
        industry = self._industry_of(d)
        names = self._universe_names()
        target = board_name or ""
        rows: list[dict[str, Any]] = []
        for _, bar in bars.iterrows():
            code = str(bar["code"])
            if industry.get(code) != target:
                continue
            rows.append({
                "代码": code,
                "名称": names.get(code, code),
                "最新价": float(bar["close"]) if bar["close"] == bar["close"] else 0.0,
                "涨跌幅": float(bar["pct_change"]) if bar["pct_change"] == bar["pct_change"] else 0.0,
                "成交额": float(bar["amount"]) if bar["amount"] == bar["amount"] else 0.0,
                "换手率": float(bar["turnover"]) if bar["turnover"] == bar["turnover"] else 0.0,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df.insert(0, "序号", range(1, len(df) + 1))
        return df

    # ------------------------------------------------------------------ #
    # 不可回放的数据面：显式空
    # ------------------------------------------------------------------ #
    def all_fund_flow_snapshot(self, fast: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def individual_fund_flow(self, symbol: str, fast: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def sector_fund_flow_rank(self, indicator: str = "今日") -> pd.DataFrame:
        return pd.DataFrame()

    def financial_indicator(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame()

    def longhu_bang(self, date: Optional[str] = None) -> pd.DataFrame:
        return pd.DataFrame()

    def strong_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 本地公告事件（事件驱动池回放）
    # ------------------------------------------------------------------ #
    def announcement_events(self, code: str, date: str, lookback_days: int = 3) -> list[dict[str, Any]]:
        """从本地巨潮公告档案读某股某日之前 lookback_days 内的事件。"""
        if not self.announcement_dir.exists():
            return []
        dates = [d for d in self._trade_dates if d <= str(date)][-lookback_days:]
        events: list[dict[str, Any]] = []
        for d in dates:
            path = self.announcement_dir / f"{d}.json"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for ev in data.get("events") or []:
                if _normalize_code(ev.get("code") or "") == _normalize_code(code):
                    events.append(ev)
        return events

    # ------------------------------------------------------------------ #
    # 兼容面
    # ------------------------------------------------------------------ #
    def get_source_quality(self) -> dict[str, Any]:
        return {
            "all_spot": {"status": "ok", "source": "replay_archive"},
            "fund_flow": {"status": "failed", "source": "replay_archive",
                          "warnings": ["回放模式无资金流数据"]},
        }

    def is_market_open(self, date: Optional[str] = None) -> bool:
        return (date or self._current_date) in self._trade_dates

    def last_trading_date(self, date: Optional[str] = None) -> str:
        d = date or self._current_date or self._trade_dates[-1]
        prior = [t for t in self._trade_dates if t <= d]
        return prior[-1] if prior else self._trade_dates[-1]

    def market_regime(self, date: Optional[str] = None, index_code: str = "equal_weight") -> Optional[dict[str, Any]]:
        """市场状态：全 A 等权指数收盘是否在 MA20 上方（由档案数据因果计算）。"""
        d = str(date or self._current_date or "")
        if not d:
            return None
        cached = self._regime_cache.get(d)
        if cached is not None:
            return dict(cached)
        if self._all_bars is None:
            return None
        closes = (
            self._all_bars[self._all_bars["date"] <= d]
            .groupby("date")["close"]
            .mean()
            .sort_index()
        )
        if len(closes) < 21:
            return None
        ma20 = closes.rolling(20).mean()
        asof = closes.index[-1]
        # MA20 斜率：与 5 个交易日前比较（严格模式下要求 MA20 上行）
        ma20_now = float(ma20.iloc[-1])
        ma20_prev = float(ma20.iloc[-6]) if len(ma20.dropna()) > 5 and not pd.isna(ma20.iloc[-6]) else ma20_now
        result = {
            "asof": str(asof),
            "close": float(closes.iloc[-1]),
            "ma20": ma20_now,
            "ma20_rising": bool(ma20_now > ma20_prev),
            "above_ma20": bool(closes.iloc[-1] > ma20_now),
            "index": "equal_weight",
        }
        self._regime_cache[d] = result
        return dict(result)
