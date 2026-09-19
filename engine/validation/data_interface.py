"""Pangu 2.0 验证层数据接口。

- :class:`ResearchData`：直接复用 ``engine.research.data_interface.ResearchData``
  协议（不可导入时定义同形协议兜底）。
- :class:`SyntheticValidationData`：确定性（seed）合成小面板，专供回测/验证
  单测使用。支持场景注入器：
  * :meth:`limit_up_open` —— 指定 (股票, 日) 开=高=低=涨停价（一字板）；
  * :meth:`suspend` —— 指定股票若干交易日停牌（面板行缺失）；
  * :meth:`ex_div` —— 除权日：preclose 偏离昨日 close，而 pct_change 保持
    真实收益（分红/拆股调整口径，百分数）。

关键数据事实（与 PITStore 一致）：
* ``pct_change`` 是分红/拆股调整后的当日收益（**百分数**）；
* ``preclose`` 是除权调整后的昨收（因此除权日 preclose != 前日 close）。

涨跌停规则：round(preclose*(1±ratio), 2)；ratio: is_st→0.05，
创业板(300)/科创板(688)→0.20，其余→0.10。
"""
from __future__ import annotations

import re
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

try:  # 复用研究层协议（首选）
    from engine.research.data_interface import ResearchData  # noqa: F401
except Exception:  # pragma: no cover - 研究层不可导入时的兜底定义
    @runtime_checkable
    class ResearchData(Protocol):  # type: ignore[no-redef]
        """与 engine.research.data_interface.ResearchData 同形的兜底协议。"""

        def daily_panel(self, start: str, end: str,
                        symbols: list[str] | None = None) -> pd.DataFrame: ...

        def universe(self, date: str) -> pd.DataFrame: ...

        def index_daily(self, code: str, start: str, end: str) -> pd.DataFrame: ...

        def trading_days(self, start: str, end: str) -> list[str]: ...


class LookaheadError(RuntimeError):
    """任何试图访问决策日之后数据的尝试。"""



PANEL_COLUMNS = (
    "open", "high", "low", "close", "preclose",
    "volume", "amount", "pct_change", "turnover", "is_st",
)


def limit_ratio(code: str, is_st: bool) -> float:
    """涨跌停幅度：ST 5%；创业板/科创板 20%；其余 10%。

    兼容 '300001.SZ' / 'sh.688001' / '300001' 三种写法（取数字部分判断）。
    """
    if is_st:
        return 0.05
    digits = re.sub(r"\D", "", str(code))
    return 0.20 if digits.startswith(("300", "688")) else 0.10


class SyntheticValidationData:
    """确定性合成面板（实现 ResearchData 协议），支持场景注入。

    参数：
        n_symbols / n_days / seed：规模与随机种子（完全确定）。
        start：起始交易日（工作日序列）。
        base_price：首日昨收基准价。
        is_st_every：每多少只取 1 只 ST（默认第 6 只，i % 6 == 5）。
    """

    def __init__(self, n_symbols: int = 8, n_days: int = 12, seed: int = 42,
                 start: str = "2025-01-06", base_price: float = 10.0,
                 is_st_every: int = 6):
        if n_symbols < 1 or n_days < 2:
            raise ValueError("synthetic validation panel too small")
        rng = np.random.default_rng(seed)
        self.dates: list[str] = [
            d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n_days)
        ]
        codes: list[str] = []
        for i in range(n_symbols):
            k, j = i % 4, i // 4 + 1
            if k == 0:
                codes.append(f"300{j:03d}.SZ")   # 创业板 20%
            elif k == 1:
                codes.append(f"688{j:03d}.SH")   # 科创板 20%
            elif k == 2:
                codes.append(f"60{j:04d}.SH")    # 沪主板 10%
            else:
                codes.append(f"00{j:04d}.SZ")    # 深主板 10%
        self.codes = codes
        self._st = {c: (i % is_st_every == is_st_every - 1) for i, c in enumerate(codes)}

        ret = rng.normal(0.0004, 0.012, size=(n_days, n_symbols))
        rows: list[dict] = []
        for jx, code in enumerate(codes):
            prev_close = float(base_price)
            for t in range(n_days):
                r = float(ret[t, jx])
                preclose = prev_close
                close = max(round(preclose * (1.0 + r), 2), 0.01)
                open_ = round(preclose * (1.0 + 0.3 * r), 2)
                high = round(max(open_, close) * 1.002, 2)
                low = round(min(open_, close) * 0.998, 2)
                volume = float(rng.integers(5_000_000, 50_000_000))
                rows.append({
                    "date": self.dates[t], "code": code,
                    "open": open_, "high": high, "low": low,
                    "close": close, "preclose": preclose,
                    "volume": volume, "amount": volume * close,
                    "pct_change": (close / preclose - 1.0) * 100.0,
                    "turnover": volume / 1e8,
                    "is_st": bool(self._st[code]),
                })
                prev_close = close
        self._df = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # 场景注入器
    # ------------------------------------------------------------------ #
    def _mask(self, symbol: str, day: str) -> pd.Series:
        return (self._df["code"] == symbol) & (self._df["date"] == day)

    def limit_prices(self, symbol: str, day: str) -> tuple[float, float]:
        """(涨停价, 跌停价)，按当前面板的 preclose/is_st 计算。"""
        rows = self._df[self._mask(symbol, day)]
        if rows.empty:
            raise KeyError(f"no synthetic bar for {symbol} @ {day}")
        r = rows.iloc[0]
        ratio = limit_ratio(symbol, bool(r["is_st"]))
        return round(float(r["preclose"]) * (1 + ratio), 2), round(float(r["preclose"]) * (1 - ratio), 2)

    def limit_up_open(self, symbol: str, day: str) -> None:
        """一字涨停板：open == high == low == 涨停价（close 不越界）。"""
        up, _ = self.limit_prices(symbol, day)
        m = self._mask(symbol, day)
        self._df.loc[m, ["open", "high", "low"]] = up
        self._df.loc[m, "close"] = self._df.loc[m, "close"].clip(upper=up)
        self._refresh_pct(symbol, day)

    def suspend(self, symbol: str, days) -> None:
        """停牌：移除指定日期的面板行。``days`` 为日期列表或前 n 个交易日(int)。"""
        if isinstance(days, int):
            days = self.dates[:days]
        drop = (self._df["code"] == symbol) & (self._df["date"].isin(list(days)))
        self._df = self._df[~drop].reset_index(drop=True)

    def ex_div(self, symbol: str, day: str, pct: float) -> None:
        """除权：preclose = round(昨收*(1-pct), 2)，pct_change 保持真实收益。"""
        m = self._mask(symbol, day)
        idx = self._df.index[m]
        if len(idx) != 1:
            raise KeyError(f"no synthetic bar for {symbol} @ {day}")
        pos = self.dates.index(day)
        if pos == 0:
            raise ValueError("ex_div needs a previous trading day")
        prev_close = float(self._df.loc[
            self._mask(symbol, self.dates[pos - 1]), "close"
        ].iloc[0])
        new_pre = round(prev_close * (1.0 - pct), 2)
        i = idx[0]
        self._df.at[i, "preclose"] = new_pre
        # pct_change 保持真实收益（含分红口径）：close / 昨收 - 1
        self._df.at[i, "pct_change"] = (float(self._df.at[i, "close"]) / prev_close - 1.0) * 100.0

    def _refresh_pct(self, symbol: str, day: str) -> None:
        m = self._mask(symbol, day)
        i = self._df.index[m][0]
        self._df.at[i, "pct_change"] = (
            float(self._df.at[i, "close"]) / float(self._df.at[i, "preclose"]) - 1.0
        ) * 100.0

    # ------------------------------------------------------------------ #
    # ResearchData 协议
    # ------------------------------------------------------------------ #
    def daily_panel(self, start: str, end: str,
                    symbols: Iterable[str] | None = None) -> pd.DataFrame:
        df = self._df[(self._df["date"] >= start) & (self._df["date"] <= end)]
        if symbols is not None:
            df = df[df["code"].isin(set(symbols))]
        out = df.sort_values(["date", "code"]).copy()
        out = out.set_index(pd.MultiIndex.from_arrays(
            [out["date"], out["code"]], names=["date", "code"]))
        return out[list(PANEL_COLUMNS)]

    def universe(self, date: str) -> pd.DataFrame:
        if date not in set(self.dates):
            raise ValueError(f"non-trading date: {date}")
        present = set(self._df[self._df["date"] == date]["code"])
        pos = self.dates.index(date)
        rows = [{
            "code": c,
            "name": f"sym{c.split('.')[0]}",
            "is_st": bool(self._st[c]),
            "suspended": c not in present,
            "listed_days": pos + 100,
            "tradable": c in present,
        } for c in self.codes]
        return pd.DataFrame(rows).set_index("code")

    def index_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        r = (self._df["pct_change"] / 100.0).groupby(self._df["date"]).mean().sort_index()
        level = 1000.0 * (1.0 + r).cumprod()
        mask = (level.index >= start) & (level.index <= end)
        return pd.DataFrame({"close": level[mask]})

    def trading_days(self, start: str, end: str) -> list[str]:
        return [d for d in self.dates if start <= d <= end]
