"""ResearchData 协议 + PITStore 适配 + 合成数据（离线测试用）。

协议（点对点契约，见模块 docstring / Pangu 2.0 任务表 P2-000）：
    daily_panel(start, end, symbols=None) -> DataFrame
        MultiIndex (date, code)，ISO 日期字符串；列：
        open, high, low, close, preclose, volume, amount, pct_change,
        turnover, is_st。
        pct_change = 日收益率（百分比，分红除权后）。**严格 date <= end**。
    universe(date) -> DataFrame
        index=code；列 name, is_st, suspended, listed_days, tradable。
    index_daily(code, start, end) -> DataFrame
        date 索引（ISO），close 列。
    trading_days(start, end) -> list[str]

PITResearchData 惰性引入 engine.data.pit_store.PITStore（并行任务在建），
尚未就绪时抛出带清晰指引的 ImportError——这是预期状态，不要等它。

SyntheticResearchData：确定性合成面板（numpy default_rng(seed)），
~120 只 × 400 个交易日，几何随机游走 + 横截面波动离散 + 少量 ST 与停牌。
注入一个设计好的 alpha：个股 quality q ~ N(0,1)，仅在样本前
`alpha_days`（默认 200）个交易日给日收益加 +q*alpha_bp（默认 2bp）漂移——
状态依赖信号，供 regime 切分机制验证。默认 2bp 接近真实弱信号；
测试需要可检测的 IC 时可用 alpha_bp 调强。
"""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

PANEL_COLUMNS = (
    "open", "high", "low", "close", "preclose",
    "volume", "amount", "pct_change", "turnover", "is_st",
)


@runtime_checkable
class ResearchData(Protocol):
    """研究层唯一数据入口（点对点契约，勿在此之外另造接口）。"""

    def daily_panel(self, start: str, end: str,
                    symbols: list[str] | None = None) -> pd.DataFrame: ...

    def universe(self, date: str) -> pd.DataFrame: ...

    def index_daily(self, code: str, start: str, end: str) -> pd.DataFrame: ...

    def trading_days(self, start: str, end: str) -> list[str]: ...


class PITResearchData:
    """PITStore -> ResearchData 适配器。

    假设 engine.data.pit_store.PITStore 提供与协议同形的四个方法
    （daily_panel / universe / index_daily / trading_days，语义见协议注释）。
    语义差异（尤其 daily_panel 的严格 end 上界）由 PITStore 侧负责；
    研究层 univariate 会再做一层 date <= asof 硬切片兜底。
    """

    def __init__(self, store=None):
        if store is None:
            try:
                from engine.data.pit_store import PITStore as _PITStore  # noqa: 惰性引入
            except Exception as exc:  # ImportError 或其内部依赖未就绪
                raise ImportError(
                    "engine.data.pit_store 尚不可用（P2 数据层并行任务未完成）。"
                    "离线研究/测试请改用 "
                    "engine.research.data_interface.SyntheticResearchData。"
                    f"原始错误: {exc}"
                ) from exc
            store = _PITStore()
        self._store = store

    # -- 协议实现（直接委托） ------------------------------------------------

    def daily_panel(self, start: str, end: str,
                    symbols: list[str] | None = None) -> pd.DataFrame:
        return self._store.daily_panel(start, end, symbols)

    def universe(self, date: str) -> pd.DataFrame:
        return self._store.universe(date)

    def index_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        return self._store.index_daily(code, start, end)

    def trading_days(self, start: str, end: str) -> list[str]:
        return list(self._store.trading_days(start, end))


class SyntheticResearchData:
    """确定性合成面板，实现 ResearchData 协议（离线测试专用）。

    参数：
        n_symbols / n_days / seed：面板规模与随机种子（完全确定）。
        alpha_bp：quality alpha 的日漂移（百分比小数的 bp，2.0 = 每日 2bp）。
        alpha_days：alpha 只作用于样本前多少个交易日（regime 依赖）。
        sigma_lo / sigma_hi：个股日波动率的均匀分布区间（横截面离散）。
        quality：注入的隐变量 q（ndarray，与 codes 对齐），供测试构造
            “揭示该 alpha”的因子（oracle）。
    """

    def __init__(self, n_symbols: int = 120, n_days: int = 400, seed: int = 7,
                 alpha_bp: float = 2.0, alpha_days: int = 200,
                 sigma_lo: float = 0.012, sigma_hi: float = 0.035):
        if n_symbols < 4 or n_days < 30:
            raise ValueError("synthetic panel too small")
        self.seed = int(seed)
        self.alpha_bp = float(alpha_bp)
        self.alpha_days = int(min(alpha_days, n_days - 1))
        rng = np.random.default_rng(seed)

        # --- 代码与板块（决定涨跌停幅度结构） -------------------------------
        codes: list[str] = []
        counters = {"300": 0, "688": 0, "60": 0, "00": 0}
        for i in range(n_symbols):
            k = i % 5
            if k == 0:      # 创业板 20%
                counters["300"] += 1
                codes.append(f"300{counters['300']:03d}.SZ")
            elif k == 1:    # 科创板 20%
                counters["688"] += 1
                codes.append(f"688{counters['688']:03d}.SH")
            elif k == 2:    # 沪主板 10%
                counters["60"] += 1
                codes.append(f"60{counters['60']:04d}.SH")
            else:           # 深主板 10%
                counters["00"] += 1
                codes.append(f"00{counters['00']:04d}.SZ")
        self.codes = codes

        # --- 隐变量与参数 ----------------------------------------------------
        self.quality = rng.normal(size=n_symbols)       # q ~ N(0,1)
        sigma = rng.uniform(sigma_lo, sigma_hi, size=n_symbols)
        start_price = 5.0 * np.exp(rng.normal(0.0, 0.6, size=n_symbols))
        shares_out = np.exp(rng.normal(18.0, 0.3, size=n_symbols))
        is_st_flags = np.array([(i % 24 == 5) for i in range(n_symbols)])
        self._is_st = dict(zip(codes, is_st_flags))

        # --- 日收益率：几何随机游走 + 前 alpha_days 注入 q*alpha_bp ---------
        eps = rng.normal(0.0, 1.0, size=(n_days, n_symbols))
        ret = sigma * eps
        ret[: self.alpha_days, :] += np.outer(
            np.ones(self.alpha_days), self.quality * (self.alpha_bp * 1e-4)
        )

        # --- 停牌缺口：少数个股在中段停牌若干天（行直接缺失） ----------------
        suspended = np.zeros((n_days, n_symbols), dtype=bool)
        for i in range(n_symbols):
            if i % 20 == 7:
                s = 150 + (i % 9) * 3
                suspended[s: s + 12, i] = True

        # --- 价格 / 成交 ------------------------------------------------------
        preclose = np.empty_like(ret)
        close = np.empty_like(ret)
        preclose[0] = start_price
        close[0] = start_price * (1.0 + ret[0])
        for t in range(1, n_days):
            preclose[t] = close[t - 1]
            close[t] = close[t - 1] * (1.0 + ret[t])
        noise = rng.normal(0.0, 1.0, size=ret.shape)
        open_ = preclose * (1.0 + 0.004 * noise)
        high = np.maximum(open_, close) * (1.0 + 0.006 * np.abs(noise))
        low = np.minimum(open_, close) * (1.0 - 0.006 * np.abs(noise))
        volume = np.exp(rng.normal(12.0, 0.5, size=ret.shape)) * (1.0 + 3.0 * np.abs(ret))
        amount = volume * close
        turnover = volume / shares_out * 100.0

        self.dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-06-03", periods=n_days)]
        self._all_panel = self._build_panel(
            self.dates, codes, open_, high, low, close, preclose,
            volume, amount, turnover, is_st_flags, suspended,
        )
        self._dates_set = set(self.dates)
        self._member = set(zip(self._all_panel.index.get_level_values("date"),
                               self._all_panel.index.get_level_values("code")))

    # -- 内部 ---------------------------------------------------------------

    @staticmethod
    def _build_panel(dates, codes, open_, high, low, close, preclose,
                     volume, amount, turnover, is_st_flags, suspended):
        frames = []
        dates_arr = np.array(dates, dtype=object)
        for j, code in enumerate(codes):
            keep = ~suspended[:, j]
            df = pd.DataFrame(
                {
                    "open": open_[keep, j], "high": high[keep, j], "low": low[keep, j],
                    "close": close[keep, j], "preclose": preclose[keep, j],
                    "volume": volume[keep, j], "amount": amount[keep, j],
                    "pct_change": ret_pct(close[keep, j], preclose[keep, j]),
                    "turnover": turnover[keep, j],
                    "is_st": np.full(int(keep.sum()), bool(is_st_flags[j])),
                },
                index=pd.Index(dates_arr[keep], name="date"),
            )
            df["code"] = code
            frames.append(df)
        panel = pd.concat(frames)
        panel = panel.set_index("code", append=True).sort_index()
        panel.index.names = ["date", "code"]
        return panel[list(PANEL_COLUMNS)]

    def _require_date(self, date: str) -> None:
        if date not in self._dates_set:
            raise ValueError(f"non-trading date for synthetic universe: {date}")

    # -- ResearchData 协议 ---------------------------------------------------

    def daily_panel(self, start: str, end: str,
                    symbols: Iterable[str] | None = None) -> pd.DataFrame:
        idx = self._all_panel.index
        mask = (idx.get_level_values("date") >= start) & (idx.get_level_values("date") <= end)
        out = self._all_panel[mask]
        if symbols is not None:
            keep = set(symbols)
            out = out[out.index.get_level_values("code").isin(keep)]
        return out.copy()

    def universe(self, date: str) -> pd.DataFrame:
        self._require_date(date)
        pos = self.dates.index(date)
        rows = []
        for code in self.codes:
            suspended = (date, code) not in self._member  # 停牌日面板行缺失
            rows.append({
                "code": code,
                "name": f"sym{code.split('.')[0]}",
                "is_st": bool(self._is_st[code]),
                "suspended": bool(suspended),
                "listed_days": pos + 500,   # 样本开始前均已上市
                "tradable": not suspended,
            })
        return pd.DataFrame(rows).set_index("code")

    def index_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        # 合成数据只有一条等权Composite：任意 code 均返回它（文档化行为）。
        r = (self._all_panel["pct_change"] / 100.0).groupby(level="date").mean()
        level = 1000.0 * (1.0 + r).cumprod()
        mask = (level.index >= start) & (level.index <= end)
        return pd.DataFrame({"close": level[mask]})

    def trading_days(self, start: str, end: str) -> list[str]:
        return [d for d in self.dates if start <= d <= end]


def ret_pct(close: np.ndarray, preclose: np.ndarray) -> np.ndarray:
    """pct_change（百分比），与 close/preclose 严格一致。"""
    return (close / preclose - 1.0) * 100.0
