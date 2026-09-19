"""BacktestV2 黄金测试：2 只股票 × 6 个交易日、手算数字、权益曲线精确到分。

情景（600001.SH 主板 10%，000002.SZ 恒为 20.00）：
- d1 收盘决策：BUY 600001 权重 0.5（max_weight 放宽到 0.5）
- d2 开盘成交：open=10.30 → 执行价 10.30*1.001=10.3103，
  qty = floor(500000/10.3103/100)*100 = 48400
- d5 收盘决策：SELL 600001 权重 0（清仓）
- d6 开盘成交：open=11.30 → 执行价 11.30*0.999=11.2887

手算（initial=1,000,000，佣金 3bp 最低 5 元，印花 5bp，过户 0.1bp）：
  notional_buy  = 48400*10.3103 = 499,018.52
  fees_buy      = 149.705556 + 4.9901852 = 154.6957412
  cash(d2..d5)  = 500,826.7842588
  权益(d2)      = 500,826.7842588 + 48400*10.55 = 1,011,446.78
  权益(d3)      = +48400*11.00 → 1,033,226.78
  权益(d4)      = +48400*11.20 → 1,042,906.78
  权益(d5)      = +48400*11.30 → 1,047,746.78
  notional_sell = 48400*11.2887 = 546,373.08
  fees_sell     = 163.911924 + 273.18654 + 5.4637308 = 442.5621948
  权益(d6)      = 500,826.7842588 + 546,373.08 - 442.5621948 = 1,046,757.30
  trade.pnl     = 546,373.08 - 442.5621948 - (499,018.52 + 154.6957412)
                = 46,757.302064
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.validation.backtest_v2 import BacktestConfig, BacktestV2, TargetOrder
from engine.validation.cross_check import cross_check

DATES = ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09",
         "2025-01-10", "2025-01-13"]

# (open, high, low, close, preclose) 逐日
P_600001 = [
    (10.00, 10.50, 9.95, 10.20, 10.00),
    (10.30, 10.60, 10.10, 10.55, 10.20),
    (10.60, 11.10, 10.45, 11.00, 10.55),
    (11.05, 11.30, 10.90, 11.20, 11.00),
    (11.25, 11.45, 11.10, 11.30, 11.20),
    (11.30, 11.50, 11.20, 11.40, 11.30),
]
SYMS = ["600001.SH", "000002.SZ"]
VOLUME = 50_000_000.0


class GoldenData:
    """字面数字小面板（实现 ResearchData 协议）。"""

    def __init__(self):
        rows = []
        for sym in SYMS:
            prev_close = P_600001[0][4] if sym == "600001.SH" else 20.00
            for t, d in enumerate(DATES):
                if sym == "600001.SH":
                    o, h, l, c, pre = P_600001[t]
                else:
                    o = h = l = c = pre = 20.00
                rows.append({
                    "date": d, "code": sym, "open": o, "high": h, "low": l,
                    "close": c, "preclose": pre, "volume": VOLUME,
                    "amount": VOLUME * c, "pct_change": (c / pre - 1) * 100.0,
                    "turnover": VOLUME / 1e8, "is_st": False,
                })
                prev_close = c
        self._df = pd.DataFrame(rows)

    def daily_panel(self, start, end, symbols=None):
        df = self._df[(self._df["date"] >= start) & (self._df["date"] <= end)]
        if symbols is not None:
            df = df[df["code"].isin(set(symbols))]
        out = df.copy()
        out = out.set_index(pd.MultiIndex.from_arrays(
            [out["date"], out["code"]], names=["date", "code"]))
        return out.drop(columns=["date", "code"]).sort_index()

    def universe(self, date):
        present = set(self._df[self._df["date"] == date]["code"])
        return pd.DataFrame([
            {"code": s, "name": s, "is_st": False, "suspended": s not in present,
             "listed_days": 100, "tradable": s in present}
            for s in SYMS]).set_index("code")

    def index_daily(self, code, start, end):
        idx = [d for d in DATES if start <= d <= end]
        return pd.DataFrame({"close": 1000.0}, index=pd.Index(idx, name="date"))

    def trading_days(self, start, end):
        return [d for d in DATES if start <= d <= end]


class GoldenStrategy:
    """d1 决策买 600001 半仓；d5 决策清仓。"""

    def rebalance(self, decision_date, history):
        if decision_date == DATES[0]:
            return [TargetOrder(symbol="600001.SH", side="BUY", weight=0.5)]
        if decision_date == DATES[4]:
            return [TargetOrder(symbol="600001.SH", side="SELL", weight=0.0)]
        return []


@pytest.fixture
def result():
    cfg = BacktestConfig(max_weight_per_stock=0.5)
    return BacktestV2(GoldenData(), cfg).run(GoldenStrategy(), DATES[0], DATES[-1])


class TestGoldenEquityCurve:
    EXPECTED_EQUITY = [1_000_000.00, 1_011_446.78, 1_033_226.78,
                       1_042_906.78, 1_047_746.78, 1_046_757.30]

    def test_equity_rows_exact_to_cent(self, result):
        eq = result.equity_curve
        assert list(eq["date"]) == DATES
        for got, want in zip(eq["equity"], self.EXPECTED_EQUITY):
            assert float(got) == pytest.approx(want, abs=0.01)

    def test_cash_and_market_value(self, result):
        eq = result.equity_curve
        assert float(eq.iloc[0]["cash"]) == pytest.approx(1_000_000.00, abs=0.01)
        # d2: 现金 500,826.78，市值 48400*10.55 = 510,620.00
        assert float(eq.iloc[1]["cash"]) == pytest.approx(500_826.78, abs=0.01)
        assert float(eq.iloc[1]["market_value"]) == pytest.approx(510_620.00, abs=0.01)
        assert float(eq.iloc[-1]["cash"]) == pytest.approx(1_046_757.30, abs=0.01)
        assert float(eq.iloc[-1]["market_value"]) == pytest.approx(0.0, abs=0.01)


class TestGoldenTrade:
    def test_trade_record_exact(self, result):
        assert len(result.trades) == 1
        tr = result.trades[0]
        assert tr["symbol"] == "600001.SH"
        assert tr["side"] == "LONG"
        assert tr["entry_date"] == "2025-01-07"
        assert tr["exit_date"] == "2025-01-13"
        assert tr["qty"] == 48400
        assert tr["entry_price"] == pytest.approx(10.3103, abs=1e-9)
        assert tr["exit_price"] == pytest.approx(11.2887, abs=1e-9)
        assert tr["fees_buy"] == pytest.approx(154.6957412, abs=1e-6)
        assert tr["fees_sell"] == pytest.approx(442.5621948, abs=1e-6)
        assert tr["pnl"] == pytest.approx(46_757.302064, abs=0.01)
        assert tr["ret"] == pytest.approx(46_757.302064 / 499_173.2157412, abs=1e-9)
        assert tr["holding_days"] == 4
        assert tr["close_reason"] == "sell_target"

    def test_summary(self, result):
        s = result.summary
        assert s["total_return"] == pytest.approx(0.0467573, abs=1e-6)
        assert s["n_trades"] == 1
        assert s["win_rate_trades"] == 1.0
        assert s["execution_rate"] == 1.0
        assert s["n_unfilled_events"] == 0


class TestGoldenCrossCheck:
    def test_independent_repricing_consistent(self, result):
        data = GoldenData()
        report = cross_check(result, data, GoldenStrategy())
        assert report["n_trades_checked"] == 1
        assert report["consistent"] is True, report
        assert report["max_rel_diff"] < 1e-6
        assert report["equity_max_abs_diff"] < 0.01
        assert report["violations"] == []
