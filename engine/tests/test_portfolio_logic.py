"""持仓跟踪模块纯逻辑测试（内存 SQLite + mock DataLoader，不依赖网络）。"""

import pandas as pd
import pytest

from engine.portfolio import PortfolioTracker


class FakeDataLoader:
    """模拟 DataLoader，只返回预设实时行情。"""

    def __init__(self, prices: dict[str, float] | None = None) -> None:
        self.prices = prices or {}

    def all_spot(self) -> pd.DataFrame:
        rows = [
            {"代码": code, "名称": name, "最新价": price}
            for code, (name, price) in self.prices.items()
        ]
        return pd.DataFrame(rows)

    def daily_kline(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()


def _make_tracker(prices: dict[str, tuple[str, float]] | None = None) -> PortfolioTracker:
    return PortfolioTracker(":memory:", data_loader=FakeDataLoader(prices))


# ---------------------------------------------------------------------- #
# 买入 / 持仓维护
# ---------------------------------------------------------------------- #
def test_record_buy_creates_holding():
    t = _make_tracker()
    t.record_buy("000001", "平安银行", 1000, 10.0, "20240110")

    holdings = t.current_holdings()
    assert len(holdings) == 1
    h = holdings[0]
    assert h.code == "000001"
    assert h.name == "平安银行"
    assert h.shares == 1000
    assert h.avg_cost == pytest.approx(10.0)
    assert h.reason == ""


def test_record_buy_updates_avg_cost():
    t = _make_tracker()
    t.record_buy("000001", "平安银行", 1000, 10.0, "20240110")
    t.record_buy("000001", "平安银行", 1000, 12.0, "20240111")

    h = t.current_holdings()[0]
    assert h.shares == 2000
    assert h.avg_cost == pytest.approx(11.0)


def test_record_buy_validation():
    t = _make_tracker()
    with pytest.raises(ValueError):
        t.record_buy("000001", "x", 0, 10.0, "20240110")
    with pytest.raises(ValueError):
        t.record_buy("000001", "x", 100, -1.0, "20240110")


# ---------------------------------------------------------------------- #
# 卖出 / 盈亏 / 清仓
# ---------------------------------------------------------------------- #
def test_record_sell_calculates_pnl_and_reduces():
    t = _make_tracker()
    t.record_buy("000001", "平安银行", 1000, 10.0, "20240110")
    pnl = t.record_sell("000001", 500, 12.0, "20240115")

    assert pnl == pytest.approx(1000.0)
    h = t.current_holdings()[0]
    assert h.shares == 500
    assert h.avg_cost == pytest.approx(10.0)


def test_record_sell_full_liquidation():
    t = _make_tracker()
    t.record_buy("000001", "平安银行", 1000, 10.0, "20240110")
    pnl = t.record_sell("000001", 1000, 9.0, "20240115")

    assert pnl == pytest.approx(-1000.0)
    assert t.current_holdings() == []


def test_record_sell_insufficient_shares():
    t = _make_tracker()
    t.record_buy("000001", "平安银行", 100, 10.0, "20240110")
    with pytest.raises(ValueError):
        t.record_sell("000001", 200, 11.0, "20240111")


# ---------------------------------------------------------------------- #
# 实时行情与浮动盈亏
# ---------------------------------------------------------------------- #
def test_current_holdings_with_realtime_price():
    t = _make_tracker({"000001": ("平安银行", 12.0)})
    t.record_buy("000001", "平安银行", 1000, 10.0, "20240110")

    h = t.current_holdings()[0]
    assert h.current_price == pytest.approx(12.0)
    assert h.market_value == pytest.approx(12000.0)
    assert h.pnl == pytest.approx(2000.0)
    assert h.pnl_pct == pytest.approx(20.0)


def test_current_holdings_fallback_price():
    """实时行情缺失时，应 fallback 到最近成交价。"""
    t = _make_tracker()
    t.record_buy("000001", "平安银行", 1000, 10.0, "20240110")

    h = t.current_holdings()[0]
    assert h.current_price == pytest.approx(10.0)
    assert h.pnl == pytest.approx(0.0)


# ---------------------------------------------------------------------- #
# 总览 / 胜率 / 持仓天数
# ---------------------------------------------------------------------- #
def test_summary_with_mixed_trades():
    t = _make_tracker({"000001": ("平安银行", 12.0)})
    t.record_buy("000001", "平安银行", 1000, 10.0, "20240110")
    t.record_buy("000002", "万科", 500, 20.0, "20240110")
    t.record_sell("000002", 500, 18.0, "20240120")  # 亏损 1000
    t.record_sell("000001", 500, 12.0, "20240120")  # 盈利 1000

    s = t.summary()
    assert s["total_invested"] == pytest.approx(20000.0)
    assert s["sell_count"] == 2
    assert s["win_rate"] == pytest.approx(50.0)
    assert s["avg_hold_days"] == pytest.approx(10.0)
    assert s["holding_count"] == 1


def test_win_rate_calculation():
    t = _make_tracker()
    t.record_buy("000001", "A", 100, 10.0, "20240101")
    t.record_buy("000002", "B", 100, 10.0, "20240101")
    t.record_buy("000003", "C", 100, 10.0, "20240101")
    t.record_sell("000001", 100, 11.0, "20240105")  # 赢
    t.record_sell("000002", 100, 9.0, "20240105")   # 亏
    t.record_sell("000003", 100, 11.0, "20240105")  # 赢

    assert t.summary()["win_rate"] == pytest.approx(66.67, abs=0.01)


def test_transactions_order_and_limit():
    t = _make_tracker()
    t.record_buy("000001", "A", 100, 10.0, "20240101")
    t.record_sell("000001", 100, 11.0, "20240102")

    rows = t.transactions(limit=10)
    assert len(rows) == 2
    assert rows[0].date == "20240102"  # 倒序
    assert rows[1].date == "20240101"


# ---------------------------------------------------------------------- #
# 归因
# ---------------------------------------------------------------------- #
def test_attribution_flags():
    t = _make_tracker()
    t.record_buy("000001", "A", 100, 10.0, "20240101")
    t.record_sell("000001", 100, 8.0, "20240102", reason="止损")

    attr = t.attribution("000001")
    assert attr["code"] == "000001"
    assert len(attr["buys"]) == 1
    assert len(attr["sells"]) == 1
    assert attr["sells"][0]["pnl"] == pytest.approx(-200.0)
    # entry_exit 计算失败（FakeDataLoader 无 K 线）时仍应返回基础字段
    assert "executed_stop_loss" in attr


# ---------------------------------------------------------------------- #
# rich 表格
# ---------------------------------------------------------------------- #
def test_to_table():
    pytest.importorskip("rich")
    t = _make_tracker({"000001": ("平安银行", 12.0)})
    t.record_buy("000001", "平安银行", 1000, 10.0, "20240110")

    table = t.to_table()
    assert table.title == "当前持仓"
    assert len(table.columns) == 8
    assert len(table.rows) == 1
