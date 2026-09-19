"""exec_model 纯函数单测：涨跌停、成交/未成交规则、费用、容量、同 bar 约定。"""

from __future__ import annotations

import pytest

from engine.validation.exec_model import (
    FeeSchedule,
    fees,
    fill_buy,
    fill_sell,
    limit_prices,
    order_capacity,
    same_bar_order,
)


class TestLimitPrices:
    def test_main_board_10pct(self):
        assert limit_prices(10.00, False, "600001.SH") == (11.00, 9.00)
        assert limit_prices(10.00, False, "000002.SZ") == (11.00, 9.00)

    def test_st_5pct(self):
        assert limit_prices(10.00, True, "600001.SH") == (10.50, 9.50)
        assert limit_prices(10.00, True, "000002.SZ") == (10.50, 9.50)

    def test_chinext_and_star_20pct(self):
        assert limit_prices(10.00, False, "300001.SZ") == (12.00, 8.00)
        assert limit_prices(10.00, False, "688001.SH") == (12.00, 8.00)

    def test_rounding(self):
        # 10.13 → up round(11.143,2)=11.14, down round(9.117,2)=9.12
        assert limit_prices(10.13, False, "600001.SH") == (11.14, 9.12)


class TestFillBuy:
    def test_normal_fill_with_slippage(self):
        f = fill_buy(10.00, 10.50, 9.95, 11.00, slippage_bps=10.0)
        assert f is not None
        assert f.raw_price == 10.00
        assert f.price == pytest.approx(10.00 * 1.001, abs=1e-12)

    def test_zero_slippage(self):
        f = fill_buy(10.00, 10.50, 9.95, 11.00, slippage_bps=0.0)
        assert f.price == 10.00

    def test_open_at_limit_up_no_allow(self):
        assert fill_buy(11.00, 11.20, 10.90, 11.00) is None

    def test_open_above_limit_up_no_allow(self):
        assert fill_buy(11.50, 11.60, 11.40, 11.00) is None

    def test_one_word_board_even_with_allow(self):
        # open == high == low == limit_up：一字板即使显式允许也不成交
        assert fill_buy(11.00, 11.00, 11.00, 11.00, allow_buy_at_limit_up=True) is None

    def test_open_at_limit_with_allow_but_range(self):
        # 触板但非一字板：显式允许时成交
        f = fill_buy(11.00, 11.20, 10.90, 11.00, allow_buy_at_limit_up=True)
        assert f is not None and f.price == pytest.approx(11.011, abs=1e-12)


class TestFillSell:
    def test_normal_fill(self):
        f = fill_sell(10.00, 10.50, 9.95, 9.00, slippage_bps=10.0)
        assert f is not None
        assert f.price == pytest.approx(10.00 * 0.999, abs=1e-12)

    def test_open_at_limit_down_no_allow(self):
        assert fill_sell(9.00, 9.20, 8.80, 9.00) is None

    def test_open_below_limit_down_no_allow(self):
        assert fill_sell(8.90, 9.00, 8.80, 9.00) is None

    def test_open_at_limit_down_with_allow(self):
        # 跌停卖出反而容易：显式允许即成交（无一字板限制）
        f = fill_sell(9.00, 9.20, 8.80, 9.00, allow_sell_at_limit_down=True)
        assert f is not None and f.price == pytest.approx(8.991, abs=1e-12)


class TestFees:
    def test_buy_no_stamp(self):
        out = fees(100_000.0, "BUY", FeeSchedule())
        assert out["commission"] == pytest.approx(30.0, abs=1e-9)
        assert out["min_commission_applied"] is False
        assert out["stamp"] == 0.0
        assert out["transfer"] == pytest.approx(1.0, abs=1e-9)
        assert out["total"] == pytest.approx(31.0, abs=1e-9)

    def test_min_commission_applied(self):
        out = fees(1_000.0, "BUY", FeeSchedule())
        assert out["commission"] == 5.0
        assert out["min_commission_applied"] is True
        assert out["total"] == pytest.approx(5.0 + 0.01, abs=1e-9)

    def test_sell_has_stamp(self):
        out = fees(100_000.0, "SELL", FeeSchedule())
        assert out["stamp"] == pytest.approx(50.0, abs=1e-9)
        assert out["total"] == pytest.approx(30.0 + 50.0 + 1.0, abs=1e-9)

    def test_duck_typed_cfg(self):
        class C:
            commission_rate = 0.001
            min_commission = 1.0
            stamp_duty_rate = 0.0
            transfer_fee_rate = 0.0

        assert fees(10_000.0, "BUY", C())["total"] == pytest.approx(10.0, abs=1e-9)


def test_same_bar_order_default_stop_first():
    assert same_bar_order() == "stop_first"
    assert same_bar_order("stop_first") == "stop_first"
    with pytest.raises(ValueError):
        same_bar_order("target_first")


class TestOrderCapacity:
    def test_cut_to_cap(self):
        assert order_capacity(1_000.0, 10_000.0, 0.02) == pytest.approx(200.0)

    def test_within_cap(self):
        assert order_capacity(100.0, 10_000.0, 0.02) == pytest.approx(100.0)

    def test_zero_or_negative_amount(self):
        assert order_capacity(1_000.0, 0.0, 0.02) == 0.0
        assert order_capacity(1_000.0, -5.0, 0.02) == 0.0
