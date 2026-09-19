"""Pangu 2.0 执行模型：纯函数（重度单测覆盖）。

语义：
- 涨停无法买入（除非显式 allow_buy_at_limit_up，但一字板 open==high==low==
  limit_up 仍不可成交——排队必排在巨量后面）；跌停无法卖出（除非显式
  allow_sell_at_limit_down；跌停一字板卖出反而容易，无特殊限制）。
- 成交价 = 开盘价 * (1 ∓ slippage_bps/1e4)。
- 费用：佣金 max(min_commission, notional*rate)，卖出加印花税，双边过户费。
- 同根 K 线同时触及止损与止盈时，保守约定 stop_first（引擎自身不做止损，
  该常量供策略层声明使用）。
- 容量：单票单日成交额不超过 day_amount * participation_cap，超出部分未成交。
"""
from __future__ import annotations

from dataclasses import dataclass

SAME_BAR_CONSERVATIVE = "stop_first"

BUY = "BUY"
SELL = "SELL"


@dataclass(frozen=True)
class Fill:
    """一次成交：raw 为原始触发价，price 为含滑点的执行价。"""
    raw_price: float
    price: float


@dataclass(frozen=True)
class FeeSchedule:
    """费用参数（BacktestConfig 同形，可鸭子类型互换）。"""
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001


def limit_prices(preclose: float, is_st: bool, code: str) -> tuple[float, float]:
    """(涨停价, 跌停价) = round(preclose*(1±ratio), 2)。"""
    from .data_interface import limit_ratio
    ratio = limit_ratio(code, is_st)
    up = round(float(preclose) * (1.0 + ratio), 2)
    down = round(float(preclose) * (1.0 - ratio), 2)
    return up, down


def fill_buy(open_: float, high: float, low: float, limit_up: float,
             allow_buy_at_limit_up: bool = False,
             slippage_bps: float = 10.0) -> Fill | None:
    """开盘买入；open >= limit_up 且未显式允许 → None；一字板一律 None。"""
    o = float(open_)
    if o >= float(limit_up) and not allow_buy_at_limit_up:
        return None
    if o >= float(limit_up) and float(high) == float(low) == o:
        return None  # 一字涨停板：挂单必然排在巨量封单之后
    slip = slippage_bps / 10_000.0
    return Fill(raw_price=o, price=o * (1.0 + slip))


def fill_sell(open_: float, high: float, low: float, limit_down: float,
              allow_sell_at_limit_down: bool = False,
              slippage_bps: float = 10.0) -> Fill | None:
    """开盘卖出；open <= limit_down 且未显式允许 → None。"""
    o = float(open_)
    if o <= float(limit_down) and not allow_sell_at_limit_down:
        return None
    slip = slippage_bps / 10_000.0
    return Fill(raw_price=o, price=o * (1.0 - slip))


def fees(notional: float, side: str, cfg: FeeSchedule | object) -> dict:
    """费用明细：commission（含最低佣金）、stamp（仅卖出）、transfer、total。

    返回 dict：commission, min_commission_applied, stamp, transfer, total。
    """
    c = cfg
    n = abs(float(notional))
    raw = n * float(getattr(c, "commission_rate", 0.0003))
    min_comm = float(getattr(c, "min_commission", 5.0))
    applied = raw < min_comm
    commission = max(raw, min_comm)
    stamp = n * float(getattr(c, "stamp_duty_rate", 0.0)) if str(side).upper() == SELL else 0.0
    transfer = n * float(getattr(c, "transfer_fee_rate", 0.0))
    return {
        "commission": commission,
        "min_commission_applied": bool(applied),
        "stamp": stamp,
        "transfer": transfer,
        "total": commission + stamp + transfer,
    }


def same_bar_order(conservative: str = SAME_BAR_CONSERVATIVE) -> str:
    """同一根 K 线同时触及止损与止盈的保守约定，默认 stop_first。"""
    if conservative != SAME_BAR_CONSERVATIVE:
        raise ValueError(f"only conservative='{SAME_BAR_CONSERVATIVE}' is supported")
    return SAME_BAR_CONSERVATIVE


def order_capacity(notional: float, day_amount: float,
                   participation_cap: float) -> float:
    """容量约束后的可成交名义额（超出部分未成交）。"""
    cap = max(0.0, float(day_amount)) * float(participation_cap)
    return min(max(0.0, float(notional)), cap)
