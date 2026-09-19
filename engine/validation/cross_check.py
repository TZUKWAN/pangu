"""Pangu 2.0 独立交叉核对（P4-011）。

对回测产出的成交清单做**独立重定价**：不经过 exec_model 内部，直接从数据
重新读取 open/preclose/is_st/amount/close，按涨跌停规则与费用公式复算每笔
交易的执行价/费用/盈亏，并从现金流重建权益曲线，与引擎输出对比。
consistent = 最大相对差异 < 1e-6 且无违规。
"""
from __future__ import annotations

import re

import pandas as pd


def _limit_ratio(code: str, is_st: bool) -> float:
    if is_st:
        return 0.05
    digits = re.sub(r"\D", "", str(code))
    return 0.20 if digits.startswith(("300", "688")) else 0.10


def _bar(data, date: str, symbol: str) -> pd.Series | None:
    sub = data.daily_panel(date, date, [symbol])
    if sub.empty:
        return None
    codes = list(sub.index.get_level_values("code"))
    if symbol not in codes:
        return None
    return sub.iloc[codes.index(symbol)]


def cross_check(result, data, strategy=None) -> dict:
    """独立复算 recorded trades 与权益曲线。

    strategy 参数保留（接口完备性）：本核对只依赖成交清单与数据。
    返回 {max_abs_diff, max_rel_diff, consistent, n_trades_checked,
    equity_max_abs_diff, violations}。
    """
    cfg = result.summary.get("config", {})
    slip = float(cfg.get("slippage_bps", 10.0)) / 10_000.0
    comm = float(cfg.get("commission_rate", 0.0003))
    min_comm = float(cfg.get("min_commission", 5.0))
    stamp = float(cfg.get("stamp_duty_rate", 0.0005))
    trans = float(cfg.get("transfer_fee_rate", 0.00001))
    allow_buy_up = bool(cfg.get("allow_buy_at_limit_up", False))
    allow_sell_dn = bool(cfg.get("allow_sell_at_limit_down", False))

    max_abs = 0.0
    max_rel = 0.0
    violations: list[str] = []
    flows: dict[str, float] = {}
    open_positions: list[dict] = []     # 每笔 trade 一个持仓段 {sym, qty, entry, exit}

    def flow(date: str, delta: float) -> None:
        flows[str(date)] = flows.get(str(date), 0.0) + delta

    for tr in result.trades:
        sym = tr["symbol"]
        qty = int(tr["qty"])
        entry_date, exit_date = str(tr["entry_date"]), str(tr["exit_date"])
        e_bar = _bar(data, entry_date, sym)
        x_bar = _bar(data, exit_date, sym)
        if e_bar is None or x_bar is None:
            violations.append(f"missing bar for {sym} {entry_date}/{exit_date}")
            continue
        # 独立涨跌停判断（不复用 exec_model）
        e_up = round(float(e_bar["preclose"]) * (1 + _limit_ratio(sym, bool(e_bar["is_st"]))), 2)
        x_dn = round(float(x_bar["preclose"]) * (1 - _limit_ratio(sym, bool(x_bar["is_st"]))), 2)
        e_open = float(e_bar["open"])
        if tr["close_reason"] == "force_close":
            x_raw = float(x_bar["close"])
        else:
            x_raw = float(x_bar["open"])
            if x_raw <= x_dn and not allow_sell_dn:
                violations.append(f"sell of {sym} at/below limit down {exit_date}")
        if e_open >= e_up and not allow_buy_up:
            violations.append(f"buy of {sym} at/above limit up {entry_date}")
        price_e = e_open * (1.0 + slip)
        price_x = x_raw * (1.0 - slip)
        notional_e = qty * price_e
        notional_x = qty * price_x
        fee_buy = max(min_comm, notional_e * comm) + notional_e * trans
        fee_sell = max(min_comm, notional_x * comm) + notional_x * (stamp + trans)
        pnl = notional_x - fee_sell - notional_e - fee_buy
        abs_d = abs(pnl - float(tr["pnl"]))
        rel_d = abs_d / max(abs(float(tr["pnl"])), 1.0)
        max_abs = max(max_abs, abs_d)
        max_rel = max(max_rel, rel_d)
        flow(entry_date, -(notional_e + fee_buy))
        flow(exit_date, notional_x - fee_sell)
        open_positions.append(
            {"sym": sym, "qty": qty, "entry": entry_date, "exit": exit_date})

    # 权益曲线重建：出场发生在出场日开盘 → 该日收盘不再持有
    equity_max_abs = 0.0
    curve = result.equity_curve
    if len(curve):
        cash = float(result.initial_capital)
        last_close: dict[str, float] = {}
        for row in curve.itertuples(index=False):
            d = str(row.date)
            active = [p for p in open_positions
                      if p["entry"] <= d and p["exit"] > d]
            cash += float(flows.get(d, 0.0))
            mv = 0.0
            for p in active:
                bar = _bar(data, d, p["sym"])
                if bar is not None:
                    c = float(bar["close"])
                    if c == c and c > 0:
                        last_close[p["sym"]] = c
                mv += p["qty"] * last_close.get(p["sym"], 0.0)
            abs_d = abs((cash + mv) - float(row.equity))
            equity_max_abs = max(equity_max_abs, abs_d)
    consistent = max_rel < 1e-6 and equity_max_abs < 1e-6 and not violations
    return {
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "equity_max_abs_diff": equity_max_abs,
        "consistent": bool(consistent),
        "n_trades_checked": len(result.trades),
        "violations": violations,
    }
