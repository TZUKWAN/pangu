"""Pangu 2.0 无未来函数回测引擎 v2（P4 阶段）。

因果语义（与 short_term_replay 一致并推广到组合层）：
- 决策日 t 收盘后策略给出目标权重列表；t+1 开盘执行；
- 策略只拿到 :class:`HistoryView`，任何访问 > 决策日数据的尝试抛
  :class:`LookaheadError`；
- 引擎内置 risk-lite：单票权重上限、最大持仓数（超出记
  ``dropped_max_positions``，绝不静默丢弃）、现金可行性（不足按比例缩减）；
- 执行层走 :mod:`engine.validation.exec_model`：涨停不可买/跌停不可卖/
  停牌不可成交/容量上限——未成交一律记入 events（unfilled_*）；
- T+1：卖出只卖 ``buy_date < t+1`` 的批次，其余记 ``unsold_t1``；
- 期末对剩余持仓按末日收盘价（含滑点）强平，记 ``force_close``。

引擎保持"笨"：止损/持有期退出是策略的责任（策略发出 SELL 目标）。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence

import pandas as pd

from .data_interface import LookaheadError, ResearchData  # noqa: F401 (re-export)
from .exec_model import fees, fill_buy, fill_sell, limit_prices, order_capacity


# --------------------------------------------------------------------------- #
# 配置与协议
# --------------------------------------------------------------------------- #

@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_bps: float = 10.0
    lot_size: int = 100
    max_positions: int = 20
    max_weight_per_stock: float = 0.10
    allow_buy_at_limit_up: bool = False
    allow_sell_at_limit_down: bool = False
    participation_cap: float = 0.02


class TargetOrder(dict):
    """目标单：{symbol, side: "BUY"/"SELL", weight=决策时权益占比 | value=名义额}。

    SELL 且 weight==0（或 weight/value 均缺省）表示清仓该票（受 T+1 约束）。
    """


class Strategy(Protocol):
    def rebalance(self, decision_date: str, history: "HistoryView") -> list[TargetOrder]: ...


# --------------------------------------------------------------------------- #
# 日期归一 + 历史视图（策略唯一的数据入口）
# --------------------------------------------------------------------------- #

def _iso(d: Any) -> str:
    s = str(d)
    if len(s) >= 10 and s[4] == "-":
        return s[:10]
    c = s.replace("-", "").replace("/", "").replace(" ", "")[:8]
    return f"{c[:4]}-{c[4:6]}-{c[6:8]}" if len(c) == 8 else s


class HistoryView:
    """包一层面板 + 交易日历；任何触及 > asof 日期的访问抛 LookaheadError。"""

    def __init__(self, data: ResearchData, panel: pd.DataFrame,
                 trading_days: Sequence[str], max_date: str):
        self._data = data
        self._panel = panel
        self._days = [str(d) for d in trading_days]
        self._max = _iso(max_date)
        self._dates = panel.index.get_level_values("date")

    @property
    def asof(self) -> str:
        return self._max

    def _guard(self, date: Any) -> None:
        d = _iso(date)
        if d > self._max:
            raise LookaheadError(
                f"attempt to access {d} beyond decision date {self._max}")

    def asof_view(self, date: str) -> "HistoryView":
        """返回进一步受限（date 不得晚于当前 asof）的视图。"""
        d = _iso(date)
        self._guard(d)
        return HistoryView(self._data, self._panel, self._days, d)

    def panel_asof(self, date: str) -> pd.DataFrame:
        """date <= asof 的全部面板行（严格 <= date）。"""
        self._guard(date)
        d = _iso(date)
        return self._panel[self._dates <= d].copy()

    def daily_frame(self, date: str) -> pd.DataFrame:
        """date 当日的全部面板行（date > asof 抛错；无数据返回空表）。"""
        self._guard(date)
        d = _iso(date)
        return self._panel[self._dates == d].copy()

    def trading_days(self, start: str, end: str) -> list[str]:
        self._guard(end)
        s, e = _iso(start), _iso(end)
        return [d for d in self._days if s <= d <= e]

    def universe(self, date: str) -> pd.DataFrame:
        self._guard(date)
        return self._data.universe(_iso(date))

    def index_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        self._guard(end)
        return self._data.index_daily(code, _iso(start), _iso(end))


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #

def _r(x: Any) -> Any:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return round(float(x), 10)
    return x


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    initial_capital: float = 0.0

    def to_dict(self) -> dict:
        """完整可序列化（确定性：同输入同字节）。"""
        return {
            "initial_capital": _r(self.initial_capital),
            "summary": {k: _r(v) for k, v in self.summary.items()},
            "trades": [{k: _r(v) for k, v in tr.items()} for tr in self.trades],
            "events": self.events,
            "equity_curve": [
                {k: _r(v) for k, v in row.items()}
                for row in self.equity_curve.to_dict("records")
            ],
        }


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #

class BacktestV2:
    """权重型日频策略的无未来函数回测。"""

    def __init__(self, data: ResearchData, cfg: BacktestConfig | None = None):
        self.data = data
        self.cfg = cfg or BacktestConfig()

    def run(self, strategy: Strategy, start: str, end: str) -> BacktestResult:
        cfg = self.cfg
        days = [str(d) for d in self.data.trading_days(start, end)]
        if not days:
            days = [str(d) for d in self.data.trading_days(
                min(_iso(start), _iso(end)), max(_iso(start), _iso(end)))]
        panel = self.data.daily_panel(days[0], days[-1]) if days else \
            self.data.daily_panel(start, end)
        by_date: dict[str, tuple[pd.DataFrame, dict[str, int]]] = {}
        for d, sub in panel.groupby(level="date"):
            codes = {c: i for i, c in enumerate(sub.index.get_level_values("code"))}
            by_date[str(d)] = (sub, codes)
        day_index = {d: i for i, d in enumerate(days)}

        cash = float(cfg.initial_capital)
        lots: dict[str, list[dict]] = {}   # symbol -> [{buy_date, qty, price, fees_buy}]
        last_price: dict[str, float] = {}
        trades: list[dict] = []
        events: list[dict] = []
        rows: list[dict] = []
        pending: list[dict] = []           # 上一决策日的目标（执行于今日开盘）
        pending_dec = ""
        pending_equity = float(cfg.initial_capital)
        turnover_notional = 0.0
        n_filled = 0
        cash_box = [cash, 0.0]             # [0]=现金, [1]=当日成交名义额

        def _close(t: str, sym: str) -> float | None:
            info = by_date.get(t)
            if not info:
                return None
            sub, codes = info
            i = codes.get(sym)
            if i is None:
                return None
            v = float(sub.iloc[i]["close"])
            return v if v == v else None

        for i, t in enumerate(days):
            # -- 1) 执行上一决策日的目标（今日开盘） --------------------------
            if pending:
                n_filled += self._execute(
                    t, pending, pending_dec, pending_equity, by_date, day_index,
                    lots, last_price, trades, events, cash_box)
                cash = cash_box[0]
                turnover_notional += cash_box[1]
                cash_box[1] = 0.0
            # -- 2) 末日：按收盘价强平 ---------------------------------------
            if i == len(days) - 1:
                for sym in sorted(lots):
                    open_qty = sum(l["qty"] for l in lots[sym])
                    if open_qty <= 0:
                        continue
                    px = _close(t, sym)          # 末日收盘优先
                    if px is None:
                        px = last_price.get(sym)  # 停牌持仓回退最近已知收盘
                    if px is None or px <= 0:
                        continue
                    price = px * (1.0 - cfg.slippage_bps / 10_000.0)
                    notional = open_qty * price
                    f = fees(notional, "SELL", cfg)
                    cash += notional - f["total"]
                    turnover_notional += notional
                    self._close_lots(sym, lots, open_qty, price, f["total"], t,
                                     day_index, "force_close", trades)
                for sym in lots:
                    lots[sym] = [l for l in lots[sym] if l["qty"] > 0]
            # -- 3) 当日收盘 mark-to-market ----------------------------------
            mv = 0.0
            n_pos = 0
            for sym in sorted(lots):
                q = sum(l["qty"] for l in lots[sym])
                if q <= 0:
                    continue
                c = _close(t, sym)
                if c is not None:
                    last_price[sym] = c
                mv += q * last_price.get(sym, 0.0)
                n_pos += 1
            equity = cash + mv
            rows.append({"date": t, "cash": cash, "market_value": mv,
                         "equity": equity, "n_positions": n_pos})
            # -- 4) 当日收盘决策（若还有下一交易日） --------------------------
            if i < len(days) - 1:
                view = HistoryView(self.data, panel, days, t)
                targets = strategy.rebalance(t, view) or []
                pending, pending_dec, pending_equity = self._enforce(
                    targets, t, equity, lots, events)
        equity_curve = pd.DataFrame(rows)
        summary = self._summary(equity_curve, trades, events, days,
                                turnover_notional, n_filled)
        summary["initial_capital"] = float(cfg.initial_capital)
        summary["config"] = {k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                             for k, v in asdict(cfg).items()}
        return BacktestResult(equity_curve=equity_curve, trades=trades,
                              events=events, summary=summary,
                              initial_capital=float(cfg.initial_capital))

    # ------------------------------------------------------------------ #
    def _enforce(self, targets, t, equity, lots, events) -> tuple[list[dict], str, float]:
        """risk-lite：单票权重/名义额上限、最大持仓数（超出记事件丢弃）。"""
        cfg = self.cfg
        cap_value = cfg.max_weight_per_stock * equity
        norm: list[dict] = []
        for tg in targets or []:
            sym = str(tg.get("symbol", ""))
            side = str(tg.get("side", "BUY")).upper()
            if not sym:
                continue
            if side == "SELL":
                item = {"symbol": sym, "side": "SELL"}
                if tg.get("value") is not None:
                    item["value"] = float(tg["value"])
                else:
                    w = float(tg.get("weight", 0.0) or 0.0)
                    item["weight"] = w
                    item["value"] = w * equity
            else:
                value = float(tg["value"]) if tg.get("value") is not None \
                    else float(tg.get("weight", 0.0) or 0.0) * equity
                item = {"symbol": sym, "side": "BUY",
                        "weight": value / equity if equity > 0 else 0.0,
                        "value": min(value, cap_value)}
            norm.append(item)
        buys = [x for x in norm if x["side"] == "BUY"]
        sells = [x for x in norm if x["side"] == "SELL"]
        held_syms = {s for s, ls in lots.items() if sum(l["qty"] for l in ls) > 0}
        new_syms = [b for b in buys if b["symbol"] not in held_syms]
        room = max(0, cfg.max_positions - len(held_syms))
        if len(new_syms) > room:
            drop = {b["symbol"] for b in new_syms[room:]}
            buys = [b for b in buys if b["symbol"] not in drop]
            events.append({"date": t, "type": "dropped_max_positions",
                           "detail": {"symbols": sorted(drop),
                                      "max_positions": cfg.max_positions}})
        return sells + buys, t, equity   # 执行日卖先买后

    # ------------------------------------------------------------------ #
    def _execute(self, t, targets, dec_date, equity_dec, by_date, day_index,
                 lots, last_price, trades, events, cash_box) -> int:
        """在 t 日开盘执行目标；返回成交笔数。现金经 cash_box[0] 更新。"""
        cfg = self.cfg
        lot = int(cfg.lot_size)
        filled = 0
        day_notional = 0.0

        def unfilled(kind, sym, detail):
            events.append({"date": t, "decision_date": dec_date,
                           "type": kind, "symbol": sym, "detail": detail})

        cash = float(cash_box[0])

        # -- 卖出（先释放现金；同批还有买入的票延后到买入之后再试，
        #     使 T+1 规则真正生效：当日买入的批次不可当日卖出） -------------
        buy_syms = {x["symbol"] for x in targets if x["side"] == "BUY"}
        sells_pure = [x for x in targets
                      if x["side"] == "SELL" and x["symbol"] not in buy_syms]
        sells_deferred = [x for x in targets
                          if x["side"] == "SELL" and x["symbol"] in buy_syms]

        def do_sell(tg) -> None:
            nonlocal cash, filled, day_notional
            sym = tg["symbol"]
            info = by_date.get(t)
            pos = info[1].get(sym) if info else None
            if pos is None:
                unfilled("unfilled_suspended", sym, {"reason": "no bar"})
                return
            row = info[0].iloc[pos]
            _, dn = limit_prices(float(row["preclose"]), bool(row["is_st"]), sym)
            f = fill_sell(float(row["open"]), float(row["high"]), float(row["low"]),
                          dn, cfg.allow_sell_at_limit_down, cfg.slippage_bps)
            if f is None:
                unfilled("unfilled_limit_down", sym,
                         {"open": float(row["open"]), "limit_down": dn})
                return
            held = [l for l in lots.get(sym, []) if l["qty"] > 0]
            held_q = sum(l["qty"] for l in held)
            if held_q <= 0:
                unfilled("unfilled_no_position", sym, {})
                return
            sellable = [l for l in held if l["buy_date"] < t]   # T+1
            sellable_q = sum(l["qty"] for l in sellable)
            w = tg.get("weight")
            if w is not None and float(w) == 0.0:
                want_q = held_q                                   # 清仓
            else:
                value = tg.get("value")
                if value is None:
                    value = float(w or 0.0) * equity_dec
                want_q = int(value / f.price // lot) * lot
            unsold_t1_q = max(0, min(want_q, held_q) - sellable_q)
            qty = min(want_q, sellable_q)
            if qty <= 0:
                if want_q > 0 and unsold_t1_q > 0:
                    unfilled("unsold_t1", sym, {"qty": int(unsold_t1_q)})
                elif want_q > 0:
                    unfilled("unfilled_rounding", sym, {"qty": int(want_q)})
                return
            notional = qty * f.price
            fee = fees(notional, "SELL", cfg)
            cash += notional - fee["total"]
            day_notional += notional
            self._close_lots(sym, lots, qty, f.price, fee["total"], t,
                             day_index, "sell_target", trades)
            filled += 1
            if unsold_t1_q > 0:
                unfilled("unsold_t1", sym, {"qty": int(unsold_t1_q)})

        for tg in sells_pure:
            do_sell(tg)

        # -- 买入 ------------------------------------------------------------
        buys = [x for x in targets if x["side"] == "BUY"]
        total = sum(x["value"] for x in buys)
        scale = 1.0
        if total > cash and total > 0:
            scale = cash / total
            events.append({"date": t, "decision_date": dec_date,
                           "type": "cash_scaled",
                           "detail": {"factor": round(scale, 10)}})
        for tg in buys:
            sym = tg["symbol"]
            info = by_date.get(t)
            pos = info[1].get(sym) if info else None
            if pos is None:
                unfilled("unfilled_suspended", sym, {"reason": "no bar"})
                continue
            row = info[0].iloc[pos]
            up, _ = limit_prices(float(row["preclose"]), bool(row["is_st"]), sym)
            f = fill_buy(float(row["open"]), float(row["high"]), float(row["low"]),
                         up, cfg.allow_buy_at_limit_up, cfg.slippage_bps)
            if f is None:
                unfilled("unfilled_limit_up", sym,
                         {"open": float(row["open"]), "limit_up": up})
                continue
            value = tg["value"] * scale
            qty = int(value / f.price // lot) * lot
            if qty <= 0:
                unfilled("unfilled_rounding", sym, {"value": float(value)})
                continue
            cap = order_capacity(qty * f.price, float(row["amount"]), cfg.participation_cap)
            if cap < qty * f.price - 1e-9:
                q2 = int(cap / f.price // lot) * lot
                unfilled("unfilled_capacity", sym,
                         {"notional": float(qty * f.price - q2 * f.price)})
                if q2 <= 0:
                    continue
                qty = q2
            while qty >= lot:   # 现金可行性（含费用）
                notional = qty * f.price
                fee = fees(notional, "BUY", cfg)
                if notional + fee["total"] <= cash + 1e-9:
                    break
                qty -= lot
            if qty <= 0:
                unfilled("unfilled_rounding", sym, {"reason": "insufficient cash"})
                continue
            notional = qty * f.price
            fee = fees(notional, "BUY", cfg)
            cash -= notional + fee["total"]
            day_notional += notional
            lots.setdefault(sym, []).append(
                {"buy_date": t, "qty": int(qty), "price": f.price,
                 "fees_buy": fee["total"]})
            last_price.setdefault(sym, f.price)
            filled += 1
        for tg in sells_deferred:      # 同日同票：买入成交后再尝试卖出（T+1）
            do_sell(tg)
        cash_box[0] = cash
        cash_box[1] = day_notional
        return filled

    # ------------------------------------------------------------------ #
    def _close_lots(self, sym, lots, qty, exit_price, fees_sell_total, exit_date,
                    day_index, reason, trades):
        """FIFO 平仓 qty 股（卖出/强平），生成 lot 级 trade 记录。"""
        remaining = int(qty)
        for l in lots.get(sym, []):
            if remaining <= 0:
                break
            if l["qty"] <= 0:
                continue
            take = min(l["qty"], remaining)
            base = max(l["qty"], 1)
            fees_buy_i = l["fees_buy"] * take / base
            lot_cost = take * l["price"] + fees_buy_i
            fees_i = fees_sell_total * take / max(1, int(qty))
            pnl = take * exit_price - fees_i - lot_cost
            ret = pnl / lot_cost if lot_cost > 0 else 0.0
            hd = (day_index.get(exit_date, 0) - day_index.get(l["buy_date"], 0)) \
                if day_index else 0
            trades.append({
                "symbol": sym, "side": "LONG",
                "entry_date": l["buy_date"], "entry_price": l["price"],
                "exit_date": exit_date, "exit_price": exit_price,
                "qty": int(take),
                "fees_buy": fees_buy_i, "fees_sell": fees_i,
                "pnl": pnl, "ret": ret,
                "holding_days": max(0, hd),
                "close_reason": reason,
            })
            l["qty"] -= take
            remaining -= take

    # ------------------------------------------------------------------ #
    def _summary(self, equity_curve, trades, events, days, turnover_notional,
                 n_filled) -> dict:
        cfg = self.cfg
        eq = equity_curve["equity"].astype(float)
        init = float(cfg.initial_capital)
        n = len(days)
        total_return = float(eq.iloc[-1] / init - 1.0) if len(eq) else 0.0
        ann_return = (1.0 + total_return) ** (252.0 / n) - 1.0 if n > 0 else 0.0
        if len(eq) > 1:
            rets = (eq / eq.shift(1) - 1.0).dropna()
        else:
            rets = pd.Series(dtype=float)
        std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
        sharpe = float(rets.mean() / std * math.sqrt(252)) if std > 0 else 0.0
        downside = rets[rets < 0]
        dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = float(rets.mean() / dstd * math.sqrt(252)) if dstd > 0 else 0.0
        peak = eq.cummax()
        dd = (peak - eq) / peak
        mdd = float(dd.max()) if len(dd) else 0.0
        mdd_end = int(dd.values.argmax()) if len(dd) else 0
        mdd_start = int(eq.values[:mdd_end + 1].argmax()) if len(dd) else 0
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] < 0]
        pf = (sum(wins) / abs(sum(losses))) if losses else (math.inf if wins else 0.0)
        n_trades = len(trades)
        npos = equity_curve["n_positions"].values if len(equity_curve) else []
        n_pos_days = int((npos > 0).sum()) if len(equity_curve) else 0
        win_days = 0
        if len(rets) and len(equity_curve) > 1:
            win_days = int(((rets > 0).values & (npos[1:] > 0)).sum())
        n_unfilled = sum(1 for e in events
                         if str(e.get("type", "")).startswith("unfilled_"))
        mean_eq = float(eq.mean()) if len(eq) else init
        years = n / 252.0 if n else 0.0
        es = float(rets[rets <= rets.quantile(0.01)].mean()) if len(rets) else 0.0
        dates = equity_curve["date"] if len(equity_curve) else pd.Series([str])
        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": mdd,
            "mdd_start": str(dates.iloc[mdd_start]) if len(equity_curve) else "",
            "mdd_end": str(dates.iloc[mdd_end]) if len(equity_curve) else "",
            "profit_factor": round(pf, 10) if math.isfinite(pf) else None,
            "win_rate_trades": (len(wins) / n_trades) if n_trades else 0.0,
            "win_rate_days": (win_days / n_pos_days) if n_pos_days else 0.0,
            "positive_day_ratio": float((rets > 0).mean()) if len(rets) else 0.0,
            "negative_day_ratio": float((rets < 0).mean()) if len(rets) else 0.0,
            "no_position_day_ratio":
                (float((equity_curve["n_positions"] == 0).sum()) / max(1, len(equity_curve)))
                if len(equity_curve) else 0.0,
            "mean_pos_day": float(rets[rets > 0].mean()) if (rets > 0).any() else 0.0,
            "mean_neg_day": float(rets[rets < 0].mean()) if (rets < 0).any() else 0.0,
            "worst_day": float(rets.min()) if len(rets) else 0.0,
            "expected_shortfall_1pct": es,
            "turnover": (turnover_notional / mean_eq / years)
                        if years > 0 and mean_eq > 0 else 0.0,
            "execution_rate": (n_filled / (n_filled + n_unfilled))
                              if (n_filled + n_unfilled) else 0.0,
            "n_trades": n_trades,
            "n_trade_dates": len({t["exit_date"] for t in trades}
                                 | {t["entry_date"] for t in trades}),
            "n_unfilled_events": n_unfilled,
        }
