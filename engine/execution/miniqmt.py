"""迅投 miniQMT (xtquant) adapter.

HONEST FAILURE RULES:
- xtquant is an optional local dependency; it only works when a locally logged-in
  QMT terminal (miniQMT) is running.  Without it the adapter reports
  health(ok=False) honestly and every call raises BrokerUnavailable.
- client_factory injection lets tests drive the full mapping with fakes shaped
  like XtQuantTrader (connect/subscribe/order_stock/cancel_order_stock/
  query_stock_orders/query_stock_trades/query_stock_asset/query_stock_positions).
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional

from engine.contracts import (
    BrokerBalance,
    BrokerOrderRecord,
    BrokerPosition,
    BrokerTradeRecord,
    BrokerUnavailable,
    CaptchaRequired,
    Order,
    OrderSide,
    OrderType,
)
from engine.execution.broker import BrokerAdapter

# xtquant constants (kept literal so xtquant is not needed at import time)
STOCK_BUY = 23
STOCK_SELL = 24
FIX_PRICE = 11  # absolute limit price
STATUS_MAP: Dict[int, str] = {
    48: "SUBMITTED",        # ORDER_UN_REPORTED
    49: "SUBMITTED",        # ORDER_REPORTED
    50: "PARTIALLY_FILLED", # ORDER_PART_SUCC
    51: "FILLED",           # ORDER_SUCCEEDED
    52: "CANCELLED",        # ORDER_CANCELED
    53: "CANCEL_PENDING",   # ORDER_CANCELING
    55: "UNKNOWN",          # ORDER_UNKNOWN
    56: "REJECTED",         # ORDER_JUNK (废单)
    57: "CANCELLED",        # ORDER_CANCEL_SUCC
    59: "UNKNOWN",
}
_CAPTCHA_KEYWORDS = ("验证码", "密码", "提示", "未登录")


def _wrap(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except CaptchaRequired:
        raise
    except Exception as exc:
        msg = str(exc)
        if any(k in msg for k in _CAPTCHA_KEYWORDS):
            raise CaptchaRequired(msg) from exc
        raise BrokerUnavailable(f"miniqmt client call failed: {type(exc).__name__}: {msg}") from exc


def _attr(obj, name, default=None):
    return getattr(obj, name, default)


class MiniQMTAdapter(BrokerAdapter):
    name = "miniqmt"

    def __init__(self, client_factory: Optional[Callable[[], object]] = None,
                 account_id: str = "", qmt_path: str = "",
                 session_id: int = 20260919) -> None:
        super().__init__()
        self._client_factory = client_factory
        self._account_id = str(account_id)
        self._qmt_path = qmt_path
        self._session_id = int(session_id)
        self._client = None

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        try:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from xtquant.xttrader import XtQuantTrader  # lazy optional import
                self._client = XtQuantTrader(self._qmt_path, self._session_id)
            if _wrap(self._client.connect) != 0:
                raise BrokerUnavailable("miniqmt: connect() failed — is the local QMT client running?")
            _wrap(self._client.subscribe, self._account_id)
            self._connected = True
            self._last_error = ""
        except BrokerUnavailable as exc:
            self._client = None
            self._connected = False
            self._last_error = str(exc)
        except Exception as exc:
            self._client = None
            self._connected = False
            msg = str(exc)
            self._last_error = "xtquant not installed" if "No module named" in msg and "xtquant" in msg \
                else f"connect failed: {msg}"

    def health(self) -> Dict:
        base = super().health()
        base["dependency"] = "xtquant (requires local QMT terminal)"
        return base

    def _require_client(self):
        self._require_connected()
        return self._client

    # -- account ------------------------------------------------------------
    def get_balance(self) -> BrokerBalance:
        client = self._require_client()
        asset = _wrap(client.query_stock_asset, self._account_id)
        if asset is None:
            raise BrokerUnavailable("miniqmt: no asset response (account not ready?)")
        return BrokerBalance(
            total_asset=float(_attr(asset, "total_asset", 0) or 0),
            available_cash=float(_attr(asset, "cash", 0) or 0),
            frozen_cash=float(_attr(asset, "frozen_cash", 0) or 0),
            market_value=float(_attr(asset, "market_value", 0) or 0),
            asof=self._now_iso())

    def get_positions(self) -> List[BrokerPosition]:
        client = self._require_client()
        out = []
        for pos in _wrap(client.query_stock_positions, self._account_id) or []:
            qty = int(_attr(pos, "volume", 0) or 0)
            if qty <= 0:
                continue
            out.append(BrokerPosition(
                symbol=str(_attr(pos, "stock_code", "")), qty=qty,
                sellable_qty=int(_attr(pos, "can_use_volume", 0) or 0),
                avg_cost=float(_attr(pos, "open_price", 0) or 0),
                market_value=float(_attr(pos, "market_value", 0) or 0),
                asof=self._now_iso()))
        return out

    # -- orders -------------------------------------------------------------
    def submit_order(self, order: Order) -> str:
        client = self._require_client()
        if order.order_type != OrderType.LIMIT or order.limit_price is None:
            raise ValueError("MiniQMT adapter requires a LIMIT order with limit_price")
        order_type = STOCK_BUY if order.side == OrderSide.BUY else STOCK_SELL
        broker_id = _wrap(client.order_stock, self._account_id, order.symbol, order_type,
                          int(order.qty), FIX_PRICE, round(float(order.limit_price), 2),
                          order.strategy_id, order.client_order_id)
        if broker_id is None or int(broker_id) < 0:
            raise BrokerUnavailable(
                f"miniqmt: order_stock returned {broker_id}; caller MUST treat order as UNKNOWN")
        return str(int(broker_id))

    def _query_orders(self) -> List[BrokerOrderRecord]:
        client = self._require_client()
        out = []
        for o in _wrap(client.query_stock_orders, self._account_id) or []:
            raw_type = int(_attr(o, "order_type", 0) or 0)
            out.append(BrokerOrderRecord(
                broker_order_id=str(_attr(o, "order_id", "")),
                symbol=str(_attr(o, "stock_code", "")),
                side=OrderSide.BUY if raw_type == STOCK_BUY else OrderSide.SELL,
                qty=int(_attr(o, "order_volume", 0) or 0),
                filled_qty=int(_attr(o, "traded_volume", 0) or 0),
                price=float(_attr(o, "price", 0) or 0) or None,
                status=STATUS_MAP.get(int(_attr(o, "order_status", 0) or 0), "UNKNOWN"),
                submitted_at=str(_attr(o, "order_time", "") or self._now_iso()),
                raw={"strategy_name": _attr(o, "strategy_name", ""),
                     "order_remark": _attr(o, "order_remark", "")}))
        return out

    def get_orders(self, trade_date: Optional[str] = None) -> List[BrokerOrderRecord]:
        # xtquant exposes today's orders; trade_date accepted for interface parity.
        return self._query_orders()

    def get_trades(self, trade_date: Optional[str] = None) -> List[BrokerTradeRecord]:
        client = self._require_client()
        out = []
        for t in _wrap(client.query_stock_trades, self._account_id) or []:
            raw_type = int(_attr(t, "order_type", 0) or 0)
            out.append(BrokerTradeRecord(
                broker_trade_id=str(_attr(t, "traded_id", "")),
                broker_order_id=str(_attr(t, "order_id", "")),
                symbol=str(_attr(t, "stock_code", "")),
                side=OrderSide.BUY if raw_type == STOCK_BUY else OrderSide.SELL,
                qty=int(_attr(t, "traded_volume", 0) or 0),
                price=float(_attr(t, "traded_price", 0) or 0),
                traded_at=str(_attr(t, "traded_time", "") or self._now_iso()), raw={}))
        return out

    def cancel_order(self, broker_order_id: str) -> bool:
        client = self._require_client()
        return int(_wrap(client.cancel_order_stock, self._account_id, int(broker_order_id))) == 0

    def query_order(self, broker_order_id: str) -> Optional[BrokerOrderRecord]:
        for rec in self._query_orders():
            if rec.broker_order_id == str(broker_order_id):
                return rec
        return None

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")
