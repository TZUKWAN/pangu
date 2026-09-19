"""同花顺 (THS) client adapter via easytrader.

HONEST FAILURE RULES (P9-004):
- easytrader is a GUI-automation library: a successful click is NOT broker
  confirmation.  submit_order() therefore verifies via today_entrusts that the
  entrust really exists before returning a broker id; otherwise it raises
  BrokerUnavailable and the OMS must treat the order as UNKNOWN.
- easytrader is NOT a pinned dependency; version drift may silently break the
  column mapping below.  Any unexpected failure is mapped to BrokerUnavailable
  (conservative) and messages containing 验证码/密码/提示 map to CaptchaRequired
  so the live loop pauses for a human.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from engine.contracts import (
    BrokerBalance,
    BrokerOrderRecord,
    BrokerPosition,
    BrokerTradeRecord,
    CaptchaRequired,
    BrokerUnavailable,
    Order,
    OrderSide,
    OrderType,
)
from engine.execution.broker import BrokerAdapter

# Chinese (THS/easytrader) -> internal field names
COLUMN_MAP = {
    "证券代码": "symbol", "证券名称": "name",
    "可用余额": "available", "可用资金": "available_cash", "可用金额": "available_cash",
    "总资产": "total_asset", "资金余额": "cash_balance", "冻结金额": "frozen", "冻结资金": "frozen",
    "证券市值": "market_value", "持仓市值": "market_value",
    "证券数量": "qty", "可用数量": "sellable_qty", "成本价": "cost", "摊薄成本价": "cost",
    "盈亏": "pnl", "市值": "market_value_row",
    "合同编号": "entrust_no", "委托编号": "entrust_no",
    "操作": "direction", "买卖标志": "direction_flag", "委托数量": "entrust_qty",
    "成交数量": "filled_qty", "委托价格": "price", "状态": "status", "委托状态": "status",
    "成交编号": "trade_no", "成交价格": "traded_price", "成交时间": "traded_time", "委托时间": "entrust_time",
}
_CAPTCHA_KEYWORDS = ("验证码", "密码", "提示")
_SIDE_WORDS = {"买入": OrderSide.BUY, "买人": OrderSide.BUY, "卖出": OrderSide.SELL}


def map_row(row: Dict) -> Dict:
    """Translate one Chinese-keyed broker row into internal keys."""
    return {COLUMN_MAP.get(str(k).strip(), str(k).strip()): v for k, v in dict(row).items()}


def map_rows(rows) -> List[Dict]:
    return [map_row(r) for r in (rows or [])]


def side_of(direction: str) -> Optional[OrderSide]:
    for word, side in _SIDE_WORDS.items():
        if word in str(direction):
            return side
    return None


def _wrap_broker_call(fn, *args, **kwargs):
    """Run one client call; map failures to CaptchaRequired/BrokerUnavailable."""
    try:
        return fn(*args, **kwargs)
    except CaptchaRequired:
        raise
    except Exception as exc:  # easytrader raises bare Exception subclasses
        msg = str(exc)
        if any(k in msg for k in _CAPTCHA_KEYWORDS):
            raise CaptchaRequired(msg) from exc
        raise BrokerUnavailable(f"ths client call failed: {type(exc).__name__}: {msg}") from exc


class TongHuaShunAdapter(BrokerAdapter):
    name = "ths_easytrader"

    def __init__(self, client_factory: Optional[Callable[[], object]] = None,
                 account: str = "", ack_timeout_s: float = 5.0,
                 poll_interval_s: float = 0.05) -> None:
        super().__init__()
        self._client_factory = client_factory
        self._account = account
        self._client = None
        self._ack_timeout_s = float(ack_timeout_s)
        self._poll = float(poll_interval_s)

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        try:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                import easytrader  # lazy: optional dependency
                self._client = easytrader.use("ths")
            self._client.connect()
            self._connected = True
            self._last_error = ""
        except Exception as exc:
            self._client = None
            self._connected = False
            msg = str(exc)
            self._last_error = "easytrader not installed" if "No module named" in msg and \
                "easytrader" in msg else f"connect failed: {msg}"

    def health(self) -> Dict:
        base = super().health()
        base["dependency"] = "easytrader"
        return base

    def _require_client(self):
        self._require_connected()
        return self._client

    # -- account ------------------------------------------------------------
    def get_balance(self) -> BrokerBalance:
        client = self._require_client()
        rows = map_rows(_wrap_broker_call(client.balance))
        if not rows:
            raise BrokerUnavailable("ths: empty balance response")
        row = rows[0]
        return BrokerBalance(
            total_asset=float(row.get("total_asset") or row.get("cash_balance") or 0),
            available_cash=float(row.get("available_cash") or row.get("available") or 0),
            frozen_cash=float(row.get("frozen") or 0),
            market_value=float(row.get("market_value") or 0),
            asof=self._now_iso())

    def get_positions(self) -> List[BrokerPosition]:
        client = self._require_client()
        out = []
        for row in map_rows(_wrap_broker_call(client.position)):
            qty = int(float(row.get("qty") or 0))
            if qty <= 0:
                continue
            out.append(BrokerPosition(
                symbol=str(row.get("symbol", "")), qty=qty,
                sellable_qty=int(float(row.get("sellable_qty", row.get("available", 0)) or 0)),
                avg_cost=float(row.get("cost") or 0),
                market_value=float(row.get("market_value_row") or 0),
                asof=self._now_iso()))
        return out

    # -- orders -------------------------------------------------------------
    def submit_order(self, order: Order) -> str:
        client = self._require_client()
        if order.order_type != OrderType.LIMIT or order.limit_price is None:
            raise ValueError("THS adapter requires a LIMIT order with limit_price")
        price = round(float(order.limit_price), 2)
        qty = int(order.qty)
        fn = client.buy if order.side == OrderSide.BUY else client.sell
        resp = _wrap_broker_call(fn, order.symbol, price, qty)
        entrust_no = self._entrust_no_from_resp(resp)
        # click-success != success: verify the entrust is visible at the broker
        deadline = time.monotonic() + self._ack_timeout_s
        while time.monotonic() < deadline:
            for row in map_rows(_wrap_broker_call(client.today_entrusts)):
                if str(row.get("symbol", "")) != str(order.symbol):
                    continue
                if side_of(str(row.get("direction", ""))) != order.side:
                    continue
                if entrust_no:
                    return entrust_no
                if qty and int(float(row.get("entrust_qty", qty) or qty)) == qty:
                    return str(row.get("entrust_no", ""))
            time.sleep(self._poll)
        raise BrokerUnavailable(
            f"ths: submit of {order.symbol} x{qty} not verifiable in today_entrusts within "
            f"{self._ack_timeout_s}s; caller MUST treat order as UNKNOWN and reconcile")

    @staticmethod
    def _entrust_no_from_resp(resp) -> str:
        if resp is None:
            return ""
        if isinstance(resp, dict):
            row = map_row(resp)
            return str(row.get("entrust_no", "") or "")
        text = str(resp).strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits or ""

    def _entrust_rows(self) -> List[Dict]:
        client = self._require_client()
        return map_rows(_wrap_broker_call(client.today_entrusts))

    @staticmethod
    def _map_status(native: str) -> str:
        if "全部成交" in native:
            return "FILLED"
        if "部成" in native:
            return "PARTIALLY_FILLED"
        if "废单" in native:
            return "REJECTED"
        if "撤" in native:
            return "CANCELLED"
        return "SUBMITTED"

    def get_orders(self, trade_date: Optional[str] = None) -> List[BrokerOrderRecord]:
        # easytrader/ths exposes today's entrusts only; trade_date ignored.
        out = []
        for row in self._entrust_rows():
            no = str(row.get("entrust_no", ""))
            out.append(BrokerOrderRecord(
                broker_order_id=no, symbol=str(row.get("symbol", "")),
                side=side_of(str(row.get("direction", ""))) or OrderSide.BUY,
                qty=int(float(row.get("entrust_qty") or 0)),
                filled_qty=int(float(row.get("filled_qty") or 0)),
                price=float(row["price"]) if row.get("price") not in (None, "") else None,
                status=self._map_status(str(row.get("status", ""))),
                submitted_at=str(row.get("entrust_time") or self._now_iso()),
                raw=row))
        return out

    def get_trades(self, trade_date: Optional[str] = None) -> List[BrokerTradeRecord]:
        client = self._require_client()
        out = []
        for row in map_rows(_wrap_broker_call(client.today_trades)):
            out.append(BrokerTradeRecord(
                broker_trade_id=str(row.get("trade_no", "")),
                broker_order_id=str(row.get("entrust_no", "")),
                symbol=str(row.get("symbol", "")),
                side=side_of(str(row.get("direction", ""))) or OrderSide.BUY,
                qty=int(float(row.get("filled_qty") or row.get("qty") or 0)),
                price=float(row.get("traded_price") or 0),
                traded_at=str(row.get("traded_time") or self._now_iso()), raw=row))
        return out

    def cancel_order(self, broker_order_id: str) -> bool:
        client = self._require_client()
        return bool(_wrap_broker_call(client.cancel_entrust, broker_order_id))

    def query_order(self, broker_order_id: str) -> Optional[BrokerOrderRecord]:
        for rec in self.get_orders():
            if rec.broker_order_id == str(broker_order_id):
                return rec
        return None

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime
        return datetime.now().isoformat(timespec="seconds")
