"""BrokerAdapter ABC: the single seam between OMS and any broker."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from engine.contracts import (
    BrokerBalance,
    BrokerOrderRecord,
    BrokerPosition,
    BrokerTradeRecord,
    BrokerUnavailable,
    Order,
)


class BrokerAdapter(ABC):
    """All concrete adapters MUST raise BrokerUnavailable when not connected.

    Honest-failure rule: an adapter that cannot reach its broker never fakes
    success; the OMS treats any BrokerError as UNKNOWN + reconciliation.
    """

    name = "adapter"

    def __init__(self) -> None:
        self._connected: bool = False
        self._last_error: str = ""

    # -- lifecycle ----------------------------------------------------------
    @abstractmethod
    def connect(self) -> None:
        """Establish the connection. On failure set _connected=False and
        record the reason (health() must report it honestly)."""

    def disconnect(self) -> None:
        self._connected = False

    def health(self) -> Dict:
        return {
            "ok": bool(self._connected),
            "adapter": self.name,
            "connected": bool(self._connected),
            "reason": self._last_error,
        }

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerUnavailable(
                f"{self.name} not connected: {self._last_error or 'call connect() first'}"
            )

    # -- account ------------------------------------------------------------
    @abstractmethod
    def get_balance(self) -> BrokerBalance: ...

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]: ...

    # -- orders -------------------------------------------------------------
    @abstractmethod
    def get_orders(self, trade_date: Optional[str] = None) -> List[BrokerOrderRecord]: ...

    @abstractmethod
    def get_trades(self, trade_date: Optional[str] = None) -> List[BrokerTradeRecord]: ...

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Return broker_order_id. Raise BrokerUnavailable/CaptchaRequired on
        any uncertainty — caller must treat the order as UNKNOWN."""

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool: ...

    @abstractmethod
    def query_order(self, broker_order_id: str) -> Optional[BrokerOrderRecord]: ...
