"""Pangu 2.0 走前验证切分 + 留出集（holdout）一次解锁策略。

- :class:`WalkForwardSplitter`：expanding / sliding 两种模式，段间 embargo
  （train 标签向前覆盖的天数）——valid 起点 >= train_end + embargo + 1，
  test 起点 >= valid_end + embargo + 1，段间不重叠。
- :class:`HoldoutPolicy`：holdout 数据默认不可见；:meth:`unlock` 只允许一次，
  并在 data/experiments/holdout_audit.jsonl 落一行审计日志。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .data_interface import LookaheadError

HOLDOUT_AUDIT_PATH = "data/experiments/holdout_audit.jsonl"


@dataclass(frozen=True)
class Split:
    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]


class WalkForwardSplitter:
    """按交易日索引切分：train / valid / test 三段 + embargo 间隔。"""

    def __init__(self, trading_days: Sequence[str], train_days: int = 120,
                 valid_days: int = 20, test_days: int = 20, step_days: int = 20,
                 embargo_days: int = 1, mode: str = "expanding"):
        if mode not in ("expanding", "sliding"):
            raise ValueError(f"mode must be expanding|sliding, got {mode}")
        if min(train_days, valid_days, test_days, step_days) < 1 or embargo_days < 0:
            raise ValueError("window sizes must be >= 1 (embargo >= 0)")
        self.trading_days = [str(d) for d in trading_days]
        self.train_days = int(train_days)
        self.valid_days = int(valid_days)
        self.test_days = int(test_days)
        self.step_days = int(step_days)
        self.embargo_days = int(embargo_days)
        self.mode = mode

    def split(self) -> list[Split]:
        days = self.trading_days
        n = len(days)
        embargo = self.embargo_days
        out: list[Split] = []
        i0 = 0
        while True:
            tr_end = i0 + self.train_days          # exclusive
            if tr_end > n:
                break
            train = days[0:tr_end] if self.mode == "expanding" else days[i0:tr_end]
            vs = tr_end + embargo
            ve = vs + self.valid_days
            if ve > n:
                break
            ts = ve + embargo
            te = ts + self.test_days
            if te > n:
                break
            out.append(Split((train[0], train[-1]),
                             (days[vs], days[ve - 1]),
                             (days[ts], days[te - 1])))
            i0 += self.step_days
        return out


class HoldoutViolation(LookaheadError):
    """任何触及 >= holdout_start 数据的查询。"""


class _HoldoutView:
    """数据包装：任何 end/date >= holdout_start 的查询直接抛 HoldoutViolation。"""

    def __init__(self, data: Any, holdout_start: str):
        self._data = data
        self._cut = str(holdout_start)

    def _guard(self, date: Any) -> None:
        if str(date) >= self._cut:
            raise HoldoutViolation(
                f"holdout violation: {date} >= holdout_start {self._cut}")

    def daily_panel(self, start: str, end: str, symbols=None):
        self._guard(end)
        return self._data.daily_panel(start, end, symbols)

    def universe(self, date: str):
        self._guard(date)
        return self._data.universe(date)

    def index_daily(self, code: str, start: str, end: str):
        self._guard(end)
        return self._data.index_daily(code, start, end)

    def trading_days(self, start: str, end: str):
        self._guard(end)
        return self._data.trading_days(start, end)


class HoldoutPolicy:
    """留出集策略：view() 给受限数据；unlock() 有且仅有一次并留审计。"""

    def __init__(self, holdout_start: str, audit_path: str = HOLDOUT_AUDIT_PATH):
        self.holdout_start = str(holdout_start)
        self.audit_path = Path(audit_path)
        self._data: Any = None
        self._unlocked = False

    @property
    def unlocked(self) -> bool:
        return self._unlocked

    def view(self, data: Any) -> _HoldoutView:
        self._data = data
        return _HoldoutView(data, self.holdout_start)

    def unlock(self, operator: str, reason: str) -> Any:
        """第二次调用抛 RuntimeError；每次解锁落一行审计日志。"""
        if self._unlocked:
            raise RuntimeError(
                "holdout already unlocked: one unlock per policy instance")
        if self._data is None:
            raise RuntimeError("nothing to unlock: view(data) was never called")
        self._unlocked = True
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "operator": str(operator),
            "reason": str(reason),
            "holdout_start": self.holdout_start,
        }
        with open(self.audit_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        return self._data
