"""Clock 抽象（Phase 2 / Task 2.1）。

核心代码禁止直接散落调用 datetime.now()；统一经 Clock 取"当前时刻"。
生产用 SystemClock；测试用 FrozenClock（可跨日推进、可注入任意时区时刻）。
统一时区：Asia/Shanghai。
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")


class Clock(Protocol):
    def now(self) -> datetime:
        """返回带时区（Asia/Shanghai）的当前时刻。"""
        ...


class SystemClock:
    """生产时钟：真实当前时间，固定上海时区。"""

    def now(self) -> datetime:
        return datetime.now(SHANGHAI)


class FrozenClock:
    """测试时钟：冻结时刻，可手动推进。"""

    def __init__(self, start: str | datetime):
        self._now = _to_shanghai(start)

    def now(self) -> datetime:
        return self._now

    def advance(self, **kwargs) -> None:
        """推进时刻，例如 advance(minutes=31)。"""
        self._now = self._now + __import__("datetime").timedelta(**kwargs)

    def set(self, value: str | datetime) -> None:
        self._now = _to_shanghai(value)


def _to_shanghai(value: str | datetime) -> datetime:
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SHANGHAI)
    return dt.astimezone(SHANGHAI)


_default_clock: Clock = SystemClock()


def now_shanghai() -> datetime:
    """模块级便捷入口（可被测试整体替换，见 set_default_clock）。"""
    return _default_clock.now()


def set_default_clock(clock: Clock) -> None:
    global _default_clock
    _default_clock = clock


def reset_default_clock() -> None:
    global _default_clock
    _default_clock = SystemClock()
