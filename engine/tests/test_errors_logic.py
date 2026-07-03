"""全局错误处理框架 + 市场时间工具测试（mock，不依赖网络）。"""

from __future__ import annotations

from datetime import datetime

import pytest

from engine.data_loader import is_market_open, last_trading_date
from engine.errors import (
    ConfigError,
    DataUnavailableError,
    MarketClosedError,
    PanguError,
    RateLimitError,
    retry_on_error,
    safe_run,
)


# ---------------------------------------------------------------------- #
# retry_on_error
# ---------------------------------------------------------------------- #
def test_retry_succeeds_first_try():
    @retry_on_error(tries=3, backoff_seconds=0)
    def ok():
        return 42

    assert ok() == 42


def test_retry_succeeds_after_failures():
    call_count = 0

    @retry_on_error(tries=3, backoff_seconds=0)
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("boom")
        return "done"

    assert flaky() == "done"
    assert call_count == 3


def test_retry_exhausts_and_raises():
    call_count = 0

    @retry_on_error(tries=2, backoff_seconds=0)
    def always_fail():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        always_fail()
    assert call_count == 2


def test_retry_only_catches_specified_exceptions():
    call_count = 0

    @retry_on_error(exceptions=(ValueError,), tries=2, backoff_seconds=0)
    def mixed():
        nonlocal call_count
        call_count += 1
        raise KeyError("key")

    with pytest.raises(KeyError):
        mixed()
    assert call_count == 1  # 不重试非指定异常


# ---------------------------------------------------------------------- #
# safe_run
# ---------------------------------------------------------------------- #
def test_safe_run_returns_value():
    def add(a, b):
        return a + b

    assert safe_run(add, 1, 2) == 3


def test_safe_run_returns_default_on_error():
    def boom():
        raise ValueError("x")

    assert safe_run(boom, default="fallback") == "fallback"


# ---------------------------------------------------------------------- #
# 异常层级
# ---------------------------------------------------------------------- #
def test_exception_hierarchy():
    assert issubclass(DataUnavailableError, PanguError)
    assert issubclass(MarketClosedError, PanguError)
    assert issubclass(ConfigError, PanguError)
    assert issubclass(RateLimitError, PanguError)


# ---------------------------------------------------------------------- #
# 市场时间工具
# ---------------------------------------------------------------------- #
def test_is_market_open_during_session():
    dt = datetime(2025, 6, 25, 10, 0)  # 周三 10:00
    assert is_market_open(dt) is True


def test_is_market_open_closed_lunch():
    dt = datetime(2025, 6, 25, 12, 0)  # 午休
    assert is_market_open(dt) is False


def test_is_market_open_weekend():
    dt = datetime(2025, 6, 28, 10, 0)  # 周六
    assert is_market_open(dt) is False


def test_is_market_open_before_open():
    dt = datetime(2025, 6, 25, 9, 0)  # 开盘前
    assert is_market_open(dt) is False


def test_last_trading_date_weekday():
    d = datetime(2025, 6, 25)  # 周三
    assert last_trading_date(d) == "20250625"


def test_last_trading_date_sunday():
    d = datetime(2025, 6, 29)  # 周日
    assert last_trading_date(d) == "20250627"
