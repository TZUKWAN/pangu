"""Pangu 全局错误处理框架。

目标：把 akshare 网络抖动、数据缺失、配置错误等「不可控外部错误」
封装成可预测、可重试、可降级的异常体系，避免直接炸崩 pipeline / REPL。

主要能力：
- 统一异常层级（PanguError 为根）
- @retry_on_error 指数退避重试
- safe_run 兜底工具：失败返回默认值 + 记录 warning
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, Iterable, Optional, TypeVar

logger = logging.getLogger("pangu.errors")


class PanguError(RuntimeError):
    """Pangu 引擎根异常。"""


class DataUnavailableError(PanguError):
    """数据源取数失败或返回空（akshare/缓存均不可用）。"""


class MarketClosedError(PanguError):
    """非交易日，或收盘前数据不全导致无法生成可靠信号。"""


class ConfigError(PanguError):
    """配置缺失/格式错误/取值非法。"""


class RateLimitError(PanguError):
    """触发数据源限流（akshare 被临时封 IP）。"""


T = TypeVar("T")


def retry_on_error(
    exceptions: Iterable[type[BaseException]] | None = None,
    tries: int = 3,
    backoff_seconds: float = 2.0,
    backoff_jitter: float = 0.2,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """指数退避重试装饰器。

    Args:
        exceptions: 触发重试的异常类型，默认 (Exception,)。
        tries: 总尝试次数（含第一次）。
        backoff_seconds: 初始退避秒数。
        backoff_jitter: 抖动比例（0~1），避免并发 herd。
        on_retry: 每次重试前的回调，签名为 (exception, attempt)。

    Raises:
        若重试用尽仍失败，抛出最后一次捕获的异常。
    """
    if exceptions is None:
        exceptions = (Exception,)
    exc_tuple = tuple(exceptions)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_err: Optional[BaseException] = None
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exc_tuple as e:  # noqa: PERF203
                    last_err = e
                    if attempt >= tries:
                        break
                    wait = backoff_seconds * (2 ** (attempt - 1))
                    wait = wait * (1 + random.uniform(-backoff_jitter, backoff_jitter))
                    wait = max(0.0, wait)
                    logger.warning(
                        "%s 失败 attempt %d/%d: %s; %.2fs 后重试",
                        func.__name__, attempt, tries, e, wait,
                    )
                    if on_retry:
                        on_retry(e, attempt)
                    time.sleep(wait)
            raise last_err  # type: ignore[misc]
        return wrapper
    return decorator


def safe_run(
    func: Callable[..., T],
    *args: Any,
    default: Optional[T] = None,
    **kwargs: Any,
) -> Optional[T]:
    """安全执行函数：失败时返回默认值并记录 warning。

    用于 REPL / pipeline 中「非关键路径」的兜底，例如打印辅助信息、
    可选指标计算等。关键路径仍应显式处理异常。
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        logger.warning("safe_run 执行 %s 失败: %s，返回默认值", func.__name__, e)
        return default
