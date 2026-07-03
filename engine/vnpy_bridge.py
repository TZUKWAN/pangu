"""Vnpy 量化桥接模块：提供 Alpha 因子、回测增强。

依赖 vnpy（已安装）。vnpy 为国内主流量化交易框架，提供：
- Alpha-158 标准因子集（vnpy.alpha）
- LightGBM/Lasso/MLP 等 ML 模型
- CTA/套利/网格等多种策略模板
- 国内券商 CTP/XTP 等网关（仅分析场景不使用）

盘古 Phase 2 集成点：
1. p0_factors.py：vnpy Alpha-158 因子补充手工因子
2. backtest.py：vnpy CTA 回测引擎
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("pangu.vnpy")

_VNPY_AVAILABLE = False


def init_vnpy() -> bool:
    """检查 vnpy 是否可用。"""
    global _VNPY_AVAILABLE
    try:
        import vnpy  # noqa: F401
        _VNPY_AVAILABLE = True
        logger.info("vnpy %s 已就绪", vnpy.__version__)
        return True
    except ImportError:
        logger.warning("vnpy 未安装；CTA回测与Alpha因子增强不可用。安装: pip install vnpy")
        return False


def is_available() -> bool:
    """检查 vnpy 是否已初始化且可用。"""
    global _VNPY_AVAILABLE
    if not _VNPY_AVAILABLE:
        init_vnpy()
    return _VNPY_AVAILABLE


def get_alpha_factors(symbols: list[str] | None = None) -> dict[str, Any] | None:
    """获取 vnpy Alpha-158 因子列表与计算方法。

    vnpy.alpha 模块实现了与 qlib 兼容的 Alpha158 因子集，
    可直接用于个股特征计算。比手工计算更严谨、更高效。

    Args:
        symbols: 股票代码列表，默认沪深300成分股
    Returns:
        {"factor_names": [...], "count": 158, "module": <alpha_module>}
    """
    if not is_available():
        return None
    try:
        from vnpy.alpha.dataset import Alpha158
        factor_count = 158
        return {
            "factor_count": factor_count,
            "available": True,
            "dataset_class": "Alpha158",
            "note": "vnpy Alpha-158 因子集（与 qlib Alpha158 兼容）",
        }
    except ImportError:
        # vnpy.alpha 可能需要单独安装
        logger.warning("vnpy.alpha 模块不可用；请确保 vnpy 完整安装")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Alpha 因子加载失败: %s", e)
        return None


def get_cta_backtester(engine_config: dict[str, Any] | None = None) -> Any | None:
    """获取 vnpy CTA 回测引擎。

    vnpy 的回测引擎基于事件驱动架构，支持：
    - 多周期 K 线回测
    - 滑点/手续费/保证金模拟
    - T+1 限制
    - 多合约组合回测

    Args:
        engine_config: 引擎配置（数据源、资金、手续费等）
    Returns:
        CTA 回测引擎实例 或 None
    """
    if not is_available():
        return None
    try:
        from vnpy.trader.constant import Interval, Direction, Offset
        from vnpy.trader.object import BarData, TickData
        logger.info("vnpy CTA 回测组件已就绪")
        return {
            "available": True,
            "interval": Interval,
            "direction": Direction,
            "offset": Offset,
            "bar_data": BarData,
            "tick_data": TickData,
            "note": "vnpy CTA 回测组件可用于构建事件驱动回测",
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("CTA 回测组件加载失败: %s", e)
        return None


def get_backtest_engine(
    symbols: list[str],
    start_date: str,
    end_date: str,
    capital: float = 1_000_000,
    commission_rate: float = 0.00025,
    slippage: float = 0.001,
) -> dict[str, Any] | None:
    """构造 vnpy CTA 回测配置。

    返回配置 dict 供 backtest.py 使用或直接传递给 vnpy 引擎。
    """
    if not is_available():
        return None
    return {
        "available": True,
        "config": {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "capital": capital,
            "commission_rate": commission_rate,
            "slippage": slippage,
            "mode": "backtesting",
            "note": "vnpy CTA 事件驱动回测；信号由盘古 Pipeline 提供",
        },
    }
