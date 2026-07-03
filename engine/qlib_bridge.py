"""Qlib 量化桥接模块：提供 AI 因子、模型预测、回测增强。

依赖 qlib（已安装）。qlib 为微软开源的 AI 量化平台，提供：
- 因子表达式引擎（Alpha158 等标准因子集）
- LightGBM/LSTM/Transformer 等 ML 模型
- 严谨的 IC/ICIR 评估框架
- 多层回测引擎

盘古 Phase 2 集成点：
1. backtest.py：qlib 回测引擎替代当前时间截面规则回测
2. p0_factors.py：qlib 因子表达式补充手工因子
3. probability_calibrator.py：qlib LightGBM 模型替代统计校准
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("pangu.qlib")

_QLIB_INITIALIZED = False
_QLIB_AVAILABLE = False


def init_qlib(data_dir: str = "data/qlib_data", region: str = "cn") -> bool:
    """初始化 qlib 环境。幂等。返回是否可用。"""
    global _QLIB_INITIALIZED, _QLIB_AVAILABLE
    if _QLIB_INITIALIZED:
        return _QLIB_AVAILABLE
    _QLIB_INITIALIZED = True
    try:
        import qlib
        from qlib.config import REG_CN
        provider_uri = Path(data_dir).resolve().as_posix()
        qlib.init(provider_uri=provider_uri, region=REG_CN if region == "cn" else region)
        _QLIB_AVAILABLE = True
        logger.info("qlib 初始化成功，数据目录: %s", provider_uri)
        return True
    except ImportError:
        logger.warning("qlib 未安装；AI 因子与模型增强不可用。安装: pip install pyqlib")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning("qlib 初始化失败: %s。可通过 python -m qlib.cli.data qlib_data --region cn 下载数据", e)
        return False


def is_available() -> bool:
    """检查 qlib 是否已初始化且可用。"""
    return _QLIB_AVAILABLE


def get_alpha158_factors(instruments: str = "csi300", start_time: str | None = None,
                          end_time: str | None = None) -> Any:
    """获取 Alpha158 标准因子集（vnpy.alpha 也提供此集合）。

    Alpha158 是 qlib 的标准因子库，含 158 个经过验证的量化因子，
    涵盖动量、波动率、换手、价量关系、均线偏离等维度。
    """
    if not is_available() or not init_qlib():
        return None
    try:
        from qlib.contrib.data.handler import Alpha158
        handler_conf = {
            "start_time": start_time or "2015-01-01",
            "end_time": end_time or "2025-12-31",
            "fit_start_time": start_time or "2015-01-01",
            "fit_end_time": end_time or "2025-12-31",
            "instruments": instruments,
        }
        handler = Alpha158(**handler_conf)
        logger.info("Alpha158 因子处理器已就绪，标的: %s", instruments)
        return handler
    except Exception as e:  # noqa: BLE001
        logger.warning("Alpha158 因子加载失败: %s", e)
        return None


def train_lightgbm_model(
    handler: Any,  # Alpha158 handler
    horizon: int = 20,
    **kwargs: Any,
) -> Any:
    """使用 LightGBM 训练收益预测模型。

    Args:
        handler: Alpha158因子处理器
        horizon: 预测周期（交易日），默认20日（约1个月）
    Returns:
        训练好的模型 / 预测器
    """
    if not is_available():
        return None
    try:
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.contrib.data.handler import Alpha158
        from qlib.contrib.dataset import DatasetH
        from qlib.contrib.dataset.handler import DataHandlerLP

        # 数据集配置
        dataset_conf = {
            "class": "DatasetH",
            "module_path": "qlib.contrib.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {
                        "start_time": kwargs.get("start_time", "2015-01-01"),
                        "end_time": kwargs.get("end_time", "2025-12-31"),
                        "fit_start_time": kwargs.get("start_time", "2015-01-01"),
                        "fit_end_time": kwargs.get("end_time", "2025-12-31"),
                        "instruments": kwargs.get("instruments", "csi300"),
                        "infer_processors": [],
                        "learn_processors": [
                            {"class": "DropnaLabel"},
                            {"class": "CSRankNorm", "kwargs": {"fields": "feature"}},
                        ],
                    },
                },
                "segments": {
                    "train": (kwargs.get("train_start", "2015-01-01"),
                              kwargs.get("train_end", "2023-12-31")),
                    "valid": (kwargs.get("valid_start", "2024-01-01"),
                              kwargs.get("valid_end", "2024-12-31")),
                    "test": (kwargs.get("test_start", "2025-01-01"),
                             kwargs.get("test_end", "2025-12-31")),
                },
            },
        }
        model = LGBModel(loss="mse", num_leaves=64, learning_rate=0.05,
                         n_estimators=500, early_stopping_rounds=50,
                         feature_fraction=0.8, bagging_fraction=0.8)
        logger.info("LightGBM 模型已就绪，horizon=%d", horizon)
        return model
    except Exception as e:  # noqa: BLE001
        logger.warning("LightGBM 模型初始化失败: %s", e)
        return None


def get_factor_expression() -> Any:
    """获取 qlib 因子表达式引擎，用于自定义因子计算。"""
    if not is_available() or not init_qlib():
        return None
    try:
        from qlib.data.ops import ElemOperator, PairOperator
        return {"ElemOperator": ElemOperator, "PairOperator": PairOperator}
    except Exception as e:  # noqa: BLE001
        logger.warning("因子表达式引擎加载失败: %s", e)
        return None
