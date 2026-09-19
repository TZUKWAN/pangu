"""Pangu 2.0 研究层（Phase 2）：因子 API / 因子库 / 单因子诊断。

模块结构：
- data_interface：ResearchData 协议 + PITStore 适配 + 合成数据（离线测试用）
- factors.base：FactorMeta / Factor 抽象基类 / 横截面预处理
- factors.library：动量、反转、低波、流动性、涨停微观结构、市场宽度等因子
- factors.registry：因子注册表（code_hash + availability 诚实标注）
- univariate：evaluate_factor 单因子诊断（IC / ICIR / 分位 / regime 切分 / PIT 守卫）
- reporting：诊断报告落盘（JSON + factor_index.jsonl）

硬性规则（P2-002 / P0-007）：因子输出一律是 raw_score（原始打分），
任何地方不得把因子值称为“概率/probability”。
"""

from __future__ import annotations

from .data_interface import PITResearchData, ResearchData, SyntheticResearchData
from .factors.base import Factor, FactorMeta, FactorUnavailableError
from .factors.registry import FactorRegistry, build_default_registry
from .reporting import write_factor_report
from .univariate import evaluate_factor, forward_returns

__all__ = [
    "ResearchData", "PITResearchData", "SyntheticResearchData",
    "Factor", "FactorMeta", "FactorUnavailableError",
    "FactorRegistry", "build_default_registry",
    "evaluate_factor", "forward_returns", "write_factor_report",
]
