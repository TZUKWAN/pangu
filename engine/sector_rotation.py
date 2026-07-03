"""板块轮动分析：识别上升通道板块 vs 退潮板块，给选股降权/加权。

短线选股的共识：选对板块比选对个股更重要。处于上升通道、资金持续流入的板块，
其成分股胜率显著高于退潮板块。本模块给每个板块算一个「轮动得分」，供：
1. trend_scanner 板块排名时叠加轮动因子
2. recommender 给候选股附加「板块强度」参考

数据源：DataLoader.sector_fund_flow_rank 的 今日/5日/10日 资金流（当前主源同花顺）。
轮动得分构成：
- 动量分（40%）：5日 vs 10日 净流入对比，5日更强=加速流入=走强
- 持续性分（40%）：连续净流入天数
- 当日强度分（20%）：当日净流入额排名

设计：取数失败返回中性 50 分，不阻断选股。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader, safe_float, find_col

logger = logging.getLogger("pangu.sector_rotation")


@dataclass
class SectorRotationResult:
    """板块轮动分析结果。"""

    scores: dict[str, float] = field(default_factory=dict)  # {板块名: 轮动得分 0-100}
    warnings: list[str] = field(default_factory=list)

    def score_of(self, board_name: str) -> float:
        """取某板块的轮动得分，未命中返回中性 50。"""
        if not board_name:
            return 50.0
        # 精确匹配 > 包含匹配
        if board_name in self.scores:
            return self.scores[board_name]
        for name, sc in self.scores.items():
            if board_name in name or name in board_name:
                return sc
        return 50.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": {k: round(v, 1) for k, v in sorted(self.scores.items(), key=lambda x: -x[1])[:20]},
            "warnings": self.warnings,
        }


class SectorRotationAnalyzer:
    """板块轮动分析器。"""

    def __init__(self, dl: DataLoader, cfg: Optional[dict] = None) -> None:
        self.dl = dl
        self.cfg = cfg or {}

    def analyze(self) -> SectorRotationResult:
        """分析全市场板块轮动，返回 {板块名: 轮动得分}。"""
        result = SectorRotationResult()
        try:
            # 取 5 日和 10 日资金流排名（含板块名+净流入额）
            fund5 = self.dl.sector_fund_flow_rank("5日")
            fund10 = self.dl.sector_fund_flow_rank("10日")
            fund_today = self.dl.sector_fund_flow_rank("今日")
        except Exception as e:  # noqa: BLE001
            result.warnings.append(f"板块资金流取数失败，轮动分析用中性值: {e}")
            return result

        if fund5.empty and fund10.empty:
            result.warnings.append("板块资金流数据为空，轮动分析用中性值")
            return result

        # 解析各周期数据 → {板块名: 净流入额}
        map5 = self._parse_fund(fund5)
        map10 = self._parse_fund(fund10)
        map_today = self._parse_fund(fund_today)

        # 汇总所有板块名
        all_boards = set(map5) | set(map10) | set(map_today)

        # 计算每个板块的轮动得分
        # 先算各周期净流入的 z-score 排名（标准化）
        today_rank = self._rank_pct(map_today)  # {板块: 当日净流入百分位 0-100}
        rank5 = self._rank_pct(map5)
        rank10 = self._rank_pct(map10)

        for board in all_boards:
            r_today = today_rank.get(board, 50)
            r5 = rank5.get(board, 50)
            r10 = rank10.get(board, 50)
            # 动量分：5日排名 > 10日排名 = 走强加速
            momentum = 50 + (r5 - r10) * 0.8
            momentum = max(0, min(100, momentum))
            # 持续性分：5日排名高 = 持续流入
            persistence = r5
            # 当日强度分
            strength = r_today
            # 加权
            score = momentum * 0.4 + persistence * 0.4 + strength * 0.2
            result.scores[board] = round(max(0, min(100, score)), 1)

        logger.info("板块轮动分析完成：%d 个板块", len(result.scores))
        return result

    def _parse_fund(self, df: pd.DataFrame) -> dict[str, float]:
        """解析资金流 DataFrame → {板块名: 净流入额}。"""
        if df is None or len(df) == 0:
            return {}
        name_col = find_col(df, ["名称", "板块名称"])
        # 净流入列名因周期不同：今日主力净流入-净额 / 5日主力净流入-净额 / 10日主力净流入-净额
        net_col = find_col(df, ["主力净流入-净额", "主力净流入额", "净流入额"])
        if name_col is None or net_col is None:
            return {}
        out = {}
        for _, row in df.iterrows():
            name = str(row.get(name_col, "")).strip()
            if not name:
                continue
            val = safe_float(row.get(net_col))
            if val == val:  # 非 NaN
                out[name] = val
        return out

    @staticmethod
    def _rank_pct(value_map: dict[str, float]) -> dict[str, float]:
        """把 {板块: 净流入额} 转成百分位排名 {板块: 0-100}。

        净流入额越大，百分位越高（资金越青睐）。
        """
        if not value_map:
            return {}
        series = pd.Series(value_map)
        # rank(pct=True) 默认升序，最大值得分接近 1.0
        ranks = series.rank(pct=True) * 100
        return {k: float(v) for k, v in ranks.items()}
