"""情绪温度计：把当日盘面情绪压缩成一个 0-100 的分数。

这是「攻防姿态」的总开关，决定今天该不该出手、出手多狠。

核心思路（短线情绪派的共识）：
- 涨停家数多 + 连板高度高 + 炸板率低 + 跌停少 → 市场亢奋，趋势做多胜率高
- 反之 → 冰点，要么观望（左侧），要么等反转信号

分项（默认权重，可经 settings.yaml 调）：
    limit_up_count    0.30  涨停家数
    consecutive_height 0.25  最高连板高度（市场风险偏好的温度计）
    broke_rate        0.20  炸板率（封板失败率，分歧度）
    limit_down_count  0.15  跌停家数（负向：恐慌）
    advance_decline   0.10  涨跌家数比（市场广度）

每项映射到 0-100，加权得总分。

姿态判定：
    < 40  冰点   → 防守/观望（情绪退潮，追高易吃面）
    40-85 正常   → 正常选股（有结构性机会）
    > 85  亢奋   → 警惕见顶（一致性太强，随时分歧退潮）

注意：本模块只读数据 + 打分，不做选股。选股在 trend_scanner + pipeline。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader, safe_float, find_col

logger = logging.getLogger("pangu.sentiment")


@dataclass
class SentimentBreakdown:
    """情绪各分项打分明细，便于前端展示和调参。"""

    limit_up_count: int = 0          # 原始：涨停家数
    limit_up_score: float = 50.0     # 打分 0-100

    consecutive_height: int = 0      # 原始：最高连板数
    consecutive_score: float = 50.0

    broke_count: int = 0             # 原始：炸板家数
    broke_rate: float = 0.0          # 原始：炸板率 = 炸板/(涨停+炸板)
    broke_rate_score: float = 50.0   # 打分（炸板率越低分越高）

    limit_down_count: int = 0        # 原始：跌停家数
    limit_down_score: float = 50.0   # 打分（跌停越少分越高）

    advance: int = 0                 # 原始：上涨家数
    decline: int = 0                 # 原始：下跌家数
    advance_decline_score: float = 50.0

    temperature: float = 50.0        # 加权总分
    posture: str = "正常"            # 姿态：冰点/正常/亢奋
    advice: str = ""                 # 一句话建议
    date: str = ""
    warnings: list[str] = field(default_factory=list)  # 数据缺失等提示

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": round(self.temperature, 1),
            "posture": self.posture,
            "advice": self.advice,
            "date": self.date,
            "components": {
                "limit_up_count": self.limit_up_count,
                "limit_up_score": round(self.limit_up_score, 1),
                "consecutive_height": self.consecutive_height,
                "consecutive_score": round(self.consecutive_score, 1),
                "broke_rate": round(self.broke_rate, 4),
                "broke_rate_score": round(self.broke_rate_score, 1),
                "limit_down_count": self.limit_down_count,
                "limit_down_score": round(self.limit_down_score, 1),
                "advance": self.advance,
                "decline": self.decline,
                "advance_decline_score": round(self.advance_decline_score, 1),
            },
            "warnings": self.warnings,
        }


class SentimentMeter:
    """情绪温度计。"""

    def __init__(self, dl: DataLoader, cfg: dict[str, Any]) -> None:
        self.dl = dl
        self.cfg = cfg or {}
        self.weights = self.cfg.get("weights", {
            "limit_up_count": 0.30,
            "consecutive_height": 0.25,
            "broke_rate": 0.20,
            "limit_down_count": 0.15,
            "advance_decline": 0.10,
        })
        self.lu_anchors = self.cfg.get("limit_up_anchors", {
            "cold": 15, "normal": 40, "hot": 80, "extreme": 120,
        })
        self.consec_anchors = self.cfg.get("consec_anchors", {
            "low": 2, "mid": 4, "high": 6, "extreme": 8,
        })

    def measure(self, date: Optional[str] = None) -> SentimentBreakdown:
        """计算当日情绪温度。date 为 None 表示今天。"""
        date = date or datetime.now().strftime("%Y%m%d")
        bd = SentimentBreakdown(date=date)

        # ---- 涨停 + 连板 ----
        zt = self.dl.limit_up_pool(date)
        if len(zt) == 0:
            bd.warnings.append("涨停池数据为空（可能非交易日或接口异常），涨停分项用中性 50")
        else:
            bd.limit_up_count = len(zt)
            bd.limit_up_score = _anchor_score(
                bd.limit_up_count,
                self.lu_anchors["cold"], self.lu_anchors["normal"],
                self.lu_anchors["hot"], self.lu_anchors["extreme"],
            )
            # 连板高度：取「涨停统计」或「连板数」列的最大值
            bd.consecutive_height = _max_consecutive(zt)
            bd.consecutive_score = _anchor_score(
                bd.consecutive_height,
                self.consec_anchors["low"], self.consec_anchors["mid"],
                self.consec_anchors["high"], self.consec_anchors["extreme"],
            )

        # ---- 炸板率 ----
        broke = self.dl.broke_pool(date)
        bd.broke_count = len(broke)
        denom = bd.limit_up_count + bd.broke_count
        if denom > 0:
            bd.broke_rate = bd.broke_count / denom
            # 炸板率越低越亢奋：0%→100分，30%→50分，60%+→0分
            bd.broke_rate_score = _clamp(100 - (bd.broke_rate / 0.60) * 100)
        else:
            bd.warnings.append("涨停+炸板均为 0，炸板率分项用中性 50")

        # ---- 跌停 ----
        dt = self.dl.limit_down_pool(date)
        bd.limit_down_count = len(dt)
        # 跌停 0→100分，20→60分，60+→0分
        bd.limit_down_score = _clamp(100 - (bd.limit_down_count / 60) * 100)

        # ---- 涨跌家数（市场广度）----
        spot = self.dl.all_spot()
        if len(spot) > 0:
            pct_col = _find_col(spot, ["涨跌幅"])
            if pct_col is not None:
                pcts = pd.to_numeric(spot[pct_col], errors="coerce").dropna()
                bd.advance = int((pcts > 0).sum())
                bd.decline = int((pcts < 0).sum())
                total = bd.advance + bd.decline
                if total > 0:
                    # 涨跌比 1:1→50，全涨→100，全跌→0
                    bd.advance_decline_score = _clamp((bd.advance / total) * 100)
            else:
                bd.warnings.append("实时行情缺涨跌幅列，涨跌比用中性 50")
        else:
            bd.warnings.append("实时行情为空，涨跌比用中性 50")

        # ---- 加权总分 ----
        bd.temperature = (
            bd.limit_up_score * self.weights.get("limit_up_count", 0) +
            bd.consecutive_score * self.weights.get("consecutive_height", 0) +
            bd.broke_rate_score * self.weights.get("broke_rate", 0) +
            bd.limit_down_score * self.weights.get("limit_down_count", 0) +
            bd.advance_decline_score * self.weights.get("advance_decline", 0)
        )
        bd.temperature = _clamp(bd.temperature)

        # ---- 姿态判定 ----
        if bd.temperature < 40:
            bd.posture = "冰点"
            bd.advice = "情绪退潮，追高易吃面。建议观望或只做超跌反弹左侧。"
        elif bd.temperature > 85:
            bd.posture = "亢奋"
            bd.advice = "一致性过强，随时分歧退潮。注意控制仓位，避免追高接力。"
        else:
            bd.posture = "正常"
            bd.advice = "情绪温和，有结构性机会。按趋势选股，严格执行止损。"

        logger.info("情绪温度 %.1f (%s) 涨停=%d 连板=%d 炸板率=%.2f%% 跌停=%d 涨跌=%d/%d",
                    bd.temperature, bd.posture, bd.limit_up_count,
                    bd.consecutive_height, bd.broke_rate * 100,
                    bd.limit_down_count, bd.advance, bd.decline)
        return bd


# ---------------------------------------------------------------------- #
# 辅助函数
# ---------------------------------------------------------------------- #
def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(x)))


def _anchor_score(value: float, a1: float, a2: float, a3: float, a4: float) -> float:
    """分段线性映射：value 在锚点 [a1,a2,a3,a4] 上分别对应 [0,50,85,100]。

    a1→0（冰点）, a2→50（正常）, a3→85（热）, a4→100（极热）。
    超出 a4 继续缓升封顶 100；低于 a1 封底 0。
    """
    points = [(a1, 0.0), (a2, 50.0), (a3, 85.0), (a4, 100.0)]
    points.sort(key=lambda p: p[0])
    if value <= points[0][0]:
        return 0.0
    if value >= points[-1][0]:
        return 100.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= value <= x1:
            if x1 == x0:
                return y1
            return _clamp(y0 + (value - x0) / (x1 - x0) * (y1 - y0))
    return 50.0


def _max_consecutive(zt_df: pd.DataFrame) -> int:
    """从涨停池里取最高连板数。

    akshare 列名可能是「涨停统计」（如"6天5板"）或「连板数」（数字）。
    """
    for col_name in ["连板数", "涨停统计"]:
        col = _find_col(zt_df, [col_name])
        if col is None:
            continue
        vals = zt_df[col].astype(str)
        if col_name == "涨停统计":
            # 格式如 "6天5板"：第一个数是累计涨停天数，第二个才是当前连板数。
            # 情绪指标要的是连板数（第二个数字），不是天数。
            parsed = vals.str.extract(r"(\d+)天(\d+)板")
            # 优先取连板数（第二列），格式不匹配时回退到第一个数字
            nums = pd.to_numeric(
                parsed[1].where(parsed[1].notna(), parsed[0]),
                errors="coerce",
            ).dropna()
        else:
            # 「连板数」列直接是连板数
            nums = pd.to_numeric(vals.str.extract(r"(\d+)").iloc[:, 0], errors="coerce").dropna()
        if len(nums) > 0:
            return int(nums.max())
    return 0


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """已弃用：请直接使用 data_loader.find_col。保留此处仅作兼容。"""
    from .data_loader import find_col
    return find_col(df, candidates)
