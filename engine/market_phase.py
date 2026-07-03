"""市场状态（情绪周期）识别模块。

将市场划分为 6 个阶段：冰点期、修复期、主升期、高潮期、分歧期、退潮期。
输出包含：当前阶段、阶段得分、允许/禁止的策略、仓位建议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader, find_col as _find_col, safe_float
from .sentiment_meter import SentimentMeter


@dataclass
class MarketPhase:
    phase: str = "未知"
    phase_score: int = 50
    allowed_strategies: list[str] = field(default_factory=list)
    forbidden_strategies: list[str] = field(default_factory=list)
    position_advice: str = "观望"
    components: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_phase": self.phase,
            "phase_score": self.phase_score,
            "allowed_strategies": self.allowed_strategies,
            "forbidden_strategies": self.forbidden_strategies,
            "position_advice": self.position_advice,
            "components": self.components,
            "warnings": self.warnings,
        }


class MarketPhaseAnalyzer:
    """根据涨停/跌停、炸板率、连板高度、指数趋势等判断市场状态。"""

    def __init__(self, dl: DataLoader, cfg: dict[str, Any] | None = None) -> None:
        self.dl = dl
        self.cfg = cfg or {}

    def analyze(self, date: Optional[str] = None) -> MarketPhase:
        date = date or pd.Timestamp.now().strftime("%Y%m%d")
        comp: dict[str, Any] = {"date": date}
        warnings: list[str] = []

        # 1. 情绪温度计数据
        try:
            bd = SentimentMeter(self.dl, self.cfg.get("sentiment", {})).measure(date)
            temperature = bd.temperature
            comp["temperature"] = round(temperature, 1)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"情绪温度获取失败: {e}")
            temperature = 50.0

        # 2. 涨停池 / 跌停池
        limit_up = self._load_limit_up(date)
        limit_down = self._load_limit_down(date)
        lu_count = len(limit_up)
        ld_count = len(limit_down)
        comp["limit_up_count"] = lu_count
        comp["limit_down_count"] = ld_count

        # 3. 连板高度
        max_consec = self._max_consecutive(limit_up)
        comp["consecutive_height"] = max_consec

        # 4. 炸板率（用涨停池里的炸板次数字段，若不可用则跳过）
        broke_rate, broke_status = self._broke_rate(limit_up, lu_count)
        comp["broke_rate"] = broke_rate
        comp["broke_rate_status"] = broke_status

        # 5. 昨日涨停今日表现（高标反馈）
        yesterday_feedback, yf_status = self._yesterday_feedback(date)
        comp["yesterday_feedback"] = yesterday_feedback
        comp["yesterday_feedback_status"] = yf_status

        # 6. 全市场涨跌分布
        breadth = self._market_breadth()
        comp["advance_decline_ratio"] = breadth.get("advance_decline_ratio")
        comp["breadth_status"] = breadth.get("status")

        # 综合判断
        return self._classify(temperature, lu_count, ld_count, max_consec, broke_rate, yesterday_feedback, breadth, comp, warnings)

    def _load_limit_up(self, date: str) -> pd.DataFrame:
        try:
            df = self.dl.limit_up_pool(date)
            return df if df is not None else pd.DataFrame()
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def _load_limit_down(self, date: str) -> pd.DataFrame:
        try:
            df = self.dl.limit_down_pool(date)
            return df if df is not None else pd.DataFrame()
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def _max_consecutive(self, limit_up: pd.DataFrame) -> int:
        if limit_up.empty:
            return 0
        col = _find_col(limit_up, ["连板数", "涨停统计"])
        if col is None:
            return 0
        nums = pd.to_numeric(limit_up[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(1).astype(int)
        return int(nums.max()) if len(nums) else 0

    def _broke_rate(self, limit_up: pd.DataFrame, lu_count: int) -> tuple[Optional[float], str]:
        broke_col = _find_col(limit_up, ["炸板次数"])
        if broke_col is None or limit_up.empty:
            return None, "unavailable"
        broke = pd.to_numeric(limit_up[broke_col], errors="coerce").fillna(0).astype(int).sum()
        denom = lu_count + broke
        if denom <= 0:
            return None, "unavailable"
        return broke / denom, "ok"

    def _yesterday_feedback(self, date: str) -> tuple[Optional[float], str]:
        from .data_loader import last_trading_date
        prev = last_trading_date(pd.Timestamp(date).to_pydatetime() if isinstance(date, str) else None)
        try:
            prev_zt = self.dl.limit_up_pool(prev)
            spot = self.dl.all_spot()
        except Exception:  # noqa: BLE001
            return None, "unavailable"
        if prev_zt is None or prev_zt.empty or spot is None or spot.empty:
            return None, "unavailable"
        code_col_prev = _find_col(prev_zt, ["代码"])
        code_col_spot = _find_col(spot, ["代码"])
        pct_col = _find_col(spot, ["涨跌幅"])
        if not code_col_prev or not code_col_spot or not pct_col:
            return None, "unavailable"
        prev_codes = set(prev_zt[code_col_prev].astype(str).str.zfill(6))
        s = spot.copy()
        s["_code"] = s[code_col_spot].astype(str).str.zfill(6)
        s["_pct"] = pd.to_numeric(s[pct_col], errors="coerce")
        hit = s[s["_code"].isin(prev_codes)]
        if hit.empty:
            return None, "unavailable"
        return round(float(hit["_pct"].mean()), 2), "ok"

    def _market_breadth(self) -> dict[str, Any]:
        try:
            spot = self.dl.all_spot()
        except Exception:  # noqa: BLE001
            return {"status": "unavailable"}
        if spot is None or spot.empty:
            return {"status": "unavailable"}
        pct_col = _find_col(spot, ["涨跌幅"])
        if pct_col is None:
            return {"status": "unavailable"}
        pcts = pd.to_numeric(spot[pct_col], errors="coerce").dropna()
        adv = int((pcts > 0).sum())
        dec = int((pcts < 0).sum())
        total = adv + dec
        return {
            "status": "ok",
            "advance": adv,
            "decline": dec,
            "advance_decline_ratio": round(adv / total, 2) if total > 0 else None,
        }

    def _classify(
        self,
        temperature: float,
        lu_count: int,
        ld_count: int,
        max_consec: int,
        broke_rate: Optional[float],
        yesterday_feedback: Optional[float],
        breadth: dict[str, Any],
        components: dict[str, Any],
        warnings: list[str],
    ) -> MarketPhase:
        adv_ratio = breadth.get("advance_decline_ratio") or 0.5
        high_feedback_bad = yesterday_feedback is not None and yesterday_feedback < -2
        broke_bad = broke_rate is not None and broke_rate > 0.35

        # 冰点期
        if temperature < 40 or ld_count > 50 or (lu_count < 15 and ld_count > lu_count):
            return MarketPhase(
                phase="冰点期",
                phase_score=max(10, int(temperature)),
                allowed_strategies=["低位修复", "红利低波防守"],
                forbidden_strategies=["追涨", "高位接力", "短线进攻"],
                position_advice="空仓或极轻仓试错",
                components=components,
                warnings=warnings + (["炸板率偏高"] if broke_bad else []),
            )

        # 高潮期 / 亢奋
        if temperature > 85 or lu_count > 100:
            if broke_bad or high_feedback_bad:
                # 高潮中出现负反馈 → 分歧期
                return MarketPhase(
                    phase="分歧期",
                    phase_score=75,
                    allowed_strategies=["龙头承接", "中军抗跌观察"],
                    forbidden_strategies=["后排追涨", "高位无承接追涨"],
                    position_advice="轻仓，只盯龙头",
                    components=components,
                    warnings=warnings + (["高位股负反馈"] if high_feedback_bad else []),
                )
            return MarketPhase(
                phase="高潮期",
                phase_score=90,
                allowed_strategies=["核心分歧回封", "等待回踩"],
                forbidden_strategies=["后排追涨", "高位无承接追涨"],
                position_advice="轻仓，只参与核心",
                components=components,
                warnings=warnings,
            )

        # 退潮期
        if high_feedback_bad and (temperature < 55 or ld_count > 20):
            return MarketPhase(
                phase="退潮期",
                phase_score=30,
                allowed_strategies=["防守观察", "红利低波"],
                forbidden_strategies=["短线进攻", "追涨", "接力"],
                position_advice="空仓或防守仓位",
                components=components,
                warnings=warnings + (["昨日涨停今日负反馈"] if high_feedback_bad else []),
            )

        # 主升期
        if 60 <= temperature <= 85 and lu_count >= 40 and max_consec >= 4 and adv_ratio >= 0.55:
            return MarketPhase(
                phase="主升期",
                phase_score=80,
                allowed_strategies=["题材龙头", "中军趋势", "补涨扩散", "连板核心"],
                forbidden_strategies=["后排杂毛", "无板块共振"],
                position_advice="积极仓位，聚焦主线",
                components=components,
                warnings=warnings,
            )

        # 修复期
        if 40 <= temperature < 60 and lu_count >= 20 and adv_ratio >= 0.5:
            return MarketPhase(
                phase="修复期",
                phase_score=65,
                allowed_strategies=["低位反转", "首板启动", "强趋势回踩"],
                forbidden_strategies=["高位一致加速", "后排追涨"],
                position_advice="轻仓试错",
                components=components,
                warnings=warnings,
            )

        # 默认：震荡/分歧
        return MarketPhase(
            phase="分歧期",
            phase_score=55,
            allowed_strategies=["龙头承接", "中军抗跌观察", "低位补涨"],
            forbidden_strategies=["后排追涨"],
            position_advice="控制仓位，聚焦核心",
            components=components,
            warnings=warnings,
        )
