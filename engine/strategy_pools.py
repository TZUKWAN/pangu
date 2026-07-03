"""A 股策略池：七类独立选股器。

每个策略池只负责在自己的选股逻辑下产出候选（未经过 QuantGuard 过滤）。
推荐闸门 `RecommendationGate` 负责把市场阶段、策略池产出、护栏结果统一成最终推荐。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader, find_col, last_trading_date, safe_float
from .trend_scanner import RPSCalculator

logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    strategy_name: str
    code: str
    name: str
    board: str = "未知"
    trigger_reason: str = ""
    score: float = 0.0
    raw_features: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    respect_market_phase: bool = True
    allow_when_phase_forbids: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "code": self.code,
            "name": self.name,
            "board": self.board,
            "trigger_reason": self.trigger_reason,
            "score": round(self.score, 2),
            "raw_features": self.raw_features,
            "warnings": self.warnings,
            "allow_when_phase_forbids": self.allow_when_phase_forbids,
        }


class StrategyPool(ABC):
    """策略池基类。"""

    name: str = ""
    # 是否受市场阶段 forbid 影响
    respect_market_phase: bool = True

    def __init__(self, dl: DataLoader, cfg: dict[str, Any] | None = None) -> None:
        self.dl = dl
        self.cfg = cfg or {}

    @abstractmethod
    def select(self, date: Optional[str] = None) -> list[StrategySignal]:
        ...

    def _market_spot(self) -> pd.DataFrame:
        df = self.dl.all_spot()
        if df is None:
            return pd.DataFrame()
        return df.copy()

    def _code_series(self, df: pd.DataFrame) -> pd.Series:
        col = find_col(df, ["代码", "code", "股票代码"])
        if col is None:
            return pd.Series(dtype=str)
        return df[col].astype(str).str.zfill(6)

    def _name_series(self, df: pd.DataFrame) -> pd.Series:
        col = find_col(df, ["名称", "name", "股票名称"])
        return df[col] if col else pd.Series(index=df.index, dtype=str)

    def _board(self, code: str) -> str:
        if code.startswith("68"):
            return "科创板"
        if code.startswith("30"):
            return "创业板"
        if code.startswith("8") or code.startswith("4"):
            return "北交所"
        if code.startswith("0"):
            return "深市主板"
        if code.startswith("60"):
            return "沪市主板"
        return "其他"

    def _daily_kline(self, code: str, n: int = 120, date: Optional[str] = None) -> pd.DataFrame:
        try:
            return self.dl.daily_kline(code, days=n, date=date)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def _is_up_limit(self, pct: float, board: str) -> bool:
        if board in ("科创板", "创业板", "北交所"):
            return pct >= 19.5
        return pct >= 9.5


# ---------------------------------------------------------------------------
# 1. 题材龙头池（主线核心）
# ---------------------------------------------------------------------------
class ThemeLeaderPool(StrategyPool):
    """识别今日涨停强度、板块集中度和连板梯队，筛选题材龙头。"""

    name = "题材龙头"

    def select(self, date: Optional[str] = None) -> list[StrategySignal]:
        date = date or pd.Timestamp.now().strftime("%Y%m%d")
        lu = self.dl.limit_up_pool(date)
        if lu is None or lu.empty:
            return []
        lu = lu.copy()
        lu["_code"] = self._code_series(lu)
        lu["_name"] = self._name_series(lu)

        # 连板数解析
        consec_col = find_col(lu, ["连板数", "涨停统计"])
        if consec_col:
            lu["_consec"] = pd.to_numeric(lu[consec_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(1).astype(int)
        else:
            lu["_consec"] = 1

        # 概念/行业
        concept_col = find_col(lu, ["所属行业", "概念", "题材"])
        if concept_col is None:
            logger.warning("题材龙头池缺少概念/行业字段，跳过")
            return []

        # 市值列（用于识别中军）
        mv_col = find_col(lu, ["总市值", "流通市值"])
        if mv_col:
            lu["_mv"] = pd.to_numeric(lu[mv_col], errors="coerce").fillna(0)
        else:
            lu["_mv"] = 0.0

        # 统计板块强度
        board_score: dict[str, float] = {}
        board_max_consec: dict[str, int] = {}
        board_mv_median: dict[str, float] = {}
        lu["_concept"] = lu[concept_col].astype(str).str.split(",").str[0]
        for concept, g in lu.groupby("_concept"):
            consec = g["_consec"].astype(int)
            board_score[concept] = len(g) + consec.sum() * 0.5
            board_max_consec[concept] = int(consec.max())
            board_mv_median[concept] = float(g["_mv"].median()) if mv_col else 0.0

        if not board_score:
            return []

        # 取前 3 主线
        top_boards = sorted(board_score.items(), key=lambda x: x[1], reverse=True)[:3]
        top_board_names = {b[0] for b in top_boards}

        signals: list[StrategySignal] = []
        for _, row in lu.iterrows():
            code = row["_code"]
            name = str(row["_name"]) if pd.notna(row.get("_name")) else ""
            board = self._board(code)
            concept = str(row[concept_col]).split(",")[0]
            if concept not in top_board_names:
                continue
            consec = int(row["_consec"])
            max_consec = board_max_consec.get(concept, 1)
            mv = safe_float(row.get("_mv")) or 0.0
            mv_median = board_mv_median.get(concept, 0.0)

            # 个股地位识别
            if consec == max_consec and consec >= 2:
                role = "龙头"
            elif mv >= mv_median * 1.2 and consec >= 1:
                role = "中军"
            elif consec >= 2:
                role = "补涨"
            elif consec == 1:
                role = "首板观察"
            else:
                role = "跟风"

            # 后排跟风/杂毛不进入推荐
            if role == "跟风":
                continue

            score = 60 + board_score.get(concept, 0) * 2 + consec * 5
            if role == "龙头":
                score += 15
            elif role == "中军":
                score += 10
            elif role == "补涨":
                score += 5
            score = min(score, 98)

            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    code=code,
                    name=name,
                    board=board,
                    trigger_reason=f"主线 {concept} {role}，{consec} 板",
                    score=score,
                    raw_features={
                        "concept": concept,
                        "role": role,
                        "consecutive_boards": consec,
                        "board_strength": round(board_score.get(concept, 0), 2),
                        "board_rank": list(top_board_names).index(concept) + 1,
                    },
                    allow_when_phase_forbids=False,
                )
            )
        return sorted(signals, key=lambda s: s.score, reverse=True)[:15]


# ---------------------------------------------------------------------------
# 2. 连板/打板池
# ---------------------------------------------------------------------------
class LimitUpPool(StrategyPool):
    """连板梯队、首板质量。"""

    name = "连板梯队"

    def select(self, date: Optional[str] = None) -> list[StrategySignal]:
        date = date or pd.Timestamp.now().strftime("%Y%m%d")
        lu = self.dl.limit_up_pool(date)
        if lu is None or lu.empty:
            return []
        lu = lu.copy()
        lu["_code"] = self._code_series(lu)
        lu["_name"] = self._name_series(lu)

        pct_col = find_col(lu, ["涨跌幅", "最新价"])
        consec_col = find_col(lu, ["连板数", "涨停统计"])
        first_seal_col = find_col(lu, ["首次封板时间", "首次涨停时间"])
        broken_col = find_col(lu, ["炸板次数"])
        seal_amt_col = find_col(lu, ["封单金额", "涨停封单额"])
        turnover_col = find_col(lu, ["换手率"])
        amount_col = find_col(lu, ["成交额"])
        if consec_col:
            lu["_consec"] = pd.to_numeric(lu[consec_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(1).astype(int)
        else:
            lu["_consec"] = 1

        signals: list[StrategySignal] = []
        for _, row in lu.iterrows():
            code = row["_code"]
            name = str(row["_name"]) if pd.notna(row.get("_name")) else ""
            board = self._board(code)
            consec = int(row["_consec"])

            # 涨停类型
            if consec == 1:
                setup = "首板启动"
            elif consec >= 2:
                setup = "连板龙头"
            else:
                setup = "涨停"

            score = 50 + consec * 8
            reasons = [f"{consec} 连板" if consec > 1 else setup]

            pct = safe_float(row.get(pct_col)) if pct_col else None
            is_20cm = pct is not None and pct > 19
            if is_20cm:
                score += 5
                reasons.append("20cm 涨停")

            # 首封时间越早越好（09:30-09:40 最佳）
            first_seal_score = 0
            if first_seal_col:
                t = str(row.get(first_seal_col, "")).replace(":", "")
                if t and t[:4].isdigit():
                    hhmm = int(t[:4])
                    if hhmm <= 930:
                        first_seal_score = 15
                    elif hhmm <= 940:
                        first_seal_score = 12
                    elif hhmm <= 1000:
                        first_seal_score = 8
                    elif hhmm <= 1030:
                        first_seal_score = 4

            # 炸板次数惩罚
            broken = 0
            if broken_col:
                broken = int(pd.to_numeric(row.get(broken_col), errors="coerce").fillna(0).iloc[0] if hasattr(row.get(broken_col), "iloc") else pd.to_numeric(row.get(broken_col), errors="coerce") or 0)
            broken_penalty = broken * 8

            # 封单金额 / 成交额 比例
            seal_ratio = None
            if seal_amt_col and amount_col:
                seal_amt = safe_float(row.get(seal_amt_col))
                amt = safe_float(row.get(amount_col))
                if seal_amt is not None and amt and amt > 0:
                    seal_ratio = seal_amt / amt

            # 换手率（充分换手更好，但过高警惕）
            turnover = safe_float(row.get(turnover_col)) if turnover_col else None
            turnover_score = 0
            if turnover is not None:
                if 3 <= turnover <= 20:
                    turnover_score = 5
                elif 1 <= turnover < 3:
                    turnover_score = 2
                elif turnover > 30:
                    turnover_score = -5

            score = score + first_seal_score - broken_penalty + turnover_score
            if seal_ratio is not None and seal_ratio > 0.1:
                score += 5
            score = max(20, min(score, 95))

            # 一字板不可买，仅作情绪锚点
            one_word = False
            if first_seal_col:
                one_word = str(row.get(first_seal_col, "")).startswith("09:25")

            raw_features = {
                "consecutive_boards": consec,
                "setup": setup,
                "first_seal_time": row.get(first_seal_col) if first_seal_col else None,
                "broken_count": broken,
                "seal_amount_ratio": round(seal_ratio, 3) if seal_ratio is not None else None,
                "turnover_rate": turnover,
                "is_20cm": is_20cm,
                "one_word_limit": one_word,
            }

            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    code=code,
                    name=name,
                    board=board,
                    trigger_reason=", ".join(reasons),
                    score=score,
                    raw_features=raw_features,
                )
            )
        return sorted(signals, key=lambda s: s.score, reverse=True)[:15]


# ---------------------------------------------------------------------------
# 3. 趋势回踩池
# ---------------------------------------------------------------------------
class TrendPullbackPool(StrategyPool):
    """强趋势股（RPS 高 + 20 日均线 + 缩量回踩/突破）。"""

    name = "趋势回踩"
    respect_market_phase = False

    def select(self, date: Optional[str] = None) -> list[StrategySignal]:
        date = date or pd.Timestamp.now().strftime("%Y%m%d")
        cfg = self.cfg.get("trend", {})
        max_candidates = cfg.get("max_candidates", 200)
        min_rps = cfg.get("min_rps", 90)
        min_turnover = cfg.get("min_turnover_rate", 1.0)

        spot = self._market_spot()
        if spot.empty:
            return []

        pct_col = find_col(spot, ["涨跌幅"])
        close_col = find_col(spot, ["最新价"])
        vol_col = find_col(spot, ["换手率", "turnover"])
        mv_col = find_col(spot, ["总市值", "流通市值"])
        if not all([pct_col, close_col, vol_col]):
            return []

        codes = self._code_series(spot).unique()
        # 限制数量防止太慢
        codes = list(codes)[:max_candidates]

        rps_engine = RPSCalculator(self.dl, self.cfg)
        rps_map = rps_engine.rps_for_codes(codes, date)

        signals: list[StrategySignal] = []
        for code in codes:
            rps_info = rps_map.get(code, {"rps": 0, "mode": "unavailable"})
            rps = safe_float(rps_info.get("rps", 0))
            mode = rps_info.get("mode", "unavailable")
            if rps < min_rps:
                continue

            row = spot[spot[self._code_series(spot) == code].index[0]]
            board = self._board(code)
            turnover = safe_float(row.get(vol_col))
            if turnover is None or turnover < min_turnover:
                continue

            # 量价/均线回踩判定
            kline = self._daily_kline(code, n=60, date=date)
            if kline.empty or len(kline) < 30:
                continue
            close = pd.to_numeric(kline["close"], errors="coerce").dropna()
            vol = pd.to_numeric(kline["volume"], errors="coerce").dropna()
            if len(close) < 30 or len(vol) < 30:
                continue
            ma20 = close.rolling(20).mean().iloc[-1]
            prev_close = close.iloc[-1]
            pullback = prev_close < ma20 * 1.03 and prev_close > ma20 * 0.95
            breakout = prev_close > ma20 * 1.05 and close.iloc[-2] <= ma20 * 1.05
            vol_shrink = vol.iloc[-1] < vol.iloc[-5:].mean() * 1.1

            if not (pullback or breakout):
                continue

            score = min(90, rps * 0.8 + (10 if pullback else 0) + (10 if breakout else 0) + (5 if vol_shrink else 0))
            reason = []
            if pullback:
                reason.append("回踩 20 日线")
            if breakout:
                reason.append("放量突破 20 日线")
            if vol_shrink:
                reason.append("缩量")
            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    code=code,
                    name=str(row.get(find_col(spot, ["名称", "name"]) or "", "")),
                    board=board,
                    trigger_reason=", ".join(reason) or "强趋势",
                    score=score,
                    raw_features={
                        "rps": round(rps, 2),
                        "rps_mode": mode,
                        "turnover_rate": turnover,
                        "close_to_ma20": round(prev_close / ma20 - 1, 4) if ma20 else None,
                    },
                )
            )
        return sorted(signals, key=lambda s: s.score, reverse=True)[:15]


# ---------------------------------------------------------------------------
# 4. 超跌反弹池
# ---------------------------------------------------------------------------
class OversoldReboundPool(StrategyPool):
    """跌深企稳、缩量十字星、止跌信号。"""

    name = "超跌反弹"

    def select(self, date: Optional[str] = None) -> list[StrategySignal]:
        date = date or pd.Timestamp.now().strftime("%Y%m%d")
        spot = self._market_spot()
        if spot.empty:
            return []
        pct_col = find_col(spot, ["涨跌幅"])
        if pct_col is None:
            return []

        codes = self._code_series(spot).unique()[:300]
        signals: list[StrategySignal] = []
        for code in codes:
            kline = self._daily_kline(code, n=60, date=date)
            if kline.empty or len(kline) < 30:
                continue
            close = pd.to_numeric(kline["close"], errors="coerce").dropna()
            high = pd.to_numeric(kline["high"], errors="coerce").dropna()
            low = pd.to_numeric(kline["low"], errors="coerce").dropna()
            vol = pd.to_numeric(kline["volume"], errors="coerce").dropna()
            if len(close) < 30:
                continue

            ret_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100
            if ret_20 > -15:
                continue
            # 缩量十字星 / 小阳线止跌
            last_body = abs(close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
            last_range = (high.iloc[-1] - low.iloc[-1]) / close.iloc[-2] * 100
            vol_shrink = vol.iloc[-1] < vol.iloc[-10:].mean() * 0.8
            hammer = last_range > 2 * last_body and close.iloc[-1] > (high.iloc[-1] + low.iloc[-1]) / 2
            if not (vol_shrink or hammer):
                continue

            rows = spot[spot[self._code_series(spot) == code]]
            if rows.empty:
                continue
            row = rows.iloc[0]
            board = self._board(code)
            score = min(88, abs(ret_20) * 1.5 + (10 if hammer else 0) + (5 if vol_shrink else 0))
            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    code=code,
                    name=str(row.get(find_col(spot, ["名称", "name"]) or "", "")),
                    board=board,
                    trigger_reason=f"20 日跌幅 {ret_20:.1f}%，" + ("锤子线" if hammer else "缩量企稳"),
                    score=score,
                    raw_features={"return_20d": round(ret_20, 2), "volume_shrink": vol_shrink, "hammer": hammer},
                )
            )
        return sorted(signals, key=lambda s: s.score, reverse=True)[:15]


# ---------------------------------------------------------------------------
# 5. 小盘优质股池
# ---------------------------------------------------------------------------
class SmallQualityPool(StrategyPool):
    """小市值 + 高换手 + 业绩不亏。"""

    name = "小盘优质"

    def select(self, date: Optional[str] = None) -> list[StrategySignal]:
        date = date or pd.Timestamp.now().strftime("%Y%m%d")
        cfg = self.cfg.get("guard", {})
        spot = self._market_spot()
        if spot.empty:
            return []
        mv_col = find_col(spot, ["流通市值", "总市值"])
        turnover_col = find_col(spot, ["换手率"])
        pct_col = find_col(spot, ["涨跌幅"])
        if not mv_col or not turnover_col:
            return []

        spot = spot.copy()
        spot["_code"] = self._code_series(spot)
        spot["_mv"] = pd.to_numeric(spot[mv_col], errors="coerce")
        spot["_turnover"] = pd.to_numeric(spot[turnover_col], errors="coerce")
        spot["_pct"] = pd.to_numeric(spot[pct_col], errors="coerce") if pct_col else 0.0

        max_mv = self.cfg.get("small_quality", {}).get("max_circ_mv_yi", 100)
        min_turnover = self.cfg.get("small_quality", {}).get("min_turnover", 2.0)
        candidates = spot[(spot["_mv"] <= max_mv * 1e8) & (spot["_turnover"] >= min_turnover)]

        signals: list[StrategySignal] = []
        for _, row in candidates.iterrows():
            code = row["_code"]
            # 过滤亏损：用财务指标（若无数据则不强制）
            finance_ok = True
            try:
                fin = self.dl.financial_indicator(code, date)
                if fin is not None and len(fin):
                    profit_col = find_col(fin, ["净利润", "归母净利润"])
                    if profit_col is not None:
                        profit = safe_float(fin.iloc[-1].get(profit_col))
                        if profit is not None and profit < 0:
                            finance_ok = False
            except Exception:  # noqa: BLE001
                pass

            if not finance_ok:
                continue

            score = 50 + (100 - row["_mv"] / 1e8) * 0.2 + row["_turnover"] * 2
            score = min(score, 85)
            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    code=code,
                    name=str(row.get(find_col(spot, ["名称", "name"]) or "", "")),
                    board=self._board(code),
                    trigger_reason=f"流通市值 {row['_mv']/1e8:.1f} 亿，换手 {row['_turnover']:.1f}%",
                    score=score,
                    raw_features={"circ_mv_yi": round(row["_mv"] / 1e8, 2), "turnover_rate": round(row["_turnover"], 2)},
                )
            )
        return sorted(signals, key=lambda s: s.score, reverse=True)[:15]


# ---------------------------------------------------------------------------
# 6. 大市值低波防守池（红利数据待接入）
# ---------------------------------------------------------------------------
class DividendLowVolPool(StrategyPool):
    """大市值 + 低波动。用于冰点/退潮期防守。

    注：当前尚未接入稳定股息率数据源，'红利'部分仅作占位。
    真实股息率接入后，再改名为'红利低波'。
    """

    name = "大市值低波"

    def select(self, date: Optional[str] = None) -> list[StrategySignal]:
        date = date or pd.Timestamp.now().strftime("%Y%m%d")
        cfg = self.cfg.get("dividend_low_vol", {})
        spot = self._market_spot()
        if spot.empty:
            return []
        mv_col = find_col(spot, ["总市值", "流通市值"])
        pct_col = find_col(spot, ["涨跌幅"])
        if not mv_col:
            return []

        spot = spot.copy()
        spot["_code"] = self._code_series(spot)
        spot["_mv"] = pd.to_numeric(spot[mv_col], errors="coerce")
        spot["_pct"] = pd.to_numeric(spot[pct_col], errors="coerce") if pct_col else pd.Series(0.0, index=spot.index)

        min_mv = cfg.get("min_mv_yi", 300)
        candidates = spot[spot["_mv"] >= min_mv * 1e8]

        signals: list[StrategySignal] = []
        for _, row in candidates.iterrows():
            code = row["_code"]
            # 波动率
            kline = self._daily_kline(code, n=60, date=date)
            if kline.empty or len(kline) < 30:
                continue
            close = pd.to_numeric(kline["close"], errors="coerce").dropna()
            vol = close.pct_change().dropna().std() * (252 ** 0.5) * 100
            if vol > cfg.get("max_volatility", 35):
                continue

            # 股息率（用财务表分红）
            dividend_yield = None
            try:
                fin = self.dl.financial_indicator(code, date)
                if fin is not None and len(fin):
                    # 简易估算：每股派息/股价
                    pass
            except Exception:  # noqa: BLE001
                pass

            score = min(80, 60 - vol + (dividend_yield or 0) * 2)
            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    code=code,
                    name=str(row.get(find_col(spot, ["名称", "name"]) or "", "")),
                    board=self._board(code),
                    trigger_reason=f"大市值，年化波动 {vol:.1f}%",
                    score=score,
                    raw_features={"market_cap_yi": round(row["_mv"] / 1e8, 2), "annualized_volatility": round(vol, 2)},
                )
            )
        return sorted(signals, key=lambda s: s.score, reverse=True)[:15]


# ---------------------------------------------------------------------------
# 7. 事件驱动池
# ---------------------------------------------------------------------------
class EventDrivenPool(StrategyPool):
    """公告/龙虎榜/新闻事件催化。

    当前覆盖：
    - 龙虎榜上榜及席位类型（机构/游资）
    - 巨潮公告关键字：业绩预增、回购、重大合同、减持、问询、立案、解禁等
    注：完整事件兑现程度判断依赖后续更系统的公告解析。
    """

    name = "事件驱动"

    POSITIVE_EVENTS = ("预增", "回购", "重大合同", "中标", "订单", "业绩快报", "股权激励", "增持")
    NEGATIVE_EVENTS = ("减持", "问询", "立案", "处罚", "亏损", "退市", "风险", "解禁", "澄清", "终止")

    def select(self, date: Optional[str] = None) -> list[StrategySignal]:
        date = date or pd.Timestamp.now().strftime("%Y%m%d")
        signals: list[StrategySignal] = []

        # 1. 龙虎榜
        try:
            lhb = self.dl.longhu_bang(date)
        except Exception:  # noqa: BLE001
            lhb = pd.DataFrame()
        if lhb is not None and not lhb.empty:
            lhb = lhb.copy()
            code_col = find_col(lhb, ["代码"])
            name_col = find_col(lhb, ["名称"])
            if code_col is not None:
                lhb["_code"] = lhb[code_col].astype(str).str.zfill(6)
                seen = set()
                for _, row in lhb.iterrows():
                    code = row["_code"]
                    if code in seen:
                        continue
                    seen.add(code)
                    # 简单席位判断：机构大买加分，机构大卖减分
                    dept = str(row.get(find_col(lhb, ["营业部", "席位"]) or "", ""))
                    is_institution = "机构专用" in dept
                    score = 75 if is_institution else 70
                    signals.append(
                        StrategySignal(
                            strategy_name=self.name,
                            code=code,
                            name=str(row.get(name_col, "")) if name_col else "",
                            board=self._board(code),
                            trigger_reason="龙虎榜" + ("机构上榜" if is_institution else "游资活跃"),
                            score=score,
                            raw_features={"source": "dragon_tiger", "institution": is_institution},
                        )
                    )

        # 2. 公告事件（基于当日涨停/异动股做少量补充，避免全市场扫公告过慢）
        try:
            spot = self._market_spot()
            if not spot.empty:
                pct_col = find_col(spot, ["涨跌幅"])
                if pct_col is not None:
                    hot_codes = spot[pd.to_numeric(spot[pct_col], errors="coerce").fillna(0) >= 5.0]
                    hot_codes = self._code_series(hot_codes).unique()[:50]
                    for code in hot_codes:
                        event_type, event_title = self._announcement_event(code)
                        if event_type:
                            name_col = find_col(spot, ["名称"])
                            name_rows = spot[spot[self._code_series(spot) == code]]
                            name = str(name_rows[name_col].iloc[0]) if name_col and not name_rows.empty else ""
                            score = 72 if event_type == "positive" else 40
                            signals.append(
                                StrategySignal(
                                    strategy_name=self.name,
                                    code=code,
                                    name=name,
                                    board=self._board(code),
                                    trigger_reason=f"公告事件: {event_title[:20]}",
                                    score=score,
                                    raw_features={"source": "announcement", "event_type": event_type, "event_title": event_title},
                                )
                            )
        except Exception:  # noqa: BLE001
            pass

        return sorted(signals, key=lambda s: s.score, reverse=True)[:15]

    def _announcement_event(self, code: str) -> tuple[Optional[str], str]:
        """读取最近一条公告，返回 (positive/negative/None, title)。"""
        try:
            rows = self._fetch_cninfo_announcements(code, page_size=5)
        except Exception:  # noqa: BLE001
            return None, ""
        if not rows:
            return None, ""
        # 取最新一条
        latest = rows[0]
        title = str(latest.get("title", ""))
        if any(k in title for k in self.NEGATIVE_EVENTS):
            return "negative", title
        if any(k in title for k in self.POSITIVE_EVENTS):
            return "positive", title
        return None, ""

    def _fetch_cninfo_announcements(self, code: str, page_size: int = 5) -> list[dict[str, Any]]:
        """简易巨潮公告抓取（无认证，可能被限流）。"""
        import requests
        org_id = self._cninfo_orgid(code)
        payload = {
            "stock": f"{code},{org_id}", "tabName": "fulltext",
            "pageSize": str(page_size), "pageNum": "1",
            "column": "", "category": "", "plate": "", "seDate": "",
            "searchkey": "", "secid": "", "sortName": "", "sortType": "",
            "isHLtitle": "true",
        }
        resp = requests.post(
            "https://www.cninfo.com.cn/new/hisAnnouncement/query",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": "https://www.cninfo.com.cn/new/disclosure"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        rows: list[dict[str, Any]] = []
        for item in data.get("announcements") or []:
            rows.append({
                "title": item.get("announcementTitle", ""),
                "type": item.get("announcementTypeName", ""),
                "date": str(item.get("announcementTime", ""))[:10],
            })
        return rows

    def _cninfo_orgid(self, code: str) -> str:
        if code.startswith("6"):
            return f"gssh0{code}"
        if code.startswith(("8", "4")):
            return f"gsbj0{code}"
        return f"gssz0{code}"


# ---------------------------------------------------------------------------
# 注册与调度
# ---------------------------------------------------------------------------
POOL_REGISTRY: dict[str, type[StrategyPool]] = {
    "题材龙头": ThemeLeaderPool,
    "连板梯队": LimitUpPool,
    "趋势回踩": TrendPullbackPool,
    "超跌反弹": OversoldReboundPool,
    "小盘优质": SmallQualityPool,
    "大市值低波": DividendLowVolPool,  # 红利数据接入前暂用此名
    "事件驱动": EventDrivenPool,
}


def list_pools() -> list[str]:
    return list(POOL_REGISTRY.keys())


def run_all_pools(dl: DataLoader, cfg: dict[str, Any] | None = None, date: Optional[str] = None) -> dict[str, list[StrategySignal]]:
    cfg = cfg or {}
    results: dict[str, list[StrategySignal]] = {}
    for name, cls in POOL_REGISTRY.items():
        try:
            results[name] = cls(dl, cfg).select(date)
        except Exception as exc:  # noqa: BLE001
            logger.exception("策略池 %s 运行失败", name)
            results[name] = []
    return results
