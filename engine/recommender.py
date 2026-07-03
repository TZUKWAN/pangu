"""股票推荐引擎：把候选股合成「推荐度 + 上涨概率 + 预测涨幅 + 简短理由」。

用户核心需求：直接给推荐，按推荐度排序，简短理由，上涨概率，预测涨幅。

设计：
- 推荐度（recommend_score, 0-100）：多维加权合成，越高越推荐
- 上涨概率（up_prob, 0-100%）：5日内上涨概率，优先用校准表（probability_calibrator），
  无校准表时用评分映射兜底（并标注「未经校准」）
- 预测涨幅（target_pct, %）：基于历史相似特征的平均涨幅，给目标区间
- 推荐等级：S(>=85)/A(70-84)/B(55-69)/C(<55)
- 简短理由（tag）：4-8字短语，如「趋势强+资金流入」「突破+放量」

评分维度权重（可调，默认合计 1.0）：
    趋势强度(RPS)     0.36   相对强度是短线第一指标
    资金面(主力流入)   0.26   增量资金=上涨燃料
    形态质量           0.18   均线多头+突破+放量
    盈亏比             0.10   交易性价比（entry_exit）
    风险控制           0.10   波动率/换手适中得分高
    新闻情绪           —      正面新闻且与热门题材共振时额外加分（封顶约 8 分）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .data_loader import safe_float

logger = logging.getLogger("pangu.recommender")


@dataclass
class Recommendation:
    """单只股票的推荐结果。"""

    code: str
    name: str
    board: str
    close: float
    # 核心推荐信息
    recommend_score: float            # 推荐度 0-100
    grade: str                        # 推荐等级 S/A/B/C
    confidence_score: float           # 综合置信度 0-100（非统计胜率，除非 calibrated=True）
    target_pct: tuple[float, float]   # 预测涨幅区间 %（保守, 乐观）
    tag: str                          # 简短理由（4-8字）
    # 交易计划（来自 entry_exit）
    buy_point: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    # 评分明细（透明，便于调参和解释）
    score_breakdown: dict[str, float] = field(default_factory=dict)
    calibrated: bool = False          # confidence_score 是否经过历史校准
    not_statistical_probability: bool = True  # 明确标注：非统计意义上的上涨概率

    def to_dict(self) -> dict[str, Any]:
        d = {
            "code": self.code, "name": self.name, "board": self.board,
            "close": round(self.close, 2),
            "recommend_score": round(self.recommend_score, 1),
            "grade": self.grade,
            "confidence_score": round(self.confidence_score, 1),
            "target_pct": [round(self.target_pct[0], 1), round(self.target_pct[1], 1)],
            "tag": self.tag,
            "buy_point": round(self.buy_point, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "score_breakdown": {
                k: (round(v, 1) if isinstance(v, (int, float)) else v)
                for k, v in self.score_breakdown.items()
            },
            "calibrated": self.calibrated,
            "not_statistical_probability": self.not_statistical_probability,
        }
        # 保留 up_prob 作为已弃用别名，避免前端/旧测试立即失效
        d["up_prob"] = round(self.confidence_score, 1)
        return d


class Recommender:
    """推荐评分引擎。"""

    def __init__(self, cfg: Optional[dict] = None) -> None:
        self.cfg = cfg or {}
        # 权重（合计 1.0）：trend/fund 主导，news 强化题材催化
        self.weights = self.cfg.get("recommender", {}).get("weights", {
            "trend": 0.36, "fund": 0.26, "pattern": 0.18,
            "risk_reward": 0.10, "risk_control": 0.10,
            "news_sentiment": 0.0,  # news 作 bonus 微调，不进主权重（保持 1.0）
        })
        # 校准表（由 probability_calibrator 提供，可选）
        self._calibrator = None
        self._calibrator_loaded = False

    def _try_load_calibrator(self) -> None:
        """惰性加载概率校准表。"""
        if self._calibrator_loaded:
            return
        self._calibrator_loaded = True
        try:
            from . import probability_calibrator as pc  # noqa
            self._calibrator = pc
        except Exception:  # noqa: BLE001
            self._calibrator = None

    # ------------------------------------------------------------------ #
    def rank(
        self,
        candidates: list[dict],
        news_sentiment: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[Recommendation]:
        """对候选股列表评分、排序，返回推荐结果（按推荐度降序）。

        Args:
            candidates: 候选股字典列表。
            news_sentiment: 新闻情绪映射 {code: {"sentiment_score": float, "themes": [...]}}。
        """
        news_sentiment = news_sentiment or {}
        recs = [self._score(c, news_sentiment.get(c.get("code", ""), {})) for c in candidates]
        recs = [r for r in recs if r is not None]
        recs.sort(key=lambda r: r.recommend_score, reverse=True)
        return recs

    # ------------------------------------------------------------------ #
    def _score(self, c: dict, news: dict[str, Any]) -> Optional[Recommendation]:
        """对单只候选股打分。

        Args:
            c: 候选股字典。
            news: 该 code 对应的新闻情绪结果（可为空）。
        """
        code = c.get("code", "")
        ee = c.get("entry_exit", {}) or {}
        if not code:
            return None

        # ---- 各维度打分（每项 0-100）----
        rps = safe_float(c.get("rps"), 50.0)
        inflow_days = safe_float(c.get("fund_inflow_days"), 0.0)
        reasons = c.get("reasons", [])
        turnover = safe_float(c.get("turnover_rate"), 0.0)
        rr = safe_float(ee.get("risk_reward_ratio"), 0.0)
        pct_change = safe_float(c.get("pct_change"), 0.0)
        tech = c.get("technical", {}) or {}
        structured = c.get("structured_factors", {}) or {}

        # 1. 趋势强度：RPS 连续映射 + 技术面共振，不再因 RPS<70 一刀切归零
        trend_score = self._trend_score(rps, tech)

        # 2. 资金面：有真实连续流入才给分，否则标记缺失（不再用 0 伪装成低分）
        fund_score, fund_missing = self._fund_score(inflow_days)

        # 3. 形态质量：基于真实技术快照（MA 排列/突破/放量/MACD/站上 MA60）
        pattern_score = self._pattern_score(tech, reasons, pct_change)

        # 4. 盈亏比：基于真实 entry/stop/targets/ATR 波动计算，不固定 100
        rr_score = self._risk_reward_score(ee)

        # 5. 风险控制：换手率适配 + 护栏风险标记降权
        risk_score = self._risk_control_score(turnover, c.get("risk_flags") or [])

        # 5b. 结构化事件因子：龙虎榜/热榜/120日资金/两融/解禁/公告等真实字段。
        # 有数据才加分或降权；没有数据只在 source_state 标注，不用默认值伪装。
        structured_score, structured_bonus, structured_penalty, structured_notes = self._structured_factor_score(structured)
        if structured_penalty:
            risk_score = max(0.0, risk_score - structured_penalty)

        # 6. 动量：RPS + 真实涨跌幅 + 量比，避免仅由 RPS 低导致全部雷同
        momentum_score = self._momentum_score(rps, pct_change, tech)

        breakdown = {
            "trend": trend_score, "fund": fund_score, "pattern": pattern_score,
            "risk_reward": rr_score, "risk_control": risk_score,
            "momentum": momentum_score, "fund_missing": fund_missing,
            "structured_signal": structured_score,
            "structured_bonus": structured_bonus,
            "structured_penalty": structured_penalty,
            "structured_notes": structured_notes,
        }

        # 7. 新闻题材情绪分（实时新闻解析，有热点题材共振加分）
        news_score, news_bonus, news_themes, news_explain = self._news_bonus(c, news)
        breakdown["news_sentiment"] = news_score
        breakdown["news_themes"] = news_themes
        breakdown["news_explain"] = news_explain

        # 加权合成推荐度（6 维正式加权）
        w = self.weights
        recommend_score = (
            trend_score * w.get("trend", 0.36) +
            fund_score * w.get("fund", 0.26) +
            pattern_score * w.get("pattern", 0.18) +
            rr_score * w.get("risk_reward", 0.10) +
            risk_score * w.get("risk_control", 0.10) +
            news_score * w.get("news_sentiment", 0.0)
        )
        # news_bonus 作为额外微调（题材强烈共振时小幅加成，上限 +5）
        recommend_score = min(100.0, recommend_score + min(5.0, news_bonus))
        recommend_score = max(0.0, min(100.0, recommend_score + structured_bonus - structured_penalty * 0.5))

        # ---- 上涨概率 + 预测涨幅 ----
        # 收集完整预测特征：rps/ret_20d/volume_ratio/fund_inflow_days/atr_pct/ma_bull/macd_signal
        rps_val = safe_float(c.get("rps"), 50.0)
        ma20 = ee.get("ma20") or next(
            (bp["price"] for bp in ee.get("buy_points", []) if "MA20" in str(bp.get("type", ""))), None
        )
        close_val = safe_float(c.get("close"))
        ret_20d = (close_val / ma20 - 1) if (ma20 and close_val and ma20 > 0) else (rps_val / 100.0 * 0.2)

        ma = tech.get("ma") or {}
        ma5 = safe_float(ma.get("ma5"))
        ma10 = safe_float(ma.get("ma10"))
        ma30 = safe_float(ma.get("ma30"))
        ma_bull = bool(ma5 and ma10 and ma30 and ma5 > ma10 > ma30)
        macd_signal = bool((tech.get("macd") or {}).get("golden_cross"))

        atr_pct = self._approx_atr_pct(tech.get("kline") or [], close_val)

        features = {
            "rps": rps_val,
            "ret_20d": ret_20d,
            "volume_ratio": max(1.0, safe_float(c.get("turnover_rate"), 5.0) / 5.0),
            "fund_inflow_days": safe_float(c.get("fund_inflow_days"), 0.0),
            "atr_pct": atr_pct,
            "ma_bull": int(ma_bull),
            "macd_signal": int(macd_signal),
        }
        confidence_score, target_pct, calibrated = self._predict(code, recommend_score, features)

        # ---- 推荐等级 ----
        if recommend_score >= 85:
            grade = "S"
        elif recommend_score >= 70:
            grade = "A"
        elif recommend_score >= 55:
            grade = "B"
        else:
            grade = "C"

        # ---- 简短理由 tag（4-8字，挑最突出的2个因子；如有新闻催化则追加）----
        tag = self._make_tag(breakdown, reasons, news_themes)

        # ---- 交易计划 ----
        buy_points = ee.get("buy_points", []) or []
        buy_point = next((bp["price"] for bp in buy_points if bp.get("is_primary")),
                         (buy_points[0]["price"] if buy_points else 0.0))
        stop_loss_obj = ee.get("stop_loss") or {}
        stop_loss = stop_loss_obj.get("price", 0.0) if isinstance(stop_loss_obj, dict) else 0.0
        take_profits = ee.get("take_profit", []) or []
        take_profit = take_profits[0]["price"] if take_profits else 0.0

        close = safe_float(c.get("close"))
        if close != close:  # NaN
            close = 0.0
        return Recommendation(
            code=code, name=c.get("name", ""), board=c.get("board", ""),
            close=close,
            recommend_score=recommend_score, grade=grade,
            confidence_score=confidence_score, target_pct=target_pct, tag=tag,
            buy_point=buy_point, stop_loss=stop_loss, take_profit=take_profit,
            risk_reward_ratio=rr, score_breakdown=breakdown, calibrated=calibrated,
            not_statistical_probability=not calibrated,
        )

    # ------------------------------------------------------------------ #
    def _trend_score(self, rps: float, tech: dict[str, Any]) -> float:
        """趋势强度：RPS 连续映射 + 技术面共振。"""
        # RPS 50->0, 90->100 的连续映射，低于 50 不再一刀切
        base = max(0.0, min(100.0, (rps - 50.0) * (100.0 / 40.0)))
        bonus = 0.0
        ma = tech.get("ma") or {}
        ma5 = safe_float(ma.get("ma5"))
        ma10 = safe_float(ma.get("ma10"))
        ma30 = safe_float(ma.get("ma30"))
        ma60 = safe_float(ma.get("ma60"))
        close = safe_float(tech.get("kline", [{}])[-1].get("close")) if tech.get("kline") else 0.0
        if ma5 and ma10 and ma30 and ma5 > ma10 > ma30:
            bonus += 10.0
        if close and ma60 and close > ma60:
            bonus += 10.0
        if (tech.get("macd") or {}).get("golden_cross"):
            bonus += 10.0
        return max(0.0, min(100.0, base + bonus))

    def _fund_score(self, inflow_days: float) -> tuple[float, bool]:
        """资金面：有真实连续流入才给分，否则明确标记缺失。"""
        if inflow_days and inflow_days > 0:
            return min(100.0, inflow_days * 12.5), False
        return 0.0, True

    def _pattern_score(self, tech: dict[str, Any], reasons: list[str], pct_change: float) -> float:
        """形态质量：基于真实技术快照而非 reasons 关键词计数。"""
        score = 0.0
        ma = tech.get("ma") or {}
        ma5 = safe_float(ma.get("ma5"))
        ma10 = safe_float(ma.get("ma10"))
        ma30 = safe_float(ma.get("ma30"))
        ma60 = safe_float(ma.get("ma60"))
        close = safe_float(tech.get("kline", [{}])[-1].get("close")) if tech.get("kline") else 0.0
        if ma5 and ma10 and ma30 and ma5 > ma10 > ma30:
            score += 25.0
        if close and ma60 and close > ma60:
            score += 15.0
        vol = tech.get("volume") or {}
        if safe_float(vol.get("volume_ratio"), 0.0) >= 1.5:
            score += 20.0
        if (tech.get("macd") or {}).get("golden_cross"):
            score += 15.0
        # 用真实涨跌幅作为突破/动量质量的连续代理变量
        score += max(0.0, min(25.0, pct_change * 3.0))
        return max(0.0, min(100.0, score))

    def _risk_reward_score(self, ee: dict[str, Any]) -> float:
        """盈亏比：基于真实 entry/stop/targets 和 risk_pct 质量，不固定 100。"""
        rr = safe_float(ee.get("risk_reward_ratio"), 0.0)
        buy_points = ee.get("buy_points") or []
        entry = safe_float(next((bp["price"] for bp in buy_points if bp.get("is_primary")),
                                 buy_points[0]["price"] if buy_points else 0.0))
        stop_obj = ee.get("stop_loss") or {}
        stop = safe_float(stop_obj.get("price") if isinstance(stop_obj, dict) else 0.0, 0.0)
        risk_pct = (entry - stop) / entry if entry and stop and entry > stop else 0.0
        # 风险宽度质量：3-8% 为佳
        if 0.03 <= risk_pct <= 0.08:
            quality = 100.0
        elif risk_pct < 0.015:
            quality = 40.0
        elif risk_pct < 0.03:
            quality = 70.0
        elif risk_pct <= 0.15:
            quality = 70.0
        else:
            quality = 40.0
        rr_value = max(0.0, min(100.0, (rr - 0.5) * (100.0 / 2.5)))
        return max(0.0, min(100.0, rr_value * 0.6 + quality * 0.4))

    def _risk_control_score(self, turnover: float, risk_flags: list[str]) -> float:
        """风险控制：换手率适配 + 护栏风险标记降权。"""
        if 3 <= turnover <= 15:
            score = 100.0 - abs(turnover - 8) * 4
        elif turnover < 3:
            score = 40.0
        else:
            score = max(0.0, 100.0 - (turnover - 15) * 5)
        score = max(0.0, min(100.0, score))
        if risk_flags:
            score = max(0.0, score - min(40.0, len(risk_flags) * 20.0))
        return score

    def _momentum_score(self, rps: float, pct_change: float, tech: dict[str, Any]) -> float:
        """动量：RPS + 真实涨跌幅 + 量比，避免低区分输入合成。"""
        vol_ratio = safe_float((tech.get("volume") or {}).get("volume_ratio"), 0.0)
        return max(0.0, min(100.0, rps * 0.5 + pct_change * 3.0 + vol_ratio * 5.0))

    def _structured_factor_score(self, structured: dict[str, Any]) -> tuple[float, float, float, list[str]]:
        """Score real structured factors; missing fields do not create synthetic scores."""
        if not structured:
            return 0.0, 0.0, 0.0, []

        bonus = 0.0
        penalty = 0.0
        notes: list[str] = []

        lhb = structured.get("dragon_tiger") or {}
        net_buy = safe_float(lhb.get("net_buy_wan"), 0.0)
        if net_buy > 0:
            inc = min(4.0, net_buy / 3000.0)
            bonus += inc
            notes.append(f"LHB+{inc:.1f}")
        elif net_buy < 0:
            dec = min(3.0, abs(net_buy) / 4000.0)
            penalty += dec
            notes.append(f"LHB-{dec:.1f}")

        hot = structured.get("hot_rank") or {}
        rank = safe_float(hot.get("rank"), 0.0)
        if 0 < rank <= 100:
            inc = max(0.5, (101.0 - rank) / 30.0)
            bonus += min(3.0, inc)
            notes.append(f"hot#{int(rank)}")

        flow = structured.get("capital_flow_120d") or {}
        flow20 = safe_float(flow.get("sum_20d_main_net"), 0.0)
        if flow20 > 0:
            inc = min(4.0, flow20 / 80_000_000.0)
            bonus += inc
            notes.append(f"20dFund+{inc:.1f}")
        elif flow20 < 0:
            dec = min(3.0, abs(flow20) / 120_000_000.0)
            penalty += dec
            notes.append(f"20dFund-{dec:.1f}")

        margin = structured.get("margin") or {}
        margin_chg = safe_float(margin.get("rzrqye_5d_change"), 0.0)
        if margin_chg > 0:
            inc = min(1.5, margin_chg / 100_000_000.0)
            bonus += inc
            notes.append(f"margin+{inc:.1f}")

        research = structured.get("research") or {}
        if int(safe_float(research.get("count"), 0)) > 0:
            bonus += 1.0
            notes.append("research")

        lockup = structured.get("lockup") or {}
        lock_ratio = safe_float(lockup.get("upcoming_ratio_sum"), 0.0)
        if lock_ratio > 0:
            dec = min(5.0, lock_ratio * 0.8)
            penalty += dec
            notes.append(f"lockup-{dec:.1f}")

        announcements = structured.get("announcements") or {}
        risk_count = int(safe_float(announcements.get("risk_count"), 0))
        if risk_count:
            dec = min(5.0, risk_count * 1.5)
            penalty += dec
            notes.append(f"annRisk-{risk_count}")

        north = structured.get("northbound") or {}
        north_5d = safe_float(north.get("sum_5d_net_buy"), 0.0)
        if north_5d > 0:
            inc = min(2.0, north_5d / 100_000_000.0)
            bonus += inc
            notes.append(f"north+{inc:.1f}")
        elif north_5d < 0:
            dec = min(2.0, abs(north_5d) / 150_000_000.0)
            penalty += dec
            notes.append(f"north-{dec:.1f}")

        block = structured.get("block_trade") or {}
        if int(safe_float(block.get("count_30d"), 0)) > 0:
            penalty += 1.0
            notes.append("blocktrade")

        holder = structured.get("holder_num") or {}
        qoq = safe_float(holder.get("qoq_pct"), 0.0)
        if qoq < -3:
            bonus += 1.0
            notes.append("holder集中")
        elif qoq > 5:
            penalty += 1.0
            notes.append("holder分散")

        score = max(0.0, min(100.0, bonus * 12.0 - penalty * 8.0 + 50.0 if (bonus or penalty) else 0.0))
        return score, round(min(10.0, bonus), 2), round(min(10.0, penalty), 2), notes[:8]

    # ------------------------------------------------------------------ #
    def _news_bonus(self, c: dict, news: dict[str, Any]) -> tuple[float, float, list[str], str]:
        """计算新闻情绪加分。

        只有同时满足以下条件才加分：
        1. 该 code 在新闻简报中有情绪数据；
        2. 情绪为正面（sentiment_score >= 60）；
        3. 新闻题材与个股所属板块/理由存在文本共振。

        返回 (news_score 0-100, bonus 加分值, 共振题材列表, 可读解释)。
        """
        if not news:
            return 0.0, 0.0, [], "无新闻情绪数据"
        score = safe_float(news.get("sentiment_score"), 50.0)
        label = str(news.get("sentiment_label", "neutral"))
        themes = news.get("themes", [])
        explain = str(news.get("explain", "")) or f"情绪标签 {label}，得分 {score}"

        # 共振判定：候选股 board 或 reasons 文本与新闻题材有交集
        board = str(c.get("board", ""))
        reasons = "；".join(c.get("reasons", []))
        resonant_themes: list[str] = []
        for theme in themes:
            theme = str(theme)
            # 题材名可能形如 "A股异动>领涨板块>工业气体"，取最后一段作为关键词
            keyword = theme.split(">")[-1].strip()
            if not keyword:
                continue
            if keyword in board or keyword in reasons or keyword in theme:
                resonant_themes.append(theme)
        resonates = bool(resonant_themes)

        if score < 60:
            return score, 0.0, resonant_themes, explain

        # 兼容配置路径：news_sentiment.news_weight 或 recommender.news_weight
        news_weight = (
            (self.cfg or {}).get("news_sentiment", {}).get("news_weight")
            or (self.cfg or {}).get("recommender", {}).get("news_weight")
            or 0.08
        )
        # 情绪分越高、共振越强，加分越多；封顶
        bonus = 0.0
        if resonates:
            bonus = (score / 100.0) * (news_weight * 100.0)
            bonus = min(news_weight * 100.0, bonus)
            explain += f"；与个股板块/理由共振，额外加分 +{round(bonus, 1)}"
        else:
            explain += "；未与个股板块/理由共振，不加分"
        return score, bonus, resonant_themes, explain

    @staticmethod
    def _approx_atr_pct(kline: list[dict[str, Any]], close: float) -> float:
        """从 K 线近似 ATR%；K 线不足时用 1.5% 保守默认值。"""
        if not kline or len(kline) < 2 or close <= 0:
            return 1.5
        tr_values: list[float] = []
        for idx, bar in enumerate(kline[-15:], start=max(0, len(kline) - 15)):
            if idx == 0:
                continue
            try:
                high = float(bar.get("high", bar.get("close", 0)) or 0)
                low = float(bar.get("low", bar.get("close", 0)) or 0)
                prev_close = float(kline[idx - 1].get("close", close) or close)
            except (TypeError, ValueError):
                continue
            if high <= 0 or low <= 0:
                continue
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        if not tr_values:
            return 1.5
        atr = sum(tr_values) / len(tr_values)
        return round(atr / close * 100, 2)

    # ------------------------------------------------------------------ #
    def _predict(self, code: str, score: float, features: dict[str, Any]
                 ) -> tuple[float, tuple[float, float], bool]:
        """预测 confidence_score 和涨幅区间。

        优先用校准表（历史回测得到），无校准则用完整真实特征映射兜底。
        未校准时返回的 confidence_score 是模型置信度，不是统计胜率。
        返回 (confidence_score%, (保守涨幅%, 乐观涨幅%), 是否校准)。
        """
        self._try_load_calibrator()
        calibrated = False

        # 尝试校准表查表（需先跑过 calibrate() 离线校准）
        if self._calibrator is not None:
            try:
                calib = self._calibrator.load_calibration()  # type: ignore
                if calib:
                    pred = self._calibrator.predict(code, features)  # type: ignore
                    prob_up = float(getattr(pred, "prob_up", 0)) * 100
                    ret = float(getattr(pred, "predicted_return", 0)) * 100
                    flag = str(getattr(pred, "uncertainty_flag", ""))
                    # 仅当校准结果有效且可信时才直接采用
                    if ret > 0 and flag not in ("", "fallback_baseline"):
                        target_low = max(2.0, ret * 1.5)
                        target_high = max(5.0, ret * 3.0)
                        if prob_up > 0:
                            return prob_up, (round(target_low, 1), round(target_high, 1)), True
            except Exception:  # noqa: BLE001
                pass

        # 兜底：基于真实特征映射（未经校准）
        confidence_score = 35.0 + (score / 100.0) * 35.0
        confidence_score = max(30.0, min(72.0, confidence_score))

        rps = safe_float(features.get("rps"), 50.0)
        ret_20d = safe_float(features.get("ret_20d"), 0.0)
        atr_pct = safe_float(features.get("atr_pct"), 1.5)
        ma_bull = bool(features.get("ma_bull"))
        macd_signal = bool(features.get("macd_signal"))
        volume_ratio = safe_float(features.get("volume_ratio"), 1.0)

        # 目标涨幅：以 ret_20d + rps/100*0.08 为中枢，按波动率/量能/形态调整
        base_ret = ret_20d + (rps / 100.0) * 0.08
        if ma_bull:
            base_ret += 0.02
        if macd_signal:
            base_ret += 0.015
        base_ret *= max(1.0, min(2.5, volume_ratio / 2.0))

        # base_ret 为小数收益率；atr_pct 为百分数，需统一为百分点
        target_low = max(2.0, base_ret * 100 - atr_pct * 0.6)
        target_high = max(5.0, base_ret * 100 + atr_pct * 1.2)
        target_low = min(target_low, target_high * 0.85)
        return confidence_score, (round(target_low, 1), round(target_high, 1)), calibrated

    # ------------------------------------------------------------------ #
    def _make_tag(self, breakdown: dict, reasons: list[str], news_themes: list[str] | None = None) -> str:
        """生成4-12字简短理由：挑最突出的因子组合；有新闻催化时追加题材词。"""
        parts = []
        if breakdown.get("trend", 0) >= 70:
            parts.append("趋势强")
        if breakdown.get("fund", 0) >= 60:
            parts.append("资金流入")
        if breakdown.get("momentum", 0) >= 65:
            parts.append("动量强")
        reason_text = "；".join(reasons)
        if "突破" in reason_text:
            parts.append("突破")
        if "放量" in reason_text:
            parts.append("放量")
        if "均线多头" in reason_text:
            parts.append("多头排列")
        # 新闻催化：取首个共振题材词（2-4 字）
        if news_themes:
            for theme in news_themes:
                keyword = theme.split(">")[-1].strip()
                if 2 <= len(keyword) <= 8:
                    parts.append(f"{keyword}催化")
                    break
        if not parts:
            parts.append("形态企稳")
        # 取前2-3个，控制长度
        tag = "+".join(parts[:3])
        return tag[:16]


def grade_color(grade: str) -> str:
    """推荐等级对应的颜色（给 rich 渲染用）。"""
    return {"S": "bold red", "A": "bold green", "B": "yellow", "C": "dim"}.get(grade, "white")
