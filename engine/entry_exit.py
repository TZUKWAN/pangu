"""买卖点自动计算模块：把趋势候选池升级为可落地的交易计划。

解决原系统的核心缺口：选股引擎只输出「候选股 + 入选理由」，没有自动化的
买点 / 止损 / 止盈 / 仓位。本模块基于 akshare 日 K，为每只个股计算结构化
交易计划，供 pipeline 直接写入 JSON，也供 LLM 做最终综合时引用。

计算清单：
- 买点：突破位（前 N 日高点）、回踩位（MA5/MA10/MA20）、支撑位（近期低点）。
- 止损：ATR 止损（2×ATR）、结构止损（跌破前低）、MA20 止损（趋势底线）。
- 止盈：盈亏比目标（2:1 / 3:1）、前高压力位、MA10 跟踪止盈参考。
- 仓位：1% 风险法则，按情绪温度动态调整仓位系数。

设计原则：
1. 可解释：每个价格都写明计算方法和触发条件。
2. 可回测：所有价格来自历史 K 线公开指标，无黑盒。
3. 可降级：K 线缺失时返回带 warning 的空计划，不阻断 pipeline。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader, safe_float, find_col as _find_col
from .trend_scanner import StockCandidate

logger = logging.getLogger("pangu.entry_exit")


# ---------------------------------------------------------------------- #
# 数据结构
# ---------------------------------------------------------------------- #
@dataclass
class BuyPoint:
    """单个买点计划。"""

    price: float
    type: str          # 突破位 / 回踩位 / 支撑位
    condition: str     # 触发条件描述
    is_primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": round(self.price, 2),
            "type": self.type,
            "condition": self.condition,
            "is_primary": self.is_primary,
        }


@dataclass
class StopLoss:
    """止损计划。"""

    price: float
    method: str        # ATR止损 / 结构止损 / MA20止损 / fallback

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": round(self.price, 2),
            "method": self.method,
        }


@dataclass
class TakeProfit:
    """止盈目标。"""

    price: float
    method: str        # 盈亏比2:1 / 盈亏比3:1 / 前高压力 / 跟踪止盈

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": round(self.price, 2),
            "method": self.method,
        }


@dataclass
class PositionPlan:
    """仓位计划。"""

    shares: int                # 建议股数（已按 100 股取整）
    risk_pct: float            # 实际承担账户风险 %
    emotion_factor: float      # 情绪温度系数
    account_size: float        # 账户规模
    base_risk_pct: float       # 基础风险 %（默认 1%）

    def to_dict(self) -> dict[str, Any]:
        return {
            "shares": self.shares,
            "risk_pct": round(self.risk_pct, 2),
            "emotion_factor": round(self.emotion_factor, 2),
            "account_size": round(self.account_size, 2),
            "base_risk_pct": round(self.base_risk_pct, 2),
        }


@dataclass
class EntryExitResult:
    """单只候选股的完整买卖点方案。"""

    code: str
    name: str
    close: float
    buy_points: list[BuyPoint] = field(default_factory=list)
    stop_loss: Optional[StopLoss] = None
    take_profit: list[TakeProfit] = field(default_factory=list)
    trailing_stop: Optional[TakeProfit] = None
    position: Optional[PositionPlan] = None
    risk_reward_ratio: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "close": round(self.close, 2),
            "buy_points": [b.to_dict() for b in self.buy_points],
            "stop_loss": self.stop_loss.to_dict() if self.stop_loss else None,
            "take_profit": [t.to_dict() for t in self.take_profit],
            "trailing_stop": self.trailing_stop.to_dict() if self.trailing_stop else None,
            "position": self.position.to_dict() if self.position else None,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------- #
# 主引擎
# ---------------------------------------------------------------------- #
class EntryExitEngine:
    """买卖点计算引擎。"""

    def __init__(self, dl: DataLoader, cfg: dict[str, Any] | None = None) -> None:
        self.dl = dl
        self.cfg = cfg or {}

        ecfg = self.cfg.get("entry_exit", {})
        self.atr_period: int = ecfg.get("atr_period", 14)
        self.breakout_lookback: int = ecfg.get("breakout_lookback", 20)
        self.support_lookback: int = ecfg.get("support_lookback", 10)
        self.resistance_lookback: int = ecfg.get("resistance_lookback", 60)
        self.ma_periods: list[int] = ecfg.get("ma_periods", [5, 10, 20])
        self.ma20_stop_buffer: float = ecfg.get("ma20_stop_buffer", 0.03)
        self.min_stop_pct: float = ecfg.get("min_stop_pct", 0.015)
        self.min_risk_pct: float = ecfg.get("min_risk_pct", 0.005)
        self.account_size: float = float(ecfg.get("account_size", 1_000_000))
        self.base_risk_pct: float = ecfg.get("base_risk_pct", 0.01)
        self.atr_multiplier: float = ecfg.get("atr_multiplier", 2.0)
        self.min_trade_amount: float = float(ecfg.get("min_trade_amount", 10_000))
        self.min_shares: int = ecfg.get("min_shares", 100)

    # ------------------------------------------------------------------ #
    def compute(
        self,
        candidate: StockCandidate | dict[str, Any],
        temperature: float = 50.0,
        account_size: float | None = None,
    ) -> EntryExitResult:
        """为单只候选股计算买卖点方案。

        Args:
            candidate: StockCandidate 或字典（必须含 code/name/close）。
            temperature: 情绪温度 0-100，用于仓位系数。
            account_size: 账户规模，None 则用配置默认值。
        """
        code, name, close = self._extract_candidate(candidate)
        res = EntryExitResult(code=code, name=name, close=close)
        account = account_size if account_size is not None else self.account_size

        # 取日 K；数据不足则直接返回降级结果
        kline = self._load_kline(code)
        if kline is None or len(kline) < max(self.ma_periods) + 2:
            res.warnings.append("日 K 数据不足，无法计算买卖点")
            return res

        try:
            highs = _numeric_series(kline, ["最高"], default_col=2)
            lows = _numeric_series(kline, ["最低"], default_col=3)
            closes = _numeric_series(kline, ["收盘"], default_col=4)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s K 线列解析失败: %s", code, e)
            res.warnings.append("K 线列解析失败")
            return res

        if len(closes) < max(self.ma_periods) + 2:
            res.warnings.append("有效收盘数据不足")
            return res

        # 核心指标
        ma_map = {n: _ma(closes, n) for n in self.ma_periods}
        atr = _atr(highs, lows, closes, self.atr_period)
        recent_high = _recent_high(highs, self.breakout_lookback)
        recent_low = _recent_low(lows, self.support_lookback)
        resistance = _recent_high(highs, self.resistance_lookback)

        if any(pd.isna(v) for v in [recent_high, recent_low, resistance]):
            res.warnings.append("高低点计算异常")
            return res

        # 买点：主买点 + 备选
        buy_points = self._build_buy_points(close, recent_high, recent_low, ma_map)
        res.buy_points = buy_points
        primary = next((b for b in buy_points if b.is_primary), buy_points[0])

        # 止损：三种方法动态选最合理的
        stop_loss = self._build_stop_loss(
            entry=primary.price,
            ma20=ma_map.get(20, close * 0.95),
            atr=atr,
            structure_low=recent_low,
        )
        res.stop_loss = stop_loss

        # 止盈：盈亏比目标 + 前高压力 + 跟踪止损参考
        res.take_profit, res.trailing_stop = self._build_take_profit(
            entry=primary.price,
            stop=stop_loss.price,
            resistance=resistance,
            ma10=ma_map.get(10, close),
        )

        # 仓位：1% 风险法则 × 情绪系数
        res.position = self._build_position(
            entry=primary.price,
            stop=stop_loss.price,
            temperature=temperature,
            account_size=account,
        )

        # 风险收益比（用 target1）
        if res.take_profit:
            target1 = res.take_profit[0].price
            risk = primary.price - stop_loss.price
            if risk > 0:
                res.risk_reward_ratio = (target1 - primary.price) / risk

        return res

    # ------------------------------------------------------------------ #
    def compute_batch(
        self,
        candidates: list[StockCandidate] | list[dict[str, Any]],
        temperature: float = 50.0,
        account_size: float | None = None,
    ) -> list[EntryExitResult]:
        """批量计算买卖点。"""
        return [
            self.compute(c, temperature=temperature, account_size=account_size)
            for c in candidates
        ]

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _extract_candidate(
        self, candidate: StockCandidate | dict[str, Any]
    ) -> tuple[str, str, float]:
        """统一从 StockCandidate 或字典提取基础字段。"""
        if isinstance(candidate, StockCandidate):
            return candidate.code, candidate.name, candidate.close
        return (
            str(candidate.get("code", "")),
            str(candidate.get("name", "")),
            safe_float(candidate.get("close"), 0.0),
        )

    def _load_kline(self, code: str) -> Optional[pd.DataFrame]:
        """加载日 K，失败返回 None（上层已做降级）。"""
        try:
            days = max(self.resistance_lookback, self.breakout_lookback) + self.atr_period + 10
            return self.dl.daily_kline(code, days=days)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s 取日 K 失败: %s", code, e)
            return None

    def _build_buy_points(
        self,
        close: float,
        recent_high: float,
        recent_low: float,
        ma_map: dict[int, Optional[float]],
    ) -> list[BuyPoint]:
        """构建买点列表，并标记主买点。"""
        points: list[BuyPoint] = []

        # 突破位：前 N 日高点（不含当日）
        points.append(BuyPoint(
            price=recent_high,
            type="突破位",
            condition=f"收盘站稳前 {self.breakout_lookback} 日高点 {recent_high:.2f}",
        ))

        # 回踩位：按 MA5→MA10→MA20 顺序，取低于现价且有效的均线
        for period in sorted(ma_map.keys()):
            ma = ma_map[period]
            if ma is None or pd.isna(ma) or ma <= 0:
                continue
            if ma < close * 1.005:  # 允许轻微上穿后回踩
                points.append(BuyPoint(
                    price=ma,
                    type="回踩位",
                    condition=f"回踩 MA{period} 不破位（{ma:.2f}）",
                ))

        # 支撑位：近期低点，作为低吸备选
        points.append(BuyPoint(
            price=recent_low,
            type="支撑位",
            condition=f"回踩近 {self.support_lookback} 日低点 {recent_low:.2f} 低吸",
        ))

        # 选主买点：短线策略只把突破位作为触发确认，不作为默认追价买点。
        # 1. 优先选择现价下方 0-8% 的回踩位；
        # 2. 其次选择现价下方 0-12% 的支撑位低吸；
        # 3. 再退一步选择最近的有效回踩/支撑；
        # 4. 只有完全没有低吸结构时才保留突破位，后续玄武池会拦截追价计划。
        pullback_points = [p for p in points if p.type == "回踩位"]
        actionable_pullbacks = [
            p for p in pullback_points
            if close * 0.92 <= p.price <= close * 1.005
        ]
        support = points[-1]
        actionable_support = support if close * 0.88 <= support.price <= close * 1.002 else None

        if actionable_pullbacks:
            primary = max(actionable_pullbacks, key=lambda p: p.price)
        elif actionable_support is not None:
            primary = actionable_support
        elif pullback_points:
            primary = max([p for p in pullback_points if p.price > 0], key=lambda p: p.price)
        elif support.price > 0:
            primary = support
        else:
            primary = points[0]

        for p in points:
            p.is_primary = (p is primary)

        return points

    def _build_stop_loss(
        self,
        entry: float,
        ma20: Optional[float],
        atr: Optional[float],
        structure_low: float,
    ) -> StopLoss:
        """选择最宽且满足最小止损距离的止损价。

        原则：止损必须给足波动空间，避免被正常波动扫出；
        同时不得低于 min_stop_pct 的硬性距离，防止止损过宽失控。
        """
        candidates: list[tuple[float, str]] = []

        # ATR 止损：entry - atr_multiplier×ATR
        if atr is not None and not pd.isna(atr) and atr > 0:
            candidates.append((
                entry - self.atr_multiplier * atr,
                f"ATR止损({self.atr_multiplier}×ATR={atr:.2f})",
            ))

        # 结构止损：近期低点
        if structure_low > 0:
            candidates.append((structure_low, f"结构止损(近{self.support_lookback}日低点)"))

        # MA20 止损：MA20 下方 buffer
        if ma20 is not None and not pd.isna(ma20) and ma20 > 0:
            candidates.append((ma20 * (1 - self.ma20_stop_buffer), "MA20趋势止损"))

        # 过滤：必须低于买点且价格有效
        valid = [(p, m) for p, m in candidates if 0 < p < entry]
        if not valid:
            # 无有效止损时给一个保守 fallback
            price = entry * (1 - self.min_stop_pct)
            return StopLoss(price=price, method="fallback(买入价下方固定比例)")

        # 选「最宽（离买点最远）且满足 min_stop_pct」的止损
        min_stop_price = entry * (1 - self.min_stop_pct)
        wide_enough = [(p, m) for p, m in valid if p <= min_stop_price]
        if wide_enough:
            # 最宽 = 价格最低
            chosen = min(wide_enough, key=lambda x: x[0])
        else:
            # 没有满足最小距离的，fallback 到硬性最小止损
            chosen = (min_stop_price, "fallback(买入价下方固定比例)")

        return StopLoss(price=chosen[0], method=chosen[1])

    def _build_take_profit(
        self,
        entry: float,
        stop: float,
        resistance: float,
        ma10: float,
    ) -> tuple[list[TakeProfit], TakeProfit]:
        """构建止盈目标与跟踪止盈参考。

        原则：盈亏比标签必须诚实——标签写"2:1"则实际就是2:1。
        阻力位只作参考提示，不能压缩到让盈亏比失真（否则标签误导）。
        """
        risk = entry - stop
        if risk <= 0:
            risk = entry * self.min_risk_pct

        target_2r = entry + 2 * risk
        target_3r = entry + 3 * risk

        # 目标1：盈亏比 2:1（不被阻力压缩，保证标签诚实）。
        # 阻力位仅作为附加提示，避免「2:1」标签下实际只有1.0的误导。
        target1 = target_2r
        method1 = "盈亏比2:1"
        if resistance > entry and resistance < target_2r:
            # 阻力位明显低于2:1目标，标注提醒（但不改目标价，保盈亏比诚实）
            method1 = f"盈亏比2:1（注意前高阻力{resistance:.2f}）"

        # 目标2：盈亏比 3:1
        target2 = target_3r
        method2 = "盈亏比3:1"

        take_profits: list[TakeProfit] = [
            TakeProfit(price=round(target1, 2), method=method1),
            TakeProfit(price=round(target2, 2), method=method2),
        ]

        # 跟踪止盈参考：MA10
        trailing = TakeProfit(price=round(ma10, 2) if ma10 else 0.0, method="MA10跟踪止盈")
        return take_profits, trailing

    def _build_position(
        self,
        entry: float,
        stop: float,
        temperature: float,
        account_size: float,
    ) -> PositionPlan:
        """1% 风险法则 + 情绪温度系数 + 最低交易金额/股数约束。"""
        emotion_factor = self._emotion_factor(temperature)

        if emotion_factor <= 0 or account_size <= 0:
            return PositionPlan(
                shares=0,
                risk_pct=0.0,
                emotion_factor=emotion_factor,
                account_size=account_size,
                base_risk_pct=self.base_risk_pct * 100,
            )

        risk_amount = account_size * self.base_risk_pct * emotion_factor

        risk_per_share = entry - stop
        if risk_per_share <= 0:
            risk_per_share = entry * self.min_risk_pct

        # 按风险金额计算股数（向下取整到 100 股）
        shares = int(risk_amount / risk_per_share // 100 * 100)

        # 约束1：最低股数
        if shares < self.min_shares:
            shares = self.min_shares

        # 约束2：最低成交金额
        if entry > 0 and entry * shares < self.min_trade_amount:
            shares = int((self.min_trade_amount / entry + 99) // 100 * 100)

        # 实际承担风险 %（反映约束后的真实股数）
        actual_risk_pct = (shares * risk_per_share) / account_size * 100 if account_size > 0 else 0.0

        return PositionPlan(
            shares=shares,
            risk_pct=actual_risk_pct,
            emotion_factor=emotion_factor,
            account_size=account_size,
            base_risk_pct=self.base_risk_pct * 100,
        )

    @staticmethod
    def _emotion_factor(temperature: float) -> float:
        """情绪温度 → 仓位系数（连续分段线性）。

        锚点：40→0, 50→0.4, 70→1.0, 85→0.6, 95→0.3。
        冰点不交易；回升到 50 开始建仓；70 附近满仓风险单位；
        85 以后进入亢奋区间，系数逐步下降防回撤。
        """
        t = float(temperature)

        if t <= 40:
            return 0.0
        if t <= 50:
            # 40→0, 50→0.4
            return 0.0 + (t - 40) * (0.4 - 0.0) / (50 - 40)
        if t <= 70:
            # 50→0.4, 70→1.0
            return 0.4 + (t - 50) * (1.0 - 0.4) / (70 - 50)
        if t <= 85:
            # 70→1.0, 85→0.6
            return 1.0 + (t - 70) * (0.6 - 1.0) / (85 - 70)
        if t <= 95:
            # 85→0.6, 95→0.3
            return 0.6 + (t - 85) * (0.3 - 0.6) / (95 - 85)
        # >95 维持 0.3
        return 0.3


# ---------------------------------------------------------------------- #
# 指标计算工具函数（纯 pandas，可独立测试）
# ---------------------------------------------------------------------- #
def _numeric_series(
    k: pd.DataFrame,
    candidates: list[str],
    default_col: int = 4,
) -> pd.Series:
    """从 K 线 DataFrame 中解析数值序列（优先按中文列名，fallback 按列索引）。"""
    col = _find_col(k, candidates)
    if col is None:
        col = k.columns[default_col]
    return pd.to_numeric(k[col], errors="coerce").dropna()


def _ma(closes: pd.Series, n: int) -> Optional[float]:
    """简单移动平均。"""
    if len(closes) < n:
        return None
    return float(closes.iloc[-n:].mean())


def _atr(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    n: int = 14,
) -> Optional[float]:
    """平均真实波幅 ATR(n)。"""
    if len(highs) < n + 1 or len(lows) < n + 1 or len(closes) < n + 1:
        return None

    # 对齐三个序列，取最后 n+1 条
    df = pd.DataFrame({"h": highs.values, "l": lows.values, "c": closes.values}).dropna()
    if len(df) < n + 1:
        return None

    prev_close = df["c"].shift(1)
    tr1 = df["h"] - df["l"]
    tr2 = (df["h"] - prev_close).abs()
    tr3 = (df["l"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).dropna()
    if len(tr) < n:
        return None
    return float(tr.iloc[-n:].mean())


def _recent_high(s: pd.Series, n: int) -> float:
    """近 n 日最高点（含当日）。"""
    if len(s) < n:
        n = len(s)
    return float(s.iloc[-n:].max())


def _recent_low(s: pd.Series, n: int) -> float:
    """近 n 日最低点（含当日）。"""
    if len(s) < n:
        n = len(s)
    return float(s.iloc[-n:].min())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """匹配列名：精确 > 前缀 > 子串（子串取最短避免误匹配）。"""
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
        for real in cols:
            if str(real).startswith(c):
                return real
        matches = [real for real in cols if c in str(real)]
        if matches:
            return min(matches, key=lambda x: len(str(x)))
    return None
