"""因子库：动量/反转/低波/流动性/市场宽度（P2-002）。

所有因子输出 raw_score；每条因子必须写明经济学假设（评审可证伪）与
缺失处理约定。earnings_quality / value_pe 无基本面数据源，注册为
availability="degraded_no_source"，compute() 抛 FactorUnavailableError
——诚实不可用，绝不伪造数据。

命名规则（P0-007）：任何因子名不得含 prob 字样、不得暗示概率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import (
    Factor,
    FactorMeta,
    FactorUnavailableError,
    PanelFactor,
    cross_section,
    group_apply,
    roll,
)


# ---------------------------------------------------------------------------
# 动量 / 趋势
# ---------------------------------------------------------------------------

class MomentumFactor(PanelFactor):
    """近 window 日累计收益（close/close.shift(window)-1）。"""

    def __init__(self, window: int):
        self._w = int(window)
        self.meta = FactorMeta(
            name=f"mom_{self._w}d", version="1.0.0", family="momentum",
            description=f"{self._w} 日价格动量",
            required_fields=("close",), lookback_days=self._w + 1,
            economic_hypothesis="行为金融下的动量效应：近期相对赢家在数日至数月内"
                                "延续超额表现（投资者反应不足、 attention 驱动）。",
            missing_rule="按面板行滚动，停牌缺口使窗口跨缺口取值；历史不足返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        mom = group_apply(panel["close"], lambda s: s / s.shift(self._w) - 1.0)
        return cross_section(mom, asof)


class Rps20dFactor(PanelFactor):
    """20 日收益的横截面百分位排名（0-100），即 RPS。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="rps_20d", version="1.0.0", family="momentum",
            description="20 日相对强度排名 RPS（0-100）",
            required_fields=("close",), lookback_days=21,
            economic_hypothesis="相对强度具有持续性：RPS 靠前的股票是资金共识赢家，"
                                "短期延续概率更高（相对强弱假说）。",
            missing_rule="截面内 rank；有效样本不足时该日整体 NaN。",
            winsorize=None, standardize="none",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        mom20 = group_apply(panel["close"], lambda s: s / s.shift(20) - 1.0)
        cs = cross_section(mom20, asof)
        return cs.rank(pct=True) * 100.0


class Ma20SlopeFactor(PanelFactor):
    """20 日均线的 5 日斜率（ma20_t / ma20_{t-5} - 1）。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="ma20_slope", version="1.0.0", family="momentum",
            description="20 日均线 5 日斜率",
            required_fields=("close",), lookback_days=26,
            economic_hypothesis="均线斜率刻画趋势的加速度而非水平：斜率为正且放大"
                                "表明趋势自我强化（趋势加速度假说）。",
            missing_rule="ma20 需至少 15 个有效样本；不足返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        ma20 = roll(panel["close"], 20, "mean", min_periods=15)
        slope = group_apply(ma20, lambda s: s / s.shift(5) - 1.0)
        return cross_section(slope, asof)


class Breakout20dFactor(PanelFactor):
    """close 相对近 20 日最高价的突破程度（<=0，触顶为 0）。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="breakout_20d", version="1.0.0", family="momentum",
            description="20 日新高突破度：close / rolling20max - 1",
            required_fields=("close",), lookback_days=21,
            economic_hypothesis="突破前期高点释放套牢盘压力，是趋势启动的技术确认"
                                "（突破假说 / range breakout）。",
            missing_rule="滚动窗口不足 10 个样本返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        hh = roll(panel["close"], 20, "max", min_periods=10)
        brk = panel["close"] / hh - 1.0
        return cross_section(brk, asof)


class High52wProximityFactor(PanelFactor):
    """52 周新高接近度：close / rolling250max - 1。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="high_52w_proximity", version="1.0.0", family="momentum",
            description="52 周新高接近度：close / rolling250max - 1",
            required_fields=("close",), lookback_days=250,
            economic_hypothesis="锚定效应：接近 52 周新高的股票因投资者锚定历史"
                                "高价而低估突破后的上行空间（52 周高点效应）。",
            missing_rule="至少 60 个有效样本；不足返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        hh = roll(panel["close"], 250, "max", min_periods=60)
        prox = panel["close"] / hh - 1.0
        return cross_section(prox, asof)


class VolAdjMom20dFactor(PanelFactor):
    """20 日动量 / 20 日已实现波动（夏普式动量）。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="vol_adj_mom_20d", version="1.0.0", family="momentum",
            description="波动率调整动量：mom20 / realized_vol20",
            required_fields=("close", "pct_change"), lookback_days=21,
            economic_hypothesis="同等动量下低波动实现的趋势质量更高，高波动动量"
                                "更多来自噪声（风险调整动量假说）。",
            missing_rule="分母波动为 0 或样本不足返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        mom20 = group_apply(panel["close"], lambda s: s / s.shift(20) - 1.0)
        vol20 = roll(panel["pct_change"] / 100.0, 20, "std", min_periods=15)
        score = mom20 / vol20.replace(0.0, np.nan)
        return cross_section(score, asof)


class TrendPersistence20dFactor(PanelFactor):
    """近 20 日上涨天数占比。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="trend_persistence_20d", version="1.0.0", family="momentum",
            description="20 日趋势持续性：上涨天数占比",
            required_fields=("pct_change",), lookback_days=20,
            economic_hypothesis="上涨天数占比高说明买盘持续介入而非单日脉冲，"
                                "趋势的可复制性更强（持续性优于幅度假说）。",
            missing_rule="窗口不足 10 个样本返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        up = (panel["pct_change"] > 0).astype(float)
        pers = roll(up, 20, "mean", min_periods=10)
        return cross_section(pers, asof)


# ---------------------------------------------------------------------------
# 反转
# ---------------------------------------------------------------------------

class ReversalFactor(PanelFactor):
    """近 window 日累计收益取负（短线反转）。"""

    def __init__(self, window: int):
        self._w = int(window)
        self.meta = FactorMeta(
            name=f"rev_{self._w}d", version="1.0.0", family="reversal",
            description=f"{self._w} 日短线反转（负 {self._w} 日收益）",
            required_fields=("close",), lookback_days=self._w + 1,
            economic_hypothesis="流动性提供者假说：短线超涨由临时买压推动，"
                                "随后回归，跌得多的反弹（短期反转效应）。",
            missing_rule="按面板行滚动；历史不足返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        mom = group_apply(panel["close"], lambda s: s / s.shift(self._w) - 1.0)
        return -cross_section(mom, asof)


class Rsi14Factor(PanelFactor):
    """14 日 RSI（0-100）。低 RSI 对应超卖。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="rsi_14", version="1.0.0", family="reversal",
            description="14 日相对强弱指标 RSI",
            required_fields=("close",), lookback_days=16,
            economic_hypothesis="超买超卖假说：RSI 极端值反映单边情绪耗竭，"
                                "低 RSI（超卖）预期均值回归反弹；预期与未来收益负相关。",
            missing_rule="至少 10 个有效样本；全涨（无亏损日）RSI 记 100。",
            winsorize=None, standardize="none",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        def _rsi(s: pd.Series) -> pd.Series:
            delta = s.diff()
            gain = delta.clip(lower=0.0)
            loss = (-delta).clip(lower=0.0)
            ag = gain.rolling(14, min_periods=10).mean()
            al = loss.rolling(14, min_periods=10).mean()
            rs = ag / al.replace(0.0, np.nan)
            out = 100.0 - 100.0 / (1.0 + rs)
            fill = pd.Series(
                np.where((al == 0) & (ag > 0), 100.0, np.nan), index=s.index
            )
            return out.fillna(fill)

        rsi = group_apply(panel["close"], _rsi)
        return cross_section(rsi, asof)


class IndexAdjRev5dFactor(PanelFactor):
    """剔除市场等权收益后的 5 日反转（指数代理=面板等权均值，见假设）。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="index_adj_rev_5d", version="1.0.0", family="reversal",
            description="市场调整 5 日反转：-(个股5日收益 - 面板等权5日收益)",
            required_fields=("close",), lookback_days=6,
            economic_hypothesis="个股特异部分的短线反转更干净：市场共同波动不回归，"
                                "只有特质超涨回归（特质反转假说）。"
                                "指数代理用当日面板等权平均收益（研究层数据接口无宽基成分权重）。",
            missing_rule="截面样本不足时该日整体 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        ret5 = group_apply(panel["close"], lambda s: s / s.shift(5) - 1.0)
        cs = cross_section(ret5, asof)
        mkt5 = cs.mean()
        return -(cs - mkt5)


# ---------------------------------------------------------------------------
# 低波动
# ---------------------------------------------------------------------------

class RealizedVol20dFactor(PanelFactor):
    """20 日已实现波动率（日收益标准差）。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="realized_vol_20d", version="1.0.0", family="low_vol",
            description="20 日已实现波动率",
            required_fields=("pct_change",), lookback_days=20,
            economic_hypothesis="低波动异象：低波动股票被杠杆约束投资者系统性"
                                "高估高波动股而相对低估，低波动组长期占优（预期与未来收益负相关）。",
            missing_rule="至少 15 个有效样本；不足返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        vol = roll(panel["pct_change"] / 100.0, 20, "std", min_periods=15)
        return cross_section(vol, asof)


class DownsideVol20dFactor(PanelFactor):
    """20 日下行波动（负收益部分的二阶矩）。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="downside_vol_20d", version="1.0.0", family="low_vol",
            description="20 日下行波动率",
            required_fields=("pct_change",), lookback_days=20,
            economic_hypothesis="投资者对下行风险厌恶不对称，下行波动大的股票"
                                "要求更高补偿（预期与未来收益负相关）。",
            missing_rule="至少 15 个有效样本；无下跌日记 0。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        r = panel["pct_change"] / 100.0

        def _dvol(s: pd.Series) -> pd.Series:
            neg = s.clip(upper=0.0)
            return np.sqrt((neg ** 2).rolling(20, min_periods=15).mean())

        return cross_section(group_apply(r, _dvol), asof)


# ---------------------------------------------------------------------------
# 流动性 / 量价
# ---------------------------------------------------------------------------

class Turnover20dAvgFactor(PanelFactor):
    """20 日平均换手率。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="turnover_20d_avg", version="1.0.0", family="liquidity",
            description="20 日平均换手率",
            required_fields=("turnover",), lookback_days=20,
            economic_hypothesis="换手率刻画关注过热度：高换手常伴随情绪顶部与"
                                "筹码派发（预期与未来收益负相关，换手率异象）。",
            missing_rule="至少 10 个有效样本；不足返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        return cross_section(roll(panel["turnover"], 20, "mean", min_periods=10), asof)


class Amihud20dFactor(PanelFactor):
    """Amihud 非流动性：mean(|ret| / amount) * 1e9。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="amihud_20d", version="1.0.0", family="liquidity",
            description="Amihud 非流动性（×1e9 缩放）",
            required_fields=("pct_change", "amount"), lookback_days=20,
            economic_hypothesis="非流动性溢价：单位成交额推动的价格变动大说明"
                                "流动性差，持有者要求流动性补偿（预期正相关）。",
            missing_rule="amount<=0 的行剔除；至少 10 个有效样本。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.DataFrame:
        illiq = (panel["pct_change"].abs() / 100.0) / panel["amount"].replace(0.0, np.nan)
        return cross_section(roll(illiq * 1e9, 20, "mean", min_periods=10), asof)


class VolumeRatio5_20Factor(PanelFactor):
    """量能比：5 日均量 / 20 日均量 - 1。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="volume_ratio_5_20", version="1.0.0", family="liquidity",
            description="量能比：5日均量/20日均量-1",
            required_fields=("volume",), lookback_days=21,
            economic_hypothesis="短期量能放大反映资金关注度脉冲；极端放量后"
                                "往往透支买盘（预期与未来收益负相关）。",
            missing_rule="分母为 0 或样本不足返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        v5 = roll(panel["volume"], 5, "mean", min_periods=4)
        v20 = roll(panel["volume"], 20, "mean", min_periods=10)
        ratio = v5 / v20.replace(0.0, np.nan) - 1.0
        return cross_section(ratio, asof)


class AmountAccel5dFactor(PanelFactor):
    """成交额加速度：5 日均额 / 20 日均额 - 1。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="amount_accel_5d", version="1.0.0", family="liquidity",
            description="成交额加速度：5日均额/20日均额-1",
            required_fields=("amount",), lookback_days=21,
            economic_hypothesis="成交额加速扩张是情绪发酵信号，加速度见顶常先于"
                                "价格见顶（资金脉冲衰减假说，预期负相关）。",
            missing_rule="分母为 0 或样本不足返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        a5 = roll(panel["amount"], 5, "mean", min_periods=4)
        a20 = roll(panel["amount"], 20, "mean", min_periods=10)
        accel = a5 / a20.replace(0.0, np.nan) - 1.0
        return cross_section(accel, asof)


class VolumePriceDiv20dFactor(PanelFactor):
    """量价背离：-corr(close, volume)（20 日滚动）。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="volume_price_div_20d", version="1.0.0", family="liquidity",
            description="量价背离度：-20日(close,volume)相关系数",
            required_fields=("close", "volume"), lookback_days=21,
            economic_hypothesis="缩量上涨（量价负相关）说明筹码锁定、抛压轻；"
                                "放量滞涨说明派发。取负相关系数使“缩量上涨”得分更高。",
            missing_rule="至少 15 个有效样本；方差为 0 返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        x, y = panel["close"], panel["volume"]
        ex = roll(x, 20, "mean", min_periods=15)
        ey = roll(y, 20, "mean", min_periods=15)
        exy = roll(x * y, 20, "mean", min_periods=15)
        sx = roll(x, 20, "std", min_periods=15)
        sy = roll(y, 20, "std", min_periods=15)
        corr = (exy - ex * ey) / (sx * sy).replace(0.0, np.nan)
        return -cross_section(corr, asof)


class TurnoverPersistence10dFactor(PanelFactor):
    """换手持续性：近 10 日换手率高于自身 20 日均线的天数占比。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="turnover_persistence_10d", version="1.0.0", family="liquidity",
            description="换手持续性：10日内换手高于20日均值的占比",
            required_fields=("turnover",), lookback_days=21,
            economic_hypothesis="持续高于常态的换手说明注意力粘性（机构持续参与），"
                                "而非一次性脉冲；粘性资金推动的行情更持久。",
            missing_rule="至少 7 个有效样本；不足返回 NaN。",
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        base = roll(panel["turnover"], 20, "mean", min_periods=10)
        flag = (panel["turnover"] >= base).astype(float)
        return cross_section(roll(flag, 10, "mean", min_periods=7), asof)


# ---------------------------------------------------------------------------
# 市场 / 宽度（全截面同值：横截面无区分度，保留给条件研究）
# ---------------------------------------------------------------------------

class _MarketFactor(Factor):
    """市场级单值因子的骨架：_market_value(asof, panel) -> float，广播到全截面。"""

    def compute(self, asof: str, universe: pd.Index, data) -> pd.Series:
        from .base import load_window  # 局部引入避免循环

        panel = load_window(data, asof, universe, self.meta.lookback_days)
        if panel.empty:
            return pd.Series(dtype=float, name=self.meta.name)
        value = self._market_value(asof, panel)
        codes = panel.index.get_level_values("code").unique()
        return pd.Series(float(value), index=pd.Index(codes, name="code"), name=self.meta.name)

    def _market_value(self, asof: str, panel: pd.DataFrame) -> float:
        raise NotImplementedError

    def finalize(self, series: pd.Series) -> pd.Series:
        return series  # 单值因子不做截面标准化


class MktBreadth5dFactor(_MarketFactor):
    """市场宽度：5 日收益为正的个股占比。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="mkt_breadth_5d", version="1.0.0", family="market",
            description="市场宽度：近5日上涨家数占比",
            required_fields=("close",), lookback_days=7,
            economic_hypothesis="宽度衡量普涨/分化：宽度收缩时涨幅集中少数权重股，"
                                "行情脆弱（宽度领先性假说）。全截面同值，仅供条件研究。",
            missing_rule="有效个股不足时返回 NaN。",
            winsorize=None, standardize="none",
        )

    def _market_value(self, asof: str, panel: pd.DataFrame) -> float:
        ret5 = group_apply(panel["close"], lambda s: s / s.shift(5) - 1.0)
        cs = cross_section(ret5, asof).dropna()
        return float((cs > 0).mean()) if len(cs) else float("nan")


class MktVol20dFactor(_MarketFactor):
    """市场 20 日等权组合波动。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="mkt_vol_20d", version="1.0.0", family="market",
            description="市场20日等权日收益波动",
            required_fields=("pct_change",), lookback_days=21,
            economic_hypothesis="市场波动刻画风险偏好环境：高波动期动量衰减、"
                                "反转增强（状态依赖假说）。全截面同值，供 regime 研究。",
            missing_rule="不足 10 个交易日返回 NaN。",
            winsorize=None, standardize="none",
        )

    def _market_value(self, asof: str, panel: pd.DataFrame) -> float:
        mkt = (panel["pct_change"] / 100.0).groupby(level="date").mean()
        vol = mkt.rolling(20, min_periods=10).std()
        return float(vol.iloc[-1]) if len(vol) else float("nan")


# ---------------------------------------------------------------------------
# 诚实不可用：无基本面数据源（degraded_no_source）
# ---------------------------------------------------------------------------

class _UnavailableFactor(Factor):
    """无数据源因子的占位实现：compute() 抛 FactorUnavailableError。"""

    def compute(self, asof: str, universe: pd.Index, data) -> pd.Series:
        raise FactorUnavailableError(
            f"{self.meta.name}: 基本面数据源未接入（availability="
            f"degraded_no_source），拒绝伪造因子值。"
        )


class EarningsQualityFactor(_UnavailableFactor):
    def __init__(self):
        self.meta = FactorMeta(
            name="earnings_quality", version="0.0.0", family="quality",
            description="盈利质量（应计/现金流占比，占位待数据源）",
            required_fields=(), lookback_days=0,
            economic_hypothesis="高质量盈利（现金流支撑、低应计）更可持续，"
                                "市场对低质量盈利定价不足（应计异象）。",
            missing_rule="无基本面数据源：availability=degraded_no_source，compute 抛错。",
            winsorize=None, standardize="none",
        )


class ValuePEFactor(_UnavailableFactor):
    def __init__(self):
        self.meta = FactorMeta(
            name="value_pe", version="0.0.0", family="value",
            description="估值（PE 分位，占位待数据源）",
            required_fields=(), lookback_days=0,
            economic_hypothesis="低估值股票被过度外推的悲观预期压制，长期均值回归"
                                "提供超额收益（价值溢价）。",
            missing_rule="无基本面数据源：availability=degraded_no_source，compute 抛错。",
            winsorize=None, standardize="none",
        )
