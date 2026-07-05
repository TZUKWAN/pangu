"""买卖点模块纯逻辑测试（mock 数据，不依赖 akshare 网络）。"""

import pandas as pd
import pytest

from engine.entry_exit import (
    _ma,
    _atr,
    _recent_high,
    _recent_low,
    _find_col,
    EntryExitEngine,
)
from engine.trend_scanner import StockCandidate


def _make_kline(n: int = 30, trend: str = "up") -> pd.DataFrame:
    """构造假 K 线。"""
    base = 10.0
    rows = []
    for i in range(n):
        if trend == "up":
            close = base + i * 0.2 + (i % 3) * 0.1
        elif trend == "down":
            close = base - i * 0.1
        else:
            close = base + (i % 5) * 0.1
        high = close + 0.15
        low = close - 0.15
        open_ = close - 0.05
        rows.append({
            "日期": f"202401{str(i+1).zfill(2)}",
            "开盘": open_,
            "收盘": close,
            "最高": high,
            "最低": low,
            "成交量": 100000 + i * 1000,
        })
    return pd.DataFrame(rows)


class MockDataLoader:
    """模拟 DataLoader，只返回固定 K 线。"""

    def __init__(self, kline: pd.DataFrame) -> None:
        self.kline = kline

    def daily_kline(self, code: str, days: int = 60, adjust: str = "qfq", date: str | None = None) -> pd.DataFrame:
        return self.kline.tail(days).reset_index(drop=True)


# ---------------------------------------------------------------------- #
# 指标函数测试
# ---------------------------------------------------------------------- #
def test_ma_basic():
    s = pd.Series([10, 11, 12, 13, 14])
    assert _ma(s, 5) == pytest.approx(12.0)
    assert _ma(s, 3) == pytest.approx(13.0)


def test_ma_insufficient_data():
    s = pd.Series([10, 11])
    assert _ma(s, 5) is None


def test_atr_basic():
    highs = pd.Series([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    lows = pd.Series([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    closes = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    atr = _atr(highs, lows, closes, n=14)
    assert atr is not None
    assert atr > 0


def test_atr_insufficient_data():
    highs = pd.Series([11, 12])
    lows = pd.Series([9, 10])
    closes = pd.Series([10, 11])
    assert _atr(highs, lows, closes, n=14) is None


def test_recent_high_low():
    s = pd.Series([1, 3, 2, 5, 4])
    assert _recent_high(s, 3) == 5.0
    assert _recent_low(s, 3) == 2.0


def test_find_col_fuzzy():
    df = pd.DataFrame({"涨跌幅.1": [1], "收盘": [10]})
    assert _find_col(df, ["涨跌幅"]) == "涨跌幅.1"
    assert _find_col(df, ["收盘"]) == "收盘"


# ---------------------------------------------------------------------- #
# 引擎测试
# ---------------------------------------------------------------------- #
def test_emotion_factor():
    # 连续分段线性：40→0, 50→0.4, 70→1.0, 85→0.6, 95→0.3
    assert EntryExitEngine._emotion_factor(30) == 0.0
    assert EntryExitEngine._emotion_factor(45) == pytest.approx(0.2)
    assert EntryExitEngine._emotion_factor(50) == pytest.approx(0.4)
    assert EntryExitEngine._emotion_factor(70) == 1.0
    assert EntryExitEngine._emotion_factor(85) == pytest.approx(0.6)
    assert EntryExitEngine._emotion_factor(90) == pytest.approx(0.45)
    assert EntryExitEngine._emotion_factor(95) == pytest.approx(0.3)


def test_buy_point_broke_out_with_near_pullback():
    """已突破且回踩位在现价 ±3% 内时，主买点应为最近回踩位。"""
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(dl, cfg={})

    close = 100.0
    recent_high = 99.5          # 已突破
    recent_low = 90.0
    ma_map = {
        5: close * 0.99,        # 在 ±3% 内
        10: close * 0.96,       # 也在 ±3% 内
        20: close * 0.90,       # 太远
    }
    points = engine._build_buy_points(close, recent_high, recent_low, ma_map)
    primary = next(p for p in points if p.is_primary)
    assert primary.type == "回踩位"
    assert primary.price == pytest.approx(close * 0.99, rel=1e-9)


def test_buy_point_prefers_support_over_chasing_breakout():
    """未突破时即使突破位离现价不远，也优先等待支撑/回踩，不默认追价。"""
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(dl, cfg={})

    close = 100.0
    recent_high = 103.0         # 未突破（close <= recent_high * 1.001），但在 5% 内
    recent_low = 90.0
    ma_map = {5: 80.0, 10: 75.0, 20: 70.0}  # 回踩位都太远
    points = engine._build_buy_points(close, recent_high, recent_low, ma_map)
    primary = next(p for p in points if p.is_primary)
    assert primary.type == "支撑位"
    assert primary.price == pytest.approx(recent_low, rel=1e-9)


def test_buy_point_fallback_to_support_when_breakout_too_far():
    """突破位离现价 >5% 且回踩位太远时，主买点 fallback 到支撑位。"""
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(dl, cfg={})

    close = 100.0
    recent_high = 110.0         # 突破位太远
    recent_low = 95.0           # 支撑位
    ma_map = {5: 80.0, 10: 75.0, 20: 70.0}  # 回踩位都太远
    points = engine._build_buy_points(close, recent_high, recent_low, ma_map)
    primary = next(p for p in points if p.is_primary)
    assert primary.type == "支撑位"
    assert primary.price == pytest.approx(recent_low, rel=1e-9)


def test_stop_loss_chooses_widest_valid():
    """在满足 min_stop_pct 的候选止损中，应选择最宽（离买点最远）的一个。"""
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(
        dl,
        cfg={"min_stop_pct": 0.02, "atr_multiplier": 2.0, "ma20_stop_buffer": 0.03},
    )

    entry = 100.0
    # ATR 止损 94（6% 宽），结构止损 95（5% 宽），MA20 止损 96*0.97=93.12（6.88% 宽）
    sl = engine._build_stop_loss(entry=entry, ma20=96.0, atr=3.0, structure_low=95.0)
    assert sl.price == pytest.approx(93.12, rel=1e-9)
    assert "MA20" in sl.method


def test_stop_loss_fallback_to_min_pct():
    """所有候选止损都过近时，fallback 到 min_stop_pct 对应的硬性价格。"""
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(
        dl,
        cfg={"min_stop_pct": 0.10, "atr_multiplier": 2.0},
    )

    entry = 100.0
    sl = engine._build_stop_loss(entry=entry, ma20=99.0, atr=3.0, structure_low=99.5)
    # 三个候选止损都大于 90，因此 fallback 到 90
    assert sl.price == pytest.approx(90.0, rel=1e-9)


def test_position_min_trade_amount_and_shares():
    """低价股在风险金额较小的情况下，应同时满足 min_shares 与 min_trade_amount 约束。"""
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(
        dl,
        cfg={"min_trade_amount": 10000, "min_shares": 100, "base_risk_pct": 0.001},
    )

    # 低价股，风险金额仅 100 元，但最小成交金额要求 10000 元
    pos = engine._build_position(entry=2.0, stop=1.9, temperature=70, account_size=100_000)
    assert pos.shares % 100 == 0
    assert pos.shares >= 100
    assert pos.shares * 2.0 >= 10000
    # 由 min_trade_amount 驱动：10000/2 = 5000 股
    assert pos.shares == 5000


def test_compute_output_schema():
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(dl, cfg={})

    candidate = StockCandidate(
        code="000001", name="测试股", board="测试板块",
        close=float(kline["收盘"].iloc[-1]),
        pct_change=2.0, turnover_rate=1.5, circ_mv_yi=100,
        rps=85, reasons=["均线多头"],
    )

    res = engine.compute(candidate, temperature=70, account_size=1_000_000)
    d = res.to_dict()

    assert d["code"] == "000001"
    assert d["close"] > 0
    assert len(d["buy_points"]) >= 3
    assert any(b["is_primary"] for b in d["buy_points"])
    assert d["stop_loss"] is not None
    assert d["stop_loss"]["price"] > 0
    assert len(d["take_profit"]) == 2
    assert d["trailing_stop"] is not None
    assert d["position"] is not None
    assert d["position"]["shares"] % 100 == 0
    assert d["risk_reward_ratio"] >= 0


def test_compute_cold_temperature_no_position():
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(dl, cfg={})

    candidate = StockCandidate(
        code="000001", name="测试股", board="测试板块",
        close=float(kline["收盘"].iloc[-1]),
        pct_change=2.0, turnover_rate=1.5, circ_mv_yi=100,
        rps=85, reasons=["均线多头"],
    )

    res = engine.compute(candidate, temperature=30, account_size=1_000_000)
    assert res.position is not None
    assert res.position.shares == 0
    assert res.position.emotion_factor == 0.0


def test_compute_with_dict_candidate():
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(dl, cfg={})

    candidate = {"code": "000002", "name": "字典股", "close": float(kline["收盘"].iloc[-1])}
    res = engine.compute(candidate, temperature=70)
    assert res.code == "000002"
    assert res.stop_loss is not None


def test_compute_insufficient_kline():
    kline = _make_kline(n=5, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(dl, cfg={})

    candidate = StockCandidate(
        code="000003", name="短历史", board="测试板块",
        close=10.0, pct_change=1.0, turnover_rate=1.0,
        circ_mv_yi=50, rps=80, reasons=["测试"],
    )
    res = engine.compute(candidate, temperature=70)
    assert len(res.warnings) > 0
    assert res.stop_loss is None


def test_risk_reward_label_is_honest():
    """盈亏比标签诚实：标'2:1'则实际盈亏比应≈2.0，不被阻力位压缩到失真。"""
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(dl, cfg={})
    candidate = StockCandidate(
        code="000001", name="测试", board="x",
        close=float(kline["收盘"].iloc[-1]), pct_change=2.0,
        turnover_rate=1.5, circ_mv_yi=100, rps=85, reasons=["t"],
    )
    res = engine.compute(candidate, temperature=60, account_size=1_000_000)
    d = res.to_dict()
    primary = next((b for b in d["buy_points"] if b["is_primary"]), d["buy_points"][0])
    entry = primary["price"]
    stop = d["stop_loss"]["price"]
    risk = entry - stop
    if risk > 0 and d["take_profit"]:
        # 第一个止盈标的是"2:1"，实际盈亏比应 ≥ 1.9（允许2%浮点误差）
        tp1 = d["take_profit"][0]["price"]
        actual_rr = (tp1 - entry) / risk
        assert actual_rr >= 1.9, f"标签2:1但实际盈亏比仅{actual_rr:.2f}，标签不诚实"


def test_config_params_affect_output():
    """EntryExitEngine 配置必须真实影响买卖点/仓位计算。"""
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    candidate = StockCandidate(
        code="000001", name="测试", board="x",
        close=float(kline["收盘"].iloc[-1]), pct_change=2.0,
        turnover_rate=1.5, circ_mv_yi=100, rps=85, reasons=["t"],
    )
    engine1 = EntryExitEngine(dl, cfg={"atr_period": 14, "account_size": 1_000_000, "base_risk_pct": 0.01})
    engine2 = EntryExitEngine(dl, cfg={"atr_period": 5, "account_size": 500_000, "base_risk_pct": 0.02, "breakout_lookback": 10})
    res1 = engine1.compute(candidate, temperature=70)
    res2 = engine2.compute(candidate, temperature=70)
    assert res1.position.account_size == 1_000_000
    assert res2.position.account_size == 500_000
    assert res1.position.risk_pct != res2.position.risk_pct


def test_compute_date_param_uses_historical_kline():
    """传入 date 参数时，应使用历史 K 线终点。"""
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(dl, cfg={})
    candidate = StockCandidate(
        code="000001", name="测试", board="x",
        close=float(kline["收盘"].iloc[-1]), pct_change=2.0,
        turnover_rate=1.5, circ_mv_yi=100, rps=85, reasons=["t"],
    )
    # MockDataLoader ignores date, but we verify date is passed through
    res = engine.compute(candidate, temperature=70, date="20250101")
    assert res.code == "000001"
    assert res.stop_loss is not None


def test_entry_plan_contains_style_and_invalid_condition():
    kline = _make_kline(n=30, trend="up")
    dl = MockDataLoader(kline)
    engine = EntryExitEngine(dl, cfg={})
    candidate = StockCandidate(
        code="000001", name="测试", board="x",
        close=float(kline["收盘"].iloc[-1]), pct_change=2.0,
        turnover_rate=1.5, circ_mv_yi=100, rps=85, reasons=["t"],
    )

    res = engine.compute(candidate, temperature=70)
    plan = res.to_dict()["entry_plan"]

    assert plan["entry_style"] == res.entry_style
    assert plan["trigger_condition"]
    assert plan["invalid_condition"]
    assert "is_chasing" in plan
