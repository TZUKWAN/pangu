"""回测模块纯逻辑测试（mock 数据，不依赖 akshare 网络）。"""

import pandas as pd
import pytest

from engine.backtest import (
    BacktestConfig,
    BacktestResult,
    Backtester,
    HistoricalDataLoader,
    TradeRecord,
    sensitivity_analysis,
)
from engine.entry_exit import EntryExitEngine
from engine.trend_scanner import StockCandidate


# ---------------------------------------------------------------------- #
# 模拟数据工厂
# ---------------------------------------------------------------------- #
def _make_uptrend_kline(
    code: str,
    n: int = 30,
    start_date: str = "20240101",
    circ_mv: float = 100e8,
) -> pd.DataFrame:
    """构造满足趋势选股条件的日 K。"""
    from datetime import datetime, timedelta

    base = datetime.strptime(start_date, "%Y%m%d")
    rows = []
    close = 10.0
    for i in range(n):
        date = (base + timedelta(days=i)).strftime("%Y%m%d")
        close = close + 0.15 + (i % 3) * 0.02
        high = close + 0.1
        low = close - 0.1
        open_ = close - 0.05
        # 最后一天放量，前 5 日均量 10000，今天 25000
        volume = 25000 if i == n - 1 else 10000 + i * 50
        rows.append({
            "日期": date,
            "股票代码": code,
            "开盘": open_,
            "收盘": close,
            "最高": high,
            "最低": low,
            "成交量": volume,
            "成交额": volume * close,
            "振幅": 1.5,
            "涨跌幅": 1.5,
            "涨跌额": 0.15,
            "换手率": 1.5,
            "流通市值": circ_mv,
        })
    return pd.DataFrame(rows)


def _make_downfall_kline(
    code: str,
    n: int = 30,
    start_date: str = "20240101",
) -> pd.DataFrame:
    """构造下跌趋势日 K（用于测试止损）。"""
    from datetime import datetime, timedelta

    base = datetime.strptime(start_date, "%Y%m%d")
    rows = []
    close = 20.0
    for i in range(n):
        date = (base + timedelta(days=i)).strftime("%Y%m%d")
        close = close - 0.3 - (i % 2) * 0.1
        high = close + 0.05
        low = close - 0.05
        open_ = close + 0.02
        rows.append({
            "日期": date,
            "股票代码": code,
            "开盘": open_,
            "收盘": close,
            "最高": high,
            "最低": low,
            "成交量": 10000,
            "成交额": 10000 * close,
            "振幅": 1.0,
            "涨跌幅": -1.0,
            "涨跌额": -0.2,
            "换手率": 1.0,
            "流通市值": 100e8,
        })
    return pd.DataFrame(rows)


class MockDataLoader:
    """模拟 DataLoader，返回固定 K 线与涨停池。"""

    def __init__(self, klines: dict[str, pd.DataFrame]) -> None:
        self.klines = klines
        self.limit_up_count = 60  # 让情绪温度 > 40
        self.broke_count = 0
        self.limit_down_count = 0

    def daily_kline(
        self,
        symbol: str,
        days: int = 60,
        adjust: str = "qfq",
        date: str | None = None,
    ) -> pd.DataFrame:
        df = self.klines.get(symbol, pd.DataFrame()).copy()
        if df.empty:
            return df
        if date:
            df = df[df["日期"] <= date]
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)
        return df

    def limit_up_pool(self, date: str | None = None) -> pd.DataFrame:
        return pd.DataFrame({
            "代码": [f"{i:06d}" for i in range(self.limit_up_count)],
            "名称": [f"涨停{i}" for i in range(self.limit_up_count)],
            "涨停统计": ["1天1板"] * self.limit_up_count,
            "连板数": [1] * self.limit_up_count,
        })

    def broke_pool(self, date: str | None = None) -> pd.DataFrame:
        return pd.DataFrame({
            "代码": [f"{i:06d}" for i in range(self.broke_count)],
            "名称": [f"炸板{i}" for i in range(self.broke_count)],
        })

    def limit_down_pool(self, date: str | None = None) -> pd.DataFrame:
        return pd.DataFrame({
            "代码": [f"{i:06d}" for i in range(self.limit_down_count)],
            "名称": [f"跌停{i}" for i in range(self.limit_down_count)],
        })

    def all_spot(self) -> pd.DataFrame:
        # HistoricalDataLoader 会覆盖 all_spot，这里保留兜底
        return pd.DataFrame()

    def concept_boards(self) -> pd.DataFrame:
        return pd.DataFrame()

    def concept_constituents(self, board_symbol: str, board_name: str | None = None) -> pd.DataFrame:
        return pd.DataFrame()


# ---------------------------------------------------------------------- #
# BacktestResult 统计函数测试
# ---------------------------------------------------------------------- #
def test_max_drawdown_calculation():
    result = BacktestResult(
        cfg=BacktestConfig(start_date="20240101", end_date="20240110", watchlist=[]),
        start_date="20240101",
        end_date="20240110",
        initial_capital=1000000,
        final_capital=900000,
        total_return=-0.1,
        annual_return=-0.5,
        win_rate=0.0,
        profit_loss_ratio=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        equity_curve=[("20240101", 100), ("20240102", 120), ("20240103", 90), ("20240104", 110)],
    )
    assert Backtester._max_drawdown([v for _, v in result.equity_curve]) == pytest.approx(0.25)


def test_sharpe_calculation():
    # 稳定上涨的净值序列
    navs = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
    sharpe = Backtester._sharpe(navs)
    assert sharpe > 0


def test_monthly_returns():
    curve = [
        ("20240102", 100),
        ("20240110", 105),
        ("20240205", 110),
        ("20240220", 99),
    ]
    monthly = Backtester._monthly_returns(curve)
    assert "202401" in monthly
    assert "202402" in monthly
    # 1 月相对初始为 0（无上个月），2 月相对 1 月末下跌
    assert monthly["202401"] == 0.0
    assert monthly["202402"] == pytest.approx((99 - 105) / 105)


def test_trade_record_pnl_calculation():
    t = TradeRecord(
        code="000001",
        name="测试",
        entry_date="20240101",
        exit_date="20240105",
        entry_price=10.0,
        exit_price=11.0,
        shares=1000,
        stop_loss=9.0,
        take_profit=12.0,
        exit_reason="止盈",
    )
    assert t.pnl == 0.0  # dataclass 默认值，业务逻辑在回测引擎中填充
    # 模拟引擎填充
    t.pnl = t.shares * (t.exit_price - t.entry_price)
    t.pnl_pct = (t.exit_price - t.entry_price) / t.entry_price
    assert t.pnl == 1000.0
    assert t.pnl_pct == pytest.approx(0.10)


# ---------------------------------------------------------------------- #
# HistoricalDataLoader 测试
# ---------------------------------------------------------------------- #
def test_historical_dataloader_reconstructs_spot():
    k1 = _make_uptrend_kline("000001", n=10, start_date="20240101")
    dl = MockDataLoader({"000001": k1})
    hdl = HistoricalDataLoader(dl, ["000001"], {"000001": "测试股"})
    hdl.preload("20240101", "20240110")
    hdl.current_date = "20240105"
    spot = hdl.all_spot()
    assert len(spot) == 1
    assert spot.iloc[0]["代码"] == "000001"
    assert spot.iloc[0]["最新价"] > 0


def test_historical_dataloader_kline_pit_safe():
    k1 = _make_uptrend_kline("000001", n=10, start_date="20240101")
    dl = MockDataLoader({"000001": k1})
    hdl = HistoricalDataLoader(dl, ["000001"])
    hdl.preload("20240101", "20240110")
    row_today = hdl.kline_on("000001", "20240105")
    row_future = hdl.kline_on("000001", "20240115")
    assert row_today is not None
    assert row_future is not None
    # 未来日期应回退到最新可用数据（PIT-safe：不偷看未来）
    assert row_future["date"] == "20240110"


# ---------------------------------------------------------------------- #
# 回测引擎端到端测试
# ---------------------------------------------------------------------- #
def _make_backtester(watchlist: list[str], klines: dict[str, pd.DataFrame]) -> Backtester:
    dl = MockDataLoader(klines)
    cfg = BacktestConfig(
        start_date="20240115",
        end_date="20240215",
        watchlist=watchlist,
        initial_capital=1_000_000.0,
        sentiment_threshold=40.0,
        rps_threshold=80.0,
        max_positions=2,
        max_holding_days=10,
        enable_progress=False,
    )
    bt = Backtester(cfg, dl=dl)
    # 注入真实 RPS 表，避免 fallback 近似失真
    bt.scanner.set_rps_map({code: 85.0 for code in watchlist})
    return bt


def test_backtest_produces_trades_and_positive_return():
    """上涨趋势应产生盈利交易。"""
    k1 = _make_uptrend_kline("000001", n=60, start_date="20231201")
    k2 = _make_uptrend_kline("000002", n=60, start_date="20231201")
    bt = _make_backtester(["000001", "000002"], {"000001": k1, "000002": k2})
    res = bt.run()

    assert res.total_trades > 0
    assert len(res.equity_curve) > 0
    assert res.equity_curve[-1][1] > 1.0
    assert res.final_capital > res.initial_capital
    assert res.win_rate == pytest.approx(1.0)


def test_backtest_stop_loss_triggers_in_downtrend():
    """下跌趋势应触发止损，胜率不为 100%。"""
    k1 = _make_uptrend_kline("000001", n=45, start_date="20231201")
    # 接上 15 天急跌，让买入后立刻止损
    k_down = _make_downfall_kline("000001", n=15, start_date="20240115")
    k_full = pd.concat([k1, k_down], ignore_index=True)
    # 修正日期递增
    from datetime import datetime, timedelta
    base = datetime.strptime("20231201", "%Y%m%d")
    k_full["日期"] = [
        (base + timedelta(days=i)).strftime("%Y%m%d")
        for i in range(len(k_full))
    ]

    bt = _make_backtester(["000001"], {"000001": k_full})
    res = bt.run()

    assert res.total_trades > 0
    # 下跌段会产生亏损交易
    assert res.winning_trades < res.total_trades
    assert res.max_drawdown > 0


def test_backtest_sentiment_threshold_skip():
    """情绪低于阈值时不选股，无交易。"""
    k1 = _make_uptrend_kline("000001", n=60, start_date="20231201")
    dl = MockDataLoader({"000001": k1})
    dl.limit_up_count = 0
    dl.broke_count = 100
    dl.limit_down_count = 100
    cfg = BacktestConfig(
        start_date="20240115",
        end_date="20240215",
        watchlist=["000001"],
        sentiment_threshold=40.0,
        rps_threshold=80.0,
        enable_progress=False,
    )
    bt = Backtester(cfg, dl=dl)
    bt.scanner.set_rps_map({"000001": 85.0})
    res = bt.run()
    assert res.total_trades == 0


def test_backtest_win_rate_and_pl_ratio():
    """验证胜率与盈亏比统计正确性。"""
    k1 = _make_uptrend_kline("000001", n=60, start_date="20231201")
    k2 = _make_uptrend_kline("000002", n=60, start_date="20231201")
    bt = _make_backtester(["000001", "000002"], {"000001": k1, "000002": k2})
    res = bt.run()
    # 全上涨序列，理论上所有交易都盈利
    assert res.win_rate == pytest.approx(1.0)
    # 盈亏比在盈利时 avg_loss=0，返回 inf，这里只验证 >=0
    assert res.profit_loss_ratio >= 0


def test_backtest_result_report_and_table():
    """to_report / to_table 输出格式检查。"""
    k1 = _make_uptrend_kline("000001", n=60, start_date="20231201")
    bt = _make_backtester(["000001"], {"000001": k1})
    res = bt.run()

    report = res.to_report()
    assert "# 盘古策略回测报告" in report
    assert "## 净值曲线" in report
    assert res.start_date in report

    table = res.to_table()
    assert table.title == "盘古策略回测结果"
    # rich Table 的 row_count 为内部属性
    assert table.row_count >= 1


# ---------------------------------------------------------------------- #
# 参数敏感性测试
# ---------------------------------------------------------------------- #
def test_backtest_take_profit_none_increases_holding():
    """关闭止盈后，交易应因最大持仓天数或回测结束退出。"""
    k1 = _make_uptrend_kline("000001", n=60, start_date="20231201")
    bt = _make_backtester(["000001"], {"000001": k1})
    bt.cfg.take_profit_method = "none"
    res = bt.run()
    assert res.total_trades > 0
    # 不应出现"止盈"退出
    reasons = {t.exit_reason for t in res.trades}
    assert "止盈" not in reasons


def test_transaction_costs_are_recorded():
    """交易成本（佣金、印花税、滑点）应被正确计入并暴露到结果中。"""
    k1 = _make_uptrend_kline("000001", n=60, start_date="20231201")
    bt = _make_backtester(["000001"], {"000001": k1})
    res = bt.run()

    assert res.total_trades > 0
    assert res.total_commission > 0
    assert res.total_stamp_duty > 0
    assert res.total_slippage > 0
    assert res.total_cost == pytest.approx(
        res.total_commission + res.total_stamp_duty + res.total_slippage
    )
    assert "total_commission" in res.to_dict()["summary"]
    assert "## 交易成本" in res.to_report()


def test_sensitivity_analysis_returns_dataframe():
    """参数敏感性分析应返回包含预期列的 DataFrame。"""
    k1 = _make_uptrend_kline("000001", n=60, start_date="20231201")
    cfg = BacktestConfig(
        start_date="20240115",
        end_date="20240215",
        watchlist=["000001"],
        enable_progress=False,
    )
    dl = MockDataLoader({"000001": k1})
    bt = Backtester(cfg, dl=dl)
    bt.scanner.set_rps_map({"000001": 85.0})

    df = sensitivity_analysis(
        cfg,
        dl=dl,
        sentiment_thresholds=[35.0, 45.0],
        rps_thresholds=[75.0],
        max_holding_days_list=[5, 10],
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    expected_cols = {
        "sentiment_threshold",
        "rps_threshold",
        "max_holding_days",
        "total_return_pct",
        "total_trades",
    }
    assert expected_cols.issubset(set(df.columns))
