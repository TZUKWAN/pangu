"""回测系统：验证「情绪>40才选股 + 趋势选股 + entry_exit买卖点 + 止损止盈」策略有效性。

设计取舍（简化版）：
- 日粒度，用历史日 K 模拟价格，不追求 tick 级精确。
- 情绪判断：用当日历史涨停/跌停/炸板数据跑 sentiment_meter。
- 趋势选股：trend_scanner 内部依赖 all_spot（仅实时），历史回看时通过
  HistoricalDataLoader 把 watchlist 的日 K 重构成 spot 快照，保证 PIT-safe。
- 买入：选出后次日开盘价执行（含滑点 + 佣金）。
- 平仓：持仓至少过夜；次日以开盘价/收盘价判断止损/止盈，跳空时按开盘价成交。
- 交易成本：佣金双边、卖出印花税、滑点均计入盈亏。
- 进度：rich.progress 显示。

本模块不修改 engine 其他文件，仅新增。
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .data_loader import DataLoader
from .entry_exit import EntryExitEngine, EntryExitResult
from .sentiment_meter import SentimentMeter
from .trend_scanner import StockCandidate, TrendScanner

logger = logging.getLogger("pangu.backtest")


# ---------------------------------------------------------------------- #
# 配置与数据结构
# ---------------------------------------------------------------------- #
@dataclass
class BacktestConfig:
    """回测参数。"""

    start_date: str
    end_date: str
    watchlist: list[str] = field(default_factory=list)
    initial_capital: float = 1_000_000.0
    sentiment_threshold: float = 40.0
    rps_threshold: float = 80.0
    max_positions: int = 5
    max_holding_days: int = 20
    position_account_fraction: float = 1.0
    stop_loss_method: str = "auto"  # auto / atr / structure / ma20
    take_profit_method: str = "2r"  # 2r / 3r / none
    enable_progress: bool = True
    benchmark_code: str = "000300"  # 沪深300
    db_path: str = "data/pangu.db"
    sentiment_cfg: dict[str, Any] = field(default_factory=dict)
    trend_cfg: dict[str, Any] = field(default_factory=dict)
    entry_exit_cfg: dict[str, Any] = field(default_factory=dict)

    # 交易成本（A 股真实成本近似）
    stamp_duty_rate: float = 0.0005   # 卖出时印花税（仅卖出收取）
    commission_rate: float = 0.00025  # 双边佣金费率
    min_commission: float = 5.0       # 单笔最低佣金（元）
    slippage_pct: float = 0.001       # 买卖滑点（成交价比例）

    @classmethod
    def from_settings(
        cls,
        start_date: str,
        end_date: str,
        settings_path: str | Path = "config/settings.yaml",
        **overrides: Any,
    ) -> "BacktestConfig":
        """从 settings.yaml 加载，再覆盖传入参数。"""
        path = Path(settings_path)
        cfg: dict[str, Any] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        kwargs = {
            "start_date": start_date,
            "end_date": end_date,
            "sentiment_cfg": cfg.get("sentiment", {}),
            "trend_cfg": cfg.get("trend", {}),
            "entry_exit_cfg": cfg.get("entry_exit", {}),
        }
        # settings.yaml 中的关键阈值可透传
        entry_cfg = cfg.get("entry_exit", {})
        trend_stock = cfg.get("trend", {}).get("stock", {})
        kwargs["rps_threshold"] = trend_stock.get("rps_min", 80.0)
        kwargs["max_holding_days"] = entry_cfg.get("max_holding_days", 20)

        # 回测相关配置（可选）
        bt_cfg = cfg.get("backtest", {})
        kwargs["initial_capital"] = bt_cfg.get("initial_capital", 1_000_000.0)
        kwargs["max_positions"] = bt_cfg.get("max_positions", 5)
        kwargs["stamp_duty_rate"] = bt_cfg.get("stamp_duty_rate", 0.0005)
        kwargs["commission_rate"] = bt_cfg.get("commission_rate", 0.00025)
        kwargs["min_commission"] = bt_cfg.get("min_commission", 5.0)
        kwargs["slippage_pct"] = bt_cfg.get("slippage_pct", 0.001)

        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class TradeRecord:
    """单笔交易记录。"""

    code: str
    name: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    stop_loss: float
    take_profit: float
    exit_reason: str  # 止损 / 止盈 / 到期 / 其他
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": round(self.entry_price, 3),
            "exit_price": round(self.exit_price, 3),
            "shares": self.shares,
            "stop_loss": round(self.stop_loss, 3),
            "take_profit": round(self.take_profit, 3),
            "exit_reason": self.exit_reason,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "holding_days": self.holding_days,
        }


@dataclass
class Position:
    """回测中的持仓（内部用）。"""

    code: str
    name: str
    entry_date: str
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    entry_idx: int  # 在交易日序列中的索引


@dataclass
class BacktestResult:
    """回测结果统计。"""

    cfg: BacktestConfig
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    win_rate: float
    profit_loss_ratio: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    benchmark_curve: list[tuple[str, float]] = field(default_factory=list)
    benchmark_return: float = 0.0
    monthly_returns: dict[str, float] = field(default_factory=dict)

    # 交易成本明细
    total_commission: float = 0.0   # 总佣金
    total_stamp_duty: float = 0.0   # 总印花税
    total_slippage: float = 0.0     # 总滑点成本
    total_cost: float = 0.0         # 交易成本合计

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "initial_capital": self.initial_capital,
                "sentiment_threshold": self.cfg.sentiment_threshold,
                "rps_threshold": self.cfg.rps_threshold,
                "max_positions": self.cfg.max_positions,
                "max_holding_days": self.cfg.max_holding_days,
                "stamp_duty_rate": self.cfg.stamp_duty_rate,
                "commission_rate": self.cfg.commission_rate,
                "min_commission": self.cfg.min_commission,
                "slippage_pct": self.cfg.slippage_pct,
            },
            "summary": {
                "final_capital": round(self.final_capital, 2),
                "total_return_pct": round(self.total_return * 100, 2),
                "annual_return_pct": round(self.annual_return * 100, 2),
                "win_rate_pct": round(self.win_rate * 100, 2),
                "profit_loss_ratio": round(self.profit_loss_ratio, 2),
                "max_drawdown_pct": round(self.max_drawdown * 100, 2),
                "sharpe_ratio": round(self.sharpe_ratio, 3),
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "benchmark_return_pct": round(self.benchmark_return * 100, 2),
                "total_commission": round(self.total_commission, 2),
                "total_stamp_duty": round(self.total_stamp_duty, 2),
                "total_slippage": round(self.total_slippage, 2),
                "total_cost": round(self.total_cost, 2),
            },
            "equity_curve": [
                {"date": d, "nav": round(v, 4)} for d, v in self.equity_curve
            ],
            "benchmark_curve": [
                {"date": d, "nav": round(v, 4)} for d, v in self.benchmark_curve
            ],
            "monthly_returns": {
                k: round(v * 100, 2) for k, v in self.monthly_returns.items()
            },
            "trades": [t.to_dict() for t in self.trades],
        }

    def to_report(self) -> str:
        """返回 Markdown 报告字符串，含收益曲线数据。"""
        lines = [
            "# 盘古策略回测报告",
            "",
            "## 参数",
            f"- 回测区间：{self.start_date} ~ {self.end_date}",
            f"- 初始资金：{self.initial_capital:,.0f}",
            f"- 情绪阈值：{self.cfg.sentiment_threshold}",
            f"- RPS 阈值：{self.cfg.rps_threshold}",
            f"- 最大持仓：{self.cfg.max_positions}",
            f"- 最大持仓天数：{self.cfg.max_holding_days}",
            f"- 佣金费率：{self.cfg.commission_rate * 10000:.2f}‱",
            f"- 印花税率：{self.cfg.stamp_duty_rate * 10000:.2f}‱（仅卖出）",
            f"- 滑点：{self.cfg.slippage_pct * 100:.2f}%",
            "",
            "## 收益统计",
            f"| 指标 | 策略 | 沪深300 |",
            f"|---|---|---|",
            f"| 总收益率 | {self.total_return*100:.2f}% | {self.benchmark_return*100:.2f}% |",
            f"| 年化收益率 | {self.annual_return*100:.2f}% | - |",
            f"| 胜率 | {self.win_rate*100:.2f}% | - |",
            f"| 盈亏比 | {self.profit_loss_ratio:.2f} | - |",
            f"| 最大回撤 | {self.max_drawdown*100:.2f}% | - |",
            f"| 夏普比率 | {self.sharpe_ratio:.3f} | - |",
            f"| 总交易笔数 | {self.total_trades} | - |",
            "",
            "## 交易成本",
            f"| 项目 | 金额（元） |",
            f"|---|---|",
            f"| 总佣金 | {self.total_commission:,.2f} |",
            f"| 总印花税 | {self.total_stamp_duty:,.2f} |",
            f"| 总滑点 | {self.total_slippage:,.2f} |",
            f"| 总成本 | {self.total_cost:,.2f} |",
            "",
            "## 净值曲线",
            "| 日期 | 策略净值 | 基准净值 |",
            "|---|---|---|",
        ]
        bench_map = dict(self.benchmark_curve)
        for d, nav in self.equity_curve:
            bnav = bench_map.get(d, 1.0)
            lines.append(f"| {d} | {nav:.4f} | {bnav:.4f} |")

        lines += [
            "",
            "## 月度收益分布",
            "| 月份 | 收益率 |",
            "|---|---|",
        ]
        for month, ret in sorted(self.monthly_returns.items()):
            lines.append(f"| {month} | {ret*100:.2f}% |")

        lines += [
            "",
            "## 交易明细",
            "| 代码 | 名称 | 买入日 | 卖出日 | 买入价 | 卖出价 | 股数 | 盈亏% | 原因 | 持仓天数 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t in self.trades:
            lines.append(
                f"| {t.code} | {t.name} | {t.entry_date} | {t.exit_date} | "
                f"{t.entry_price:.2f} | {t.exit_price:.2f} | {t.shares} | "
                f"{t.pnl_pct*100:.2f}% | {t.exit_reason} | {t.holding_days} |"
            )
        lines.append("")
        return "\n".join(lines)

    def to_table(self) -> Table:
        """返回 rich Table。"""
        table = Table(title="盘古策略回测结果")
        table.add_column("指标", justify="left")
        table.add_column("数值", justify="right")
        table.add_row("回测区间", f"{self.start_date} ~ {self.end_date}")
        table.add_row("初始资金", f"{self.initial_capital:,.0f}")
        table.add_row("最终资金", f"{self.final_capital:,.2f}")
        table.add_row("总收益率", f"{self.total_return*100:.2f}%")
        table.add_row("年化收益率", f"{self.annual_return*100:.2f}%")
        table.add_row("胜率", f"{self.win_rate*100:.2f}%")
        table.add_row("盈亏比", f"{self.profit_loss_ratio:.2f}")
        table.add_row("最大回撤", f"{self.max_drawdown*100:.2f}%")
        table.add_row("夏普比率", f"{self.sharpe_ratio:.3f}")
        table.add_row("总交易笔数", str(self.total_trades))
        table.add_row("基准收益率", f"{self.benchmark_return*100:.2f}%")
        table.add_row("总佣金", f"{self.total_commission:,.2f}")
        table.add_row("总印花税", f"{self.total_stamp_duty:,.2f}")
        table.add_row("总滑点", f"{self.total_slippage:,.2f}")
        table.add_row("总成本", f"{self.total_cost:,.2f}")
        return table


# ---------------------------------------------------------------------- #
# PIT-safe 数据包装器
# ---------------------------------------------------------------------- #
class HistoricalDataLoader:
    """为回测构造 PIT-safe 数据视图。

    trend_scanner / sentiment_meter 内部会调用 all_spot / concept_boards 等实时接口，
    历史回看时本包装器把这些接口替换为 watchlist 日 K 快照。
    涨停/跌停/炸板仍透传给底层 DataLoader（akshare 仅保留近期，历史为空属已知限制）。
    """

    def __init__(
        self,
        dl: DataLoader,
        watchlist: list[str],
        watchlist_names: dict[str, str] | None = None,
    ) -> None:
        self.dl = dl
        self.watchlist = [c.strip() for c in watchlist]
        self.watchlist_names = watchlist_names or {}
        self.current_date: Optional[str] = None
        self._klines: dict[str, pd.DataFrame] = {}
        self._date_index: pd.DatetimeIndex | None = None

    def preload(self, start_date: str, end_date: str) -> None:
        """预加载 watchlist 全部日 K，并生成交易日序列。"""
        logger.info("预加载 %d 只标的日 K ...", len(self.watchlist))
        all_dates: set[str] = set()
        for code in self.watchlist:
            df = self.dl.daily_kline(code, days=252 * 2, date=end_date)
            if len(df) == 0:
                logger.warning("%s 无历史 K 线", code)
                continue
            df = self._normalize_kline(df)
            self._klines[code] = df
            all_dates.update(df["date"].astype(str).tolist())

        # 交易日序列：落在 [start, end] 区间且至少有一只票有数据
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        dates = sorted(
            d for d in all_dates
            if start_dt <= datetime.strptime(d, "%Y%m%d") <= end_dt
        )
        self._date_index = pd.DatetimeIndex(
            pd.to_datetime(dates, format="%Y%m%d")
        )
        logger.info("预加载完成：共 %d 个交易日", len(dates))

    def trading_days(self, start_date: str, end_date: str) -> list[str]:
        """返回区间内的交易日列表（YYYYMMDD）。"""
        if self._date_index is None:
            raise RuntimeError("请先调用 preload()")
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        return [
            d.strftime("%Y%m%d")
            for d in self._date_index
            if start_dt <= d <= end_dt
        ]

    def kline(self, code: str) -> pd.DataFrame:
        return self._klines.get(code, pd.DataFrame())

    def kline_on(self, code: str, date: str) -> pd.Series | None:
        """取某股票某日的 K 线（PIT-safe：只取到该日为止的数据）。"""
        df = self._klines.get(code)
        if df is None or len(df) == 0:
            return None
        mask = df["date"] <= date
        if not mask.any():
            return None
        # 返回该日所在行（若该日无交易则取最近一个交易日）
        return df[mask].iloc[-1]

    def kline_after(self, code: str, date: str) -> pd.Series | None:
        """取 date 之后第一个交易日的 K 线。"""
        df = self._klines.get(code)
        if df is None or len(df) == 0:
            return None
        mask = df["date"] > date
        if not mask.any():
            return None
        return df[mask].iloc[0]

    # ------------------------------------------------------------------ #
    # 覆盖 DataLoader 方法
    # ------------------------------------------------------------------ #
    def all_spot(self) -> pd.DataFrame:
        """把 watchlist 重构成 spot 快照。"""
        if self.current_date is None:
            return pd.DataFrame()
        rows = []
        for code in self.watchlist:
            row = self.kline_on(code, self.current_date)
            if row is None:
                continue
            rows.append({
                "代码": code,
                "名称": self.watchlist_names.get(code, code),
                "最新价": float(row.get("收盘", row.get("close", 0))),
                "涨跌幅": float(row.get("涨跌幅", row.get("pct_change", 0))),
                "换手率": float(row.get("换手率", row.get("turnover", 0))),
                "流通市值": float(row.get("流通市值", row.get("circ_mv", 0))),
            })
        return pd.DataFrame(rows)

    def concept_boards(self) -> pd.DataFrame:
        """返回一个 synthetic 板块，供趋势扫描使用。"""
        return pd.DataFrame({
            "排名": [1],
            "板块名称": ["回测板块"],
            "板块代码": ["BK9999"],
            "最新价": [1.0],
            "涨跌额": [0.0],
            "涨跌幅": [0.0],
            "总市值": [1.0],
            "换手率": [0.0],
            "上涨家数": [len(self.watchlist)],
            "下跌家数": [0],
            "领涨股票": [""],
            "涨跌幅.1": [0.0],
        })

    def concept_constituents(self, board_symbol: str, board_name: str | None = None) -> pd.DataFrame:
        """返回 watchlist 作为该 synthetic 板块成分股。"""
        rows = []
        for code in self.watchlist:
            row = self.kline_on(code, self.current_date or "20991231")
            if row is None:
                continue
            rows.append({
                "序号": len(rows) + 1,
                "代码": code,
                "名称": self.watchlist_names.get(code, code),
                "最新价": float(row.get("收盘", row.get("close", 0))),
                "涨跌幅": float(row.get("涨跌幅", row.get("pct_change", 0))),
                "涨跌额": 0.0,
                "成交量": float(row.get("成交量", row.get("volume", 0))),
                "成交额": float(row.get("成交额", row.get("amount", 0))),
                "振幅": float(row.get("振幅", row.get("amplitude", 0))),
                "最高": float(row.get("最高", row.get("high", 0))),
                "最低": float(row.get("最低", row.get("low", 0))),
                "换手率": float(row.get("换手率", row.get("turnover", 0))),
                "市盈率-动态": 0.0,
                "市净率": 0.0,
                "流通市值": float(row.get("流通市值", row.get("circ_mv", 0))),
            })
        return pd.DataFrame(rows)

    def sector_fund_flow_rank(self, indicator: str = "今日") -> pd.DataFrame:
        """历史资金流不可得，返回空。"""
        return pd.DataFrame()

    def individual_fund_flow(self, symbol: str, fast: bool = False) -> pd.DataFrame:
        """历史资金流不可得，返回空（trend_scanner 资金项不强制淘汰）。"""
        return pd.DataFrame()

    def limit_up_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        return self.dl.limit_up_pool(date=date)

    def broke_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        return self.dl.broke_pool(date=date)

    def limit_down_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        return self.dl.limit_down_pool(date=date)

    def daily_kline(
        self,
        symbol: str,
        days: int = 60,
        adjust: str = "qfq",
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """优先用预加载 K 线切片，不足再透传。"""
        df = self._klines.get(symbol)
        if df is not None and len(df) > 0:
            end = date or self.current_date
            if end:
                df = df[df["date"] <= end].copy()
            if len(df) > days:
                df = df.tail(days).reset_index(drop=True)
            return df
        return self.dl.daily_kline(symbol, days=days, adjust=adjust, date=date)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_kline(df: pd.DataFrame) -> pd.DataFrame:
        """把 akshare 日 K 统一成标准列名，便于后续计算。

        为兼容 engine.trend_scanner / engine.entry_exit 的列名查找逻辑，
        价格/成交量列保留中文名（它们优先按中文列名匹配，fallback 才用列索引）。
        额外增加 "date" 字符串列和 "流通市值" 列供 HistoricalDataLoader 内部使用。
        """
        # 保留原中文列，增加统一访问列
        if "日期" in df.columns and "date" not in df.columns:
            df["date"] = pd.to_datetime(df["日期"]).dt.strftime("%Y%m%d")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")

        numeric_cols = {
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
        for cn, en in numeric_cols.items():
            if cn in df.columns:
                df[cn] = pd.to_numeric(df[cn], errors="coerce")
            elif en in df.columns and cn not in df.columns:
                # 若原表是英文列，复制一列中文名，保证下游 _find_col 能命中
                df[cn] = pd.to_numeric(df[en], errors="coerce")
            # 统一增加英文别名，方便 HistoricalDataLoader 内部按英文访问
            if cn in df.columns and en not in df.columns:
                df[en] = df[cn]

        # 流通市值：akshare hist 不返回，用收盘价 × 固定股本近似（仅用于过滤）。
        # 回测场景下该值只影响趋势扫描的市值过滤，不影响盈亏。
        # 已知限制：HistoricalDataLoader 无法提供历史 PE/PB 估值数据，
        # 回测时 quant_guard 的估值过滤会跳过（spot_row 为 None）。
        close = pd.to_numeric(df.get("收盘", df.get("close", pd.Series(dtype=float))), errors="coerce")
        if "流通市值" not in df.columns or df["流通市值"].isna().all():
            df["流通市值"] = close * 1e9  # 假设 10 亿股本，仅做占位过滤
        if "circ_mv" not in df.columns or df["circ_mv"].isna().all():
            df["circ_mv"] = df["流通市值"]
        # 估值字段：历史日 K 不提供 PE/PB，回测时 quant_guard 会跳过估值检查
        if "市盈率-动态" not in df.columns:
            df["市盈率-动态"] = float("nan")
        if "市净率" not in df.columns:
            df["市净率"] = float("nan")
        return df


# ---------------------------------------------------------------------- #
# 回测引擎
# ---------------------------------------------------------------------- #
class Backtester:
    """策略回测引擎。"""

    def __init__(
        self,
        cfg: BacktestConfig,
        dl: Optional[DataLoader] = None,
    ) -> None:
        self.cfg = cfg
        self.dl = dl or DataLoader()
        self.hdl = HistoricalDataLoader(
            self.dl,
            cfg.watchlist,
            {c: c for c in cfg.watchlist},
        )
        self.sentiment = SentimentMeter(self.hdl, cfg.sentiment_cfg)
        self.scanner = TrendScanner(self.hdl, cfg.trend_cfg)
        self.entry_exit = EntryExitEngine(self.hdl, cfg.entry_exit_cfg)

        # 交易成本累计
        self.total_commission: float = 0.0
        self.total_stamp_duty: float = 0.0
        self.total_slippage: float = 0.0

    # ------------------------------------------------------------------ #
    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> BacktestResult:
        """跑完整回测。"""
        start = start_date or self.cfg.start_date
        end = end_date or self.cfg.end_date
        self.hdl.preload(start, end)
        dates = self.hdl.trading_days(start, end)
        if len(dates) < 2:
            logger.error("回测区间交易日不足：%d 天", len(dates))
            return self._empty_result(start, end)

        # 注入真实 RPS 表（若存在）
        try:
            rps_map = self._load_rps_map(dates)
            if rps_map:
                self.scanner.set_rps_map(rps_map)
                logger.info("已加载真实 RPS 表：%d 只", len(rps_map))
        except Exception as e:  # noqa: BLE001
            logger.debug("RPS 表加载失败：%s", e)

        cash = self.cfg.initial_capital
        active: list[Position] = []
        pending: list[tuple[StockCandidate, EntryExitResult]] = []
        trades: list[TradeRecord] = []
        equity_curve: list[tuple[str, float]] = []

        progress = self._build_progress()
        with progress:
            task = progress.add_task(f"[cyan]回测 {start}~{end}", total=len(dates))
            for i, date in enumerate(dates):
                self.hdl.current_date = date
                progress.update(task, description=f"[cyan]{date} 持仓{len(active)} 待买{len(pending)}")

                # 1. 执行昨日决策的买入（今日开盘价，含滑点/佣金）
                for cand, ee in pending:
                    row_next = self.hdl.kline_on(cand.code, date)
                    if row_next is None or pd.isna(row_next.get("open")):
                        continue
                    raw_entry = float(row_next["open"])
                    exec_entry = raw_entry * (1 + self.cfg.slippage_pct)
                    shares = self._calc_shares(cand, ee, cash)
                    if shares <= 0 or exec_entry <= 0:
                        continue
                    cost = shares * exec_entry
                    commission = max(cost * self.cfg.commission_rate, self.cfg.min_commission)
                    if cost + commission > cash:
                        # 现金不足，按可用资金缩放（至少 100 股）
                        shares = int(cash / exec_entry // 100 * 100)
                        if shares <= 0:
                            continue
                        cost = shares * exec_entry
                        commission = max(cost * self.cfg.commission_rate, self.cfg.min_commission)
                        if cost + commission > cash:
                            continue
                    cash -= cost + commission
                    self.total_commission += commission
                    self.total_slippage += (exec_entry - raw_entry) * shares
                    active.append(Position(
                        code=cand.code,
                        name=cand.name,
                        entry_date=date,
                        entry_price=exec_entry,
                        shares=shares,
                        stop_loss=ee.stop_loss.price if ee.stop_loss else exec_entry * 0.95,
                        take_profit=self._select_take_profit(ee, exec_entry),
                        entry_idx=i,
                    ))
                pending.clear()

                # 2. 持仓平仓检查（次日规则：当日买入的仓位不检查；
                #   其余仓位用今日开盘价/收盘价判断止损/止盈/到期）
                still_active: list[Position] = []
                for pos in active:
                    row = self.hdl.kline_on(pos.code, date)
                    if row is None:
                        still_active.append(pos)
                        continue

                    # 当日买入的仓位最早明天才能平仓
                    if pos.entry_idx >= i:
                        still_active.append(pos)
                        continue

                    open_ = float(row["open"])
                    close_ = float(row.get("close", open_))
                    raw_exit: float | None = None
                    reason = ""

                    if open_ <= pos.stop_loss:
                        raw_exit = open_
                        reason = "止损"
                    elif open_ >= pos.take_profit and self.cfg.take_profit_method != "none":
                        raw_exit = open_
                        reason = "止盈"
                    elif close_ <= pos.stop_loss:
                        raw_exit = close_
                        reason = "止损"
                    elif close_ >= pos.take_profit and self.cfg.take_profit_method != "none":
                        raw_exit = close_
                        reason = "止盈"
                    elif i - pos.entry_idx >= self.cfg.max_holding_days:
                        raw_exit = open_
                        reason = "到期"

                    if raw_exit is not None:
                        exec_exit = raw_exit * (1 - self.cfg.slippage_pct)
                        proceeds = exec_exit * pos.shares
                        commission = max(proceeds * self.cfg.commission_rate, self.cfg.min_commission)
                        stamp_duty = proceeds * self.cfg.stamp_duty_rate
                        cash += proceeds - commission - stamp_duty
                        self.total_commission += commission
                        self.total_stamp_duty += stamp_duty
                        self.total_slippage += (raw_exit - exec_exit) * pos.shares
                        holding_days = i - pos.entry_idx
                        pnl = pos.shares * (exec_exit - pos.entry_price)
                        trades.append(TradeRecord(
                            code=pos.code,
                            name=pos.name,
                            entry_date=pos.entry_date,
                            exit_date=date,
                            entry_price=pos.entry_price,
                            exit_price=exec_exit,
                            shares=pos.shares,
                            stop_loss=pos.stop_loss,
                            take_profit=pos.take_profit,
                            exit_reason=reason,
                            pnl=pnl,
                            pnl_pct=(exec_exit - pos.entry_price) / pos.entry_price,
                            holding_days=holding_days,
                        ))
                    else:
                        still_active.append(pos)
                active = still_active

                # 3. 收盘后权益
                mtm = self._mark_to_market(active, date)
                equity = cash + mtm
                equity_curve.append((date, equity))

                # 4. 产生明日买入信号（最后一天不选股）
                if i < len(dates) - 1:
                    new_pending = self._select_candidates(
                        date, len(active) + len(pending), equity,
                    )
                    pending.extend(new_pending)

                progress.advance(task)

        # 尾盘强平剩余持仓
        if active:
            last_date = dates[-1]
            for pos in active:
                row = self.hdl.kline_on(pos.code, last_date)
                raw_exit = float(row["close"]) if row is not None else pos.entry_price
                exec_exit = raw_exit * (1 - self.cfg.slippage_pct)
                proceeds = exec_exit * pos.shares
                commission = max(proceeds * self.cfg.commission_rate, self.cfg.min_commission)
                stamp_duty = proceeds * self.cfg.stamp_duty_rate
                cash += proceeds - commission - stamp_duty
                self.total_commission += commission
                self.total_stamp_duty += stamp_duty
                self.total_slippage += (raw_exit - exec_exit) * pos.shares
                holding_days = len(dates) - 1 - pos.entry_idx
                trades.append(TradeRecord(
                    code=pos.code,
                    name=pos.name,
                    entry_date=pos.entry_date,
                    exit_date=last_date,
                    entry_price=pos.entry_price,
                    exit_price=exec_exit,
                    shares=pos.shares,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    exit_reason="回测结束",
                    pnl=pos.shares * (exec_exit - pos.entry_price),
                    pnl_pct=(exec_exit - pos.entry_price) / pos.entry_price,
                    holding_days=holding_days,
                ))
            active.clear()
            mtm = self._mark_to_market(active, last_date)
            equity_curve[-1] = (last_date, cash + mtm)

        return self._build_result(start, end, equity_curve, trades)

    # ------------------------------------------------------------------ #
    # 选股信号
    # ------------------------------------------------------------------ #
    def _select_candidates(
        self,
        date: str,
        current_exposure: int,
        account_size: float,
    ) -> list[tuple[StockCandidate, EntryExitResult]]:
        """在 date 收盘后选股，返回明日待买入列表。"""
        # 情绪过滤
        bd = self.sentiment.measure(date)
        if bd.temperature < self.cfg.sentiment_threshold:
            return []

        # 趋势扫描
        trend = self.scanner.scan(date=date)
        if not trend.candidates:
            return []

        slots = self.cfg.max_positions - current_exposure
        if slots <= 0:
            return []

        pending: list[tuple[StockCandidate, EntryExitResult]] = []
        for cand in trend.candidates:
            if cand.rps < self.cfg.rps_threshold:
                continue
            ee = self.entry_exit.compute(
                cand,
                temperature=bd.temperature,
                account_size=account_size,
            )
            if ee.stop_loss is None or not ee.buy_points:
                continue
            # 按配置选定止损方法（entry_exit 已动态选择，这里做校验/覆盖提示）
            ee = self._apply_stop_loss_method(ee)
            pending.append((cand, ee))
            if len(pending) >= slots:
                break
        return pending

    def _apply_stop_loss_method(self, ee: EntryExitResult) -> EntryExitResult:
        """根据配置调整止损（entry_exit 已自动选最合理止损，这里仅做校验）。"""
        if self.cfg.stop_loss_method == "auto" or ee.stop_loss is None:
            return ee
        # TODO: 如需强制指定单一止损方法，可在此根据 ee.buy_points + 原始 K 线重新计算。
        # 当前 engine.entry_exit 的动态选择已足够合理，避免重复实现。
        return ee

    def _select_take_profit(self, ee: EntryExitResult, entry_price: float) -> float:
        """根据止盈方法选择目标价。"""
        if self.cfg.take_profit_method == "none" or not ee.take_profit:
            return entry_price * 1e6  # 永不触发
        if self.cfg.take_profit_method == "3r" and len(ee.take_profit) > 1:
            return ee.take_profit[1].price
        return ee.take_profit[0].price

    # ------------------------------------------------------------------ #
    # 仓位与市值
    # ------------------------------------------------------------------ #
    def _calc_shares(
        self,
        cand: StockCandidate,
        ee: EntryExitResult,
        cash: float,
    ) -> int:
        """计算买入股数，100 股取整。"""
        if ee.position is not None and ee.position.shares > 0:
            return ee.position.shares
        # fallback：均分资金
        alloc = cash * self.cfg.position_account_fraction / self.cfg.max_positions
        entry = cand.close
        if entry <= 0:
            return 0
        return int(alloc / entry // 100 * 100)

    def _mark_to_market(self, active: list[Position], date: str) -> float:
        """按收盘价计算持仓市值。"""
        total = 0.0
        for pos in active:
            row = self.hdl.kline_on(pos.code, date)
            if row is None:
                continue
            total += pos.shares * float(row.get("close", pos.entry_price))
        return total

    # ------------------------------------------------------------------ #
    # 结果统计
    # ------------------------------------------------------------------ #
    def _build_result(
        self,
        start: str,
        end: str,
        equity_curve: list[tuple[str, float]],
        trades: list[TradeRecord],
    ) -> BacktestResult:
        initial = self.cfg.initial_capital
        final = equity_curve[-1][1] if equity_curve else initial
        total_ret = (final - initial) / initial if initial else 0.0

        start_dt = datetime.strptime(start, "%Y%m%d")
        end_dt = datetime.strptime(end, "%Y%m%d")
        years = max((end_dt - start_dt).days / 365.25, 1e-6)
        # 避免 total_ret <= -1 时幂运算异常
        annual_ret = (1 + max(total_ret, -0.9999)) ** (1 / years) - 1

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        navs = [v for _, v in equity_curve]
        max_dd = self._max_drawdown(navs)
        sharpe = self._sharpe(navs)

        monthly = self._monthly_returns(equity_curve)

        bench_curve, bench_ret = self._benchmark_curve(start, end)

        # 权益曲线统一为净值（初始=1.0），便于前端展示
        nav_curve = [(d, v / initial) for d, v in equity_curve] if initial else equity_curve

        total_cost = self.total_commission + self.total_stamp_duty + self.total_slippage

        return BacktestResult(
            cfg=self.cfg,
            start_date=start,
            end_date=end,
            initial_capital=initial,
            final_capital=final,
            total_return=total_ret,
            annual_return=annual_ret,
            win_rate=win_rate,
            profit_loss_ratio=pl_ratio,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            trades=trades,
            equity_curve=nav_curve,
            benchmark_curve=bench_curve,
            benchmark_return=bench_ret,
            monthly_returns=monthly,
            total_commission=self.total_commission,
            total_stamp_duty=self.total_stamp_duty,
            total_slippage=self.total_slippage,
            total_cost=total_cost,
        )

    def _empty_result(self, start: str, end: str) -> BacktestResult:
        return BacktestResult(
            cfg=self.cfg,
            start_date=start,
            end_date=end,
            initial_capital=self.cfg.initial_capital,
            final_capital=self.cfg.initial_capital,
            total_return=0.0,
            annual_return=0.0,
            win_rate=0.0,
            profit_loss_ratio=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )

    @staticmethod
    def _max_drawdown(equity: list[float]) -> float:
        """最大回撤。"""
        if not equity:
            return 0.0
        peak = equity[0]
        dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = max(dd, (peak - v) / peak)
        return dd

    @staticmethod
    def _sharpe(equity: list[float], risk_free: float = 0.0) -> float:
        """简化夏普：按日收益率年化。"""
        if len(equity) < 2:
            return 0.0
        rets = np.diff(equity) / np.array(equity[:-1])
        if len(rets) == 0 or rets.std() == 0:
            return 0.0
        return (rets.mean() - risk_free / 252) / rets.std() * np.sqrt(252)

    @staticmethod
    def _monthly_returns(equity_curve: list[tuple[str, float]]) -> dict[str, float]:
        """月度收益分布。"""
        monthly: dict[str, list[float]] = {}
        for date, nav in equity_curve:
            month = date[:6]
            monthly.setdefault(month, []).append(nav)
        result: dict[str, float] = {}
        prev_nav: float | None = None
        for month in sorted(monthly):
            navs = monthly[month]
            if prev_nav is None:
                result[month] = 0.0
            else:
                result[month] = (navs[-1] - prev_nav) / prev_nav
            prev_nav = navs[-1]
        return result

    # ---------------------------------------------------------------------- #
    # 基准
    # ---------------------------------------------------------------------- #
    def _benchmark_curve(
        self,
        start: str,
        end: str,
    ) -> tuple[list[tuple[str, float]], float]:
        """取沪深300同期净值曲线。"""
        try:
            import akshare as ak
            start_f = datetime.strptime(start, "%Y%m%d").strftime("%Y%m%d")
            end_f = datetime.strptime(end, "%Y%m%d").strftime("%Y%m%d")
            # 优先用指数日行情
            df = pd.DataFrame()
            for func_name, kwargs in [
                ("stock_zh_index_daily_em", {"symbol": self.cfg.benchmark_code,
                                              "start_date": start_f, "end_date": end_f}),
                ("stock_zh_index_daily", {"symbol": f"sh{self.cfg.benchmark_code}"}),
            ]:
                func = getattr(ak, func_name, None)
                if func is None:
                    continue
                try:
                    df = func(**kwargs)
                    if len(df) > 0:
                        break
                except Exception:  # noqa: BLE001
                    continue
            if len(df) == 0:
                return self._flat_benchmark(start, end)

            close_col = "收盘" if "收盘" in df.columns else (
                "close" if "close" in df.columns else df.columns[-1]
            )
            date_col = "日期" if "日期" in df.columns else (
                "date" if "date" in df.columns else df.columns[0]
            )
            df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y%m%d")
            df = df[(df[date_col] >= start) & (df[date_col] <= end)].sort_values(date_col)
            closes = pd.to_numeric(df[close_col], errors="coerce").dropna()
            if len(closes) == 0:
                return self._flat_benchmark(start, end)

            base = float(closes.iloc[0])
            curve = [(d, float(c) / base) for d, c in zip(df[date_col], closes)]
            ret = float(closes.iloc[-1]) / base - 1
            return curve, ret
        except Exception as e:  # noqa: BLE001
            logger.warning("基准数据获取失败：%s", e)
            return self._flat_benchmark(start, end)

    def _flat_benchmark(self, start: str, end: str) -> tuple[list[tuple[str, float]], float]:
        """无基准数据时返回 1.0 平线。"""
        days = self.hdl.trading_days(start, end)
        return [(d, 1.0) for d in days], 0.0

    # ---------------------------------------------------------------------- #
    # 辅助
    # ---------------------------------------------------------------------- #
    def _build_progress(self) -> Progress:
        if not self.cfg.enable_progress:
            return Progress(disable=True)
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

    def _load_rps_map(self, dates: list[str]) -> dict[str, float]:
        """加载回测期内最新的真实 RPS 表。"""
        from . import rps as rps_mod
        # 取区间内最近一个有数据的交易日
        for d in reversed(dates):
            rps_map = rps_mod.load_rps_map(d, self.cfg.db_path)
            if rps_map:
                return rps_map
        return {}


# ---------------------------------------------------------------------- #
# 参数敏感性分析
# ---------------------------------------------------------------------- #
def sensitivity_analysis(
    cfg: BacktestConfig,
    dl: Optional[DataLoader] = None,
    *,
    sentiment_thresholds: Optional[list[float]] = None,
    rps_thresholds: Optional[list[float]] = None,
    max_holding_days_list: Optional[list[int]] = None,
) -> pd.DataFrame:
    """扫描 (sentiment_threshold, rps_threshold, max_holding_days) 参数组合。

    每个组合独立跑一次回测，汇总总收益、年化收益、胜率、最大回撤、夏普、交易次数、总成本。
    默认网格较小，方便快速对比；生产环境可传入更密集的列表。
    """
    sentiment_thresholds = sentiment_thresholds or [30.0, 40.0, 50.0]
    rps_thresholds = rps_thresholds or [70.0, 80.0, 90.0]
    max_holding_days_list = max_holding_days_list or [10, 20, 30]

    rows: list[dict[str, Any]] = []
    for st, rps, mhd in itertools.product(
        sentiment_thresholds, rps_thresholds, max_holding_days_list
    ):
        run_cfg = replace(
            cfg,
            sentiment_threshold=st,
            rps_threshold=rps,
            max_holding_days=mhd,
            enable_progress=False,
        )
        bt = Backtester(run_cfg, dl=dl)
        res = bt.run()
        rows.append({
            "sentiment_threshold": st,
            "rps_threshold": rps,
            "max_holding_days": mhd,
            "total_return_pct": round(res.total_return * 100, 2),
            "annual_return_pct": round(res.annual_return * 100, 2),
            "win_rate_pct": round(res.win_rate * 100, 2),
            "max_drawdown_pct": round(res.max_drawdown * 100, 2),
            "sharpe_ratio": round(res.sharpe_ratio, 3),
            "total_trades": res.total_trades,
            "total_cost": round(res.total_cost, 2),
        })

    return pd.DataFrame(rows)
