"""持仓跟踪模块：SQLite 持久化的实际买卖记录、实时盈亏与事后归因。

设计原则：
1. 单一事实源：所有成交写入 transactions，持仓由 transactions 派生维护。
2. 买/卖接口幂等可重放：重复调用会累加持仓/交易明细。
3. 实时价降级：akshare 取不到最新价时，用最近成交价为参考，保证本地账本始终可读。
4. 与 entry_exit 联动：attribution() 对比系统建议的止损/止盈，帮助用户复盘执行纪律。
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

from .data_loader import DataLoader, safe_float
from .entry_exit import EntryExitEngine

logger = logging.getLogger("pangu.portfolio")

try:
    from rich.table import Table
except ImportError:  # pragma: no cover - 无 rich 时 to_table 不可用
    Table = None


# ---------------------------------------------------------------------- #
# 数据结构
# ---------------------------------------------------------------------- #
@dataclass
class Holding:
    """数据库 holdings 表的内存映射。"""

    code: str
    name: str
    shares: int
    avg_cost: float
    first_buy_date: str
    reason: str = ""


@dataclass
class Transaction:
    """数据库 transactions 表的内存映射。"""

    id: int
    code: str
    name: str
    action: str
    shares: int
    price: float
    amount: float
    date: str
    pnl: Optional[float]
    reason: str = ""


@dataclass
class PositionSnapshot:
    """实时持仓快照（含当前价与浮动盈亏）。"""

    code: str
    name: str
    shares: int
    avg_cost: float
    current_price: float
    market_value: float
    pnl: float
    pnl_pct: float
    reason: str = ""


# ---------------------------------------------------------------------- #
# 主类
# ---------------------------------------------------------------------- #
class PortfolioTracker:
    """持仓跟踪器。

    Args:
        db_path: SQLite 数据库路径，默认 ``data/pangu.db``。
        data_loader: 可选外部 DataLoader（测试时注入 mock）。
    """

    def __init__(
        self,
        db_path: str = "data/pangu.db",
        data_loader: Optional[DataLoader] = None,
    ) -> None:
        self.db_path = db_path
        self.dl = data_loader if data_loader is not None else DataLoader()
        self.entry_exit = EntryExitEngine(self.dl, {})

        # 内存数据库需要复用同一个连接，否则每次 connect 都是新的空库
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    # 数据库连接管理
    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """统一连接管理：内存库复用连接，文件库每次新建并自动关闭。"""
        if self._persistent_conn is not None:
            yield self._persistent_conn
            return
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # 数据库初始化
    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        """创建 holdings / transactions 表。"""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS holdings (
                    code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    shares INTEGER NOT NULL DEFAULT 0,
                    avg_cost REAL NOT NULL DEFAULT 0,
                    first_buy_date TEXT NOT NULL,
                    reason TEXT DEFAULT ''
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('buy','sell')),
                    shares INTEGER NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    date TEXT NOT NULL,
                    pnl REAL,
                    reason TEXT DEFAULT ''
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tx_code ON transactions(code)"
            )
            conn.commit()
        logger.debug("持仓数据库初始化完成: %s", self.db_path)

    # ------------------------------------------------------------------ #
    # 写入：买入 / 卖出
    # ------------------------------------------------------------------ #
    def record_buy(
        self,
        code: str,
        name: str,
        shares: int,
        price: float,
        date: str,
        reason: str = "",
    ) -> None:
        """记录一笔买入，写入 transactions 并更新/新建 holdings。

        Args:
            code: 6 位股票代码，如 "000001"。
            name: 股票名称。
            shares: 买入股数（>0）。
            price: 买入单价（>0）。
            date: 交易日期，格式 ``YYYYMMDD``。
            reason: 买入理由，可选。

        Raises:
            ValueError: 数量或价格不合法。
        """
        if shares <= 0:
            raise ValueError("买入股数必须大于 0")
        if price <= 0:
            raise ValueError("买入价格必须大于 0")

        amount = round(shares * price, 2)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO transactions (code, name, action, shares, price, amount, date, pnl, reason)
                VALUES (?, ?, 'buy', ?, ?, ?, ?, NULL, ?)
                """,
                (code, name, shares, price, amount, date, reason),
            )

            row = cur.execute(
                "SELECT shares, avg_cost, first_buy_date FROM holdings WHERE code = ?",
                (code,),
            ).fetchone()
            if row:
                old_shares, old_avg_cost, first_buy_date = row
                total_cost = old_shares * old_avg_cost + shares * price
                total_shares = old_shares + shares
                new_avg_cost = round(total_cost / total_shares, 3)
                if date < first_buy_date:
                    first_buy_date = date
                cur.execute(
                    """
                    UPDATE holdings
                    SET name = ?, shares = ?, avg_cost = ?, first_buy_date = ?, reason = ?
                    WHERE code = ?
                    """,
                    (name, total_shares, new_avg_cost, first_buy_date, reason, code),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO holdings (code, name, shares, avg_cost, first_buy_date, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (code, name, shares, price, date, reason),
                )
            conn.commit()

        logger.info("买入 %s %s %d 股 @ %.2f", code, name, shares, price)

    def record_sell(
        self,
        code: str,
        shares: int,
        price: float,
        date: str,
        reason: str = "",
    ) -> float:
        """记录一笔卖出，计算已实现盈亏并更新/清仓 holdings。

        Args:
            code: 6 位股票代码。
            shares: 卖出股数（>0）。
            price: 卖出单价（>0）。
            date: 交易日期，格式 ``YYYYMMDD``。
            reason: 卖出理由，可选。

        Returns:
            这笔卖出的已实现盈亏（元，已四舍五入到分）。

        Raises:
            ValueError: 数量/价格不合法，或持仓不足。
        """
        if shares <= 0:
            raise ValueError("卖出股数必须大于 0")
        if price <= 0:
            raise ValueError("卖出价格必须大于 0")

        with self._connect() as conn:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT name, shares, avg_cost FROM holdings WHERE code = ?",
                (code,),
            ).fetchone()
            if row is None:
                raise ValueError(f"没有 {code} 的持仓，无法卖出")
            name, old_shares, avg_cost = row
            if old_shares < shares:
                raise ValueError(
                    f"{code} 持仓不足：持有 {old_shares} 股，尝试卖出 {shares} 股"
                )

            pnl = round((price - avg_cost) * shares, 2)
            amount = round(shares * price, 2)
            cur.execute(
                """
                INSERT INTO transactions (code, name, action, shares, price, amount, date, pnl, reason)
                VALUES (?, ?, 'sell', ?, ?, ?, ?, ?, ?)
                """,
                (code, name, shares, price, amount, date, pnl, reason),
            )

            new_shares = old_shares - shares
            if new_shares == 0:
                cur.execute("DELETE FROM holdings WHERE code = ?", (code,))
            else:
                cur.execute(
                    "UPDATE holdings SET shares = ? WHERE code = ?",
                    (new_shares, code),
                )
            conn.commit()

        logger.info("卖出 %s %s %d 股 @ %.2f，盈亏 %.2f", code, name, shares, price, pnl)
        return pnl

    # ------------------------------------------------------------------ #
    # 查询：持仓 / 交易明细 / 总览
    # ------------------------------------------------------------------ #
    def current_holdings(self) -> list[PositionSnapshot]:
        """返回当前持仓列表，按 akshare 实时价计算浮动盈亏。

        若实时价获取失败或找不到该代码，则 fallback 到最近成交价格，
        并在日志中给出警告。
        """
        prices = self._fetch_realtime_prices()
        snapshots: list[PositionSnapshot] = []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT code, name, shares, avg_cost, reason FROM holdings ORDER BY code"
            ).fetchall()

        for code, name, shares, avg_cost, reason in rows:
            current_price = prices.get(code)
            if current_price is None or current_price <= 0 or pd.isna(current_price):
                current_price = self._last_trade_price(code)
                if current_price <= 0:
                    current_price = avg_cost

            market_value = round(shares * current_price, 2)
            cost = round(shares * avg_cost, 2)
            pnl = round(market_value - cost, 2)
            pnl_pct = round(pnl / cost * 100, 2) if cost > 0 else 0.0

            snapshots.append(
                PositionSnapshot(
                    code=code,
                    name=name,
                    shares=shares,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    reason=reason or "",
                )
            )
        return snapshots

    def transactions(self, limit: int = 100) -> list[Transaction]:
        """返回最近 N 条交易明细，按日期倒序。"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, code, name, action, shares, price, amount, date, pnl, reason
                FROM transactions
                ORDER BY date DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [Transaction(*row) for row in rows]

    def summary(self) -> dict[str, Any]:
        """总览统计：总投入、当前市值、总盈亏、胜率、平均持仓天数。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE action = 'buy'"
            ).fetchone()
            total_invested = float(row[0]) if row else 0.0

            sells = conn.execute(
                "SELECT date, code, pnl FROM transactions WHERE action = 'sell'"
            ).fetchall()

        positions = self.current_holdings()
        market_value = round(sum(p.market_value for p in positions), 2)
        unrealized_pnl = round(sum(p.pnl for p in positions), 2)

        realized_pnl = round(sum(pnl for _, _, pnl in sells if pnl is not None), 2)
        total_pnl = round(realized_pnl + unrealized_pnl, 2)

        wins = sum(1 for _, _, pnl in sells if pnl is not None and pnl > 0)
        sell_count = len(sells)
        win_rate = round(wins / sell_count * 100, 2) if sell_count > 0 else 0.0

        avg_hold_days = self._calc_avg_hold_days(sells)

        return {
            "total_invested": round(total_invested, 2),
            "market_value": market_value,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_hold_days": avg_hold_days,
            "holding_count": len(positions),
            "sell_count": sell_count,
        }

    # ------------------------------------------------------------------ #
    # 归因：对比 entry_exit
    # ------------------------------------------------------------------ #
    def attribution(self, code: str) -> dict[str, Any]:
        """个股盈亏归因：返回买卖记录，并对比系统建议的止损/止盈。

        Args:
            code: 6 位股票代码。

        Returns:
            字典，包含 buys/sells、entry_exit 计划、executed_stop_loss/
            executed_take_profit 等字段。
        """
        buys: list[dict[str, Any]] = []
        sells: list[dict[str, Any]] = []
        name = code

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT action, shares, price, amount, date, pnl, reason
                FROM transactions WHERE code = ? ORDER BY date, id
                """,
                (code,),
            ).fetchall()
            for action, shares, price, amount, date, pnl, reason in rows:
                rec = {
                    "shares": shares,
                    "price": price,
                    "amount": amount,
                    "date": date,
                    "reason": reason or "",
                }
                if action == "buy":
                    buys.append(rec)
                else:
                    rec["pnl"] = pnl
                    sells.append(rec)

            name_row = conn.execute(
                "SELECT name FROM transactions WHERE code = ? LIMIT 1", (code,)
            ).fetchone()
            if name_row:
                name = name_row[0]

        ee_result = self._compute_entry_exit(code, name)
        stop_price: Optional[float] = None
        tp_prices: list[float] = []
        if ee_result and ee_result.stop_loss:
            stop_price = round(ee_result.stop_loss.price, 2)
        if ee_result:
            tp_prices = [round(t.price, 2) for t in ee_result.take_profit]

        executed_stop = False
        executed_tp = False
        for s in sells:
            if stop_price is not None and s["price"] <= stop_price:
                executed_stop = True
            if tp_prices and s["price"] >= tp_prices[0]:
                executed_tp = True

        suggestion = "未触发系统建议止损/止盈"
        if executed_stop:
            suggestion = "已执行系统建议止损"
        elif executed_tp:
            suggestion = "已执行系统建议止盈"

        return {
            "code": code,
            "name": name,
            "buys": buys,
            "sells": sells,
            "entry_exit": ee_result.to_dict() if ee_result else None,
            "executed_stop_loss": executed_stop,
            "executed_take_profit": executed_tp,
            "suggestion": suggestion,
        }

    # ------------------------------------------------------------------ #
    # 展示
    # ------------------------------------------------------------------ #
    def to_table(self, positions: Optional[list[PositionSnapshot]] = None) -> Any:
        """返回 rich.table.Table 对象，用于 REPL/CLI 美化展示。

        Args:
            positions: 可选外部传入的持仓快照列表；默认调用 current_holdings()。

        Returns:
            rich.table.Table 实例。
        """
        if Table is None:
            raise ImportError("rich 未安装，无法生成表格")

        table = Table(title="当前持仓")
        table.add_column("代码", style="cyan")
        table.add_column("名称")
        table.add_column("数量", justify="right")
        table.add_column("成本价", justify="right")
        table.add_column("现价", justify="right")
        table.add_column("市值", justify="right")
        table.add_column("盈亏", justify="right")
        table.add_column("盈亏率%", justify="right")

        positions = positions if positions is not None else self.current_holdings()
        for p in positions:
            pnl_style = "red" if p.pnl < 0 else "green"
            table.add_row(
                p.code,
                p.name,
                str(p.shares),
                f"{p.avg_cost:.2f}",
                f"{p.current_price:.2f}",
                f"{p.market_value:.2f}",
                f"[{pnl_style}]{p.pnl:.2f}[/{pnl_style}]",
                f"{p.pnl_pct:.2f}",
            )
        return table

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _fetch_realtime_prices(self) -> dict[str, float]:
        """调用 akshare 全市场快照，解析为 {code: 最新价}。

        失败时返回空字典，由上层 fallback 到最近成交价。
        """
        try:
            df = self.dl.all_spot()
            if df is None or len(df) == 0:
                return {}
        except Exception as e:  # noqa: BLE001
            logger.warning("实时行情获取异常: %s", e)
            return {}

        code_col = self._find_col(df, ["代码", "股票代码", "code"])
        price_col = self._find_col(df, ["最新价", "现价", "收盘价", "close"])
        if code_col is None or price_col is None:
            logger.warning("实时行情列不匹配，无法解析价格")
            return {}

        prices: dict[str, float] = {}
        for _, row in df.iterrows():
            raw_code = str(row[code_col]).strip()
            price = safe_float(row[price_col], 0.0)
            if not raw_code or price <= 0:
                continue
            # 统一为 6 位代码
            code6 = raw_code[-6:] if raw_code[-6:].isdigit() else raw_code
            prices[code6] = price
        return prices

    def _last_trade_price(self, code: str) -> float:
        """取某代码最近一笔成交价格；无任何记录返回 0。"""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT price FROM transactions
                WHERE code = ? ORDER BY date DESC, id DESC LIMIT 1
                """,
                (code,),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def _calc_avg_hold_days(self, sells: list[tuple[str, str, Optional[float]]]) -> float:
        """根据卖出记录与对应最早买入日期计算平均持仓天数。"""
        if not sells:
            return 0.0

        hold_days: list[int] = []
        with self._connect() as conn:
            for sell_date, code, _ in sells:
                row = conn.execute(
                    """
                    SELECT MIN(date) FROM transactions
                    WHERE code = ? AND action = 'buy' AND date <= ?
                    """,
                    (code, sell_date),
                ).fetchone()
                if row is None or row[0] is None:
                    continue
                buy_date = row[0]
                try:
                    d1 = datetime.strptime(str(buy_date), "%Y%m%d")
                    d2 = datetime.strptime(str(sell_date), "%Y%m%d")
                    hold_days.append(max(0, (d2 - d1).days))
                except ValueError:
                    continue

        return round(sum(hold_days) / len(hold_days), 2) if hold_days else 0.0

    def _compute_entry_exit(self, code: str, name: str) -> Optional[Any]:
        """为该代码计算当前 entry_exit 买卖点方案，失败返回 None。"""
        try:
            close = self._last_trade_price(code)
            if close <= 0:
                close = 10.0  # 兜底，避免 entry_exit 输入为 0
            candidate = {"code": code, "name": name, "close": close}
            return self.entry_exit.compute(candidate, temperature=50.0)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s 买卖点计算失败: %s", code, e)
            return None

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        """模糊匹配 DataFrame 列名。"""
        cols = list(df.columns)
        for c in candidates:
            for real in cols:
                if c in str(real):
                    return real
        return None


# ---------------------------------------------------------------------- #
# 命令行测试入口
# ---------------------------------------------------------------------- #
def _cmd_buy(args: argparse.Namespace, tracker: PortfolioTracker) -> int:
    tracker.record_buy(
        args.code,
        args.name,
        args.shares,
        args.price,
        args.date,
        reason=args.reason or "",
    )
    print(f"已记录买入 {args.code} {args.shares} 股")
    return 0


def _cmd_sell(args: argparse.Namespace, tracker: PortfolioTracker) -> int:
    pnl = tracker.record_sell(
        args.code,
        args.shares,
        args.price,
        args.date,
        reason=args.reason or "",
    )
    print(f"已记录卖出 {args.code} {args.shares} 股，盈亏 {pnl:.2f}")
    return 0


def _cmd_holdings(args: argparse.Namespace, tracker: PortfolioTracker) -> int:
    positions = tracker.current_holdings()
    if args.rich:
        from rich.console import Console
        console = Console()
        console.print(tracker.to_table(positions))
    else:
        for p in positions:
            print(
                f"{p.code} {p.name} {p.shares}股 成本{p.avg_cost:.2f} "
                f"现价{p.current_price:.2f} 市值{p.market_value:.2f} "
                f"盈亏{p.pnl:.2f}({p.pnl_pct:.2f}%)"
            )
    return 0


def _cmd_transactions(args: argparse.Namespace, tracker: PortfolioTracker) -> int:
    rows = tracker.transactions(limit=args.limit)
    for tx in rows:
        pnl_str = f" 盈亏={tx.pnl:.2f}" if tx.pnl is not None else ""
        print(
            f"{tx.date} {tx.action.upper()} {tx.code} {tx.name} "
            f"{tx.shares}股 @ {tx.price:.2f} 金额={tx.amount:.2f}{pnl_str}"
        )
    return 0


def _cmd_summary(args: argparse.Namespace, tracker: PortfolioTracker) -> int:
    import json
    print(json.dumps(tracker.summary(), ensure_ascii=False, indent=2))
    return 0


def _cmd_attribution(args: argparse.Namespace, tracker: PortfolioTracker) -> int:
    import json
    print(json.dumps(tracker.attribution(args.code), ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """命令行入口：支持 buy/sell/holdings/transactions/summary/attribution。"""
    parser = argparse.ArgumentParser(
        prog="portfolio",
        description="持仓跟踪：记录买卖、查看实时盈亏与事后归因",
    )
    parser.add_argument("--db", default="data/pangu.db", help="SQLite 路径")
    parser.add_argument("--no-rich", action="store_true", help=" holdings 输出不用 rich 表格")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_buy = sub.add_parser("buy", help="记录买入")
    p_buy.add_argument("code", help="股票代码")
    p_buy.add_argument("name", help="股票名称")
    p_buy.add_argument("shares", type=int, help="股数")
    p_buy.add_argument("price", type=float, help="买入价")
    p_buy.add_argument("date", help="日期 YYYYMMDD")
    p_buy.add_argument("--reason", default="", help="买入理由")
    p_buy.set_defaults(func=_cmd_buy)

    p_sell = sub.add_parser("sell", help="记录卖出")
    p_sell.add_argument("code", help="股票代码")
    p_sell.add_argument("shares", type=int, help="股数")
    p_sell.add_argument("price", type=float, help="卖出价")
    p_sell.add_argument("date", help="日期 YYYYMMDD")
    p_sell.add_argument("--reason", default="", help="卖出理由")
    p_sell.set_defaults(func=_cmd_sell)

    p_hold = sub.add_parser("holdings", help="当前持仓")
    p_hold.set_defaults(func=_cmd_holdings)

    p_tx = sub.add_parser("transactions", help="交易明细")
    p_tx.add_argument("--limit", type=int, default=50, help="条数")
    p_tx.set_defaults(func=_cmd_transactions)

    p_sum = sub.add_parser("summary", help="总览统计")
    p_sum.set_defaults(func=_cmd_summary)

    p_attr = sub.add_parser("attribution", help="个股归因")
    p_attr.add_argument("code", help="股票代码")
    p_attr.set_defaults(func=_cmd_attribution)

    args = parser.parse_args(argv)
    args.rich = not args.no_rich
    tracker = PortfolioTracker(db_path=args.db)
    return args.func(args, tracker)


if __name__ == "__main__":
    import sys
    sys.exit(main())
