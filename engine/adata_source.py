"""adata 数据源封装：作为 akshare/同花顺 失败时的兜底。

adata 本身是多源聚合 SDK（东财/新浪/腾讯/百度/同花顺），对 A 股常用数据有稳定
封装。本模块把它纳入 `DataLoader` 的降级链，避免单点失败时无数据可用。

覆盖：
- 全市场实时行情快照（all_spot）
- 个股历史日 K（daily_kline）
- 个股历史资金流向（individual_fund_flow）
- 概念板块资金流向（concept_fund_flow）

所有函数失败返回空 DataFrame，不抛异常。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from . import _proxy_patch  # noqa: F401  确保 requests 绕过系统代理

logger = logging.getLogger("pangu.adata_source")


def _is_available() -> bool:
    try:
        import adata  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------- #
# 全市场实时行情
# ---------------------------------------------------------------------- #
def all_spot(batch_size: int = 1000, timeout_sec: float = 30.0) -> pd.DataFrame:
    """用 adata 取全市场实时行情快照。

    返回列统一为：代码 / 名称 / 最新价 / 涨跌幅 / 涨跌额 / 成交量 / 成交额
    （换手率、市值等字段 adata 不提供，置空或 0）。
    """
    if not _is_available():
        return pd.DataFrame()
    import adata

    codes: list[str] = []
    try:
        codes_df = adata.stock.info.all_code()
        if codes_df is not None and not codes_df.empty:
            col = "stock_code" if "stock_code" in codes_df.columns else codes_df.columns[0]
            codes = codes_df[col].astype(str).str.strip().str.zfill(6).tolist()
    except Exception as e:  # noqa: BLE001
        logger.debug("adata 全市场代码列表失败: %s", e)
        return pd.DataFrame()
    if not codes:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    deadline = time.time() + timeout_sec
    for i in range(0, len(codes), batch_size):
        if time.time() > deadline:
            logger.debug("adata all_spot 超时，已取 %d 只", len(rows))
            break
        batch = codes[i:i + batch_size]
        try:
            df = adata.stock.market.list_market_current(code_list=batch)
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                code = str(r.get("stock_code", "")).zfill(6)
                if not code:
                    continue
                rows.append({
                    "代码": code,
                    "名称": r.get("short_name", ""),
                    "最新价": _to_float(r.get("price")),
                    "涨跌幅": _to_float(r.get("change_pct")),
                    "涨跌额": _to_float(r.get("change")),
                    "成交量": _to_float(r.get("volume")),
                    "成交额": _to_float(r.get("amount")),
                    "换手率": pd.NA,
                    "turnover_rate": pd.NA,
                    "turnover_missing": True,
                    "turnover_source": "missing",
                    "turnover_status": "missing",
                    "流通市值": 0.0,
                    "总市值": 0.0,
                    "主力净流入-净额": 0.0,
                })
        except Exception as e:  # noqa: BLE001
            logger.debug("adata list_market_current 批次失败: %s", e)
            continue

    if not rows:
        return pd.DataFrame()
    logger.info("adata 全市场行情：%d 只", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# 个股历史日 K
# ---------------------------------------------------------------------- #
def daily_kline(
    symbol: str,
    days: int = 60,
    adjust: str = "qfq",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date: Optional[str] = None,
) -> pd.DataFrame:
    """用 adata 取个股历史日 K（前复权）。

    返回列：日期 / 股票代码 / 开盘 / 收盘 / 最高 / 最低 / 成交量 / 成交额 / 换手率
    """
    if not _is_available():
        return pd.DataFrame()
    import adata

    symbol = symbol.strip().zfill(6)
    if end_date is None:
        end_dt = datetime.strptime(str(date), "%Y%m%d") if date else datetime.now()
    else:
        end_dt = datetime.strptime(str(end_date), "%Y%m%d")
    if start_date is None:
        start_dt = end_dt - timedelta(days=days * 2)
    else:
        start_dt = datetime.strptime(str(start_date), "%Y%m%d")

    k_type = 1  # 日 K
    adjust_type = 1 if adjust in ("qfq", "hfq", "") else 0

    try:
        df = adata.stock.market.get_market(
            stock_code=symbol,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
            k_type=k_type,
            adjust_type=adjust_type,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("adata K线 %s 失败: %s", symbol, e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    col_map: dict[str, str] = {}
    for c in df.columns:
        c = str(c)
        if "time" in c.lower() or "date" in c.lower():
            col_map[c] = "日期"
        elif c in ("open", "开盘价"):
            col_map[c] = "开盘"
        elif c in ("close", "收盘价"):
            col_map[c] = "收盘"
        elif c in ("high", "最高价"):
            col_map[c] = "最高"
        elif c in ("low", "最低价"):
            col_map[c] = "最低"
        elif "volume" in c.lower() or "成交量" in c:
            col_map[c] = "成交量"
        elif "amount" in c.lower() or "成交额" in c:
            col_map[c] = "成交额"
        elif "turnover" in c.lower() or "换手率" in c:
            col_map[c] = "换手率"
    df = df.rename(columns=col_map)

    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y%m%d")
    df["股票代码"] = symbol
    keep = ["日期", "股票代码", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA if c == "换手率" else 0.0
    if "换手率" in df.columns:
        turnover = pd.to_numeric(df["换手率"], errors="coerce")
        df["turnover_rate"] = turnover.where(turnover.notna(), pd.NA)
        df["turnover_missing"] = turnover.isna()
        df["turnover_source"] = "adata"
        df["turnover_status"] = df["turnover_missing"].map(lambda missing: "missing" if bool(missing) else "ok")
    meta_cols = ["turnover_rate", "turnover_missing", "turnover_source", "turnover_status"]
    return df[keep + [c for c in meta_cols if c in df.columns]].copy()


# ---------------------------------------------------------------------- #
# 个股历史资金流向
# ---------------------------------------------------------------------- #
def individual_fund_flow(symbol: str, days: int = 60) -> pd.DataFrame:
    """用 adata 取个股历史资金流向（东财）。

    返回列：日期 / 主力净流入-净额 / 主力净流入-净占比 / 股票代码
    """
    if not _is_available():
        return pd.DataFrame()
    import adata

    symbol = symbol.strip().zfill(6)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days * 2)
    try:
        df = adata.stock.market.get_capital_flow(
            stock_code=symbol,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("adata 个股资金流 %s 失败: %s", symbol, e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    col_map: dict[str, str] = {}
    for c in df.columns:
        c = str(c)
        if "date" in c.lower():
            col_map[c] = "日期"
        elif "main_net" in c.lower() or "main" in c.lower():
            col_map[c] = "主力净流入-净额"
    df = df.rename(columns=col_map)

    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y%m%d")
    df["股票代码"] = symbol
    if "主力净流入-净占比" not in df.columns:
        df["主力净流入-净占比"] = 0.0
    keep = ["日期", "主力净流入-净额", "主力净流入-净占比", "股票代码"]
    for c in keep:
        if c not in df.columns:
            df[c] = 0.0
    return df[keep].copy()


# ---------------------------------------------------------------------- #
# 概念板块资金流向
# ---------------------------------------------------------------------- #
def concept_fund_flow(indicator: str = "今日") -> pd.DataFrame:
    """用 adata 取概念板块资金流向（东财）。

    indicator: 今日/5日/10日
    """
    if not _is_available():
        return pd.DataFrame()
    import adata

    days_map = {"今日": 1, "5日": 5, "10日": 10}
    days_type = days_map.get(indicator, 1)
    try:
        df = adata.stock.market.all_capital_flow_east(days_type=days_type)
    except Exception as e:  # noqa: BLE001
        logger.debug("adata 概念资金流失败: %s", e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df


# ---------------------------------------------------------------------- #
def _to_float(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
