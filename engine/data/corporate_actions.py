"""公司行为（除权除息日）推断与登记。

PIT 铁律（务必遵守）：
- 静态整体前复权/后复权（forward/backward adjustment）被【禁止】——那会改写
  全部历史价格，破坏因果性；历史收益一律直接用 breadth_raw.pct_change
  （已是分红/拆股调整后的真实收益，百分数）。
- 这里推断出的除权候选仅作【诊断/审计】用途（例如解释某日价格跳空、
  校验数据管道），不得用于重算价格序列。

推断规则：对每个 (date, code)，若
    |close_t / preclose_t - 1 - pct_change_t / 100| > 1e-6
则价格隐含收益与调整后收益不一致，说明当日发生了除权/除息（或数据异常），
登记为 kind='ex_right' 候选；ratio_implied = (1 + pct_change/100) /
(close_t/preclose_t) 为隐含调整比例。
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .pit_store import _connect_ro, normalize_code

logger = logging.getLogger("pangu.corporate_actions")

DEFAULT_DB = "data/market_breadth/raw.sqlite3"
DEFAULT_OUT = Path("data/pit/corporate_actions.sqlite")
EPS = 1e-6


def detect_ex_right_candidates(
    db_path: str | Path = DEFAULT_DB,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """扫描日线档案，返回价格隐含收益 != pct_change 的除权候选行。

    返回列：date(紧凑), code(6位), kind='ex_right', ratio_implied。
    """
    where = ["preclose IS NOT NULL", "close IS NOT NULL", "pct_change IS NOT NULL",
             "close != 0", "preclose != 0"]
    params: list = []
    if start:
        where.append("date >= ?")
        params.append(str(start).replace("-", ""))
    if end:
        where.append("date <= ?")
        params.append(str(end).replace("-", ""))
    query = (
        "SELECT date, code, close, preclose, pct_change FROM breadth_raw "
        f"WHERE {' AND '.join(where)} ORDER BY date, code"
    )
    with _connect_ro(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return pd.DataFrame(columns=["date", "code", "kind", "ratio_implied"])
    price_ret = df["close"] / df["preclose"] - 1.0
    adj_ret = df["pct_change"] / 100.0
    mask = (price_ret - adj_ret).abs() > EPS
    hit = df[mask]
    out = pd.DataFrame({
        "date": hit["date"].astype(str),
        "code": hit["code"].map(normalize_code),
        "kind": "ex_right",
        "ratio_implied": (1.0 + hit["pct_change"] / 100.0) / (hit["close"] / hit["preclose"]),
    }).reset_index(drop=True)
    return out


def persist_corporate_actions(
    db_path: str | Path = DEFAULT_DB,
    out_path: str | Path = DEFAULT_OUT,
) -> pd.DataFrame:
    """检测并持久化除权候选到 data/pit/corporate_actions.sqlite。"""
    df = detect_ex_right_candidates(db_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS corporate_actions (
                   date TEXT NOT NULL,
                   code TEXT NOT NULL,
                   kind TEXT NOT NULL,
                   ratio_implied REAL,
                   PRIMARY KEY (date, code, kind))"""
        )
        if not df.empty:
            conn.executemany(
                "INSERT OR REPLACE INTO corporate_actions VALUES (?,?,?,?)",
                list(df.itertuples(index=False, name=None)),
            )
        conn.commit()
    finally:
        conn.close()
    logger.info("除权候选已登记：%d 条 → %s", len(df), out)
    return df
