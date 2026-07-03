"""RPS（相对强度排名）预计算与查询。

背景：trend_scanner 原来的 _rps 用「当日全市场涨跌幅」近似 20 日累计涨幅的
百分位，尺度不匹配，经模拟验证与真实 RPS 相关系数仅 0.19，严重失真。

根治方案：
1. 离线批量算全市场每只股票的真实 20 日累计涨幅，存 SQLite，得到真实百分位。
2. scan 时直接查表，毫秒级。

两种用法：
    # 离线预计算（盘后跑一次，建议交易日 15:30 后）
    python -m engine.cli rps-build
    # 之后 scan 自动用真实 RPS（表存在时），否则回退到旧的近似版

性能：全市场 ~5000 只，单只 K 线 ~0.09s，10 线程并行约 1.5-2 分钟。
表很小（股票数 × 几列），存 SQLite 合适。
"""

from __future__ import annotations

import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .data_loader import DataLoader

logger = logging.getLogger("pangu.rps")

RPS_WINDOW = 20  # RPS 回看窗口（交易日）


def _db_path(db_path: str = "data/pangu.db") -> Path:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_table(db_path: str = "data/pangu.db") -> None:
    """建 rps 表（如不存在）。"""
    with sqlite3.connect(_db_path(db_path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rps (
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                ret_20d REAL,           -- 近 20 日累计涨幅
                rps REAL,               -- 百分位排名 0-100
                PRIMARY KEY (code, date)
            )
        """)
        conn.commit()


def compute_all_rps(
    dl: DataLoader,
    date: Optional[str] = None,
    db_path: str = "data/pangu.db",
    workers: int = 10,
    batch_size: int = 200,
) -> dict:
    """批量算全市场 20 日 RPS，存 SQLite。

    Args:
        date: 基准日 YYYYMMDD（默认今天）
        workers: 并发线程数
        batch_size: 分批提交大小（避免一次写太多）
    Returns:
        {date, total, ok, fail, elapsed}
    """
    date = date or datetime.now().strftime("%Y%m%d")
    ensure_table(db_path)

    spot = dl.all_spot()
    if len(spot) == 0:
        logger.error("实时行情为空，无法计算 RPS")
        return {"date": date, "total": 0, "ok": 0, "fail": 0, "elapsed": 0}

    code_col = None
    for c in ["代码", "code"]:
        if c in spot.columns:
            code_col = c
            break
    if code_col is None:
        code_col = spot.columns[1]

    codes = [str(x).strip() for x in spot[code_col].tolist() if str(x).strip()]
    logger.info("开始计算 %d 只股票的 %d 日 RPS（%d 线程）", len(codes), RPS_WINDOW, workers)

    t0 = time.time()
    rets: dict[str, float] = {}
    fail = 0

    def _one(code: str) -> tuple[str, Optional[float]]:
        try:
            k = dl.daily_kline(code, days=RPS_WINDOW + 5)
            close_col = "收盘" if "收盘" in k.columns else (k.columns[4] if len(k.columns) > 4 else None)
            if close_col is None or len(k) < RPS_WINDOW + 1:
                return code, None
            closes = pd.to_numeric(k[close_col], errors="coerce").dropna()
            if len(closes) < RPS_WINDOW + 1:
                return code, None
            ret = float(closes.iloc[-1] / closes.iloc[-RPS_WINDOW - 1] - 1)
            return code, ret
        except Exception:  # noqa: BLE001
            return code, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, c): c for c in codes}
        done = 0
        for fut in as_completed(futures):
            code, ret = fut.result()
            done += 1
            if ret is not None:
                rets[code] = ret
            else:
                fail += 1
            if done % batch_size == 0:
                logger.info("RPS 进度 %d/%d（成功 %d）", done, len(codes), len(rets))

    # 算百分位：ret 在全市场的排名分位
    if rets:
        series = pd.Series(rets)
        ranks = series.rank(pct=True) * 100  # 0-100
    else:
        ranks = pd.Series(dtype=float)

    # 写库（先删当日旧数据再插）
    with sqlite3.connect(_db_path(db_path)) as conn:
        conn.execute("DELETE FROM rps WHERE date = ?", (date,))
        rows = [(code, date, rets.get(code), float(ranks.get(code, 50.0))) for code in rets]
        conn.executemany("INSERT INTO rps (code, date, ret_20d, rps) VALUES (?, ?, ?, ?)", rows)
        conn.commit()

    elapsed = time.time() - t0
    logger.info("RPS 计算完成：%d 只成功，%d 失败，耗时 %.1fs", len(rets), fail, elapsed)
    return {"date": date, "total": len(codes), "ok": len(rets), "fail": fail, "elapsed": round(elapsed, 1)}


def load_rps_map(date: Optional[str] = None, db_path: str = "data/pangu.db") -> dict[str, float]:
    """读当日全市场 RPS 表，返回 {code: rps}。表不存在或空则返回 {}。"""
    date = date or datetime.now().strftime("%Y%m%d")
    p = _db_path(db_path)
    if not p.exists():
        return {}
    try:
        with sqlite3.connect(p) as conn:
            df = pd.read_sql("SELECT code, rps FROM rps WHERE date = ?", conn, params=(date,))
    except Exception:  # noqa: BLE001
        return {}
    if len(df) == 0:
        return {}
    return dict(zip(df["code"].astype(str), df["rps"].astype(float)))


def is_available(date: Optional[str] = None, db_path: str = "data/pangu.db") -> bool:
    """当日 RPS 表是否可用（已预计算）。"""
    return len(load_rps_map(date, db_path)) > 0
