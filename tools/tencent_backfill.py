"""Backfill 2022-01 → 2025-12-14 daily history from Tencent fqkline (hfq+raw).

Why hfq: 后复权 historical values never change when future dividends occur —
PIT-safe by construction (unlike 静态前复权).  factor_t = hfq_t/raw_t uses only
info available at t; pct_change_t = hfq_t/hfq_{t-1} - 1 is the total return.

Writes raw fetches into new tables (tencent_hfq / tencent_raw), then merges
into breadth_raw filling ONLY dates that baostock did not cover, with
source='tencent' (amount approximated as volume×close — documented).
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DB = "data/market_breadth/raw.sqlite3"
WINDOW = ("2022-01-01", "2025-12-14")
HALVES = [("2022-01-01", "2023-12-31"), ("2024-01-01", "2025-12-14")]
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(code: str, start: str, end: str, fq: str, retries: int = 3) -> list:
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={code},day,{start},{end},800,{fq}")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            d = json.loads(urllib.request.urlopen(req, timeout=20).read())
            data = d["data"][code]
            key = "hfqday" if fq == "hfq" else "day"
            return data.get(key) or data.get("day") or []
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return []


def to_compact(d: str) -> str:
    return d.replace("-", "")


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS tencent_hfq (
        date TEXT NOT NULL, code TEXT NOT NULL, close REAL,
        PRIMARY KEY(date, code))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS tencent_raw (
        date TEXT NOT NULL, code TEXT NOT NULL, open REAL, high REAL, low REAL,
        close REAL, volume REAL, PRIMARY KEY(date, code))""")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(breadth_raw)")]
    if "source" not in cols:
        conn.execute("ALTER TABLE breadth_raw ADD COLUMN source TEXT DEFAULT 'baostock'")
    conn.commit()


def fetch_code(code: str) -> str:
    """code like sh.600000 → tencent sh600000."""
    tcode = code.replace(".", "")
    hfq, raw = [], []
    for (s, e) in HALVES:
        hfq.extend(fetch(tcode, s, e, "hfq"))
        raw.extend(fetch(tcode, s, e, ""))
    conn = sqlite3.connect(DB, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany("INSERT OR IGNORE INTO tencent_hfq VALUES (?,?,?)",
                     [(to_compact(r[0]), code, float(r[2])) for r in hfq if len(r) >= 5])
    conn.executemany("INSERT OR IGNORE INTO tencent_raw VALUES (?,?,?,?,?,?,?)",
                     [(to_compact(r[0]), code, float(r[1]), float(r[3]), float(r[4]),
                       float(r[2]), float(r[5])) for r in raw if len(r) >= 6])
    conn.commit()
    conn.close()
    return code


def codes_needing_repair(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("""SELECT code, COUNT(*) FROM breadth_raw
        WHERE date >= '20220101' AND date <= '20251214' GROUP BY code""").fetchall()
    have = {c for c, _ in rows}
    need = sorted(c for c, n in rows if n < 800)
    all_codes = {r[0] for r in conn.execute("SELECT DISTINCT code FROM breadth_raw")}
    need += sorted(all_codes - have)
    return sorted(set(need))


def merge(conn: sqlite3.Connection) -> int:
    """Fill breadth_raw gaps from tencent tables.

    修复版：不跳过整个代码，而是逐日期判定——任何 tencent 有而 breadth_raw
    没有的日期都补入（source='tencent'）。baostock 已有行保持不变（baostock
    胜出），插值链基于 tencent hfq 序列自身（连续、含除权），与 baostock 行
    是否相邻无关。
    """
    added = 0
    codes = [r[0] for r in conn.execute("SELECT DISTINCT code FROM tencent_hfq")]
    for i, code in enumerate(codes):
        hfq = dict(conn.execute("SELECT date, close FROM tencent_hfq WHERE code=?", (code,)).fetchall())
        raw = conn.execute("""SELECT date, open, high, low, close, volume FROM tencent_raw
                              WHERE code=? ORDER BY date""", (code,)).fetchall()
        if not raw or len(hfq) < 2:
            continue
        have_any = {r[0] for r in conn.execute(
            "SELECT date FROM breadth_raw WHERE code=?", (code,))}
        prev_hfq = prev_raw = prev_factor = None
        rows = []
        for (d, o, h, l, c, v) in raw:
            factor = hfq.get(d)
            if factor is None or not c:
                prev_hfq = prev_raw = prev_factor = None
                continue
            factor = factor / c
            if prev_hfq is None or not prev_raw:
                prev_hfq, prev_raw, prev_factor = hfq[d], c, factor
                continue
            if d in have_any:
                prev_hfq, prev_raw, prev_factor = hfq[d], c, factor
                continue
            pct = (hfq[d] / prev_hfq - 1.0) * 100.0
            preclose = prev_raw * (factor / prev_factor)
            amount = v * c  # 近似：成交额≈volume×close（已文档化）
            rows.append((d, code, o, h, l, c, round(preclose, 4), v, amount,
                         round(pct, 4), None, 0, "tencent"))
            prev_hfq, prev_raw, prev_factor = hfq[d], c, factor
        if rows:
            conn.executemany("""INSERT OR IGNORE INTO breadth_raw
                (date, code, open, high, low, close, preclose, volume, amount,
                 pct_change, turnover, is_st, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
            conn.commit()
            added += len(rows)
        if (i + 1) % 200 == 0:
            print(f"merged {i+1}/{len(codes)} codes, added {added} rows", flush=True)
    return added


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    conn = sqlite3.connect(DB, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_tables(conn)
    if mode == "fetch":
        codes = codes_needing_repair(conn)
        print(f"codes to fetch: {len(codes)}", flush=True)
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=4) as ex:
            for _ in ex.map(fetch_code, codes):
                done += 1
                if done % 100 == 0:
                    rate = done / max(time.time() - t0, 1)
                    print(f"fetch {done}/{len(codes)} ({rate:.2f}/s ETA {(len(codes)-done)/max(rate,.01)/60:.0f}m)",
                          flush=True)
        print(f"fetch done in {time.time()-t0:.0f}s", flush=True)
    elif mode == "merge":
        n = merge(conn)
        print(f"merge done, added {n} rows", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
