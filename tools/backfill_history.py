"""Backfill baostock daily history into data/market_breadth/raw.sqlite3.

Extends breadth_raw backwards (default from 2025-12-15 to 2022-01-01) so that
walk-forward research has enough trading days.  Also backfills:
  - breadth_universe  : PIT daily membership (query_all_stock per day),
    includes later-delisted codes because membership is per historical date.
  - index_daily       : benchmark indices (sh.000001/000300/000905/000852, sz.399006).

All writes are INSERT OR IGNORE so re-running is safe.  Progress goes to
breadth_progress (status column) and stdout (tee to tmp/backfill.log).
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
import time

import baostock as bs

DB_PATH = "data/market_breadth/raw.sqlite3"
KFIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg,turn,isST"
SAMPLE_DAYS = ["2022-01-04", "2022-07-01", "2023-01-03", "2023-07-03",
               "2024-01-02", "2024-07-01", "2025-01-02", "2025-06-02"]
INDEX_CODES = ["sh.000001", "sh.000300", "sh.000905", "sh.000852", "sz.399006", "sz.399001"]


def is_stock_code(code: str) -> bool:
    if code.startswith("sh.6") or code.startswith("sz.0") or code.startswith("sz.3"):
        return True
    return False


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS index_daily (
        date TEXT NOT NULL, code TEXT NOT NULL, open REAL, high REAL, low REAL,
        close REAL, preclose REAL, volume REAL, amount REAL, pct_change REAL,
        PRIMARY KEY(date, code))""")
    conn.commit()


def existing_codes(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT DISTINCT code FROM breadth_raw")}


def historical_codes(conn: sqlite3.Connection) -> set[str]:
    codes: set[str] = set()
    lg = bs.login()
    for day in SAMPLE_DAYS:
        rs = bs.query_all_stock(day=day)
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            code = row[0]
            if is_stock_code(code):
                codes.add(code)
        print(f"universe sample {day}: cumulative {len(codes)}", flush=True)
    return codes


def backfill_stock(conn: sqlite3.Connection, code: str, start: str, end: str) -> tuple[str, str]:
    rs = bs.query_history_k_data_plus(code, KFIELDS, start_date=start, end_date=end,
                                      frequency="d", adjustflag="3")
    rows = []
    while rs.error_code == "0" and rs.next():
        d = rs.get_row_data()
        try:
            rows.append((d[0].replace("-", ""), d[1],
                         float(d[2] or None) if d[2] else None,
                         float(d[3]) if d[3] else None, float(d[4]) if d[4] else None,
                         float(d[5]) if d[5] else None, float(d[6]) if d[6] else None,
                         float(d[7]) if d[7] else None, float(d[8]) if d[8] else None,
                         float(d[9]) if d[9] else None, float(d[10]) if d[10] else None,
                         int(d[11]) if d[11] else 0))
        except (ValueError, IndexError):
            continue
    if rows:
        conn.executemany("""INSERT OR IGNORE INTO breadth_raw
            (date, code, open, high, low, close, preclose, volume, amount, pct_change, turnover, is_st)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        conn.commit()
    return code, f"{len(rows)}"


def backfill_universe_day(conn: sqlite3.Connection, day: str) -> int:
    rs = bs.query_all_stock(day=day)
    rows = []
    while rs.error_code == "0" and rs.next():
        code, name = rs.get_row_data()[0], rs.get_row_data()[1]
        if is_stock_code(code):
            rows.append((day.replace("-", ""), code, name))
    if rows:
        conn.executemany("INSERT OR IGNORE INTO breadth_universe (date, code, name) VALUES (?,?,?)", rows)
        conn.commit()
    return len(rows)


def backfill_indices(conn: sqlite3.Connection, start: str, end: str) -> None:
    for code in INDEX_CODES:
        rs = bs.query_history_k_data_plus(code, "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                          start_date=start, end_date=end, frequency="d", adjustflag="3")
        rows = []
        while rs.error_code == "0" and rs.next():
            d = rs.get_row_data()
            try:
                rows.append((d[0].replace("-", ""), d[1], float(d[2]) if d[2] else None, float(d[3]) if d[3] else None,
                             float(d[4]) if d[4] else None, float(d[5]) if d[5] else None,
                             float(d[6]) if d[6] else None, float(d[7]) if d[7] else None,
                             float(d[8]) if d[8] else None, float(d[9]) if d[9] else None))
            except (ValueError, IndexError):
                continue
        if rows:
            conn.executemany("""INSERT OR IGNORE INTO index_daily
                (date, code, open, high, low, close, preclose, volume, amount, pct_change)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", rows)
            conn.commit()
        print(f"index {code}: {len(rows)} rows", flush=True)


def main() -> None:
    start, end = sys.argv[1] if len(sys.argv) > 1 else "2022-01-01", \
                 sys.argv[2] if len(sys.argv) > 2 else "2025-12-14"
    only_universe = len(sys.argv) > 3 and sys.argv[3] == "universe-only"
    lg = bs.login()
    print(f"login: {lg.error_code} {lg.error_msg}", flush=True)
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    have = existing_codes(conn)
    hist = historical_codes(conn)
    codes = sorted(hist | have)
    print(f"total codes to backfill: {len(codes)} (archive {len(have)}, historical-only {len(hist - have)})", flush=True)

    t0 = time.time()
    done = 0
    for i, code in enumerate(codes):
        if not only_universe:
            try:
                _, n = backfill_stock(conn, code, start, end)
            except Exception as e:  # noqa: BLE001
                conn.execute("""INSERT OR REPLACE INTO breadth_progress (code, status, error, updated_at)
                                VALUES (?,?,?,?)""", (code, "error", str(e)[:200], dt.datetime.now().isoformat()))
                conn.commit()
                continue
        done += 1
        if done % 100 == 0:
            rate = done / max(time.time() - t0, 1)
            print(f"progress {done}/{len(codes)} ({rate:.1f}/s, elapsed {time.time()-t0:.0f}s)", flush=True)
    print(f"stock backfill done: {done} codes in {time.time()-t0:.0f}s", flush=True)

    # PIT universe per trading day
    rs = bs.query_trade_dates(start_date=start, end_date="2026-09-30")
    trade_days = []
    while rs.error_code == "0" and rs.next():
        if rs.get_row_data()[1] == "1":
            trade_days.append(rs.get_row_data()[0])
    have_days = {r[0] for r in conn.execute("SELECT DISTINCT date FROM breadth_universe")}
    todo_days = [d for d in trade_days if start <= d <= "2026-01-27" and d not in have_days]
    print(f"universe days to backfill: {len(todo_days)}", flush=True)
    for i, day in enumerate(todo_days):
        try:
            n = backfill_universe_day(conn, day)
        except Exception as e:  # noqa: BLE001
            print(f"universe {day} error: {e}", flush=True)
            continue
        if (i + 1) % 50 == 0:
            print(f"universe progress {i+1}/{len(todo_days)} (last {day}: {n})", flush=True)
    print("universe backfill done", flush=True)

    backfill_indices(conn, start, "2026-09-19")
    bs.logout()
    n = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM breadth_raw").fetchone()
    print(f"breadth_raw now: {n}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
