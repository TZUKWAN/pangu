"""Sharded parallel backfill worker (v2).

Step 1: python tools/backfill_shard.py codelist        # writes tmp/backfill_codes.txt
Step 2: python tools/backfill_shard.py stocks <i> <n>  # slice i of n from codelist file
Step 3: python tools/backfill_shard.py meta            # universe days + indices
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
import time

import baostock as bs

sys.path.insert(0, "tools")
from backfill_history import (DB_PATH, backfill_indices, backfill_stock,  # noqa: E402
                              backfill_universe_day, ensure_schema, is_stock_code)

CODELIST = "tmp/backfill_codes.txt"


def build_codelist() -> None:
    lg = bs.login()
    print(f"login: {lg.error_code} {lg.error_msg}", flush=True)
    codes: set[str] = set()
    for day in ["2022-01-04", "2022-07-01", "2023-01-03", "2023-07-03",
                "2024-01-02", "2024-07-01", "2025-01-02", "2025-06-02"]:
        rs = bs.query_all_stock(day=day)
        n = 0
        while rs.error_code == "0" and rs.next():
            code = rs.get_row_data()[0]
            if is_stock_code(code):
                codes.add(code)
                n += 1
        print(f"{day}: +{n} cumulative {len(codes)}", flush=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    have = {r[0] for r in conn.execute("SELECT DISTINCT code FROM breadth_raw")}
    union = sorted(codes | have)
    with open(CODELIST, "w") as f:
        f.write("\n".join(union))
    print(f"codelist written: {len(union)} (archive {len(have)})", flush=True)
    conn.close()
    bs.logout()


def main() -> None:
    mode = sys.argv[1]
    lg = bs.login()
    print(f"[{mode}] login: {lg.error_code} {lg.error_msg}", flush=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_schema(conn)

    if mode == "stocks":
        shard, total = int(sys.argv[2]), int(sys.argv[3])
        with open(CODELIST) as f:
            all_codes = [c for c in f.read().split() if c]
        # skip only codes whose backfill already fully covered the target range
        covered = {r[0] for r in conn.execute(
            "SELECT code FROM breadth_progress WHERE status='backfilled22'")}
        codes = [c for c in all_codes[shard::total] if c not in covered]
        print(f"[stocks {shard}/{total}] {len(codes)} of {len(all_codes)}", flush=True)
        t0 = time.time()
        for i, code in enumerate(codes):
            try:
                backfill_stock(conn, code, "2022-01-01", "2025-12-14")
                conn.execute("""INSERT OR REPLACE INTO breadth_progress
                                (code, status, error, updated_at, covered_start, covered_end)
                                VALUES (?, 'backfilled22', '', ?, '2022-01-01', '2025-12-14')""",
                             (code, dt.datetime.now().isoformat(timespec="seconds")))
                conn.commit()
            except Exception as e:  # noqa: BLE001
                conn.execute("""INSERT OR REPLACE INTO breadth_progress
                                (code, status, error, updated_at) VALUES (?, 'error', ?, ?)""",
                             (code, str(e)[:200], dt.datetime.now().isoformat()))
                conn.commit()
            if (i + 1) % 100 == 0:
                rate = (i + 1) / max(time.time() - t0, 1)
                eta = (len(codes) - i - 1) / max(rate, 0.01)
                print(f"[stocks {shard}] {i+1}/{len(codes)} ({rate:.2f}/s ETA {eta/60:.0f}m)", flush=True)
        print(f"[stocks {shard}] DONE in {time.time()-t0:.0f}s", flush=True)
    elif mode == "meta":
        rs = bs.query_trade_dates(start_date="2022-01-01", end_date="2026-09-30")
        trade_days = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            if len(row) >= 2 and row[1] == "1":
                trade_days.append(row[0].replace("-", ""))
        have_days = {r[0] for r in conn.execute("SELECT DISTINCT date FROM breadth_universe")}
        todo = [d for d in trade_days if d <= "20260127" and d not in have_days]
        print(f"[meta] universe days to fill: {len(todo)}", flush=True)
        for i, day in enumerate(todo):
            try:
                backfill_universe_day(conn, day)
            except Exception as e:  # noqa: BLE001
                print(f"[meta] universe {day} error: {e}", flush=True)
            if (i + 1) % 50 == 0:
                print(f"[meta] universe {i+1}/{len(todo)}", flush=True)
        backfill_indices(conn, "2022-01-01", "2026-09-19")
        print("[meta] DONE", flush=True)
    conn.close()
    bs.logout()


if __name__ == "__main__":
    if mode_arg := (sys.argv[1] if len(sys.argv) > 1 else ""):
        if mode_arg == "codelist":
            build_codelist()
        else:
            main()
