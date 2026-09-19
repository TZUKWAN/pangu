"""Verify per-code 2022→2025 coverage and repair truncated codes (single
process, sequential — the earlier concurrent run hit server throttling which
silently truncated most histories)."""
from __future__ import annotations
import datetime as dt
import sqlite3
import sys
import time
import baostock as bs
sys.path.insert(0, "tools")
from backfill_history import backfill_stock

DB = "data/market_breadth/raw.sqlite3"

def main() -> None:
    shard, total = int(sys.argv[1]), int(sys.argv[2])
    conn = sqlite3.connect(DB, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("PRAGMA journal_mode=WAL")
    rows = conn.execute("""SELECT code, COUNT(*) FROM breadth_raw
        WHERE date >= '20220101' AND date <= '20251231' GROUP BY code""").fetchall()
    codes = sorted(c for c, n in rows if n < 800)
    # codes with no pre-2026 rows at all are missing from that GROUP BY; add them
    all_codes = {r[0] for r in conn.execute("SELECT DISTINCT code FROM breadth_raw")}
    have = {c for c, _ in rows}
    codes = sorted((set(codes) | (all_codes - have)))[shard::total]
    print(f"[repair {shard}/{total}] {len(codes)} codes to repair", flush=True)
    lg = bs.login()
    print(f"login {lg.error_code}", flush=True)
    t0 = time.time()
    fails = 0
    for i, code in enumerate(codes):
        try:
            n_before = conn.execute("SELECT COUNT(*) FROM breadth_raw WHERE code=?", (code,)).fetchone()[0]
            backfill_stock(conn, code, "2022-01-01", "2025-12-14")
            n_after = conn.execute("SELECT COUNT(*) FROM breadth_raw WHERE code=?", (code,)).fetchone()[0]
            if n_after - n_before < 800:
                fails += 1
                raise RuntimeError(f"truncated again: got {n_after-n_before} rows")
            fails = 0
        except Exception as e:  # noqa: BLE001
            print(f"[repair {shard}] {code}: {e}", flush=True)
            if fails >= 2:
                print(f"[repair {shard}] relogin + 20s cooldown", flush=True)
                try: bs.logout()
                except Exception: pass
                time.sleep(20)
                bs.login()
                fails = 0
        if (i + 1) % 50 == 0:
            rate = (i+1) / max(time.time()-t0, 1)
            print(f"[repair {shard}] {i+1}/{len(codes)} ({rate:.2f}/s ETA {(len(codes)-i-1)/max(rate,.01)/60:.0f}m)", flush=True)
    print(f"[repair {shard}] DONE in {time.time()-t0:.0f}s", flush=True)
    conn.close()
    bs.logout()

if __name__ == "__main__":
    main()
