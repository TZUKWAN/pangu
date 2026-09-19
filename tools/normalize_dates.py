"""Normalize date formats in raw.sqlite3 to canonical YYYYMMDD (compact).

Backfill wrote ISO 'YYYY-MM-DD' while the original archive and live writer use
'YYYYMMDD'.  Mixed formats break ordering and replay snapping.  Idempotent.
"""
from __future__ import annotations

import sqlite3

DB = "data/market_breadth/raw.sqlite3"
TABLES = {
    "breadth_raw": "date, code, open, high, low, close, preclose, volume, amount, pct_change, turnover, is_st",
    "breadth_universe": "date, code, name",
    "industry_membership": "date, code, industry, classification, update_date",
    "index_daily": "date, code, open, high, low, close, preclose, volume, amount, pct_change",
}


def main() -> None:
    conn = sqlite3.connect(DB, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("PRAGMA journal_mode=WAL")
    for table, cols in TABLES.items():
        try:
            n_dash = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE date LIKE '%-%'").fetchone()[0]
        except sqlite3.OperationalError:
            print(f"{table}: missing, skip")
            continue
        if not n_dash:
            print(f"{table}: already normalized")
            continue
        col_list = ", ".join(c.strip() for c in cols.split(","))
        norm_list = ", ".join(
            f"replace({c.strip()}, '-', '') AS {c.strip()}" if c.strip() == "date"
            else c.strip() for c in cols.split(","))
        conn.execute("DROP TABLE IF EXISTS _norm")
        conn.execute(f"CREATE TEMP TABLE _norm AS SELECT {norm_list} FROM {table} WHERE date LIKE '%-%'")
        conn.execute(f"INSERT OR IGNORE INTO {table} ({col_list}) SELECT {col_list} FROM _norm")
        conn.execute(f"DELETE FROM {table} WHERE date LIKE '%-%'")
        conn.commit()
        print(f"{table}: normalized {n_dash} rows")
    conn.execute("DROP TABLE IF EXISTS _norm")
    print("breadth_raw range:", conn.execute(
        "SELECT COUNT(*), MIN(date), MAX(date) FROM breadth_raw").fetchone())
    conn.close()


if __name__ == "__main__":
    main()
