"""Convert the Pangu PIT archive (raw.sqlite3) into qlib binary format.

PIT adjustment policy (docs/pangu2/DATA_DICTIONARY.md §8):
- Returns come from pct_change (baostock ex-right adjusted, known at close of t).
- adjclose_t = 100 * cumprod(1 + pct_change/100)  → only past returns used.
- factor_t   = adjclose_t / close_t               → known at t.
- O/H/L/C dumped = raw * factor_t                 → per-date factors, no future leak.
This is total-return style pricing, PIT-valid at every date.

Output:
  data/qlib_data/calendars/day.txt
  data/qlib_data/instruments/all.txt      (symbol  start  end)
  data/qlib_data/features/<SYMBOL>/*.bin  (open, high, low, close, volume, amount)

Symbols: sh.600000 -> SH600000, sz.000001 -> SZ000001 (qlib cn convention).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DB = "data/market_breadth/raw.sqlite3"
OUT = Path("data/qlib_data")
CSV_DIR = Path("tmp/qlib_csv")


def to_qlib_symbol(code: str) -> str:
    market, num = code.split(".")
    return ("SH" if market == "sh" else "SZ") + num


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2022-01-04"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-09-04"
    start_c, end_c = start.replace("-", ""), end.replace("-", "")

    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT date, code, open, high, low, close, volume, amount, pct_change, is_st "
        "FROM breadth_raw WHERE date >= ? AND date <= ? ORDER BY date",
        conn, params=(start_c, end_c))
    conn.close()
    df["date"] = df["date"].astype(str)
    print(f"rows: {len(df)}, symbols: {df['code'].nunique()}, {df['date'].min()} → {df['date'].max()}")

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "calendars").mkdir(parents=True, exist_ok=True)
    (OUT / "instruments").mkdir(parents=True, exist_ok=True)
    (OUT / "features").mkdir(parents=True, exist_ok=True)

    # calendar
    calendar = sorted(df["date"].unique())
    (OUT / "calendars" / "day.txt").write_text("\n".join(calendar) + "\n")

    instruments = []
    for code, g in df.groupby("code"):
        g = g.dropna(subset=["close"])
        g = g[g["close"] > 0]
        if len(g) < 20:
            continue
        r = (g["pct_change"].fillna(0.0) / 100.0).clip(-0.5, 0.5)
        # day-1 return cannot be derived → start chain at first close
        adjclose = 100.0 * (1.0 + r).cumprod()
        factor = adjclose / g["close"]
        iso = pd.to_datetime(g["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        out = pd.DataFrame({
            "date": iso.values,
            "open": (g["open"] * factor).values,
            "high": (g["high"] * factor).values,
            "low": (g["low"] * factor).values,
            "close": adjclose.values,
            "volume": g["volume"].values,
            "amount": g["amount"].values,
        })
        sym = to_qlib_symbol(code)
        out.to_csv(CSV_DIR / f"{sym}.csv", index=False)
        instruments.append((sym, f"{g['date'].min()[:4]}-{g['date'].min()[4:6]}-{g['date'].min()[6:8]}",
                            f"{g['date'].max()[:4]}-{g['date'].max()[4:6]}-{g['date'].max()[6:8]}"))
    with open(OUT / "instruments" / "all.txt", "w") as f:
        for sym, s, e in sorted(instruments):
            f.write(f"{sym}\t{s}\t{e}\n")
    print(f"wrote {len(instruments)} instruments, calendar {len(calendar)} days")

    # dump to qlib binary via upstream script
    script = Path("_refs/qlib/scripts/dump_bin.py")
    if script.exists():
        import subprocess
        cmd = [sys.executable, str(script), "dump_all",
               f"--data_path={CSV_DIR}", f"--qlib_dir={OUT}",
               "--date_field_name=date", "--exclude_fields=date"]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("qlib dump done")
    else:
        print("dump_bin.py not found; CSVs written only", file=sys.stderr)


if __name__ == "__main__":
    main()
