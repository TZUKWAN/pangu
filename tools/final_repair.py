"""Final repair: single-threaded, per-half-year windows, only codes whose
breadth_raw coverage ends before 20251201 but the stock still traded (truly
truncated by earlier throttling).  Writes directly into breadth_raw with
source='tencent' using the hfq chain (PIT-safe)."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.request

DB = "data/market_breadth/raw.sqlite3"
HEADERS = {"User-Agent": "Mozilla/5.0"}
WINS = [("2022-01-01", "2022-06-30"), ("2022-07-01", "2022-12-31"),
        ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"),
        ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-14")]


def fetch(code: str, start: str, end: str, fq: str) -> list:
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={code},day,{start},{end},800,{fq}")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            d = json.loads(urllib.request.urlopen(req, timeout=25).read())
            data = d["data"][code]
            return data.get("hfqday" if fq == "hfq" else "day") or []
        except Exception:  # noqa: BLE001
            time.sleep(2.0 * (attempt + 1))
    return []


def main() -> None:
    conn = sqlite3.connect(DB, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("PRAGMA journal_mode=WAL")
    short = [r[0] for r in conn.execute(
        """SELECT code FROM (SELECT code, MAX(date) mx FROM breadth_raw GROUP BY code)
           WHERE mx < '20251201' AND mx > '20221001'""")]
    print(f"codes to final-repair: {len(short)}", flush=True)
    added_total = 0
    for k, pcode in enumerate(short):
        tcode = pcode.replace(".", "")
        # verify the stock still traded recently on tencent; if not skip (delisted)
        probe = fetch(tcode, "2025-09-01", "2025-12-14", "")
        if not probe:
            print(f"[{pcode}] no recent data → skip (delisted?)", flush=True)
            continue
        have_any = {r[0] for r in conn.execute(
            "SELECT date FROM breadth_raw WHERE code=?", (pcode,))}
        # collect both series across windows (sequential, throttled source)
        hfq_rows, raw_rows = {}, {}
        for (s, e) in WINS:
            for r in fetch(tcode, s, e, "hfq"):
                hfq_rows[r[0].replace("-", "")] = float(r[2])
            for r in fetch(tcode, s, e, ""):
                raw_rows[r[0].replace("-", "")] = (float(r[1]), float(r[3]), float(r[4]),
                                                   float(r[2]), float(r[5]))
            time.sleep(0.8)
        dates = sorted(set(hfq_rows) & set(raw_rows))
        prev = None
        rows = []
        for d in dates:
            o, h, l, c, v = raw_rows[d]
            f = hfq_rows[d] / c
            if prev is None:
                prev = (d, hfq_rows[d], c, f)
                continue
            if d in have_any:
                prev = (d, hfq_rows[d], c, f)
                continue
            pd_, phfq, praw, pf = prev
            pct = (hfq_rows[d] / phfq - 1.0) * 100.0
            preclose = praw * (f / pf)
            rows.append((d, pcode, o, h, l, c, round(preclose, 4), v, v * c,
                         round(pct, 4), None, 0, "tencent"))
            prev = (d, hfq_rows[d], c, f)
        if rows:
            conn.executemany("""INSERT OR IGNORE INTO breadth_raw
                (date, code, open, high, low, close, preclose, volume, amount,
                 pct_change, turnover, is_st, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
            conn.commit()
            added_total += len(rows)
        if (k + 1) % 25 == 0:
            print(f"[{k+1}/{len(short)}] added {added_total} rows", flush=True)
    print(f"final repair done, added {added_total} rows", flush=True)
    n = conn.execute("""SELECT COUNT(*) FROM (SELECT code, COUNT(*) c FROM breadth_raw
        WHERE date>='20220101' AND date<='20251214' GROUP BY code HAVING c<800)""").fetchone()[0]
    print(f"remaining short codes: {n}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
