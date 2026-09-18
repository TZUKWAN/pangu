"""验收达标线复验脚本（一条命令完成：回填档案 → 全窗回放 → 对照达标线）。

达标线（docs/验收报告_20260905.md §七，用户授权采用的实测诚实口径）：
    事件驱动档案累积 >= 30 笔成交后，复核 PF >= 1.3 且成功率 >= 42%。

用法::

    python tools/acceptance_check.py                # 回填+全窗复验+判定
    python tools/acceptance_check.py --no-backfill  # 跳过回填（档案已最新时）
    python tools/acceptance_check.py --start 20260801   # 快速冒烟（短窗口）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 直接以 `python tools/acceptance_check.py` 运行时，把项目根加进 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

EVENT_DRIVEN_OVERRIDE = {
    "entry_exit": {
        "primary_style_preference": "breakout_confirm",
        "entry_zone_width": 0.04,
    },
    "strategy_framework": {"pools": ["事件驱动"]},
}
MIN_TRADES = 30
MIN_PROFIT_FACTOR = 1.3
MIN_SUCCESS_RATE = 0.42


def _backfill_archive(today: str) -> str:
    """增量回填本地档案到最近交易日，返回新的档案截止日。"""
    import sqlite3

    db = Path("data/market_breadth/raw.sqlite3")
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT MAX(date) FROM breadth_raw").fetchone()
    last = (row[0] if row and row[0] else "20260101")
    last_dt = datetime.strptime(last, "%Y%m%d")
    today_dt = datetime.strptime(today, "%Y%m%d")
    if (today_dt - last_dt).days < 3:
        print(f"[回填] 档案已最新（{last}），跳过")
        return last
    start = (last_dt + timedelta(days=1)).strftime("%Y%m%d")
    print(f"[回填] BaoStock 日线 {start} ~ {today}（断点续传，首次约 1 小时）...")
    from engine.market_breadth_archive import BaoStockBreadthArchive

    BaoStockBreadthArchive("data/market_breadth").build(start, today)
    print("[回填] 行业成员...")
    from engine.industry_trend_archive import BaoStockIndustryTrendArchive

    BaoStockIndustryTrendArchive("data/market_breadth").build(start, today)
    print("[回填] 巨潮公告...")
    from engine.announcement_archive import CninfoAnnouncementArchive

    CninfoAnnouncementArchive("data/announcement_archive").backfill(start, today)
    print("[回填] RPS 档案直算...")
    from engine.rps import compute_rps_from_archive

    compute_rps_from_archive()
    return today


def _latest_trade_date() -> str:
    import sqlite3

    with sqlite3.connect("data/market_breadth/raw.sqlite3") as conn:
        row = conn.execute("SELECT MAX(date) FROM breadth_raw").fetchone()
    return row[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="验收达标线复验")
    parser.add_argument("--start", default="20260410", help="复验窗口起点")
    parser.add_argument("--no-backfill", action="store_true", help="跳过档案回填")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y%m%d")
    if not args.no_backfill:
        _backfill_archive(today)
    end = _latest_trade_date()
    print(f"[复验] 事件驱动档案 {args.start} ~ {end} ...")

    from engine.replay_backtest import ReplayBacktester, ReplayBacktestConfig

    cfg = ReplayBacktestConfig(
        start_date=args.start,
        end_date=end,
        settings_override=json.loads(json.dumps(EVENT_DRIVEN_OVERRIDE)),
        output_dir="data/research",
        min_trades_for_claim=MIN_TRADES,
        enable_progress=True,
        tag="acceptance",
    )
    report = ReplayBacktester(cfg).run()

    sm = report["success_metrics"]
    sg = report["signals"]
    closed = sg.get("closed") or 0
    success = sm.get("success_rate")
    pf = sm.get("profit_factor")

    print("\n===== 验收达标线复验 =====")
    print(f"窗口: {args.start} ~ {end}")
    print(f"信号 {sg.get('total')} 笔 / 成交 {closed} 笔 / 成功率 {success} / PF {pf}")
    print(f"达标线: 成交>={MIN_TRADES} 且 PF>={MIN_PROFIT_FACTOR} 且 成功率>={MIN_SUCCESS_RATE}")
    if closed < MIN_TRADES:
        print(
            f"结论: 未达标——样本 {closed}/{MIN_TRADES} 不足。"
            "保持每日 `python -m engine.cli daily` 累积推荐，稍后重跑本脚本。"
        )
        return 2
    passed = (pf is not None and pf >= MIN_PROFIT_FACTOR) and (
        success is not None and success >= MIN_SUCCESS_RATE
    )
    print("结论: 达标 ✅" if passed else "结论: 未达标 ❌（样本足够但指标未达线）")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
