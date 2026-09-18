"""并行网格搜索驱动（Windows 兼容：spawn + 每进程独立 loader）。

用法::

    python tools/run_optimize.py --start 20260601 --end 20260630 --workers 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("run_optimize")



# 网格：2^3 = 8 组（第四轮：止损结构/持有期/防御组合；走前验证口径）
GRID_KEYS = [
    ("short_term_agent.horizon_days", [1, 2]),
    ("entry_exit.atr_multiplier", [1.5, 2.5]),
    ("strategy_framework.pools", [
        ["题材龙头", "连板梯队", "趋势回踩", "超跌反弹", "小盘优质", "大市值低波", "事件驱动"],
        ["超跌反弹", "大市值低波", "事件驱动"],
    ]),
]
BASE_OVERRIDE = {
    "structured_data": {"enabled": False},
    "entry_exit": {"primary_style_preference": "breakout_confirm", "entry_zone_width": 0.04},
    "strategy_framework": {"gate": {"market_regime_filter": True}},
    "guard": {"exclude_one_word_limit": True},
}


def build_combos() -> list[dict]:
    keys = [k for k, _ in GRID_KEYS]
    out = []
    for values in itertools.product(*[v for _, v in GRID_KEYS]):
        combo = {}
        for k, v in zip(keys, values):
            node = combo
            parts = k.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = v
        out.append(combo)
    return out


def worker(args_tuple) -> str:
    (wid, combos, start, end, output_dir) = args_tuple
    # Windows spawn 不继承父进程 sys.path，手动补项目根
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from engine.optimize_replay import grid_search
    from engine.replay_loader import ReplayDataLoader

    loader = ReplayDataLoader()
    df = grid_search(
        start, end,
        fixed_combos=combos,
        base_override=BASE_OVERRIDE,
        output_dir=output_dir,
        min_trades=20,
        loader=loader,
        tag_prefix=f"w{wid}_",
        csv_name=f"optimize_part{wid}.csv",
    )
    return f"worker {wid}: {len(df)} rows"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="data/research")
    parser.add_argument("--optimize-only", action="store_true",
                        help="只打印组合不执行")
    args = parser.parse_args()

    combos = build_combos()
    logger.info("共 %d 个组合", len(combos))
    if args.optimize_only:
        for c in combos:
            print(json.dumps(c, ensure_ascii=False))
        return 0

    chunks = [combos[i::args.workers] for i in range(args.workers)]
    tasks = [(i, chunk, args.start, args.end, args.output_dir)
             for i, chunk in enumerate(chunks) if chunk]
    with Pool(processes=args.workers) as pool:
        for msg in pool.imap_unordered(worker, tasks):
            logger.info(msg)
    # 合并各分片结果
    import pandas as pd
    parts = sorted(Path(args.output_dir).glob("optimize_part*.csv"))
    if parts:
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df = df.sort_values(
            ["success_rate", "profit_factor", "closed"],
            ascending=[False, False, False], na_position="last",
        ).reset_index(drop=True)
        out = Path(args.output_dir) / f"optimize_{args.start}_{args.end}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        logger.info("合并结果 → %s\n%s", out, df.head(10).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
