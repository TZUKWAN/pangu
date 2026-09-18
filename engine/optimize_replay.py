"""回放回测参数优化器：网格扫描 + 排名。

用法（示例）::

    from engine.optimize_replay import OptimizeGrid, grid_search
    result = grid_search(
        start="20260410", end="20260612",   # 优化期（留出期另行验证）
        grid=OptimizeGrid.build_default(),
    )
    print(result.head(20).to_string())

设计约束：
- 所有配置共享同一个 PIT 数据面（一次内存装载）；
- 每个网格点独立写报告文件（data/research/opt_<tag>.json）；
- 排名指标 = 成功率（净费用后），并列时按 (盈亏比, 成交样本) 破平；
  样本 < min_trades 的配置标记 insufficient，不参与排名头部。
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .replay_backtest import ReplayBacktester, ReplayBacktestConfig
from .replay_loader import ReplayDataLoader

logger = logging.getLogger("pangu.optimize_replay")


@dataclass
class OptimizeGrid:
    """参数网格：键为 settings.yaml 覆盖路径（点分），值为候选列表。"""

    params: dict[str, list[Any]] = field(default_factory=dict)

    def items(self) -> list[tuple[str, list[Any]]]:
        return list(self.params.items())

    @staticmethod
    def build_default() -> "OptimizeGrid":
        return OptimizeGrid(params={
            # 入场可执行性：区间半宽、主买点深度上限
            "entry_exit.entry_zone_width": [0.015, 0.03],
            "entry_exit.primary_max_below_pct": [0.04, 0.08],
            # 退出：最长持有天数（1-3，经 short_term_agent.horizon_days 生效）
            "short_term_agent.horizon_days": [2, 3],
            # 反追涨整体灵敏度：距 MA5 阈值
            "anti_chase.thresholds.default.dist_ma5": [0.10, 0.14],
        })

    def combinations(self) -> list[dict[str, Any]]:
        keys = list(self.params.keys())
        out = []
        for values in itertools.product(*(self.params[k] for k in keys)):
            override: dict[str, Any] = {}
            for k, v in zip(keys, values):
                node = override
                parts = k.split(".")
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = v
            out.append(override)
        return out


def _flatten_override(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    override: dict[str, Any] = {}
    for k, v in pairs:
        node = override
        parts = k.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = v
    return override


def grid_search(
    start: str,
    end: str,
    grid: Optional[OptimizeGrid] = None,
    *,
    fixed_combos: Optional[list[dict[str, Any]]] = None,
    settings_path: str = "config/settings.yaml",
    base_override: Optional[dict[str, Any]] = None,
    output_dir: str = "data/research",
    min_trades: int = 20,
    max_combos: Optional[int] = None,
    loader: Optional[ReplayDataLoader] = None,
    tag_prefix: str = "opt",
    csv_name: Optional[str] = None,
) -> pd.DataFrame:
    """扫描网格（或直接给定 fixed_combos），返回按成功率排名的 DataFrame。"""
    if fixed_combos is not None:
        combos = [dict(c) for c in fixed_combos]
    else:
        grid = grid or OptimizeGrid.build_default()
        combos = grid.combinations()
    if max_combos:
        combos = combos[:max_combos]
    if loader is None:
        loader = ReplayDataLoader()
    rows: list[dict[str, Any]] = []
    total = len(combos)
    for i, combo in enumerate(combos):
        merged: dict[str, Any] = json.loads(json.dumps(base_override or {}))
        for k, v in combo.items():
            node = merged
            parts = k.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = v
        tag = f"{tag_prefix}{i:02d}"
        cfg = ReplayBacktestConfig(
            start_date=start, end_date=end,
            settings_path=settings_path,
            settings_override=merged,
            output_dir=output_dir,
            min_trades_for_claim=min_trades,
            enable_progress=False,
            tag=tag,
        )
        t0 = time.time()
        try:
            bt = ReplayBacktester(cfg, loader=loader)
            report = bt.run()
        except Exception as e:  # noqa: BLE001
            logger.warning("网格点 %s 失败: %s", tag, e)
            continue
        sm = report.get("success_metrics", {})
        sg = report.get("signals", {})
        rows.append({
            "tag": tag,
            "override": json.dumps(combo, ensure_ascii=False),
            "signals": sg.get("total"),
            "closed": sg.get("closed"),
            "execution_rate": sg.get("execution_rate"),
            "success_rate": sm.get("success_rate"),
            "wilson_lo": sm.get("wilson_95_lower"),
            "avg_net_return": sm.get("avg_net_return"),
            "profit_factor": sm.get("profit_factor"),
            "max_drawdown": sm.get("max_drawdown"),
            "insufficient": sm.get("insufficient_sample"),
            "elapsed_s": round(time.time() - t0, 1),
        })
        logger.info(
            "[%d/%d] %s 成功率=%s 成交=%s PF=%s (%.0fs)",
            i + 1, total, tag, sm.get("success_rate"), sg.get("closed"),
            sm.get("profit_factor"), time.time() - t0,
        )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(
            ["success_rate", "profit_factor", "closed"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
        out = Path(output_dir) / (csv_name or f"optimize_{start}_{end}.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        logger.info("优化结果已写入 %s", out)
    return df
