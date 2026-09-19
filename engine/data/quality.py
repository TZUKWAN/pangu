"""数据质量报告：量化 asof 日的数据完整度/新鲜度并判定 PIT 安全性。

指标定义：
- data_completeness：宇宙成员中当日有 bar 的比例。
- missing_ratio：1 - data_completeness。
- freshness：请求日是交易日且当日确有 bar → 1.0，否则 0.0。
- source_count：近 30 个自然日内有数据的源表数量（日线/宇宙/行业/指数）。
- revision_risk：0.0（档案为一次性快照，无后续修订通道；接入修订流后重估）。
- pit_safe：completeness >= COMPLETENESS_MIN 且 missing <= MISSING_MAX
  且 freshness >= 1.0，任一阈值不满足即为 False（fail closed）。

报告写入 <pit_dir>/quality/<YYYYMMDD>.json（out_dir 可覆盖）。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .lookahead import to_compact, to_iso
from .pit_store import DEFAULT_PIT_DIR, MANIFEST_TABLES, PITStore


class DataQualityReport:
    COMPLETENESS_MIN = 0.80
    MISSING_MAX = 0.20
    FRESHNESS_MIN = 1.0
    SOURCE_COUNT_MIN = 2

    @classmethod
    def build(
        cls,
        store: PITStore,
        date: Any,
        out_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        d = to_compact(date)
        uni = store.universe(d)
        panel = store.daily_panel(d, d)

        members = len(uni)
        if members and not panel.empty:
            traded = set(panel.index.get_level_values("code"))
            with_bar = sum(1 for code in uni.index if code in traded)
        else:
            with_bar = 0
        completeness = round(with_bar / members, 4) if members else 0.0
        missing_ratio = round(1.0 - completeness, 4)
        freshness = 1.0 if (d in set(store._trade_dates) and not panel.empty) else 0.0

        window_start = (datetime.strptime(d, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
        source_count = sum(
            1 for t in MANIFEST_TABLES if store.row_count(t, window_start, d) > 0
        )

        pit_safe = bool(
            completeness >= cls.COMPLETENESS_MIN
            and missing_ratio <= cls.MISSING_MAX
            and freshness >= cls.FRESHNESS_MIN
            and source_count >= cls.SOURCE_COUNT_MIN
        )
        report = {
            "date": to_iso(d),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "data_completeness": completeness,
            "freshness": freshness,
            "source_count": source_count,
            "missing_ratio": missing_ratio,
            "revision_risk": 0.0,
            "pit_safe": pit_safe,
            "thresholds": {
                "completeness_min": cls.COMPLETENESS_MIN,
                "missing_max": cls.MISSING_MAX,
                "freshness_min": cls.FRESHNESS_MIN,
                "source_count_min": cls.SOURCE_COUNT_MIN,
            },
            "detail": {
                "universe_members": int(members),
                "members_with_bar": int(with_bar),
            },
        }
        cls.write(report, out_dir if out_dir is not None else (Path(getattr(store, "pit_dir", DEFAULT_PIT_DIR)) / "quality"))
        return report

    @staticmethod
    def write(report: Dict[str, Any], out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{to_compact(report['date'])}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
