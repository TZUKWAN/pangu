"""PIT 可投资宇宙构建：在 asof 日只使用当时已知的信息筛票。

过滤条件（全部 PIT 安全）：
- ``min_listed_days``：档案内首见至今的交易日数不足 → 剔除（次新股）。
- ``exclude_st``：当日 ST 标记为真 → 剔除（历史 ST 状态，不做全期静态过滤）。
- 停牌（成员但当日无 bar）/ 不可交易 → 剔除。

每次构建向 <pit_dir>/universe_manifest.jsonl 追加一行 manifest，便于审计
（哪一天、用了什么阈值、剔了多少、为什么）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List

import pandas as pd

from .lookahead import to_compact, to_iso
from .pit_store import DEFAULT_PIT_DIR, PITStore

DEFAULT_MIN_LISTED_DAYS = 60


class UniverseBuilder:
    """基于 PITStore 的 asof 宇宙构建器。"""

    def __init__(self, store: PITStore, pit_dir: str | Path | None = None) -> None:
        self.store = store
        self.pit_dir = Path(pit_dir) if pit_dir else Path(getattr(store, "pit_dir", DEFAULT_PIT_DIR))

    def asof(
        self,
        date: Any,
        min_listed_days: int = DEFAULT_MIN_LISTED_DAYS,
        exclude_st: bool = True,
    ) -> pd.DataFrame:
        """返回 asof 宇宙快照（全量行 + included 布尔列 + exclude_reason）。

        返回的是带原因标注的全量成员表：included=True 的行为当日可投资宇宙；
        included=False 的行带 exclude_reason（分号连接），便于解释剔除原因。
        """
        d = to_compact(date)
        uni = self.store.universe(d).copy()
        reasons: List[str] = []
        for _, row in uni.iterrows():
            row_reasons: List[str] = []
            if bool(row["suspended"]):
                row_reasons.append("suspended")
            if not bool(row["tradable"]):
                row_reasons.append("not_tradable")
            if int(row["listed_days"]) < int(min_listed_days):
                row_reasons.append(f"listed_days<{min_listed_days}")
            if exclude_st and bool(row["is_st"]):
                row_reasons.append("st")
            reasons.append(";".join(row_reasons))
        uni["included"] = [not r for r in reasons]
        uni["exclude_reason"] = reasons
        self._append_manifest(d, uni, min_listed_days, exclude_st)
        return uni

    def _append_manifest(
        self, d: str, uni: pd.DataFrame, min_listed_days: int, exclude_st: bool
    ) -> None:
        included = int(uni["included"].sum()) if not uni.empty else 0
        entry = {
            "date": to_iso(d),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "min_listed_days": int(min_listed_days),
            "exclude_st": bool(exclude_st),
            "universe_size": int(len(uni)),
            "included": included,
            "excluded": int(len(uni) - included),
            "reason_counts": self._reason_counts(uni),
        }
        try:
            self.pit_dir.mkdir(parents=True, exist_ok=True)
            with open(self.pit_dir / "universe_manifest.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            # manifest 是审计附加物，绝不阻塞主链路
            pass

    @staticmethod
    def _reason_counts(uni: pd.DataFrame) -> dict:
        counts: dict = {}
        if uni.empty:
            return counts
        for reason in uni["exclude_reason"]:
            if not reason:
                continue
            for part in str(reason).split(";"):
                counts[part] = counts.get(part, 0) + 1
        return counts
