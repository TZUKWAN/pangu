"""Pangu 2.0 实验登记簿：append-only JSONL（data/experiments/registry.jsonl）。

铁律：
- register() 校验必填字段（缺则 :class:`MissingExperimentFields`），自动附
  ts + registry_seq（仅 register 行递增）；
- mark_failed() 追加 status 行（同 id 后写覆盖读出时的 status）；
- **没有** delete / update-in-place API——纠错只能追加新行。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

REGISTRY_PATH = "data/experiments/registry.jsonl"

REQUIRED_FIELDS = (
    "experiment_id", "hypothesis", "economic_rationale", "data", "pit_status",
    "universe", "decision_time", "execution_time", "features", "label",
    "train_range", "validation_range", "test_range", "costs", "slippage",
    "baseline", "parameters", "optimization_method", "n_variants_tried",
    "metrics", "leakage_audit", "independent_backtest", "conclusion", "status",
)


class MissingExperimentFields(ValueError):
    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__(f"missing experiment fields: {self.missing}")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ExperimentRegistry:
    """append-only JSONL 登记簿。"""

    def __init__(self, path: str = REGISTRY_PATH):
        self.path = Path(path)

    # ------------------------------------------------------------------ #
    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _count_registers(self) -> int:
        return sum(1 for r in self._read() if r.get("kind") == "register")

    # ------------------------------------------------------------------ #
    def register(self, experiment: dict) -> dict:
        exp = dict(experiment)
        missing = [k for k in REQUIRED_FIELDS if k not in exp]
        if missing:
            raise MissingExperimentFields(missing)
        rec = {"kind": "register", "ts": _now(),
               "registry_seq": self._count_registers() + 1, **exp}
        self._append(rec)
        return rec

    def mark_failed(self, experiment_id: str, reason: str) -> dict:
        rec = {"kind": "status", "ts": _now(),
               "experiment_id": str(experiment_id),
               "status": "failed", "reason": str(reason)}
        self._append(rec)
        return rec

    def query(self, status: Optional[str] = None,
              family: Optional[str] = None) -> list[dict]:
        """读出登记条目（status 行覆盖原 status），按 status/family 过滤。"""
        entries = []
        latest: dict[str, dict] = {}
        for rec in self._read():
            if rec.get("kind") == "register":
                entries.append(dict(rec))
            elif rec.get("kind") == "status":
                latest[str(rec.get("experiment_id"))] = rec
        out = []
        for e in entries:
            eff = dict(e)
            upd = latest.get(str(e.get("experiment_id")))
            if upd:
                eff["status"] = upd.get("status", eff.get("status"))
                eff["status_update_reason"] = upd.get("reason")
            if status is not None and eff.get("status") != status:
                continue
            if family is not None and e.get("family") != family:
                continue
            out.append(eff)
        return out
