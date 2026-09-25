"""运行历史持久化（Phase 12）。

每次 /pangu 决策落盘 data/decision_runs/<run_id>.json，保证"当时为什么推荐"
可完全重建（asof/freshness/evidence ids/top20/holding/entry-exit/warnings）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from engine.decision.contracts import DecisionRun

RUNS_DIR = Path("data/decision_runs")


class DecisionRunStore:
    def __init__(self, runs_dir: Path = RUNS_DIR):
        self.dir = Path(runs_dir)

    def save(self, run: DecisionRun) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"{run.run_id}.json"
        p.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=1),
                     encoding="utf-8")
        return p

    def load(self, run_id: str) -> Optional[DecisionRun]:
        p = self.dir / f"{run_id}.json"
        if not p.exists():
            return None
        try:
            return DecisionRun.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError):
            return None

    def latest(self, execution_date: Optional[str] = None) -> Optional[DecisionRun]:
        runs = self.list_runs(execution_date=execution_date)
        return self.load(runs[-1][0]) if runs else None

    def list_runs(self, execution_date: Optional[str] = None
                  ) -> List[tuple[str, str]]:
        """[(run_id, saved_iso)] 按文件名（含时间戳语义）升序。"""
        if not self.dir.exists():
            return []
        out = []
        for p in sorted(self.dir.glob("pangu-*.json")):
            run_id = p.stem
            if execution_date and not run_id.startswith(
                    f"pangu-{execution_date.replace('-', '')}"):
                continue
            out.append((run_id, __import__("datetime").datetime
                        .fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")))
        return out
