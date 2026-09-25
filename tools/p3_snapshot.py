"""Pangu 3.0 Phase 0 / Task 0.3 — behavior snapshot fixtures.

Captures current pipeline behavior on a FIXED historical date (PIT replay) as
regression fixtures, plus the strategy registry state.  Real-current-data
outputs are NOT captured here (they change daily); live behavior is covered by
explicit smoke runs recorded in docs/pangu3/BASELINE.md.

Usage: .venv/Scripts/python tools/p3_snapshot.py [date=YYYYMMDD]
Output: engine/tests/fixtures/pangu3_snapshot/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

OUT = Path("engine/tests/fixtures/pangu3_snapshot")


def main() -> None:
    date = (sys.argv[1] if len(sys.argv) > 1 else "20260720").replace("-", "")
    OUT.mkdir(parents=True, exist_ok=True)

    from engine.config import load_config
    from engine.pipeline_factory import PipelineFactory

    cfg = load_config()
    pipe = PipelineFactory.from_config(cfg, mode="replay")
    result = pipe.run(date, replay=True)
    d = result.to_dict()

    # pipeline shape snapshot (structure + content on fixed date)
    (OUT / f"pipeline_replay_{date}.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # separated slices the task list names explicitly
    def dump(name, obj):
        (OUT / f"{name}_{date}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    dump("final_recommendations", d.get("final_recommendations"))
    dump("watchlist", d.get("watchlist"))
    dump("candidate_evidence", d.get("candidate_evidence"))
    dump("source_status", d.get("source_status"))
    dump("strategy_signals", d.get("strategy_signals"))
    dump("market_phase", d.get("sentiment"))

    # strategy registry state
    from engine.strategies.registry import StrategyRegistry
    reg = StrategyRegistry()
    rows = []
    for m in reg.list():
        rows.append(m.to_dict())
    (OUT / "strategy_registry.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print("snapshot written to", OUT)
    print("final:", len(d.get("final_recommendations") or []),
          "watch:", len(d.get("watchlist") or []),
          "candidates:", len(d.get("candidates") or []))


if __name__ == "__main__":
    main()
