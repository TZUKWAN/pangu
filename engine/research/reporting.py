"""因子诊断报告落盘（P2-004）。

- 每因子一份 JSON：data/experiments/factors/<name>.json（内含 pit_checks）。
- 向 data/experiments/factor_index.jsonl 追加一行索引：
  {name, version, family, horizons, status:"evaluated", ts}。

诚实规则：报告缺 pit_checks / feature_used_only_past 时拒绝写盘。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_OUT_DIR = "data/experiments/factors"


def write_factor_report(factor, report: dict,
                        out_dir: Optional[str] = None) -> dict:
    """写单因子 JSON 报告 + 追加 experiments 索引行。返回落盘路径。"""
    out = Path(out_dir or DEFAULT_OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    pit = report.get("pit_checks")
    if not isinstance(pit, dict) or "feature_used_only_past" not in pit:
        raise ValueError(
            "report must carry pit_checks.feature_used_only_past (PIT 诚实留痕，拒绝无守卫的报告)"
        )

    meta = factor.meta
    payload = {
        "name": meta.name,
        "version": meta.version,
        "family": meta.family,
        "description": meta.description,
        "economic_hypothesis": meta.economic_hypothesis,
        "status": "evaluated",
        **report,
        "pit_checks": {
            **pit,
            "feature_used_only_past": bool(pit.get("feature_used_only_past")),
        },
    }
    report_path = out / f"{meta.name}.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    horizons = payload.get("horizons") or {}
    index_line = {
        "name": meta.name,
        "version": meta.version,
        "family": meta.family,
        "horizons": [str(h) for h in horizons.keys()],
        "status": "evaluated",
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    index_path = out.parent / "factor_index.jsonl"
    with open(index_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(index_line, ensure_ascii=False) + "\n")

    return {"report_path": str(report_path), "index_path": str(index_path),
            "index_line": index_line}
