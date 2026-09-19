"""Factor study on the real PIT archive (research window only).

Discipline (docs/pangu2/RESEARCH_PROTOCOL.md):
- Research window: 2022-01-04 → 2025-12-31.  HOLDOUT 2026-06-01→2026-09-04 and
  test 2026-01-01→2026-05-29 are NOT touched here.
- Every factor evaluation is registered in data/experiments/registry.jsonl,
  including failures/degraded ones.

Usage: .venv/Scripts/python tools/run_factor_study.py [family_filter]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from engine.research.data_interface import PITResearchData  # noqa: E402
from engine.research.factors.registry import build_default_registry  # noqa: E402
from engine.research.univariate import evaluate_factor  # noqa: E402
from engine.research.reporting import write_factor_report  # noqa: E402
from engine.validation.experiment_registry import ExperimentRegistry  # noqa: E402
from engine.data.pit_store import PITStore  # noqa: E402

START, END = "2022-01-04", "2025-12-31"
OUT_DIR = Path("data/experiments/factors")


def main() -> None:
    family_filter = sys.argv[1] if len(sys.argv) > 1 else None
    store = PITStore()
    data = PITResearchData(store)
    registry = build_default_registry()
    exp_reg = ExperimentRegistry()
    factors = registry.list()
    print(f"factors: {len(factors)}")

    # preload panel once for speed via data interface
    t_all = time.time()
    for row in factors:
        name = row["name"]
        if family_filter and family_filter not in (row.get("family") or "") and family_filter != name:
            continue
        factor = registry.get(name)
        meta = factor.meta
        t0 = time.time()
        try:
            if row.get("availability", "ready") == "degraded_no_source":
                raise FactorUnavailableLike(name)
            report = evaluate_factor(factor, data, START, END)
            report["pit_checks"] = report.get("pit_checks", {})
            report["pit_checks"]["research_window"] = [START, END]
            try:
                write_factor_report(factor, report, out_dir=str(OUT_DIR))
            except TypeError:
                write_factor_report(factor, report, out_dir=OUT_DIR)
            h5 = report.get("by_horizon", {}).get("5", report.get("horizons", {}).get("5", {}))
            metrics = _extract(report)
            status = "evaluated"
            conclusion = _conclude(metrics)
        except Exception as e:  # noqa: BLE001
            metrics = {}
            status = "failed"
            conclusion = f"degraded_no_source" if isinstance(e, FactorUnavailableLike) else f"error: {e!r}"
            print(f"[{name}] {conclusion}", flush=True)
        try:
            exp_reg.register({
                "experiment_id": f"factor_{name}_v{getattr(meta, 'version', '1')}",
                "hypothesis": getattr(meta, "economic_hypothesis", ""),
                "economic_rationale": getattr(meta, "description", ""),
                "data": "PITStore breadth_raw 2022-01-04→2025-12-31 (pct_change returns)",
                "pit_status": "strict_asof; universe=panel rows; lookahead-guarded",
                "universe": "all A-share panel rows (incl. later-delisted); suspended (no bar) excluded by construction; ST NOT excluded (documented limitation)",
                "decision_time": "15:05 T",
                "execution_time": "T+1 open (engine)",
                "features": [name],
                "label": f"forward return h=1/3/5/10/20 (pct_change sum)",
                "train_range": "n/a (univariate diagnostic)",
                "validation_range": "n/a",
                "test_range": "n/a",
                "costs": "not applied (IC study)",
                "slippage": "n/a",
                "capacity": "n/a",
                "baseline": "zero-IC null",
                "parameters": {"lookback": getattr(meta, "lookback_days", None),
                               "winsorize": getattr(meta, "winsorize", None)},
                "optimization_method": "none (pre-registered definition)",
                "n_variants_tried": 1,
                "metrics": metrics,
                "leakage_audit": report.get("pit_checks", {}) if status == "evaluated" else {"not_run": True},
                "independent_backtest": "pending (strategy stage)",
                "conclusion": conclusion,
                "status": status,
                "family": getattr(meta, "family", ""),
            })
        except Exception as e:  # noqa: BLE001
            print(f"[{meta.name}] registry error: {e!r}", flush=True)
        if status == "evaluated":
            print(f"[{name}] ok {time.time()-t0:.1f}s metrics={json.dumps(metrics)[:220]}", flush=True)
    print(f"factor study done in {time.time()-t_all:.0f}s")


def _extract(report: dict) -> dict:
    """Pull a compact metric dict out of the report shape (defensive)."""
    out = {}
    block = report.get("horizons") or {}
    for h, vals in block.items():
        if isinstance(vals, dict):
            out[f"h{h}_rank_ic"] = vals.get("ic_mean_spearman")
            out[f"h{h}_icir"] = vals.get("icir")
            ls = vals.get("long_short_spread")
            if ls is not None:
                out[f"h{h}_ls_spread"] = ls
            if vals.get("ic_first_half") is not None:
                out[f"h{h}_ic_1st_half"] = vals.get("ic_first_half")
                out[f"h{h}_ic_2nd_half"] = vals.get("ic_second_half")
    return out


def _conclude(metrics: dict) -> str:
    ic5 = metrics.get("h5_rank_ic")
    if ic5 is None:
        return "no_metrics"
    if abs(ic5) < 0.01:
        return "ic_not_significant"
    direction = "positive" if ic5 > 0 else "negative"
    icir = metrics.get("h5_icir") or 0
    if abs(icir) < 0.3:
        return f"{direction}_weak_icir_below_0.3"
    return f"{direction}_candidate_for_strategy_study"


class FactorUnavailableLike(Exception):
    pass


if __name__ == "__main__":
    main()
