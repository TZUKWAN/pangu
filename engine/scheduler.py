"""每日盘后调度入口（标准库实现，不依赖 apscheduler）。

典型用法：
    python -m engine.scheduler           # 跑完整盘后链路
    python -m engine.scheduler --dry-run # 只检查步骤与通知配置，不执行耗时取数
    python -m engine.scheduler --date 20260701 --skip-rps

链路：
    1. rps-build（可选，默认执行，约 1-2 分钟）
    2. snapshot-build（收盘快照）
    3. scan（选股 Pipeline）
    4. report（生成 Markdown 简报）
    5. notify（如果 PANGU_NOTIFY_WEBHOOK 已配置）

状态与日志：
    - 写 JSON：data/scheduler/YYYYMMDD_status.json
    - 写文本日志：data/scheduler/scheduler.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import build_data_loader, load_config
from .pipeline import Pipeline
from .report import save_report
from .snapshot import SnapshotBuilder

logger = logging.getLogger("pangu.scheduler")

_DEFAULT_STATUS_DIR = Path("data/scheduler")


@dataclass
class StepResult:
    name: str
    status: str  # ok / skipped / failed
    duration_seconds: float = 0.0
    output: Any = None
    error: str = ""


class DailyScheduler:
    """盘后链路调度器。"""

    def __init__(
        self,
        cfg: dict[str, Any],
        date: str | None = None,
        skip_rps: bool = False,
        skip_snapshot: bool = False,
        skip_notify: bool = False,
        dry_run: bool = False,
        workers: int = 10,
        status_dir: str | Path = _DEFAULT_STATUS_DIR,
    ) -> None:
        self.cfg = cfg
        self.date = date or datetime.now().strftime("%Y%m%d")
        self.skip_rps = skip_rps
        self.skip_snapshot = skip_snapshot
        self.skip_notify = skip_notify
        self.dry_run = dry_run
        self.workers = workers
        self.status_dir = Path(status_dir)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[StepResult] = []
        self.pipeline_result: dict[str, Any] | None = None
        self.report_path: Path | None = None

    def _run_step(self, name: str, fn: Callable[[], Any], skip: bool = False) -> StepResult:
        """执行单个步骤并计时。"""
        if skip:
            logger.info("[scheduler] 跳过 %s", name)
            return StepResult(name=name, status="skipped")
        logger.info("[scheduler] 开始 %s", name)
        start = time.time()
        try:
            output = fn()
            dur = time.time() - start
            logger.info("[scheduler] %s 完成，耗时 %.2fs", name, dur)
            return StepResult(name=name, status="ok", duration_seconds=round(dur, 2), output=output)
        except Exception as e:  # noqa: BLE001
            dur = time.time() - start
            logger.exception("[scheduler] %s 失败", name)
            return StepResult(
                name=name,
                status="failed",
                duration_seconds=round(dur, 2),
                error=f"{type(e).__name__}: {e}",
            )

    def _step_rps_build(self) -> dict[str, Any]:
        from . import rps as rps_mod
        dl = build_data_loader(self.cfg)
        return rps_mod.compute_all_rps(
            dl,
            date=self.date,
            db_path=self.cfg.get("output", {}).get("db_path", "data/pangu.db"),
            workers=self.workers,
        )

    def _step_snapshot_build(self) -> dict[str, Any]:
        dl = build_data_loader(self.cfg)
        snapshot_dir = self.cfg.get("data", {}).get("snapshot_dir", "data/snapshots")
        builder = SnapshotBuilder(dl, snapshot_dir=snapshot_dir)
        result = builder.build(self.date)
        return result.to_dict()

    def _step_scan(self) -> dict[str, Any]:
        pipe = self._build_pipeline()
        result = pipe.run(self.date)
        data = json.loads(result.to_json())
        self.pipeline_result = data
        return {"date": result.date, "candidates": len(result.candidates), "warnings": result.warnings}

    def _step_report(self) -> dict[str, Any]:
        if self.dry_run:
            return {"dry_run": True, "report_path": None}
        if self.pipeline_result is None:
            # report 依赖 scan，如果 scan 被跳过或失败则无法生成
            raise RuntimeError("无选股结果，无法生成报告")
        # 从 dict 重建 PipelineResult 以复用 save_report
        from .pipeline import PipelineResult
        result = PipelineResult(
            date=self.pipeline_result["date"],
            sentiment=self.pipeline_result["sentiment"],
            boards=self.pipeline_result["boards"],
            candidates=self.pipeline_result["candidates"],
            rejected=self.pipeline_result["rejected"],
            posture_advice=self.pipeline_result["posture_advice"],
            warnings=self.pipeline_result.get("warnings", []),
            news=self.pipeline_result.get("news", {}),
            market_modules=self.pipeline_result.get("market_modules", {}),
            source_status=self.pipeline_result.get("source_status", self.pipeline_result.get("source_state", {})),
            xuanwu_pool=self.pipeline_result.get("xuanwu_pool", {}),
            recommendation_allowed=self.pipeline_result.get("recommendation_allowed", False),
            historical_mode=self.pipeline_result.get("historical_mode", "live"),
        )
        result.watchlist = self.pipeline_result.get("watchlist", [])
        result.final_recommendations = self.pipeline_result.get("final_recommendations", [])
        result.strategy_signals = self.pipeline_result.get("strategy_signals", {})
        result.strategy_candidates = self.pipeline_result.get("strategy_candidates", [])
        report_dir = self.cfg.get("output", {}).get("report_dir", "data/reports")
        self.report_path = save_report(result, report_dir)
        return {"report_path": str(self.report_path)}

    def _step_notify(self) -> dict[str, Any]:
        from .notifier import Notifier
        notifier = Notifier.from_env()
        if not notifier.enabled:
            return {"enabled": False, "reason": "PANGU_NOTIFY_WEBHOOK 未配置，通知已跳过"}
        if self.dry_run:
            return {"enabled": True, "dry_run": True, "reason": "dry-run 模式不发送真实通知"}
        summary = self._make_notify_summary()
        return notifier.send(summary)

    def _make_notify_summary(self) -> dict[str, Any]:
        data = self.pipeline_result or {}
        candidates = data.get("candidates") or []
        top = [
            {
                "code": c.get("code"),
                "name": c.get("name"),
                "grade": (c.get("recommend") or {}).get("grade"),
                "score": (c.get("recommend") or {}).get("recommend_score"),
            }
            for c in candidates[:5]
        ]
        sentiment = data.get("sentiment") or {}
        return {
            "date": self.date,
            "target_trade_date": data.get("date") or self.date,
            "temperature": sentiment.get("temperature"),
            "posture": sentiment.get("posture"),
            "top_candidates": top,
            "candidate_count": len(candidates),
            "report_path": str(self.report_path) if self.report_path else None,
            "warnings": (data.get("warnings") or [])[:3],
        }

    def _build_pipeline(self) -> Pipeline:
        from .cli import _build_pipeline as build
        return build(self.cfg)

    def run(self) -> dict[str, Any]:
        """执行完整盘后链路，返回状态摘要。"""
        self.results = []
        overall_start = time.time()

        # 1. RPS 预计算
        self.results.append(self._run_step(
            "rps_build",
            self._step_rps_build,
            skip=self.skip_rps or self.dry_run,
        ))

        # 2. 收盘快照
        self.results.append(self._run_step(
            "snapshot_build",
            self._step_snapshot_build,
            skip=self.skip_snapshot or self.dry_run,
        ))

        # 3. 选股
        self.results.append(self._run_step(
            "scan",
            self._step_scan,
            skip=self.dry_run,
        ))

        # 4. 报告
        self.results.append(self._run_step(
            "report",
            self._step_report,
            skip=self.dry_run,
        ))

        # 5. 通知
        self.results.append(self._run_step(
            "notify",
            self._step_notify,
            skip=self.skip_notify,
        ))

        overall_dur = time.time() - overall_start
        failed = [r for r in self.results if r.status == "failed"]
        summary = {
            "date": self.date,
            "run_at": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "overall_status": "failed" if failed else "ok",
            "overall_duration_seconds": round(overall_dur, 2),
            "steps": [
                {
                    "name": r.name,
                    "status": r.status,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error,
                }
                for r in self.results
            ],
            "report_path": str(self.report_path) if self.report_path else None,
            "candidate_count": len(self.pipeline_result.get("candidates", [])) if self.pipeline_result else 0,
        }
        self._save_status(summary)
        return summary

    def _save_status(self, summary: dict[str, Any]) -> None:
        status_file = self.status_dir / f"{self.date}_status.json"
        try:
            status_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("[scheduler] 状态已保存: %s", status_file)
        except Exception as e:  # noqa: BLE001
            logger.warning("[scheduler] 状态保存失败: %s", e)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    _DEFAULT_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _DEFAULT_STATUS_DIR / "scheduler.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pangu-daily", description="盘古每日盘后调度")
    parser.add_argument("-c", "--config", default=None, help="配置文件路径")
    parser.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    parser.add_argument("--skip-rps", action="store_true", help="跳过 RPS 预计算")
    parser.add_argument("--skip-snapshot", action="store_true", help="跳过收盘快照")
    parser.add_argument("--skip-notify", action="store_true", help="跳过通知")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置与通知，不执行耗时取数")
    parser.add_argument("--workers", type=int, default=10, help="RPS 预计算并发数")
    parser.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config(args.config)
    scheduler = DailyScheduler(
        cfg=cfg,
        date=args.date,
        skip_rps=args.skip_rps,
        skip_snapshot=args.skip_snapshot,
        skip_notify=args.skip_notify,
        dry_run=args.dry_run,
        workers=args.workers,
    )
    summary = scheduler.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["overall_status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
