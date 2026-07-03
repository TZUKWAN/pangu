"""每日调度逻辑测试：不依赖真实 akshare/LLM，验证步骤编排与状态保存。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.scheduler import DailyScheduler


@pytest.fixture
def cfg():
    return {
        "data": {"cache_dir": "data/cache", "snapshot_dir": "data/snapshots"},
        "output": {"report_dir": "data/reports", "db_path": "data/pangu.db", "pick_count": 5},
    }


@pytest.fixture
def status_dir(tmp_path):
    return tmp_path / "scheduler"


def test_dry_run_skips_data_fetch(cfg, status_dir):
    """dry-run 不执行取数，只检查配置与通知，状态为 ok。"""
    scheduler = DailyScheduler(cfg, date="20260101", dry_run=True, status_dir=status_dir)
    summary = scheduler.run()
    assert summary["dry_run"] is True
    assert summary["overall_status"] == "ok"
    step_names = {s["name"] for s in summary["steps"]}
    assert step_names == {"rps_build", "snapshot_build", "scan", "report", "notify"}
    for s in summary["steps"]:
        if s["name"] in ("rps_build", "snapshot_build", "scan"):
            assert s["status"] == "skipped", f"{s['name']} 应在 dry-run 跳过"
    # 状态文件已写入
    assert (status_dir / "20260101_status.json").exists()


def test_skip_rps_and_snapshot(cfg, status_dir):
    """可单独跳过 RPS 与快照。"""
    scheduler = DailyScheduler(
        cfg, date="20260101", skip_rps=True, skip_snapshot=True, dry_run=True, status_dir=status_dir
    )
    summary = scheduler.run()
    statuses = {s["name"]: s["status"] for s in summary["steps"]}
    assert statuses["rps_build"] == "skipped"
    assert statuses["snapshot_build"] == "skipped"


def test_scheduler_status_json_content(cfg, status_dir):
    """状态 JSON 包含必要字段。"""
    scheduler = DailyScheduler(cfg, date="20260101", dry_run=True, status_dir=status_dir)
    scheduler.run()
    content = (status_dir / "20260101_status.json").read_text(encoding="utf-8")
    data = json.loads(content)
    assert data["date"] == "20260101"
    assert "run_at" in data
    assert "overall_duration_seconds" in data
    assert data["overall_status"] == "ok"


def test_report_step_fails_without_scan_result(cfg, status_dir):
    """report 步骤在 scan 失败/跳过时应当失败。"""
    scheduler = DailyScheduler(cfg, date="20260101", dry_run=False, status_dir=status_dir)
    # 直接调用 report 步骤， pipeline_result 为 None
    result = scheduler._run_step("report", scheduler._step_report)
    assert result.status == "failed"
    assert "无选股结果" in result.error


def test_scheduler_respects_no_notify_env(cfg, status_dir, monkeypatch):
    """未配置 PANGU_NOTIFY_WEBHOOK 时通知步骤安全跳过。"""
    monkeypatch.delenv("PANGU_NOTIFY_WEBHOOK", raising=False)
    scheduler = DailyScheduler(cfg, date="20260101", dry_run=True, status_dir=status_dir)
    summary = scheduler.run()
    notify_step = next(s for s in summary["steps"] if s["name"] == "notify")
    assert notify_step["status"] == "ok"
