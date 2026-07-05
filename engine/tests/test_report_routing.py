"""报告路由测试：降级报告不污染 latest_ok.json 指针。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.report import save_report


def _result(data_quality: str = "ok", tradable: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        date="20260703",
        sentiment={"temperature": 50, "posture": "震荡", "advice": "观望", "components": {}},
        boards=[], candidates=[], rejected=[],
        posture_advice="观望。", warnings=[],
        source_state={"all_spot": {"status": "ok"}},
        news={},
        data_quality=data_quality,
        tradable=tradable,
        no_trade_reason="",
        block_reasons=[],
        to_dict=lambda: {},
    )


def test_ok_report_updates_latest_ok(tmp_path: Path) -> None:
    result = _result("ok", True)
    save_report(result, report_dir=str(tmp_path))
    assert (tmp_path / "20260703.md").exists()
    assert (tmp_path / "20260703.json").exists()
    latest = json.loads((tmp_path / "latest_ok.json").read_text(encoding="utf-8"))
    assert latest["date"] == "20260703"
    assert latest["data_quality"] == "ok"


def test_degraded_report_routed_to_degraded_dir(tmp_path: Path) -> None:
    result = _result("degraded", False)
    save_report(result, report_dir=str(tmp_path))
    assert (tmp_path / "degraded" / "20260703.md").exists()
    assert (tmp_path / "degraded" / "20260703.json").exists()
    assert not (tmp_path / "latest_ok.json").exists()


def test_force_degraded_overrides_ok(tmp_path: Path) -> None:
    result = _result("ok", True)
    save_report(result, report_dir=str(tmp_path), force_degraded=True)
    assert (tmp_path / "degraded" / "20260703.md").exists()
    assert not (tmp_path / "latest_ok.json").exists()
