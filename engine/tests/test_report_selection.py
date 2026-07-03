"""报告生产/选择路径测试：_report_is_complete / _report_sort_key / _find_latest_report。

覆盖领导收口要求：
- 旧/外部 ``_p0.json`` 不劫持更新的正式 ``{date}.json``；
- 不完整 ``_p0.json`` 被跳过；
- 完整 P0 在更新时可优先；
- 跨日期时新报告优先。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from engine.web import server


def _complete_report(date: str = "20260701", n: int = 3) -> dict:
    """受控完整报告（经 _enrich 形态：带 source_status.structured_data）。"""
    return {
        "date": date,
        "candidates": [
            {"code": f"00000{i}", "name": f"s{i}",
             "recommend": {"recommend_score": 70.0 + i, "grade": "A"}}
            for i in range(n)
        ],
        "source_status": {"structured_data": "ok", "market_data": "ok"},
    }


def _write(path: Path, data: dict, mtime_offset: float = 0.0) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    ts = time.time() + mtime_offset
    os.utime(path, (ts, ts))


@pytest.fixture
def reports_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_REPORT_DIR", tmp_path)
    return tmp_path


# ---------------------------- _report_is_complete 单元 ----------------------------

def test_report_is_complete_true_for_full(reports_dir):
    assert server._report_is_complete(_complete_report()) is True


def test_report_is_complete_false_for_empty_candidates(reports_dir):
    d = _complete_report()
    d["candidates"] = []
    assert server._report_is_complete(d) is False


def test_report_is_complete_false_for_missing_scores(reports_dir):
    d = {"date": "20260701",
         "candidates": [{"code": "000001", "recommend": {}}],
         "source_status": {"structured_data": "ok"}}
    assert server._report_is_complete(d) is False


def test_report_is_complete_false_for_no_structured(reports_dir):
    d = _complete_report()
    d.pop("source_status")
    assert server._report_is_complete(d) is False


def test_report_is_complete_accepts_source_state_form(reports_dir):
    """_p0.json 直落形态用 source_state.structured_data（dict），也应判完整。"""
    d = _complete_report()
    d.pop("source_status")
    d["source_state"] = {"structured_data": {"dragon_tiger_daily": {"status": "ok"}}}
    assert server._report_is_complete(d) is True


# ---------------------------- 排序/选择策略 ----------------------------

def test_old_p0_does_not_override_newer_same_date_json(reports_dir):
    """同 date：外部旧 _p0 vs scan 新写的 .json → 选 mtime 更新的 .json。"""
    _write(reports_dir / "20260701_p0.json", _complete_report(), mtime_offset=-100)
    _write(reports_dir / "20260701.json", _complete_report(), mtime_offset=0)
    assert server._list_report_paths()[0].name == "20260701.json"
    assert server._find_latest_report() is not None


def test_incomplete_p0_is_skipped(reports_dir):
    """不完整 _p0（空候选）即便 mtime 最新也跳过，落回完整 .json。"""
    bad = _complete_report()
    bad["candidates"] = []
    _write(reports_dir / "20260701_p0.json", bad, mtime_offset=1000)
    _write(reports_dir / "20260701.json", _complete_report(), mtime_offset=0)
    got = server._find_latest_report()
    assert got is not None and got["candidates"], "不完整 _p0 应被跳过"


def test_complete_p0_preferred_when_newest(reports_dir):
    """完整 _p0（mtime 最新）+ 完整旧 .json → 选 _p0（更新）。"""
    _write(reports_dir / "20260701.json", _complete_report(), mtime_offset=-100)
    _write(reports_dir / "20260701_p0.json", _complete_report(), mtime_offset=0)
    assert server._list_report_paths()[0].name == "20260701_p0.json"
    assert server._find_latest_report() is not None


def test_cross_date_newer_report_wins(reports_dir):
    """跨 date：20260702 完整 vs 20260701_p0 完整（即便 mtime 更新）→ 选 20260702。"""
    _write(reports_dir / "20260701_p0.json", _complete_report("20260701"), mtime_offset=1000)
    _write(reports_dir / "20260702.json", _complete_report("20260702"), mtime_offset=0)
    got = server._find_latest_report()
    assert got is not None and got["date"] == "20260702"


def test_all_incomplete_returns_none(reports_dir):
    """全部不完整时返回 None（不返回残件）。"""
    bad = _complete_report()
    bad["candidates"] = []
    _write(reports_dir / "20260701_p0.json", bad, mtime_offset=0)
    _write(reports_dir / "20260701.json", bad, mtime_offset=0)
    assert server._find_latest_report() is None
