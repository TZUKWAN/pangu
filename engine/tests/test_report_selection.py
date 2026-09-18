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
    """符合 PipelineResult.to_dict 新契约的正式报告。"""
    candidates = [
        {"code": f"00000{i}", "name": f"s{i}",
         "recommend": {"recommend_score": 70.0 + i, "grade": "A"}}
        for i in range(n)
    ]
    return {
        "date": date,
        "data_quality": "ok",
        "tradable": False,
        "candidates": candidates,
        "final_recommendations": [],
        "watchlist": [],
        "rejected": [],
        "final_count": 0,
        "watch_count": 0,
        "raw_candidate_count": n,
        "candidate_evidence": {c["code"]: {"code": c["code"]} for c in candidates},
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


def test_report_is_complete_false_for_invalid_raw_count(reports_dir):
    """原始候选计数小于实际展示数的残件必须被拒绝。"""
    d = _complete_report()
    d["raw_candidate_count"] = 0
    assert server._report_is_complete(d) is False


def test_report_is_complete_accepts_data_ok_without_candidates(reports_dir):
    """数据完整但无低风险买点仍是正式报告。"""
    assert server._report_is_complete(_complete_report(n=0)) is True


def test_report_is_complete_false_for_missing_scores(reports_dir):
    d = {"date": "20260701",
         "candidates": [{"code": "000001", "recommend": {}}],
         "source_status": {"structured_data": "ok"}}
    assert server._report_is_complete(d) is False


def test_report_is_complete_false_for_no_structured(reports_dir):
    d = _complete_report()
    d.pop("source_status")
    assert server._report_is_complete(d) is False


def test_report_is_complete_rejects_legacy_unknown_quality(reports_dir):
    """旧报告没有明确 data_quality=ok，不能伪装成 latest。"""
    d = _complete_report()
    d.pop("data_quality")
    assert server._report_is_complete(d) is False


def test_report_is_complete_rejects_degraded(reports_dir):
    d = _complete_report()
    d["data_quality"] = "degraded"
    assert server._report_is_complete(d) is False


# ---------------------------- 排序/选择策略 ----------------------------

def test_old_p0_does_not_override_newer_same_date_json(reports_dir):
    """同 date：外部旧 _p0 vs scan 新写的 .json → 选 mtime 更新的 .json。"""
    _write(reports_dir / "20260701_p0.json", _complete_report(), mtime_offset=-100)
    _write(reports_dir / "20260701.json", _complete_report(), mtime_offset=0)
    assert server._list_report_paths()[0].name == "20260701.json"
    assert server._find_latest_report() is not None


def test_incomplete_p0_is_skipped(reports_dir):
    """不完整 _p0（无明确质量）即便 mtime 最新也跳过，落回完整 .json。"""
    bad = _complete_report()
    bad.pop("data_quality")
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
    bad["data_quality"] = "degraded"
    _write(reports_dir / "20260701_p0.json", bad, mtime_offset=0)
    _write(reports_dir / "20260701.json", bad, mtime_offset=0)
    assert server._find_latest_report() is None
