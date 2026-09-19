"""实验登记簿测试：必填字段校验、append-only、status 覆盖、query 过滤。"""

from __future__ import annotations

import json

import pytest

from engine.validation.experiment_registry import (
    ExperimentRegistry,
    MissingExperimentFields,
    REQUIRED_FIELDS,
)


def make_exp(eid="exp-001", **kw):
    base = {k: "x" for k in REQUIRED_FIELDS}
    base.update({
        "experiment_id": eid,
        "family": "momentum",
        "status": "proposed",
        "metrics": {"sharpe": 1.2},
    })
    base.update(kw)
    return base


@pytest.fixture
def reg(tmp_path):
    return ExperimentRegistry(str(tmp_path / "registry.jsonl"))


class TestRegister:
    def test_valid_register_appends_seq_and_ts(self, reg, tmp_path):
        rec = reg.register(make_exp())
        assert rec["registry_seq"] == 1
        assert rec["ts"]
        lines = (tmp_path / "registry.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["experiment_id"] == "exp-001"
        assert loaded["registry_seq"] == 1

    def test_seq_increments_across_instances(self, tmp_path):
        p = str(tmp_path / "registry.jsonl")
        ExperimentRegistry(p).register(make_exp("a"))
        ExperimentRegistry(p).register(make_exp("b"))
        reg3 = ExperimentRegistry(p)
        rec = reg3.register(make_exp("c"))
        assert rec["registry_seq"] == 3
        assert [e["experiment_id"] for e in reg3.query()] == ["a", "b", "c"]

    def test_missing_field_rejected(self, reg, tmp_path):
        bad = make_exp()
        del bad["hypothesis"]
        with pytest.raises(MissingExperimentFields) as ei:
            reg.register(bad)
        assert "hypothesis" in ei.value.missing
        assert not (tmp_path / "registry.jsonl").exists()

    def test_all_required_fields_enforced(self, reg):
        for field in REQUIRED_FIELDS:
            bad = make_exp()
            del bad[field]
            with pytest.raises(MissingExperimentFields):
                reg.register(bad)


class TestStatusAndQuery:
    def test_mark_failed_overlays_status(self, reg):
        reg.register(make_exp("e1"))
        rec = reg.mark_failed("e1", "lookahead found in feature build")
        assert rec["status"] == "failed"
        got = reg.query(status="failed")
        assert len(got) == 1
        assert got[0]["experiment_id"] == "e1"
        assert got[0]["status"] == "failed"
        assert got[0]["status_update_reason"] == "lookahead found in feature build"
        assert reg.query(status="proposed") == []

    def test_query_family_filter(self, reg):
        reg.register(make_exp("e1", family="momentum"))
        reg.register(make_exp("e2", family="mean_revert"))
        fams = {e["experiment_id"] for e in reg.query(family="mean_revert")}
        assert fams == {"e2"}

    def test_query_on_missing_file(self, tmp_path):
        assert ExperimentRegistry(str(tmp_path / "none.jsonl")).query() == []


class TestAppendOnly:
    def test_no_delete_or_update_api(self, reg):
        for meth in ("delete", "update", "remove", "patch", "rewrite",
                     "replace", "set_status", "update_in_place", "pop", "clear"):
            assert not hasattr(reg, meth), f"registry must not expose {meth}()"

    def test_updates_are_new_lines_only(self, reg, tmp_path):
        reg.register(make_exp("e1"))
        reg.mark_failed("e1", "because")
        lines = (tmp_path / "registry.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 2                      # 只追加，不改写
        orig = json.loads(lines[0])
        assert orig["status"] == "proposed"         # 原行未被改写
        assert json.loads(lines[1])["kind"] == "status"
