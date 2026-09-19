"""走前切分与 holdout 策略测试。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.validation.data_interface import SyntheticValidationData
from engine.validation.walk_forward import (
    HoldoutPolicy,
    HoldoutViolation,
    Split,
    WalkForwardSplitter,
)


def _days(n=100):
    return [d.strftime("%Y-%m-%d")
            for d in pd.bdate_range("2025-01-06", periods=n)]


class TestWalkForwardSplitter:
    def make(self, mode="expanding", n=100, train=40, valid=10, test=10,
             step=10, embargo=2):
        days = _days(n)
        sp = WalkForwardSplitter(days, train_days=train, valid_days=valid,
                                 test_days=test, step_days=step,
                                 embargo_days=embargo, mode=mode)
        return days, sp.split()

    def test_splits_exist_and_cover(self):
        days, splits = self.make()
        assert splits
        # 首个 split 从头开始
        assert splits[0].train[0] == days[0]
        # 连续 fold 步进 step_days
        for a, b in zip(splits, splits[1:]):
            ia, ib = days.index(a.valid[0]), days.index(b.valid[0])
            assert ib - ia == 10

    def test_embargo_respected_no_overlap(self):
        days, splits = self.make(embargo=2)
        for s in splits:
            assert isinstance(s, Split)
            ti = days.index(s.train[1])
            vi = days.index(s.valid[0])
            ve = days.index(s.valid[1])
            si = days.index(s.test[0])
            se = days.index(s.test[1])
            # valid 起点 = train_end + embargo + 1
            assert vi - ti == 3
            # test 起点 = valid_end + embargo + 1
            assert si - ve == 3
            assert ti < vi <= ve < si <= se   # 段间不重叠且有序

    def test_expanding_train_grows_from_start(self):
        days, splits = self.make(mode="expanding")
        assert all(s.train[0] == days[0] for s in splits)
        assert days.index(splits[1].train[1]) > days.index(splits[0].train[1])

    def test_sliding_fixed_train_window(self):
        days, splits = self.make(mode="sliding")
        for s in splits:
            assert days.index(s.train[1]) - days.index(s.train[0]) + 1 == 40
        assert splits[1].train[0] != splits[0].train[0]

    def test_tail_does_not_fit_is_skipped(self):
        days, splits = self.make(n=68)  # 最后一个 fold 的 test 段放不下
        assert splits
        for s in splits:
            assert days.index(s.test[1]) < len(days)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            WalkForwardSplitter(_days(30), mode="anchored")


class TestHoldoutPolicy:
    def test_view_refuses_holdout_dates(self, tmp_path):
        data = SyntheticValidationData(n_symbols=4, n_days=10)
        pol = HoldoutPolicy(data.dates[7],
                            audit_path=str(tmp_path / "holdout_audit.jsonl"))
        v = pol.view(data)
        with pytest.raises(HoldoutViolation):
            v.daily_panel(data.dates[0], data.dates[7])
        with pytest.raises(HoldoutViolation):
            v.trading_days(data.dates[0], data.dates[9])
        with pytest.raises(HoldoutViolation):
            v.universe(data.dates[8])
        with pytest.raises(HoldoutViolation):
            v.index_daily("sh000001", data.dates[0], data.dates[9])
        # holdout 之前的数据可用
        panel = v.daily_panel(data.dates[0], data.dates[6])
        assert len(panel) > 0
        assert pol.unlocked is False

    def test_unlock_once_and_audit_line(self, tmp_path):
        data = SyntheticValidationData(n_symbols=4, n_days=10)
        path = tmp_path / "holdout_audit.jsonl"
        pol = HoldoutPolicy(data.dates[7], audit_path=str(path))
        pol.view(data)
        full = pol.unlock("alice", "final acceptance review")
        assert full is data
        with pytest.raises(RuntimeError):
            pol.unlock("bob", "second look")
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["operator"] == "alice"
        assert rec["reason"] == "final acceptance review"
        assert rec["holdout_start"] == data.dates[7]
        assert "ts" in rec

    def test_unlock_before_view_raises(self, tmp_path):
        pol = HoldoutPolicy("2025-12-31",
                            audit_path=str(tmp_path / "h.jsonl"))
        with pytest.raises(RuntimeError):
            pol.unlock("eve", "too early")
