"""入口一致性契约测试（P0-005）：

cli / repl / PipelineFactory("cli"|"repl") 四条构造路径必须产出配置完全
一致的 Pipeline（pick_count / guard_cfg / entry_exit_cfg / sentiment_cfg /
db_path / full_cfg），防止某个入口悄悄漏传 full_cfg 之类的漂移回归。

在线数据面由 monkeypatch 的 engine.config.build_data_loader 替身提供。
"""

from __future__ import annotations

import pytest

from engine import cli, config, repl
from engine.pipeline_factory import PipelineFactory


class StubDataLoader:
    pass


@pytest.fixture()
def stub_loader(monkeypatch):
    def _factory(cfg):
        return StubDataLoader()

    monkeypatch.setattr(config, "build_data_loader", _factory)
    return _factory


def entry_cfg():
    return {
        "sentiment": {"weights": {"advance_decline": 0.5}},
        "trend": {"stock": {"rps_min": 77}},
        "guard": {"exclude_new_days": 9, "pe_max": 88},
        "entry_exit": {"atr_period": 21, "breakout_lookback": 10},
        "output": {"pick_count": 7, "db_path": "data/entry_test.db"},
        "xuanwu_pool": {"debate_top_n": 6},
    }


def _fingerprint(pipe):
    """从构造好的 Pipeline 提取配置指纹（子引擎各自保存 cfg）。"""
    return {
        "pick_count": pipe.pick_count,
        "db_path": pipe.db_path,
        "full_cfg": pipe.full_cfg,
        "sentiment_cfg": pipe.meter.cfg,
        "guard_cfg": pipe.guard.cfg,
        "entry_exit_cfg": {
            "atr_period": pipe.entry_exit.atr_period,
            "breakout_lookback": pipe.entry_exit.breakout_lookback,
        },
        "debate_candidate_limit": pipe.debate_candidate_limit,
    }


def test_cli_repl_and_factory_paths_are_identical(stub_loader):
    c = entry_cfg()
    paths = {
        "cli": cli._build_pipeline(c),
        "repl": repl._build_scan_pipeline(c),
        "factory_cli": PipelineFactory.from_config(c, mode="cli"),
        "factory_repl": PipelineFactory.from_config(c, mode="repl"),
    }
    fps = {name: _fingerprint(p) for name, p in paths.items()}
    reference = fps["cli"]
    for name, fp in fps.items():
        assert fp == reference, f"{name} 与 cli 入口配置不一致"

    # 关键字段逐一显式断言（失败时可读性更好）
    for pipe in paths.values():
        assert pipe.pick_count == 7
        assert pipe.db_path == "data/entry_test.db"
        assert pipe.full_cfg is c                     # full_cfg 必须原样透传
        assert pipe.guard.cfg == c["guard"]
        assert pipe.meter.cfg == c["sentiment"]
        assert pipe.entry_exit.atr_period == 21
        assert pipe.debate_candidate_limit == 6


def test_factory_cli_matches_legacy_inline_construction(stub_loader):
    """工厂输出必须与 cli 历史内联构造逐字段等价（含 entry_exit 兜底语义）。"""
    c = entry_cfg()
    pipe = cli._build_pipeline(c)
    assert isinstance(pipe.dl, StubDataLoader)
    assert pipe.pick_count == c["output"]["pick_count"]
    assert pipe.db_path == c["output"]["db_path"]
    assert pipe.full_cfg is c


def test_factory_entry_exit_fallback_to_full_cfg(stub_loader):
    """cfg 无 entry_exit 段时，与 cli 一致回退为整个 cfg（kimi engine 契约）。"""
    c = {"output": {"pick_count": 3}, "atr_period": 33}
    pipe = PipelineFactory.from_config(c, mode="cli")
    assert pipe.entry_exit.atr_period == 33           # 回退后从整 cfg 顶层取值
