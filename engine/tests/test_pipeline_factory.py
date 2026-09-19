"""PipelineFactory 单元测试：全部模式构造行为 + 非法模式拒绝。

monkeypatch engine.config.build_data_loader —— 工厂经模块属性调用它，
因此在线路径不会触碰真实数据源。
"""

from __future__ import annotations

import pytest

from engine.pipeline_factory import PipelineFactory


class StubDataLoader:
    """最小鸭子类型：Pipeline.__init__ 只保存引用。"""


@pytest.fixture()
def stub_loader(monkeypatch):
    created = []

    def _factory(cfg):
        dl = StubDataLoader()
        created.append(dl)
        return dl

    monkeypatch.setattr("engine.config.build_data_loader", _factory)
    return created


def cfg():
    return {
        "sentiment": {"weights": {"limit_up_count": 0.9}},
        "trend": {"stock": {"rps_min": 80}},
        "guard": {"exclude_st": False, "pe_max": 123},
        "entry_exit": {"atr_period": 21, "breakout_lookback": 10},
        "output": {"pick_count": 7, "db_path": "data/test_pangu.db"},
    }


def test_factory_cli_builds_pipeline_with_full_config(stub_loader):
    c = cfg()
    pipe = PipelineFactory.from_config(c, mode="cli")
    assert isinstance(stub_loader[0], StubDataLoader)
    assert pipe.dl is stub_loader[0]
    assert pipe.full_cfg is c
    assert pipe.pick_count == 7
    assert pipe.db_path == "data/test_pangu.db"
    assert pipe.meter.cfg == c["sentiment"]
    assert pipe.guard.cfg == c["guard"]
    assert pipe.guard.exclude_st is False
    assert pipe.entry_exit.atr_period == 21
    assert pipe.entry_exit.breakout_lookback == 10


def test_factory_repl_and_web_and_scheduler_share_cli_semantics(stub_loader):
    c = cfg()
    for mode in ("repl", "web", "scheduler"):
        pipe = PipelineFactory.from_config(c, mode=mode)
        assert pipe.dl is not None
        assert pipe.full_cfg is c
        assert pipe.pick_count == 7
        assert pipe.guard.cfg == c["guard"]
        assert pipe.entry_exit.atr_period == 21


def test_factory_replay_mode_has_no_live_loader(stub_loader):
    c = cfg()
    pipe = PipelineFactory.from_config(c, mode="replay")
    assert pipe.replay_requested is True
    assert pipe.full_cfg is c
    assert pipe.pick_count == 7
    # 不构造在线数据面（ReplayDataLoader 由 replay_backtest / 运行时挂载）
    assert not isinstance(pipe.dl, StubDataLoader)
    assert stub_loader == []  # build_data_loader 从未被调用


def test_factory_rejects_unknown_mode(stub_loader):
    with pytest.raises(ValueError, match="未知 Pipeline 模式"):
        PipelineFactory.from_config(cfg(), mode="cron")
