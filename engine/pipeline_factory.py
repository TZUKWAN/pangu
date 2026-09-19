"""Pipeline 统一构造工厂（P0-005 / P0-004）。

Pipeline 只允许在这里构造一次：cli / repl / web / scheduler / replay 全部
委托给 ``PipelineFactory.from_config(cfg, mode)``，保证各入口的
sentiment/trend/guard/entry_exit/pick_count/db_path/full_cfg 配置注入
完全一致（有 test_pipeline_entrypoint_consistency.py 锁住该契约）。

mode 语义：
- "cli" / "repl" / "web" / "scheduler"：实时链路，build_data_loader(cfg)
  构造在线数据面，完整注入各子配置与 full_cfg。
- "replay"：PIT 历史回放。传 replay=True 且不带在线 DataLoader——
  replay_backtest.py 会自行把 ReplayDataLoader 挂到 pipeline.dl
  （Pipeline.run(replay=True) 亦会经 _activate_replay 惰性挂载）。
"""
from __future__ import annotations

from typing import Any, Optional

from . import config as _config
from .pipeline import Pipeline

VALID_MODES = ("cli", "repl", "web", "replay", "scheduler")


class PipelineFactory:
    """从配置 dict 构造 Pipeline 的唯一入口。"""

    @staticmethod
    def from_config(cfg: dict, mode: str = "cli") -> Pipeline:
        """按入口模式构造配置一致的 Pipeline。

        注意 build_data_loader 经模块属性访问（``_config.build_data_loader``），
        测试可 monkeypatch ``engine.config.build_data_loader``。
        """
        if mode not in VALID_MODES:
            raise ValueError(f"未知 Pipeline 模式: {mode!r}（可选: {VALID_MODES}）")
        cfg = cfg or {}
        if mode == "replay":
            # 回放链路：不构造在线数据面；ReplayDataLoader 由回放器/运行时挂载
            return Pipeline(full_cfg=cfg, replay=True)
        return Pipeline(
            dl=_config.build_data_loader(cfg),
            sentiment_cfg=cfg.get("sentiment", {}),
            trend_cfg=cfg.get("trend", {}),
            guard_cfg=cfg.get("guard", {}),
            entry_exit_cfg=cfg.get("entry_exit", cfg),  # kimi 的 engine 期望整个 cfg
            pick_count=cfg.get("output", {}).get("pick_count", 5),
            db_path=cfg.get("output", {}).get("db_path", "data/pangu.db"),
            full_cfg=cfg,
        )


def build_pipeline(cfg: dict, mode: str = "cli") -> Pipeline:
    """便捷函数包装（等价 PipelineFactory.from_config）。"""
    return PipelineFactory.from_config(cfg, mode=mode)
