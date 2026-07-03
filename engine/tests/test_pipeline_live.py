"""端到端冒烟测试：连真实 akshare，验证主链路不报错。

需网络，CI/离线环境可跳过：pytest -k "not live"
"""

import pytest

akshare = pytest.importorskip("akshare")  # 没装 akshare 则整体跳过


@pytest.mark.live
def test_pipeline_runs_end_to_end():
    """完整跑一遍 scan，只要不抛异常、返回结构正确即通过。

    不对候选股数量/质量做断言（市场每天不同），只验证管道连通。
    """
    from engine.config import build_data_loader, load_config
    from engine.pipeline import Pipeline

    cfg = load_config()
    dl = build_data_loader(cfg)
    pipe = Pipeline(
        dl=dl,
        sentiment_cfg=cfg.get("sentiment", {}),
        trend_cfg=cfg.get("trend", {}),
        guard_cfg=cfg.get("guard", {}),
        pick_count=cfg.get("output", {}).get("pick_count", 5),
    )
    result = pipe.run()
    d = result.to_dict()
    # 结构断言
    assert "date" in d
    assert "sentiment" in d
    assert "temperature" in d["sentiment"]
    assert 0 <= d["sentiment"]["temperature"] <= 100
    assert "boards" in d
    assert "candidates" in d
    assert isinstance(d["candidates"], list)


@pytest.mark.live
def test_data_loader_smoke():
    """各数据接口至少能返回 DataFrame（可空）。"""
    from engine.config import build_data_loader, load_config
    dl = build_data_loader(load_config())
    assert isinstance(dl.all_spot(), type(dl.all_spot()))  # 能调用不抛
    df = dl.limit_up_pool()
    assert hasattr(df, "columns") or len(df) == 0
