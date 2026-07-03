"""Pipeline 配置层逻辑测试。"""

from engine.pipeline import Pipeline


class FakeDataLoader:
    pass


def test_debate_candidate_limit_defaults_to_small_llm_budget():
    pipe = Pipeline(
        dl=FakeDataLoader(),
        full_cfg={
            "output": {"pick_count": 5},
            "structured_data": {"deep_candidate_limit": 100},
        },
    )

    assert pipe.debate_candidate_limit == 5


def test_debate_candidate_limit_respects_explicit_xuanwu_config_without_watch_inflation():
    pipe = Pipeline(
        dl=FakeDataLoader(),
        full_cfg={
            "output": {"pick_count": 5},
            "structured_data": {"deep_candidate_limit": 100},
            "xuanwu_pool": {"debate_top_n": 12},
        },
    )

    assert pipe.debate_candidate_limit == 12


def test_debate_candidate_limit_respects_larger_explicit_xuanwu_config():
    pipe = Pipeline(
        dl=FakeDataLoader(),
        full_cfg={
            "output": {"pick_count": 5},
            "structured_data": {"deep_candidate_limit": 100},
            "xuanwu_pool": {"debate_top_n": 80},
        },
    )

    assert pipe.debate_candidate_limit == 80


def test_xuanwu_source_status_uses_pipeline_source_state():
    pipe = Pipeline(dl=FakeDataLoader(), full_cfg={})

    status = pipe._xuanwu_source_status({
        "news": {"status": "ok", "warnings": []},
        "structured_data": {"status": "degraded", "warnings": ["结构化源降级"]},
    })

    assert status["news"] == "ok"
    assert status["structured_data"] == "degraded"
    assert status["market_data"] == "degraded"
    assert "结构化源降级" in status["warnings"]
