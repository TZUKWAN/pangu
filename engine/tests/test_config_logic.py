"""配置加载逻辑测试：环境变量占位符与 LLM key 读取。"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from engine.config import _resolve_env, load_config, save_strategy_settings


def test_resolve_env_placeholder(monkeypatch):
    """${ENV:NAME} 被替换为对应环境变量值。"""
    monkeypatch.setenv("PANGU_TEST_KEY", "secret_from_env")
    assert _resolve_env("${ENV:PANGU_TEST_KEY}") == "secret_from_env"


def test_resolve_env_missing_returns_empty():
    """缺失的环境变量返回空字符串。"""
    assert _resolve_env("${ENV:PANGU_TEST_MISSING_XYZ}") == ""


def test_resolve_env_nested():
    """递归解析 dict/list 中的占位符。"""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PANGU_NESTED", "v1")
    try:
        assert _resolve_env({"a": "${ENV:PANGU_NESTED}", "b": ["${ENV:PANGU_NESTED}"]}) == {
            "a": "v1",
            "b": ["v1"],
        }
    finally:
        monkeypatch.undo()


def test_load_config_with_env_override(tmp_path, monkeypatch):
    """配置文件中 api_key 为空时，环境变量仍可被 build_llm_client 读取。"""
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text("llm:\n  api_key: ''\n  base_url: https://api.example.com\n  model: test\n", encoding="utf-8")
    cfg = load_config(str(cfg_file))
    assert cfg["llm"]["api_key"] == ""


def test_load_config_no_hardcoded_secret(tmp_path):
    """确认配置文件不包含明文 sk- key（即使通过占位符）。"""
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text("llm:\n  api_key: '${ENV:PANGU_LLM_API_KEY}'\n", encoding="utf-8")
    content = cfg_file.read_text(encoding="utf-8")
    assert "sk-" not in content


def test_save_strategy_settings_sanitizes_and_load_config_merges(tmp_path, monkeypatch):
    """策略覆盖只保存白名单数值，并在 load_config 时合并生效。"""
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text(
        "xuanwu_pool:\n  min_total_score: 75\ntrend:\n  stock:\n    fallback_top_n: 600\n",
        encoding="utf-8",
    )
    override = tmp_path / "strategy_settings.json"
    monkeypatch.setattr("engine.config.STRATEGY_OVERRIDE", override)

    saved = save_strategy_settings(
        {
            "xuanwu_pool": {
                "min_total_score": 82,
                "rule_min_total_score": 63,
                "debate_rounds": 1,
                "debate_max_tokens": 1600,
                "judge_max_tokens": 1200,
                "turnover_max": 500,
                "unsafe": "x",
            },
            "trend": {"stock": {"fallback_top_n": 900}},
            "agent_prompts": {"bull": "看多 {stock_name}", "risk_judge": "风险裁决 {plan_summary}"},
            "llm": {"api_key": "should-not-save"},
        }
    )
    assert saved["xuanwu_pool"]["min_total_score"] == 82.0
    assert saved["xuanwu_pool"]["rule_min_total_score"] == 63.0
    assert saved["xuanwu_pool"]["debate_rounds"] == 1
    assert saved["xuanwu_pool"]["debate_max_tokens"] == 1600
    assert saved["xuanwu_pool"]["judge_max_tokens"] == 1200
    assert "turnover_max" not in saved["xuanwu_pool"]
    assert "llm" not in saved
    assert saved["agent_prompts"]["bull"] == "看多 {stock_name}"
    assert saved["agent_prompts"]["risk_judge"] == "风险裁决 {plan_summary}"

    cfg = load_config(str(cfg_file))
    assert cfg["xuanwu_pool"]["min_total_score"] == 82.0
    assert cfg["xuanwu_pool"]["rule_min_total_score"] == 63.0
    assert cfg["xuanwu_pool"]["debate_rounds"] == 1
    assert cfg["trend"]["stock"]["fallback_top_n"] == 900
    assert cfg["agent_prompts"]["bull"] == "看多 {stock_name}"
