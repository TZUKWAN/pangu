"""插件自测器回归测试：所有宿主适配清单必须能按配置拉起 MCP server。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.mcp.selftest import _manifests, selftest  # noqa: E402


def test_all_manifests_present():
    names = [n for n, cfg in _manifests() if "_error" not in cfg]
    for must in ("ZCode(.zcode-plugin)", "KimiCode",
                 "WorkBuddy(workbuddy/)"):
        assert any(must in n for n in names), f"缺少必需宿主清单: {must}"


def test_selftest_all_manifests_pass():
    ok, fails = selftest(fast=True, per_timeout=120)
    assert not fails, f"插件自测失败: {fails}"
    assert ok >= 5
