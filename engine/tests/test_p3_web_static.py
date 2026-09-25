"""Web 静态资源回归测试：内联 <script> 必须通过 node --check。

来源：Round 1 发现 loadDecisionStatus 的 join('\n') 被写成真实换行导致整页
JS 失效（switchView 未定义）。此测试永久防回归（node 不可用时 skip）。
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = Path(__file__).parent.parent / "web" / "static" / "index.html"


def test_inline_scripts_pass_syntax_check():
    if shutil.which("node") is None:
        pytest.skip("node 不可用，跳过静态语法检查")
    s = HTML.read_text(encoding="utf-8")
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", s, re.S)
    assert scripts, "index.html 应包含内联脚本"
    for i, sc in enumerate(scripts):
        tmp = Path(__file__).parent.parent.parent / "tmp" / f"_syntax_check_{i}.js"
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_text(sc, encoding="utf-8")
        r = subprocess.run(["node", "--check", str(tmp)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"script block {i} 语法错误: {r.stderr[:300]}"
        # JS 内禁止真实换行打断字符串（历史 Bug 形态）
        assert "\n'" not in sc or "join('\n')" not in sc


def test_nav_has_default_three_views():
    """默认导航只保留 今日Top20 / 个股 / 系统；高级分组保留原研究能力。"""
    s = HTML.read_text(encoding="utf-8")
    for anchor in ("nav-today", "nav-stock", "nav-system",
                   "nav-execution", "nav-strategies", "nav-research"):
        assert anchor in s, f"导航缺少 {anchor}"
    assert "今日Top20" in s
