"""Pangu 2.0 泄漏审计：静态模式扫描 + 运行时标签对齐断言。

静态扫描在 engine.data.lookahead 基础模式之上增加：
- 全期 cummax / cumsum 喂入特征；
- 帧反转 .iloc[::-1]；
- label 列疑似被用作特征。
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .data_interface import LookaheadError

# (regex, 说明)。新写法出现时在此登记。
LEAKAGE_PATTERNS: list[tuple[str, str]] = [
    (r"shift\(\s*-\s*\d", "负向 shift：直接取未来行"),
    (r"shift\(\s*periods\s*=\s*-\s*\d", "负向 shift（periods 形式）：取未来行"),
    (r"direction\s*=\s*[\"']forward[\"']", "merge_asof direction='forward'：向前（未来）查找"),
    (r"\bbfill\s*\(|method\s*=\s*[\"']bfill[\"']|\bbackfill\s*\(", "向后填充：用未来值填补当前行"),
    (r"\bcummax\s*\(", "全期 cummax：最大值含未来信息，喂特征前须确认仅用过去窗口"),
    (r"\bcumsum\s*\(", "全期 cumsum：累计量含未来信息，喂特征前须确认仅用过去窗口"),
    (r"\.iloc\[::\s*-\s*1\]", "帧反转：索引顺序操纵易引入未来行"),
    (r"features.*[\"']label[\"']|[\"']label[\"'].*features",
     "label 列疑似进入特征构造"),
]


def scan_source(text: str) -> list[dict]:
    findings: list[dict] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern, reason in LEAKAGE_PATTERNS:
            if re.search(pattern, line):
                findings.append({"line_no": line_no, "line": line.strip(),
                                 "pattern": pattern, "reason": reason})
    return findings


def audit_module(path: str | Path) -> list[dict]:
    """扫描 .py 模块的泄漏模式，返回 findings（无文件返回空列表）。"""
    p = Path(path)
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return []
    return scan_source(text)


def assert_label_alignment(feature_index, label_index) -> None:
    """标签日期必须逐行严格晚于特征日期，否则抛 LookaheadError。"""
    f = [str(x) for x in list(feature_index)]
    l = [str(x) for x in list(label_index)]
    if len(f) != len(l):
        raise LookaheadError(
            f"feature/label index length mismatch: {len(f)} vs {len(l)}")
    for i, (a, b) in enumerate(zip(f, l)):
        if not str(b) > str(a):
            raise LookaheadError(
                f"label date {b} must be strictly after feature date {a} (row {i})")
