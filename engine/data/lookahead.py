"""前视偏差（lookahead）防护工具。

PIT 铁律：
- 任何进入决策的数据行，其日期必须 <= 决策日（asof）。
- 代码层面禁止负向 shift / forward 合并 / 后向填充等"偷看未来"的写法。

本模块提供三层防线：
1. ``assert_no_future_rows(df, asof_date)``：运行时断言 DataFrame 不含未来行。
2. ``scan_module_for_lookahead(path)``：静态扫描源码中的前视模式（诊断用）。
3. 日期归一化助手（``to_compact`` / ``to_iso``），被 PIT 底座复用。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import pandas as pd

# (regex, 说明)。新写法出现时在此登记。
LOOKAHEAD_PATTERNS: List[Tuple[str, str]] = [
    (r"shift\(\s*-\s*\d", "负向 shift：直接取未来行"),
    (r"direction\s*=\s*[\"']forward[\"']", "merge_asof direction='forward'：向前（未来）查找"),
    (r"\bbfill\s*\(|method\s*=\s*[\"']bfill[\"']|\bbackfill\s*\(", "向后填充：用未来值填补当前行"),
    (r"shift\(\s*periods\s*=\s*-\s*\d", "负向 shift（periods 形式）：取未来行"),
]


def to_compact(d: Any) -> str:
    """'2026-01-05' / '20260105' / Timestamp → '20260105'（内部统一紧凑格式）。"""
    s = str(d).strip()
    return s.replace("-", "").replace("/", "").split("T")[0].split(" ")[0][:8]


def to_iso(d: Any) -> str:
    """'20260105' / '2026-1-5' → '2026-01-05'（对外统一 ISO 格式）。"""
    c = to_compact(d)
    if len(c) == 8:
        return f"{c[:4]}-{c[4:6]}-{c[6:8]}"
    return str(d)


def _collect_date_values(df: pd.DataFrame) -> List[Iterable[Any]]:
    """收集 DataFrame 中所有可能的日期序列：'date' 列 + 索引中的 date 级。"""
    series: List[Iterable[Any]] = []
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if "date" in getattr(df, "columns", []):
        series.append(df["date"])
    idx = getattr(df, "index", None)
    if isinstance(idx, pd.MultiIndex):
        for name in idx.names:
            if name is not None and "date" in str(name).lower():
                series.append(idx.get_level_values(name))
    elif idx is not None and not isinstance(idx, pd.RangeIndex):
        if (idx.name is not None and "date" in str(idx.name).lower()) or (
            pd.api.types.is_datetime64_any_dtype(idx)
        ):
            series.append(idx)
    return series


def assert_no_future_rows(df: pd.DataFrame, asof_date: Any) -> None:
    """断言 df 中没有任何日期行晚于 asof_date，否则抛 ValueError。

    同时检查 'date' 列与索引中的 date 级（含 MultiIndex 与 DatetimeIndex）。
    这是 PIT 底座 ``PITStore._guard_asof`` 的通用实现。
    """
    limit = to_compact(asof_date)
    violations: List[str] = []
    for dates in _collect_date_values(df):
        for v in dates:
            try:
                c = to_compact(v)
            except Exception:  # noqa: BLE001
                continue
            if len(c) == 8 and c > limit:
                violations.append(str(v))
                if len(violations) >= 5:
                    break
        if len(violations) >= 5:
            break
    if violations:
        raise ValueError(
            f"PIT guard: 发现晚于 asof={to_iso(limit)} 的未来行 {len(violations)}+ 条: {violations}"
        )


def scan_source(text: str) -> List[dict]:
    """对源码文本做前视模式正则扫描，返回 findings 列表。"""
    findings: List[dict] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern, reason in LOOKAHEAD_PATTERNS:
            if re.search(pattern, line):
                findings.append(
                    {"line_no": line_no, "line": line.strip(), "pattern": pattern, "reason": reason}
                )
    return findings


def scan_module_for_lookahead(path: str | Path) -> List[dict]:
    """扫描一个 .py 模块文件中的前视模式，返回 findings 列表（诊断用）。"""
    p = Path(path)
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return []
    return scan_source(text)
