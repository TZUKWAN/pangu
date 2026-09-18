"""Small dataframe helpers with no market-provider import side effects."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        return default if pd.isna(number) else number
    except (TypeError, ValueError):
        return default


def find_col(frame: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if frame is None or frame.empty:
        return None
    columns = list(frame.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
        for column in columns:
            if str(column).startswith(candidate):
                return column
        matches = [column for column in columns if candidate in str(column)]
        if matches:
            return min(matches, key=lambda item: len(str(item)))
    return None
