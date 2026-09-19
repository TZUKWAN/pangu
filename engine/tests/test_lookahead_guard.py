"""前视偏差防护测试：assert_no_future_rows + 源码扫描。"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data.lookahead import (
    assert_no_future_rows,
    scan_module_for_lookahead,
    scan_source,
)


def _panel(dates, codes=("000001", "000002")) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    return pd.DataFrame({"close": range(len(idx))}, index=idx)


def test_assert_no_future_rows_passes_on_clean_frames():
    df = _panel(["2026-01-05", "2026-01-06", "2026-01-07"])
    assert_no_future_rows(df, "2026-01-07")
    assert_no_future_rows(df, "20260107")            # 紧凑格式 asof
    assert_no_future_rows(df, "2027-01-01")          # 宽松 asof 更不抛


def test_assert_no_future_rows_raises_on_future_index_level():
    df = _panel(["2026-01-05", "2026-01-06", "2026-01-08"])
    with pytest.raises(ValueError, match="PIT guard"):
        assert_no_future_rows(df, "2026-01-07")


def test_assert_no_future_rows_raises_on_future_date_column():
    df = pd.DataFrame({"date": ["2026-01-05", "2026-01-09"], "close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        assert_no_future_rows(df, "20260108")


def test_assert_no_future_rows_handles_datetime_index():
    df = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-01-05", "2026-01-07"], name="date"),
    )
    assert_no_future_rows(df, "2026-01-07")
    with pytest.raises(ValueError):
        assert_no_future_rows(df, "2026-01-06")


def test_scan_finds_negative_shift_in_sample_source(tmp_path):
    src = (
        "import pandas as pd\n"
        "def leak(df):\n"
        "    return df['close'].shift(-1) > df['close']\n"
        "def ok(df):\n"
        "    return df['close'].shift(1) > df['close']\n"
    )
    findings = scan_source(src)
    hits = [f for f in findings if "shift(-1)" in f["line"].replace(" ", "")]
    assert len(hits) == 1
    assert hits[0]["line_no"] == 3
    assert "shift" in hits[0]["reason"]


def test_scan_finds_forward_merge_and_backfill(tmp_path):
    src = (
        "m = pd.merge_asof(left, right, on='date', direction='forward')\n"
        "df = df.fillna(method='bfill')\n"
        "x = df.bfill()\n"
    )
    reasons = {f["reason"] for f in scan_source(src)}
    assert any("merge_asof" in r for r in reasons)
    assert any("填充" in r for r in reasons)


def test_scan_module_for_lookahead_reads_file_and_clean_file_passes(tmp_path):
    bad = tmp_path / "bad_module.py"
    bad.write_text("y = df.shift(-2)\n", encoding="utf-8")
    findings = scan_module_for_lookahead(bad)
    assert len(findings) == 1
    assert findings[0]["line_no"] == 1

    good = tmp_path / "good_module.py"
    good.write_text("y = df.shift(1)\n", encoding="utf-8")
    assert scan_module_for_lookahead(good) == []
    assert scan_module_for_lookahead(tmp_path / "missing.py") == []
