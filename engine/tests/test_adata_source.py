"""adata_source 单元测试（使用 monkeypatch 模拟 adata 返回值）。"""

from __future__ import annotations

import pandas as pd
import pytest

from engine import adata_source


def _make_spot_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"stock_code": "000001", "short_name": "平安", "price": 10.0,
         "change": 0.1, "change_pct": 1.0, "volume": 1000, "amount": 10000.0},
    ])


def _make_kline_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"trade_time": "2025-06-25", "open": 10.0, "close": 10.2,
         "high": 10.3, "low": 9.9, "volume": 1000, "amount": 10000.0,
         "turnover_ratio": 1.2},
    ])


def _make_fund_flow_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"trade_date": "2025-06-25", "main_net_inflow": 1234.0},
    ])


@pytest.fixture
def mock_adata(monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟 adata 模块及关键函数。"""
    class FakeAdata:
        class stock:
            class info:
                @staticmethod
                def all_code() -> pd.DataFrame:
                    return pd.DataFrame({"stock_code": ["000001", "600000"]})
            class market:
                @staticmethod
                def list_market_current(code_list: list[str]) -> pd.DataFrame:
                    return _make_spot_df()
                @staticmethod
                def get_market(*args: object, **kwargs: object) -> pd.DataFrame:
                    return _make_kline_df()
                @staticmethod
                def get_capital_flow(*args: object, **kwargs: object) -> pd.DataFrame:
                    return _make_fund_flow_df()
                @staticmethod
                def all_capital_flow_east(*args: object, **kwargs: object) -> pd.DataFrame:
                    return pd.DataFrame({"板块": ["芯片"], "净流入": [1e6]})
    monkeypatch.setattr(adata_source, "_is_available", lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "adata", FakeAdata())


def test_all_spot(mock_adata: None) -> None:
    df = adata_source.all_spot(batch_size=10)
    assert not df.empty
    assert "代码" in df.columns
    assert df.iloc[0]["代码"] == "000001"


def test_daily_kline(mock_adata: None) -> None:
    df = adata_source.daily_kline("000001", days=10, date="20250630")
    assert not df.empty
    assert set(df.columns) >= {"日期", "股票代码", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"}


def test_individual_fund_flow(mock_adata: None) -> None:
    df = adata_source.individual_fund_flow("000001")
    assert not df.empty
    assert "主力净流入-净额" in df.columns
    assert df.iloc[0]["股票代码"] == "000001"


def test_concept_fund_flow(mock_adata: None) -> None:
    df = adata_source.concept_fund_flow("今日")
    assert not df.empty
