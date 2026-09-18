"""量化护栏纯逻辑测试（mock 数据）。"""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from engine.anti_chase_guard import AntiChaseGuard
from engine.data_loader import DataLoader
from engine.quant_guard import QuantGuard, GuardResult
from engine.trend_scanner import StockCandidate


class StubDL(DataLoader):
    """绕过 DataLoader 的 akshare 依赖，返回预设数据。"""

    def __init__(self, spot_df, kline_df=None, fin_df=None):
        # 不调 super().__init__，避免触发 akshare 检查
        self._spot = spot_df
        self._kline = kline_df if kline_df is not None else pd.DataFrame(
            {"日期": pd.date_range("2020-01-01", periods=100).astype(str), "收盘": range(100)}
        )
        self._fin = fin_df if fin_df is not None else pd.DataFrame()

    def all_spot(self):
        return self._spot

    def daily_kline(self, symbol, days=60, adjust="qfq", date=None):
        return self._kline

    def individual_fund_flow(self, symbol):
        return pd.DataFrame()

    def financial_indicator(self, symbol):
        return self._fin


def make_candidate(name="测试股", code="000001"):
    return StockCandidate(
        code=code, name=name, board="测试", close=10, pct_change=2,
        turnover_rate=1, circ_mv_yi=100, rps=85, reasons=["x"], fund_inflow_days=2,
    )


def test_exclude_st():
    spot = pd.DataFrame({"代码": ["000001"], "名称": ["ST测试"], "市盈率-动态": [10], "市净率": [1]})
    g = QuantGuard(StubDL(spot), {"exclude_st": True, "exclude_loss": False, "debt_ratio_max": 1.0})
    c = make_candidate(name="ST测试")
    r = g.filter([c])
    # 硬护栏：ST 被明确剔除，不得进入 kept
    assert len(r.kept) == 0
    assert len(r.rejected) == 1
    assert "ST" in r.rejected[0]["reason"]


def test_exclude_high_pe():
    spot = pd.DataFrame({"代码": ["000001"], "名称": ["测试"], "市盈率-动态": [500], "市净率": [2]})
    g = QuantGuard(StubDL(spot), {"exclude_st": True, "pe_max": 200, "exclude_loss": False, "debt_ratio_max": 1.0})
    r = g.filter([make_candidate()])
    assert len(r.kept) == 0
    assert len(r.rejected) == 1
    assert "估值过高" in r.rejected[0]["reason"]


def test_exclude_loss_pe_negative():
    spot = pd.DataFrame({"代码": ["000001"], "名称": ["测试"], "市盈率-动态": [-50], "市净率": [2]})
    g = QuantGuard(StubDL(spot), {"exclude_st": True, "pe_min": 0, "exclude_loss": True, "debt_ratio_max": 1.0})
    r = g.filter([make_candidate()])
    assert len(r.kept) == 0
    assert len(r.rejected) == 1
    assert "亏损" in r.rejected[0]["reason"]


def test_pass_normal_stock():
    spot = pd.DataFrame({"代码": ["000001"], "名称": ["测试"], "市盈率-动态": [20], "市净率": [2]})
    g = QuantGuard(StubDL(spot), {"exclude_st": True, "pe_max": 200, "exclude_loss": False, "debt_ratio_max": 1.0})
    r = g.filter([make_candidate()])
    assert len(r.kept) == 1
    assert len(r.rejected) == 0
    assert r.kept[0].risk_flags == []


def test_exclude_high_debt():
    spot = pd.DataFrame({"代码": ["000001"], "名称": ["测试"], "市盈率-动态": [20], "市净率": [2]})
    fin = pd.DataFrame({"资产负债率(%)": [90]})
    g = QuantGuard(StubDL(spot, fin_df=fin), {"exclude_st": True, "pe_max": 200, "exclude_loss": False, "debt_ratio_max": 0.80})
    r = g.filter([make_candidate()])
    assert len(r.kept) == 0
    assert len(r.rejected) == 1
    assert "资产负债率" in r.rejected[0]["reason"]


def test_guard_result_to_dict():
    r = GuardResult(kept=[], watch=[], rejected=[{"code": "1", "name": "x", "reason": "y"}])
    d = r.to_dict()
    assert d["rejected"][0]["reason"] == "y"


def test_watch_on_missing_valuation():
    spot = pd.DataFrame({"代码": ["000001"], "名称": ["测试"]})
    g = QuantGuard(StubDL(spot), {"exclude_st": True, "pe_max": 200, "exclude_loss": False, "debt_ratio_max": 1.0})
    r = g.filter([make_candidate()])
    assert len(r.kept) == 0
    assert len(r.watch) == 1
    assert any("估值数据缺失" in flag for flag in r.watch[0].risk_flags)


def test_guard_uncovered_financial_candidates_are_watch_only():
    """财务数据源可用时，超出覆盖上限的候选仍降观察。"""
    spot = pd.DataFrame({
        "代码": ["000001", "000002"], "名称": ["A", "B"],
        "市盈率-动态": [20, 20], "市净率": [2, 2],
    })
    fin = pd.DataFrame({
        "选项": ["净利润"], "日期": ["2026-06-30"], "净利润": [1.0e8],
        "资产负债率": [40.0],
    })
    guard = QuantGuard(
        StubDL(spot, fin_df=fin),
        {
            "exclude_new_days": 0,
            "financial_check_limit": 1,
            "workers": 2,
            "financial_risk": {"exclude_loss": True, "debt_ratio_max": 0.9},
        },
    )
    result = guard.filter([make_candidate(code="000001"), make_candidate(code="000002")])
    assert len(result.kept) == 1
    assert len(result.watch) == 1
    assert any("财务排雷未覆盖" in flag for flag in result.watch[0].risk_flags)


def test_guard_financial_source_unavailable_does_not_demote_candidates():
    """财务数据源整体不可用（如回放档案）时不再把「未排雷」当成候选问题。"""
    spot = pd.DataFrame({
        "代码": ["000001", "000002"], "名称": ["A", "B"],
        "市盈率-动态": [20, 20], "市净率": [2, 2],
    })
    guard = QuantGuard(
        StubDL(spot),
        {
            "exclude_new_days": 0,
            "financial_check_limit": 1,
            "workers": 2,
            "financial_risk": {"exclude_loss": True, "debt_ratio_max": 0.9},
        },
    )
    result = guard.filter([make_candidate(code="000001"), make_candidate(code="000002")])
    assert len(result.kept) == 2
    assert len(result.watch) == 0


def test_anti_chase_reuses_technical_kline_without_network_call():
    dl = MagicMock()
    rows = [
        {"close": 10 + i * 0.1, "volume": 1000 - i * 10}
        for i in range(10)
    ]
    item = {
        "code": "000001",
        "close": 10.9,
        "pct_change": 1.0,
        "technical": {"ma": {"ma5": 10.7, "ma20": 10.0}, "kline": rows},
    }
    AntiChaseGuard(dl, {}).guard([item], date="20260710")
    dl.daily_kline.assert_not_called()
    assert item["anti_chase"]["status"] in {"ok", "watch", "blocked"}
