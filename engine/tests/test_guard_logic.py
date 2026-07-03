"""量化护栏纯逻辑测试（mock 数据）。"""

import pandas as pd
import pytest

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
    # 软护栏：保留候选但标记风险，前端根据风险标签过滤
    assert len(r.kept) == 1
    assert any("ST" in flag for flag in r.kept[0].risk_flags)
    assert "ST" in r.rejected[0]["reason"]


def test_exclude_high_pe():
    spot = pd.DataFrame({"代码": ["000001"], "名称": ["测试"], "市盈率-动态": [500], "市净率": [2]})
    g = QuantGuard(StubDL(spot), {"exclude_st": True, "pe_max": 200, "exclude_loss": False, "debt_ratio_max": 1.0})
    r = g.filter([make_candidate()])
    assert len(r.kept) == 1
    assert any("估值过高" in flag for flag in r.kept[0].risk_flags)
    assert "估值过高" in r.rejected[0]["reason"]


def test_exclude_loss_pe_negative():
    spot = pd.DataFrame({"代码": ["000001"], "名称": ["测试"], "市盈率-动态": [-50], "市净率": [2]})
    g = QuantGuard(StubDL(spot), {"exclude_st": True, "pe_min": 0, "exclude_loss": True, "debt_ratio_max": 1.0})
    r = g.filter([make_candidate()])
    assert len(r.kept) == 1
    assert any("亏损" in flag for flag in r.kept[0].risk_flags)
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
    assert len(r.kept) == 1
    assert any("资产负债率" in flag for flag in r.kept[0].risk_flags)
    assert "资产负债率" in r.rejected[0]["reason"]


def test_guard_result_to_dict():
    r = GuardResult(kept=[], rejected=[{"code": "1", "name": "x", "reason": "y"}])
    d = r.to_dict()
    assert d["rejected"][0]["reason"] == "y"
