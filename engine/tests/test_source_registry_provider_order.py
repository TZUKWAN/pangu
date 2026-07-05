from __future__ import annotations

from engine.sources.registry import build_default_registry


def test_daily_kline_chain_order_with_baostock_before_stale() -> None:
    names = [p.name for p in build_default_registry().providers("daily_kline")]

    assert "baostock_daily_kline" in names
    assert "tencent_kline_qfq" in names
    assert "mootdx_daily_kline" in names
    assert "baidu_gushitong_daily" in names
    # 东财已移除：确认 chain 里没有 eastmoney_push2his
    assert "eastmoney_push2his_daily" not in names
    # baostock 在所有真实源之后、最后的 stale_cache(live) 之前
    # 找最后一个 stale_cache 的位置
    stale_positions = [i for i, n in enumerate(names) if n == "stale_cache"]
    assert len(stale_positions) >= 2  # diagnostic + live
    live_stale_idx = stale_positions[-1]
    assert names.index("baostock_daily_kline") < live_stale_idx


def test_all_spot_chain_order() -> None:
    names = [p.name for p in build_default_registry().providers("all_spot")]

    # 前 6 个：snapshot × 2 + stale + ths + tencent + sina
    assert names[:6] == [
        "snapshot_exact",
        "snapshot_exact",
        "stale_cache",
        "ths_all_spot",
        "tencent_qt_all_spot",
        "sina_hq_all_spot",
    ]
    assert "baidu_gushitong_all_spot" in names
    assert "adata_all_spot" in names
    # 百度 / adata 在 tencent 之后
    assert names.index("tencent_qt_all_spot") < names.index("adata_all_spot")


def test_fund_flow_chain_ends_with_unavailable() -> None:
    names = [p.name for p in build_default_registry().providers("fund_flow")]

    assert names[0] == "snapshot_exact"
    assert "ths_fund_flow" in names
    assert "adata_fund_flow" in names
    assert names[-1] == "unavailable"
    assert names.index("ths_fund_flow") < names.index("unavailable")
