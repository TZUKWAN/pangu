"""确保主链路不再调用东方财富涨跌停接口。"""

import pandas as pd

from engine import data_loader as dl_mod
from engine.data_loader import DataLoader


def test_limit_pools_use_ths_without_eastmoney(monkeypatch, tmp_path):
    calls = []

    def forbidden_call_pool(self, *args, **kwargs):
        raise AssertionError("Eastmoney akshare pool interface should not be called")

    def fake_limit_up(date):
        calls.append(("up", date))
        return pd.DataFrame({"代码": ["000001"], "名称": ["涨停股"]})

    def fake_limit_pool(date, pool_type):
        calls.append((pool_type, date))
        return pd.DataFrame({"代码": ["000002"], "名称": ["跌停股"]})

    def fake_strong(date):
        calls.append(("strong", date))
        return pd.DataFrame({"代码": ["000003"], "名称": ["强势股"]})

    monkeypatch.setattr(DataLoader, "_call_pool", forbidden_call_pool)
    monkeypatch.setattr(dl_mod, "_ths_limit_up_pool", fake_limit_up)
    monkeypatch.setattr(dl_mod, "_ths_limit_pool", fake_limit_pool)
    monkeypatch.setattr(dl_mod, "_ths_strong_pool", fake_strong)

    loader = DataLoader(cache_dir=tmp_path, retry_times=1, backoff_seconds=0)

    assert loader.limit_up_pool("20260703").iloc[0]["代码"] == "000001"
    assert loader.limit_down_pool("20260703").iloc[0]["代码"] == "000002"
    assert loader.strong_pool("20260703").iloc[0]["代码"] == "000003"
    assert calls == [("up", "20260703"), ("down", "20260703"), ("strong", "20260703")]
