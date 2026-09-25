"""Phase 4/5 Ranker 测试：合成面板 + 注入研究登记。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.decision.asof import TradingCalendar, build_asof_context
from engine.decision.clock import FrozenClock
from engine.decision.contracts import (DecisionAction, DecisionRequest,
                                       MarketStatus)
from engine.decision.ranker import (FactorEnsemble, Top20Ranker,
                                    load_factor_specs)
from engine.evidence.model import (Directness, EvidenceCategory,
                                   EvidenceDirection, EvidenceItem, SourceTier)


# --------------------------------------------------------------------------- #
# 合成 PITStore（与真实 PITStore duck-typing 兼容）
# --------------------------------------------------------------------------- #
class FakeStore:
    def __init__(self, panel: pd.DataFrame, index_close: pd.Series, names: dict):
        self._panel = panel
        self._idx = index_close
        self._names = names

    def daily_panel(self, start, end, symbols=None):
        df = self._panel
        df = df[(df.index.get_level_values("date") >= start)
                & (df.index.get_level_values("date") <= end)]
        return df

    def index_daily(self, code, start, end):
        s = self._idx.loc[(self._idx.index >= start) & (self._idx.index <= end)]
        return pd.DataFrame({"close": s})

    def universe(self, date):
        idx = pd.Index(list(self._names.keys()), name="code")
        return pd.DataFrame({"name": list(self._names.values())}, index=idx)


def make_panel(n_days=80, n_codes=30, seed=7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=LAST_DAY, periods=n_days).strftime("%Y-%m-%d")
    codes = [f"{600000 + i}" for i in range(n_codes)]
    rets = rng.normal(0, 0.02, size=(n_days, n_codes))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                         index=days, columns=codes)
    amount = pd.DataFrame(rng.uniform(5e7, 5e8, size=(n_days, n_codes)),
                          index=days, columns=codes)
    preclose = close.shift(1)
    pct = (close / preclose - 1) * 100
    volume = amount / close
    frames = []
    for col in ["close", "amount", "pct_change", "volume", "preclose"]:
        df = {"close": close, "amount": amount, "pct_change": pct,
              "volume": volume, "preclose": preclose}[col]
        frames.append(df.stack().rename(col))
    panel = pd.concat(frames, axis=1)
    panel.index.names = ["date", "code"]
    panel["is_st"] = 0
    index_close = close.mean(axis=1)
    names = {c: f"股票{c[-2]}" for c in codes}
    return panel, index_close, names


def inject_alpha(panel: pd.DataFrame, last_day: str, top_codes, strength=3.0):
    """在最后 5 天给指定代码注入确定性负收益（反转因子的正向输入）。"""
    close = panel["close"].unstack("code")
    pct = panel["pct_change"].unstack("code")
    loc = close.index.get_loc(last_day)
    per_day = -0.012 * strength               # 每天 -3.6%（strength=3）
    for c in top_codes:
        j = close.columns.get_loc(c)
        for k in range(loc - 4, loc + 1):
            # 逐日累计缩放（含末日）：ret5 = (1+per_day)^5 - 1
            close.iloc[k, j] = close.iloc[k - 1, j] * (1 + per_day)
            pct.iloc[k, j] = per_day * 100
    frames = [close.stack().rename("close"), pct.stack().rename("pct_change")]
    others = [col for col in panel.columns if col not in ("close", "pct_change")]
    for col in others:
        frames.append(panel[col].unstack("code").stack().rename(col))
    out = pd.concat(frames, axis=1)
    out.index.names = ["date", "code"]
    return out


def write_fake_research(tmp_path: Path):
    """登记驱动测试：写最小因子报告 + registry，验证集成从登记读取方向。"""
    fdir = tmp_path / "factors"
    fdir.mkdir(parents=True)
    (fdir / "rev_5d.json").write_text(json.dumps(
        {"horizons": {"5": {"ic_mean_spearman": 0.037}}}), encoding="utf-8")
    (fdir / "realized_vol_20d.json").write_text(json.dumps(
        {"horizons": {"5": {"ic_mean_spearman": -0.065}}}), encoding="utf-8")
    (fdir / "weak_factor.json").write_text(json.dumps(
        {"horizons": {"5": {"ic_mean_spearman": 0.001}}}), encoding="utf-8")
    return FactorEnsemble(factor_dir=str(fdir), registry_path=str(
        tmp_path / "missing_registry.jsonl"))


def make_event(code, strength=0.8, tier=SourceTier.B, direction=EvidenceDirection.positive,
               half_life=14, published="2026-08-13T10:00:00+08:00"):
    return EvidenceItem(
        evidence_id=f"ev-{code}-{half_life}-{direction.value}", entity_type="stock",
        entity_id=code, category=EvidenceCategory.news, event_type="order_win",
        direction=direction, magnitude=strength, source="财联社",
        source_tier=tier, published_at=published, effective_at=None,
        fetched_at="2026-08-13T15:05:00+08:00", expiry_at=None,
        confidence=0.8, directness=Directness.direct,
        extra={"half_life_days": half_life})


ASOF_ISO = "2026-08-13T15:05:00+08:00"
LAST_DAY = "2026-08-13"


def _ctx():
    cal = TradingCalendar(known_days=["20260813", "20260814"], allow_online=False)
    return build_asof_context(clock=FrozenClock(ASOF_ISO), cal=cal)


@pytest.fixture(scope="module")
def env():
    panel, idx, names = make_panel()
    panel = inject_alpha(panel, LAST_DAY, ["600003", "600007"], strength=0.5)
    store = FakeStore(panel, idx, names)
    ens = write_fake_research(Path(_tmpdir()))
    ranker = Top20Ranker(store, ensemble=ens)
    return ranker, store, names


def _tmpdir():
    import tempfile
    d = tempfile.mkdtemp()
    return d


class TestRanker:
    def test_specs_read_from_registry_not_hardcoded(self, tmp_path):
        ens = write_fake_research(tmp_path)
        names = [s["factor"] for s in ens.specs]
        assert "rev_5d" in names and "realized_vol_20d" in names
        weak = load_factor_specs(factor_dir=str(tmp_path / "factors"),
                                 registry_path=str(tmp_path / "none.jsonl"),
                                 min_abs_ic=0.02)
        assert all(s["factor"] != "weak_factor" for s in weak)  # 未支持/未登记的不进
        rev = next(s for s in ens.specs if s["factor"] == "rev_5d")
        assert rev["direction"] == 1.0
        vol = next(s for s in ens.specs if s["factor"] == "realized_vol_20d")
        assert vol["direction"] == -1.0

    def test_full_rank_run(self, env):
        ranker, store, names = env
        ctx = _ctx()
        run = ranker.rank(DecisionRequest(limit=10), ctx, evidence=[],
                          data_status="ok")
        assert run.model_version == "pangu_ranker_v1"
        assert run.execution_date == "2026-08-14"
        assert len(run.recommendations.decisions) == 10
        ranks = [d.rank for d in run.recommendations.decisions]
        assert ranks == list(range(1, 11))
        for d in run.recommendations.decisions:
            assert d.expected_holding_days in (1, 3, 5, 10, 20)
            assert d.holding_range and d.holding_range[0] <= d.expected_holding_days
            assert "hard_stop@" in " ".join(d.exit_conditions)
            assert d.score_breakdown.get("factor_alpha") is not None
        # 稳定排序：同输入两次运行 Top 集一致
        run2 = ranker.rank(DecisionRequest(limit=10), ctx, evidence=[])
        assert [d.code for d in run2.recommendations.decisions] == \
               [d.code for d in run.recommendations.decisions]

    def test_injected_alpha_ranks_top(self, env):
        ranker, _, _ = env
        ctx = _ctx()
        run = ranker.rank(DecisionRequest(limit=10), ctx)
        codes = [d.code for d in run.recommendations.decisions]
        # 平缓回落（高质量反转画像）应进入前 8（30 只的前 27% 分位）
        assert "600003" in codes[:8] and "600007" in codes[:8]

    def test_event_beats_no_event(self, env):
        ranker, _, _ = env
        ctx = _ctx()
        base = ranker.rank(DecisionRequest(limit=10), ctx)
        weakest = base.recommendations.decisions[-1].code
        boosted = ranker.rank(DecisionRequest(limit=10), ctx,
                              evidence=[make_event(weakest)])
        pos_b = next(d for d in boosted.recommendations.decisions if d.code == weakest)
        assert pos_b.score > base.recommendations.decisions[-1].score

    def test_risk_event_demotes_buy(self, env):
        ranker, _, _ = env
        ctx = _ctx()
        run = ranker.rank(DecisionRequest(limit=10), ctx)
        buys = [d for d in run.recommendations.decisions if d.decision == DecisionAction.BUY]
        if not buys:
            pytest.skip("no BUY in this synthetic window")
        target = buys[0].code
        run2 = ranker.rank(DecisionRequest(limit=10), ctx, evidence=[
            make_event(target, direction=EvidenceDirection.negative,
                       half_life=10, strength=0.9)])
        d2 = next(d for d in run2.recommendations.decisions if d.code == target)
        assert d2.decision in (DecisionAction.WATCH, DecisionAction.AVOID)

    def test_data_failed_forbids_buy(self, env):
        ranker, _, _ = env
        ctx = _ctx()
        run = ranker.rank(DecisionRequest(limit=10), ctx, data_status="failed")
        assert all(d.decision != DecisionAction.BUY
                   for d in run.recommendations.decisions)
        assert run.recommendations.data_status == "failed"

    def test_limit_smaller_than_universe(self, env):
        ranker, _, _ = env
        ctx = _ctx()
        run = ranker.rank(DecisionRequest(limit=3), ctx)
        assert len(run.recommendations.decisions) == 3

    def test_single_code_request(self, env):
        ranker, _, _ = env
        ctx = _ctx()
        run = ranker.rank(DecisionRequest(limit=5, codes=["600001"]), ctx)
        assert len(run.recommendations.decisions) == 1
        assert run.recommendations.decisions[0].rank == 1
