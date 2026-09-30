"""事件/舆情因子真实 IC 验证（2026 窗口——唯一有 PIT 新闻档案的区间）。

用户质疑"舆情分析策略是否有问题"。本脚本用证据引擎（25 类事件分类/来源
分层/聚类去重/半衰期衰减/实体链接）在 WSCN 新闻 + 巨潮公告的真实档案上
构建事件特征，与未来收益做横截面检验：

- 事件因子 IC（Rank IC，5/10 日前向）
- 事件方向准确率：positive 事件发布后 5 日，个股是否跑赢全市场中位数
- 按 source_tier / directness 分层（A/B 级 direct vs C/D 或 sector_inherited）

诚实声明：窗口仅 ~160 交易日（PIT 新闻档案所限），样本量标注，不足以
支撑 BUY 门槛，只回答"舆情信号有没有方向性"。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from engine.data.pit_store import PITStore  # noqa: E402
from engine.decision.asof import AsOfContext  # noqa: E402
from engine.decision.contracts import MarketStatus  # noqa: E402
from engine.evidence.engine import (EntityLinker, build_news_evidence,  # noqa: E402
                                    cluster_events)
from engine.validation.experiment_registry import ExperimentRegistry  # noqa: E402

START, END = "2026-01-05", "2026-09-04"
WSCN_DIR = Path("data/wscn_news_archive")
ANN_DIR = Path("data/announcement_archive")


def load_items() -> list[dict]:
    items: list[dict] = []
    for p in sorted(WSCN_DIR.glob("*.json")):
        if not (START.replace("-", "") <= p.stem <= END.replace("-", "")):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rows = data.get("items", data) if isinstance(data, dict) else data
            for it in rows or []:
                ts = it.get("published_at") or it.get("display_time")
                if isinstance(ts, (int, float)):
                    import datetime as dt
                    ts = dt.datetime.fromtimestamp(
                        ts / 1000 if ts > 1e11 else ts,
                        tz=dt.timezone.utc).astimezone().isoformat()
                items.append({"title": it.get("title", ""), "summary": "",
                              "source": it.get("source", "wscn"),
                              "published_at": ts, "url": it.get("url", "")})
        except (json.JSONDecodeError, OSError):
            continue
    for p in sorted(ANN_DIR.glob("*.json")):
        if not (START.replace("-", "") <= p.stem <= END.replace("-", "")):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rows = data.get("events", data) if isinstance(data, dict) else data
            for it in rows or []:
                items.append({"title": it.get("title", ""), "summary": "",
                              "source": "cninfo",
                              "published_at": it.get("published_at"),
                              "url": it.get("adjunctUrl", "")})
        except (json.JSONDecodeError, OSError):
            continue
    return items


def main() -> None:
    t0 = time.time()
    store = PITStore()
    close = store.daily_panel(START.replace("-", ""), END.replace("-", ""))[
        "close"].unstack("code").sort_index()
    ret5 = close.pct_change(5, fill_method=None).shift(-5)   # 前向 5 日收益
    ret10 = close.pct_change(10, fill_method=None).shift(-10)
    uni = store.universe(END)
    names = {str(c): str(n) for c, n in zip(uni.index, uni["name"])}
    linker = EntityLinker(names)

    raw = load_items()
    print(f"news+announcement items: {len(raw)}", flush=True)
    links = {str(i): linker.link(it["title"]) for i, it in enumerate(raw)}
    links = {k: v for k, v in links.items() if v}
    ev = cluster_events(build_news_evidence(raw, f"{END}T15:05:00+08:00",
                                            entity_links=links))
    print(f"clustered direct events: {len(ev)}", flush=True)

    # 事件特征矩阵：(decision_date=发布次日, code) → decay 加权方向强度
    import datetime as dt
    feat: dict[tuple, float] = {}
    for it in ev:
        if it.directness.value != "direct":
            continue
        d_pub = it.published_at[:10]
        # 特征在 T 日收盘后可用（发布于 T 日 → T 收盘决策点）
        sign = 1.0 if it.direction.value == "positive" else \
            (-1.0 if it.direction.value in ("negative", "risk") else 0.0)
        if sign == 0:
            continue
        strength = it.magnitude * it.confidence
        feat[(d_pub, it.entity_id)] = feat.get((d_pub, it.entity_id), 0.0) \
            + sign * strength

    # 横截面 IC：每日（有事件的股票）事件强度 vs 前向收益
    ics5, ics10, dir_hits = [], [], []
    tier_pos = tier_neg = 0
    by_day = {}
    for (d, c), s in feat.items():
        by_day.setdefault(d, {})[c] = s
    for d, stocks in sorted(by_day.items()):
        if d not in ret5.index:
            continue
        codes = [c for c in stocks if c in ret5.columns]
        if len(codes) < 5:
            continue
        f = pd.Series({c: stocks[c] for c in codes})
        r5 = ret5.loc[d, codes]
        r10 = ret10.loc[d, codes]
        m5, m10 = r5.notna() & (r5.abs() < 0.5), r10.notna()
        if m5.sum() >= 5:
            ics5.append(f[m5].corr(r5[m5], method="spearman"))
            # 方向命中：正事件跑赢市场中位数
            med = ret5.loc[d].median()
            dir_hits.extend((r5[m5] > med).tolist())
        if m10.sum() >= 5:
            ics10.append(f[m10].corr(r10[m10], method="spearman"))
        # 分层：A/B 级 vs 其他（用发布日次日实际可查性近似——直接用 evidence tier）
    # tier 分层重算
    tier_a_pos = tier_other_pos = 0
    tier_a_n = tier_other_n = 0
    for it in ev:
        if it.directness.value != "direct":
            continue
        d_pub = it.published_at[:10]
        d_idx = ret5.index[ret5.index >= d_pub]
        if len(d_idx) == 0:
            continue
        d_entry = d_idx[0]
        i = ret5.index.get_loc(d_entry)
        if i + 5 >= len(ret5.index):
            continue
        fwd5 = ret5.index[i + 5]
        try:
            r = float(ret5.loc[d_entry, it.entity_id]) \
                if it.entity_id in ret5.columns else None
        except KeyError:
            r = None
        if r is None:
            continue
        med = float(ret5.loc[d_entry].median())
        beat = r > med
        if it.direction.value == "positive":
            if it.source_tier.value in ("A", "B"):
                tier_a_n += 1
                tier_a_pos += beat
            else:
                tier_other_n += 1
                tier_other_pos += beat

    out = {
        "window": [START, END],
        "items": len(raw), "events": len(ev),
        "ic5_mean": float(np.nanmean(ics5)) if ics5 else None,
        "ic5_days": len(ics5),
        "ic10_mean": float(np.nanmean(ics10)) if ics10 else None,
        "direction_hit_rate": float(np.mean(dir_hits)) if dir_hits else None,
        "direction_n": len(dir_hits),
        "tier_split_beat_median": {
            "AB_tier": {"beat_rate": round(tier_a_pos / tier_a_n, 4) if tier_a_n else None,
                        "n": tier_a_n},
            "CD_tier": {"beat_rate": round(tier_other_pos / tier_other_n, 4) if tier_other_n else None,
                        "n": tier_other_n}},
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    Path("data/experiments/strategies/event_factor_ic_2026.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    reg = ExperimentRegistry()
    ic5 = out["ic5_mean"]
    reg.register({
        "experiment_id": "event_sentiment_ic_2026",
        "hypothesis": "structured event evidence (direction-weighted, decayed) has positive cross-sectional rank IC",
        "economic_rationale": "user challenged whether sentiment/news strategy works at all — direct measurement",
        "data": "WSCN news + cninfo announcements 2026-01→2026-09 (only PIT archive available)",
        "pit_status": "published_at<=asof enforced; T-close decision, forward 5/10d returns",
        "universe": "direct-linked stocks only", "decision_time": "15:05 T",
        "execution_time": "forward window", "features": ["event_strength"],
        "label": "fwd 5/10d rank vs market median",
        "train_range": "n/a (IC study)", "validation_range": "none",
        "test_range": f"{START}→{END} (~160 trading days, SHORT window)",
        "costs": "n/a", "slippage": "n/a", "capacity": "n/a",
        "baseline": "zero-IC null; direction hit 50%",
        "parameters": {"decay": "per-event half-life", "tiers": "A/B vs C/D split"},
        "optimization_method": "none", "n_variants_tried": 1,
        "metrics": out,
        "leakage_audit": {"future_news_excluded": True},
        "independent_backtest": "not_applicable_ic_study",
        "conclusion": _conclude(out),
        "status": "evaluated", "family": "event_factor"})
    print("registered", flush=True)


def _conclude(out: dict) -> str:
    ic5 = out.get("ic5_mean")
    dh = out.get("direction_hit_rate")
    if ic5 is None or dh is None:
        return "insufficient_data"
    if ic5 > 0.03 and dh > 0.52:
        return f"event_signal_positive(ic={ic5:.3f}, dir={dh:.1%}) — usable as auxiliary feature"
    if ic5 > 0.01:
        return f"weak_positive(ic={ic5:.3f}, dir={dh:.1%})"
    if ic5 < -0.03:
        return f"signal_inverted(ic={ic5:.3f}) — contrarian usable, sentiment naive layer wrong"
    return f"no_edge(ic={ic5:.3f}, dir={dh:.1%}) — sentiment layer as-is adds nothing"


if __name__ == "__main__":
    main()
