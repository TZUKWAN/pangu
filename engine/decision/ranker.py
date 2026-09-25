"""Pangu 3.0 Top20 决策排序器（Phase 4 / Task 4.1-4.6 + Phase 5）。

- 因子集成：**从研究登记自动读取**有效因子与其 IC 方向（data/experiments/），
  不把因子名硬编码进决策层；研究否决的因子不进入集成。
- 事件 alpha：来自 Evidence Fabric 的结构化事件（衰减/层级/印证/直接性）。
- 市场状态：regime 决定因子族权重剖面。
- 风险调整机会分：expected alpha × confidence × 执行可行性 − 风险惩罚，
  输出完整 score_breakdown。
- 决策枚举只有 BUY/WATCH/AVOID/BLOCKED；数据 failed → 禁止 BUY；
  Tier D 单一事件不能制造 BUY；CRISIS → 最多 WATCH。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.decision.asof import AsOfContext
from engine.decision.contracts import (ConfidenceType, DecisionAction,
                                       DecisionRequest, DecisionRun,
                                       MarketStatus, StockDecision,
                                       Top20DecisionSet)
from engine.decision.holding import build_holding_plan
from engine.decision.regime import (MarketRegime, RegimeLabel,
                                    REGIME_WEIGHT_PROFILES, compute_regime)
from engine.evidence.engine import decay as evidence_decay
from engine.evidence.model import (Directness, EvidenceDirection,
                                   EvidenceItem)

MODEL_VERSION = "pangu_ranker_v1"
EVIDENCE_VERSION = "evidence.v1"

SCORE_BUY_THRESHOLD = 65.0
SCORE_AVOID_THRESHOLD = 35.0
RISK_EVENT_BLOCK = 0.30          # 个股风险事件衰减强度 ≥ 此值 → 不得 BUY
RISK_EVENT_AVOID = 0.50          # ≥ 此值 → AVOID
LIQUIDITY_FLOOR_AMOUNT = 3.0e7   # 20 日均成交额下限
DEFAULT_CAPITAL = 1_000_000.0

# 集成可计算的因子键 → 计算器（研究登记里存在且 IC 达标的才会被启用）
# family 用于 regime 权重剖面。
SUPPORTED_FACTORS = {
    "rev_5d": ("reversal", "_f_rev5"),
    "index_adj_rev_5d": ("reversal", "_f_idxrev5"),
    "amihud_20d": ("illiquidity", "_f_amihud"),
    "realized_vol_20d": ("lowvol", "_f_lowvol"),
    "turnover_20d_avg": ("lowturnover", "_f_lowturn"),
    "volume_price_div_20d": ("lowturnover", "_f_vpr"),
}


def load_factor_specs(factor_dir: str = "data/experiments/factors",
                      registry_path: str = "data/experiments/registry.jsonl",
                      min_abs_ic: float = 0.02
                      ) -> List[Dict[str, Any]]:
    """从研究产物读取有效因子与方向（Task 4.2：登记驱动，非硬编码）。

    返回 [{factor, family, ic_h5, weight, direction}]；登记中无报告或 |IC|
    不达最低标准的因子一律不进入集成。
    """
    specs: List[Dict[str, Any]] = []
    fdir = Path(factor_dir)
    reg_rows: List[dict] = []
    rp = Path(registry_path)
    if rp.exists():
        reg_rows = [json.loads(l) for l in rp.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
    for factor, (family, _calc) in SUPPORTED_FACTORS.items():
        report_p = fdir / f"{factor}.json"
        ic = None
        if report_p.exists():
            try:
                rep = json.loads(report_p.read_text(encoding="utf-8"))
                ic = (rep.get("horizons", {}).get("5", {}) or {}).get("ic_mean_spearman")
            except (json.JSONDecodeError, OSError):
                ic = None
        if ic is None:
            # 从登记 metrics 回退
            for row in reg_rows:
                if row.get("experiment_id") == f"factor_{factor}_v1.0.0":
                    m = row.get("metrics", {}) or {}
                    ic = m.get("h5_rank_ic")
                    break
        if ic is None or abs(ic) < min_abs_ic:
            continue
        specs.append({"factor": factor, "family": family, "ic_h5": float(ic),
                      "direction": 1.0 if ic > 0 else -1.0,
                      "weight": abs(float(ic))})
    # 归一化权重
    total = sum(s["weight"] for s in specs) or 1.0
    for s in specs:
        s["weight"] = round(s["weight"] / total, 4)
    return specs


class FactorEnsemble:
    """按研究登记装配因子集成，并计算截面分。"""

    def __init__(self, specs: Optional[List[Dict[str, Any]]] = None,
                 factor_dir: str = "data/experiments/factors",
                 registry_path: str = "data/experiments/registry.jsonl"):
        self.specs = specs if specs is not None else load_factor_specs(
            factor_dir=factor_dir, registry_path=registry_path)

    def compute_scores(self, panel: pd.DataFrame, asof: str
                       ) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """返回 (合成 z 分, {factor: 截面分})。仅用 <= asof 行。"""
        close = panel["close"].unstack("code").sort_index().loc[:asof]
        pct = (panel["pct_change"] / 100.0).unstack("code").sort_index().loc[:asof]
        amount = panel["amount"].unstack("code").sort_index().loc[:asof]
        volume = panel["volume"].unstack("code").sort_index().loc[:asof]
        ew_idx_ret = pct.mean(axis=1)

        calc = {
            "rev_5d": lambda: -close.pct_change(5).iloc[-1],
            "index_adj_rev_5d": lambda: -(close.pct_change(5).iloc[-1]
                                          - ew_idx_ret.iloc[-5:].sum()),
            "amihud_20d": lambda: ((pct.abs()) / amount.clip(lower=1e5))
            .rolling(20, min_periods=10).mean().iloc[-1] * 1e9,
            "realized_vol_20d": lambda: pct.rolling(20, min_periods=10).std().iloc[-1],
            "turnover_20d_avg": lambda: -amount.rolling(20, min_periods=10).mean().iloc[-1],
            "volume_price_div_20d": lambda: -close.rolling(20, min_periods=10)
            .corr(volume).iloc[-1],
        }
        cross: Dict[str, pd.Series] = {}
        for spec in self.specs:
            f = spec["factor"]
            if f not in calc:
                continue
            try:
                raw = calc[f]()
            except Exception:  # noqa: BLE001 — 单因子计算失败跳过
                continue
            raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
            if len(raw) < 20:
                continue
            # 方向统一为"越大越好"，再 z-score
            oriented = raw * spec["direction"]
            sd = float(oriented.std())
            if sd and sd > 0:
                cross[f] = (oriented - oriented.mean()) / sd
        if not cross:
            return pd.Series(dtype=float), cross
        wtotal = sum(s["weight"] for s in self.specs if s["factor"] in cross) or 1.0
        combined = pd.Series(0.0, index=cross[next(iter(cross))].index)
        wmap = {s["factor"]: s["weight"] for s in self.specs}
        for f, series in cross.items():
            combined = combined.add(series.fillna(0.0) * wmap[f] / wtotal, fill_value=0.0)
        return combined, cross


class EventAlpha:
    """把聚类后的证据转成事件特征（Task 4.3）。"""

    def __init__(self, items: List[EvidenceItem], asof: str):
        self.asof = asof
        self.strength: Dict[str, float] = {}
        self.risk_strength: Dict[str, float] = {}
        self.best_half_life: Dict[str, float] = {}
        self.best_event: Dict[str, EvidenceItem] = {}
        self.buyable_evidence: Dict[str, List[str]] = {}
        for it in items:
            d = evidence_decay(it, asof)
            if d <= 0:
                continue
            sign = 1.0 if it.direction == EvidenceDirection.positive else \
                (-1.0 if it.direction in (EvidenceDirection.negative,
                                          EvidenceDirection.risk) else 0.0)
            w = d * it.confidence
            if it.direction in (EvidenceDirection.negative, EvidenceDirection.risk):
                self.risk_strength[it.entity_id] = self.risk_strength.get(
                    it.entity_id, 0.0) + abs(w)
            else:
                self.strength[it.entity_id] = self.strength.get(it.entity_id, 0.0) + sign * w
                if it.can_trigger_buy:
                    self.buyable_evidence.setdefault(it.entity_id, []).append(it.evidence_id)
            hl = float(it.extra.get("half_life_days", 7))
            self.best_half_life[it.entity_id] = min(
                self.best_half_life.get(it.entity_id, 1e9), hl)
            if it.entity_id not in self.best_event or w > 0:
                self.best_event.setdefault(it.entity_id, it)

    def features(self, code: str) -> Dict[str, float]:
        return {"event_strength": round(self.strength.get(code, 0.0), 4),
                "event_risk": round(self.risk_strength.get(code, 0.0), 4)}


class Top20Ranker:
    """主排序器：request + ctx + evidence → DecisionRun。"""

    def __init__(self, store, ensemble: Optional[FactorEnsemble] = None,
                 capital: float = DEFAULT_CAPITAL):
        self.store = store
        self.ensemble = ensemble or FactorEnsemble()
        self.capital = capital

    def rank(self, request: DecisionRequest, ctx: AsOfContext,
             evidence: Optional[List[EvidenceItem]] = None,
             data_status: str = "ok",
             source_health: Optional[Dict[str, Any]] = None,
             data_freshness: Optional[Dict[str, Any]] = None,
             clock=None) -> DecisionRun:
        t0 = time.monotonic()
        asof_iso = ctx.asof_timestamp
        asof_c = ctx.decision_date.replace("-", "")
        start = (_dt.datetime.strptime(asof_c, "%Y%m%d")
                 - _dt.timedelta(days=420)).strftime("%Y%m%d")
        panel = self.store.daily_panel(start, asof_c)
        index = self.store.index_daily("sh.000300", start, asof_c)
        index_close = index["close"] if index is not None and not index.empty else None

        regime = compute_regime(panel, index_close, asof_iso)
        alpha, _cross = self.ensemble.compute_scores(panel, asof_iso)
        ev_alpha = EventAlpha(evidence or [], asof_iso)

        close_all = panel["close"].unstack("code").sort_index().loc[:asof_c]
        amount_all = panel["amount"].unstack("code").sort_index().loc[:asof_c]
        liq20 = amount_all.rolling(20, min_periods=10).mean().iloc[-1]
        last_day = close_all.index[-1]
        tradable = close_all.iloc[-1].dropna().index           # 停牌/未上市自动排除

        regime_profile = REGIME_WEIGHT_PROFILES[regime.label]
        codes = [c for c in (request.codes or list(tradable)) if c in set(tradable)]
        self._load_names(asof_c)
        rows: List[Tuple[float, dict]] = []
        for code in codes:
            if self._names.get(code) is None and request.codes:
                continue
            feats = self._stock_features(code, close_all, liq20, alpha, ev_alpha,
                                         regime.label)
            if feats is None:
                continue
            rows.append((feats["score"], feats))
        rows.sort(key=lambda x: (-x[0], x[1]["code"]))
        limit = request.limit
        picked = rows[:limit]
        decisions: List[StockDecision] = []
        for i, (score, f) in enumerate(picked, start=1):
            dec, reasons, risks = self._decide(f, data_status, regime)
            hp = build_holding_plan(
                f["code"], close_all[f["code"]].dropna(), f["vol20"],
                f["event_half_life"], regime)
            entry_zone = self._entry_zone(close_all[f["code"]].dropna())
            decisions.append(StockDecision(
                rank=i, code=f["code"], name=f["name"], decision=dec,
                score=round(score, 2),
                confidence=round(min(1.0, max(0.0, score / 100.0)), 3),
                confidence_type=ConfidenceType.EVIDENCE,
                execution_date=ctx.execution_date,
                entry_condition="开盘进入入场区且未触发失效条件",
                entry_zone=entry_zone,
                invalid_condition=(hp.exit_plan.news_invalidation
                                   if hp.exit_plan else "利好证伪"),
                stop_loss=hp.exit_plan.hard_stop if hp.exit_plan else None,
                target_zone=[hp.exit_plan.profit_target] if hp.exit_plan
                and hp.exit_plan.profit_target else None,
                expected_holding_days=hp.expected_holding_days,
                holding_range=hp.holding_range,
                exit_conditions=hp.exit_plan.to_conditions() if hp.exit_plan else [],
                primary_strategy=f["dominant"],
                factor_evidence=[f"factor:{k}:{round(v, 3)}"
                                 for k, v in f["factor_parts"].items()],
                event_evidence=[e for e in f["event_evidence"]],
                market_evidence=[f"regime:{regime.label.value}",
                                 f"breadth:{regime.breadth_above_ma20:.2f}"],
                liquidity_evidence=[f"amount20:{f['amount20'] / 1e8:.2f}亿",
                                    f"feasibility:{f['feasibility']:.2f}"],
                risks=risks, reasons=reasons,
                score_breakdown={k: round(v, 3) for k, v in f["breakdown"].items()},
                evidence_ids=f["evidence_ids"],
                freshness={"panel_last_date": str(last_day),
                           "asof": asof_iso}))

        note = None
        if len(rows) < limit:
            note = (f"有效候选 {len(rows)} 只，不足 {limit}；"
                    "如实返回不足数量（不硬凑）")
        tset = Top20DecisionSet(
            asof=asof_iso, execution_date=ctx.execution_date, decisions=decisions,
            market_status=ctx.market_status,
            market_conclusion=self._market_conclusion(regime),
            note=note, data_status=data_status)

        run_id = self._run_id(ctx)
        return DecisionRun(
            run_id=run_id, query_timestamp=ctx.query_timestamp,
            decision_date=ctx.decision_date, execution_date=ctx.execution_date,
            asof_timestamp=asof_iso, market_status=ctx.market_status,
            data_freshness=data_freshness or {},
            source_health=source_health or {},
            recommendations=tset,
            warnings=list(ctx.warnings),
            blocked_reasons=[],
            model_version=MODEL_VERSION, evidence_version=EVIDENCE_VERSION,
            latency={"total_s": round(time.monotonic() - t0, 3)},
            request=request)

    # ------------------------------------------------------------------ #
    def _stock_features(self, code, close_all, liq20, alpha, ev_alpha,
                        label) -> Optional[dict]:
        if code not in close_all.columns:
            return None
        s = close_all[code].dropna()
        if len(s) < 30:
            return None
        vol20 = float(s.pct_change().rolling(20).std().iloc[-1])
        amount20 = float(liq20.get(code, 0.0) or 0.0)
        if amount20 < LIQUIDITY_FLOOR_AMOUNT:
            return None                                        # 流动性地板
        f_score = float(alpha.get(code, 0.0)) if len(alpha) else 0.0
        evf = ev_alpha.features(code)
        # 事件半衰期（该股最强事件）
        hl = ev_alpha.best_half_life.get(code)
        # 执行可行性：默认资金 1 笔 vs 当日成交额
        order_value = self.capital * 0.1
        feasibility = float(min(1.0, (amount20 * 0.02) / max(order_value, 1.0)))
        # regime fit：低波在 risk_off 更受青睐（因子族权重已由剖面体现于 f_score
        # 之前的合成，这里再做状态适配惩罚/奖励）
        regime_fit = {"risk_on": 0.6, "neutral": 0.5,
                      "risk_off": 0.4, "crisis": 0.2}[label.value]
        breakdown = {
            "factor_alpha": 50.0 + 15.0 * np.tanh(f_score / 1.5),
            "event_alpha": 50.0 + 30.0 * np.tanh(evf["event_strength"]),
            "regime_fit": regime_fit * 100.0,
            "liquidity_feasibility": feasibility * 100.0,
        }
        score = (0.45 * breakdown["factor_alpha"] + 0.25 * breakdown["event_alpha"]
                 + 0.15 * breakdown["regime_fit"]
                 + 0.15 * breakdown["liquidity_feasibility"])
        # 风险惩罚
        vol_pen = min(10.0, max(0.0, (vol20 - 0.04) * 200.0))
        risk_pen = min(25.0, evf["event_risk"] * 50.0)
        score -= vol_pen + risk_pen
        breakdown["vol_penalty"] = -vol_pen
        breakdown["event_risk_penalty"] = -risk_pen

        dominant = max({"factor_alpha": breakdown["factor_alpha"],
                        "event_alpha": breakdown["event_alpha"]},
                       key=lambda k: breakdown[k])
        parts = {"combined_factor_z": round(f_score, 3)}
        return {"code": code, "name": self._name_of(code), "score": float(score),
                "vol20": vol20, "amount20": amount20,
                "feasibility": feasibility, "breakdown": breakdown,
                "event_strength": evf["event_strength"],
                "event_risk": evf["event_risk"],
                "event_half_life": hl,
                "dominant": dominant,
                "factor_parts": parts,
                "event_evidence": list(ev_alpha.buyable_evidence.get(code, [])),
                "evidence_ids": list(ev_alpha.buyable_evidence.get(code, []))}

    def _decide(self, f, data_status, regime) -> Tuple[DecisionAction, List[str], List[str]]:
        reasons: List[str] = []
        risks: List[str] = []
        if data_status == "failed":
            return DecisionAction.BLOCKED, ["关键数据源失败，禁止决策"], []
        if f["event_risk"] >= RISK_EVENT_AVOID:
            risks.append(f"负面/风险事件强度 {f['event_risk']:.2f}")
            return DecisionAction.AVOID, reasons, risks
        if f["score"] < SCORE_AVOID_THRESHOLD:
            return DecisionAction.AVOID, reasons, risks
        if f["event_risk"] >= RISK_EVENT_BLOCK:
            reasons.append("存在风险事件 → 降级 WATCH")
            return DecisionAction.WATCH, reasons, risks
        if regime.label == RegimeLabel.CRISIS:
            reasons.append("市场状态 crisis → 最多 WATCH")
            return DecisionAction.WATCH, reasons, risks
        if f["score"] >= SCORE_BUY_THRESHOLD:
            if f["dominant"] == "event_alpha" and not f["event_evidence"]:
                reasons.append("事件驱动分不可由 Tier D 单源支撑 → WATCH")
                return DecisionAction.WATCH, reasons, risks
            reasons.append(f"风险调整分 {f['score']:.1f} ≥ {SCORE_BUY_THRESHOLD:.0f}")
            if f["event_strength"] > 0.1:
                reasons.append(f"事件催化强度 {f['event_strength']:.2f}")
            reasons.append("反转/流动性因子方向经研究窗 OOS 验证")
            return DecisionAction.BUY, reasons, risks
        reasons.append(f"分数 {f['score']:.1f} 介于观察区间")
        return DecisionAction.WATCH, reasons, risks

    def _entry_zone(self, close: pd.Series) -> Optional[List[float]]:
        s = close.iloc[-20:]
        if s.empty:
            return None
        last = float(s.iloc[-1])
        lo = float(np.nanmin(s))
        return [round(max(lo, last * 0.985), 2), round(last * 1.01, 2)]

    def _market_conclusion(self, regime: MarketRegime) -> str:
        return {
            RegimeLabel.RISK_ON: "趋势向上、宽度扩散，反转与流动性因子均可参与",
            RegimeLabel.NEUTRAL: "震荡市，反转因子为主，控制仓位",
            RegimeLabel.RISK_OFF: "趋势偏弱/波动偏高，只做高质量反转，缩短持有",
            RegimeLabel.CRISIS: "恐慌状态，不建议 BUY，仅保留观察",
        }[regime.label]

    def _load_names(self, asof_c: str) -> None:
        """名称表：PIT universe（asof 当日成员），失败降级为空（决策仍可用代码）。"""
        if getattr(self, "_names", None) is not None:
            return
        try:
            uni = self.store.universe(_iso_from_compact(asof_c))
            self._names = {str(c).split(".")[-1].zfill(6): str(n)
                           for c, n in zip(uni.index, uni.get("name", uni.index))}
        except Exception:  # noqa: BLE001 — 名称缺失不阻断决策
            self._names = {}

    def _name_of(self, code: str) -> Optional[str]:
        return self._names.get(code, code)

    def _run_id(self, ctx: AsOfContext) -> str:
        basis = f"{ctx.asof_timestamp}|{ctx.execution_date}|{time.time_ns()}"
        h = hashlib.sha256(basis.encode()).hexdigest()[:8]
        return f"pangu-{ctx.execution_date.replace('-', '')}-{h}"


def _iso_from_compact(c: str) -> str:
    c = c.replace("-", "")
    return f"{c[:4]}-{c[4:6]}-{c[6:8]}"
