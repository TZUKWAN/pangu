"""Pangu 2.0 daily trading loop: pipeline result → portfolio plan → risk →
OMS staged/submitted orders → reconcile → persisted LoopResult.

Honest-failure design (fail closed at every step):
- Scenario F  data not pit_safe                        → status="blocked" (reason="data_quality")
- Scenario H  no executable strategy (current reality:  research-only pools) → status="no_executable_strategy", nothing placed
- Scenario I  risk engine veto                         → order NOT staged, reason recorded
- Scenario G/H LIVE without full compliance/infra      → status="live_blocked" + failed_checks
- kill switch / manual intervention / trading blocked  → status="blocked"
- SHADOW mode plans orders but never submits (CREATED only)

Default execution mode is PAPER and the loop is OFF unless
cfg["execution"]["enabled"] is true — turning it on never changes the
recommendation/report behaviour of the scheduler.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from engine.contracts import (
    EXECUTABLE_STRATEGY_STATUS,
    ExecutionMode,
    Order,
    OrderStatus,
    OrderType,
    PortfolioTarget,
    RiskDecision,
)
from engine.data.lookahead import to_compact
from engine.data.quality import DataQualityReport
from engine.execution.broker import BrokerAdapter
from engine.execution.oms import NeedsReconciliation, OrderManagementSystem
from engine.execution.paper import PaperBroker
from engine.execution.quotes import quote_provider_factory
from engine.execution.risk_controls import (
    CashCheckGuard,
    DailyLossCircuitBreaker,
    DuplicateOrderGuard,
    PositionCheckGuard,
    PriceDeviationGuard,
    RateLimitGuard,
)
from engine.portfolio_engine.constructor import (
    PortfolioConfig,
    PortfolioConstructor,
    PortfolioPlanContext,
)
from engine.portfolio_engine.risk import RiskContext, RiskEngine, RiskEngineConfig
from engine.portfolio_engine.state import PortfolioState
from engine.strategies.gates import PromotionContext  # noqa: F401  (re-export convenience)
from engine.strategies.registry import StrategyRegistry, seed_registry

# headroom over the reference price used when sizing qty so that slippage +
# fees cannot push a sized order into an avoidable insufficient_cash reject
_PRICE_HEADROOM = 1.005

_STATUS_LIFECYCLE = {
    "idea": 0, "research": 1, "validated": 2, "paper": 3,
    "shadow_live": 4, "limited_live": 5, "live": 6,
}


# ---------------------------------------------------------------------------
# Config / result contracts
# ---------------------------------------------------------------------------

@dataclass
class ExecutionLoopConfig:
    enabled: bool = False
    execution_mode: str = ExecutionMode.PAPER.value   # ExecutionMode string
    db_path: str = "data/execution.db"                # OMS + strategy registry
    paper_db_path: str = "data/paper_broker.db"
    portfolio_state_path: str = "data/portfolio_state.db"
    pit_db_path: str = "data/market_breadth/raw.sqlite3"
    compliance_db_path: str = "data/compliance.db"
    compliance_audit_dir: str = "data/audit"
    output_dir: str = "data/execution"
    equity: float = 1_000_000.0
    max_weight_per_stock: float = 0.10
    max_positions: int = 20
    rate_limits: Dict[str, float] = field(
        default_factory=lambda: {"max_per_day": 200, "min_interval_seconds": 2})
    daily_loss_soft: float = -0.02
    daily_loss_hard: float = -0.05
    require_pit_safe: bool = True
    # LIVE only: name of a configured live broker adapter; absent → fail closed
    live_adapter: Optional[str] = None


@dataclass
class LoopResult:
    status: str                    # ok / blocked / no_executable_strategy / live_blocked / disabled
    decision_date: str = ""
    reason: str = ""
    execution_mode: str = ""
    plans: List[Dict[str, Any]] = field(default_factory=list)
    risk_decisions: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    reconcile: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# PaperBroker → BrokerAdapter seam (constructor injection, kept here so that
# engine/execution/ stays free of loop-specific glue)
# ---------------------------------------------------------------------------

class PaperBrokerAsAdapter(BrokerAdapter):
    """Adapt a PaperBroker instance to the BrokerAdapter ABC."""

    name = "paper"

    def __init__(self, broker: PaperBroker) -> None:
        super().__init__()
        self._broker = broker

    @property
    def broker(self) -> PaperBroker:
        return self._broker

    def connect(self) -> None:
        self._broker.connect()
        self._connected = True

    def set_date(self, date: str) -> None:
        self._broker.set_date(date)

    def get_balance(self):
        return self._broker.get_balance()

    def get_positions(self):
        return self._broker.get_positions()

    def get_orders(self, trade_date=None):
        return self._broker.get_orders(trade_date)

    def get_trades(self, trade_date=None):
        return self._broker.get_trades(trade_date)

    def submit_order(self, order: Order) -> str:
        return self._broker.submit_order(order)

    def cancel_order(self, broker_order_id: str) -> bool:
        return self._broker.cancel_order(broker_order_id)

    def query_order(self, broker_order_id: str):
        return self._broker.query_order(broker_order_id)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

class DailyExecutionLoop:
    """One decision date's execution: pipeline → targets → plan → risk → OMS."""

    def __init__(
        self,
        config: ExecutionLoopConfig,
        pipeline_result: Any = None,
        settings: Optional[Dict[str, Any]] = None,
        quote_provider: Optional[Callable[[str, str], Optional[Dict]]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.config = config
        self.settings = dict(settings or {})
        self.now_fn = now_fn or datetime.now
        self._pipeline_result = self._normalize_pipeline_result(pipeline_result)
        # quote provider: injected stub wins; otherwise PIT-backed (lazy store)
        self._quote_provider = quote_provider or quote_provider_factory(config.pit_db_path)
        self._pit_store: Any = None
        self.state = PortfolioState(config.portfolio_state_path)
        self.registry = StrategyRegistry(config.db_path)
        seed_registry(self.registry)  # idempotent
        self._oms_config: Dict[str, Any] = {
            "equity": float(config.equity),
            "daily_pnl": 0.0,
            "commission_rate": 0.0003,
            "min_commission": 5.0,
            "stamp_duty_rate": 0.0005,
        }
        self.adapter: Optional[BrokerAdapter] = None
        self.oms = self._build_oms()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        cfg_dict: Optional[Dict[str, Any]],
        pipeline_result: Any = None,
        quote_provider: Optional[Callable[[str, str], Optional[Dict]]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> "DailyExecutionLoop":
        cfg = cfg_dict or {}
        raw = dict(cfg.get("execution") or {})
        known = {f.name for f in ExecutionLoopConfig.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in raw.items() if k in known}
        return cls(ExecutionLoopConfig(**kwargs), pipeline_result=pipeline_result,
                   settings=cfg, quote_provider=quote_provider, now_fn=now_fn)

    # ------------------------------------------------------------------ #
    # component builders
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_pipeline_result(pr: Any) -> Dict[str, Any]:
        if pr is None:
            return {}
        if isinstance(pr, dict):
            return pr
        to_dict = getattr(pr, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        return {}

    @property
    def pit_store(self):
        """Lazily constructed PIT reader; None when the archive is unavailable
        (fail-closed: the data-quality gate treats that as not pit_safe)."""
        if self._pit_store is None:
            try:
                from engine.data.pit_store import PITStore  # local import: heavy module

                self._pit_store = PITStore(self.config.pit_db_path)
            except Exception:  # noqa: BLE001 — missing/corrupt archive
                self._pit_store = False
        return self._pit_store or None

    def _build_oms(self) -> OrderManagementSystem:
        broker = PaperBroker(
            db_path=self.config.paper_db_path,
            quote_provider=self._quote_provider,
            initial_cash=float(self.config.equity),
            now_fn=self.now_fn,
        )
        self.adapter = PaperBrokerAsAdapter(broker)
        self.adapter.connect()  # paper broker is local; connect() is idempotent
        rate = self.config.rate_limits or {}
        checks = [
            DuplicateOrderGuard(window_minutes=5),
            PriceDeviationGuard(max_deviation_pct=0.03),
            CashCheckGuard(),
            PositionCheckGuard(),
            RateLimitGuard(max_per_day=int(rate.get("max_per_day", 200)),
                           min_interval_seconds=float(rate.get("min_interval_seconds", 2))),
            DailyLossCircuitBreaker(soft_stop=float(self.config.daily_loss_soft),
                                    hard_stop=float(self.config.daily_loss_hard)),
        ]
        return OrderManagementSystem(
            adapter=self.adapter,
            checks=checks,
            db_path=self.config.db_path,
            config=self._oms_config,
            now_fn=self.now_fn,
        )

    # ------------------------------------------------------------------ #
    # data quality (Scenario F)
    # ------------------------------------------------------------------ #
    def _check_data_quality(self, date: str) -> "tuple[bool, Dict[str, Any]]":
        try:
            report = DataQualityReport.build(self.pit_store, date)
        except Exception as exc:  # noqa: BLE001 — missing archive etc: fail closed
            return False, {"pit_safe": False, "error": f"{type(exc).__name__}: {exc}"}
        return bool(report.get("pit_safe")), report

    # ------------------------------------------------------------------ #
    # strategy authority (Scenario H)
    # ------------------------------------------------------------------ #
    def _collect_targets(self, date: str) -> tuple[List[PortfolioTarget], List[str]]:
        recs = self._pipeline_result.get("final_recommendations") or []
        executable_ids = set(self.registry.executable_strategy_ids())
        targets: List[PortfolioTarget] = []
        notes: List[str] = []
        seen: set = set()
        for item in recs:
            if not isinstance(item, dict):
                continue
            pool = str(item.get("strategy") or item.get("strategy_pool")
                       or item.get("pool") or "").strip()
            symbol = str(item.get("code") or "").strip()
            if not pool or not symbol:
                notes.append(f"skip_final_rec: missing pool/symbol (code={item.get('code')!r})")
                continue
            key = (pool, symbol)
            if key in seen:
                continue
            seen.add(key)
            if pool not in executable_ids:
                status = "unregistered"
                try:
                    status = self.registry.get(pool).approval_state.value
                except KeyError:
                    pass
                notes.append(f"skip {symbol}: strategy '{pool}' not executable (status={status})")
                continue
            # fail-closed re-read of the manifest at loop time
            manifest = self.registry.get(pool)
            if manifest.approval_state not in EXECUTABLE_STRATEGY_STATUS:
                notes.append(f"skip {symbol}: strategy '{pool}' status "
                             f"{manifest.approval_state.value} not executable")
                continue

            raw_score = item.get("score")
            if raw_score is None:
                raw_score = item.get("recommend_score")
            if raw_score is None:
                raw_score = (item.get("recommend") or {}).get("recommend_score")
            try:
                raw_score = float(raw_score) if raw_score is not None else None
            except (TypeError, ValueError):
                raw_score = None

            calibrated = bool(item.get("calibrated")
                              or (item.get("recommend") or {}).get("calibrated"))
            confidence = None
            if calibrated:
                conf = item.get("confidence_score")
                if conf is not None:
                    try:
                        confidence = float(conf)
                    except (TypeError, ValueError):
                        confidence = None

            weight = self._size_weight(item) or (1.0 / max(1, self.config.max_positions))
            weight = max(0.0, min(weight, float(self.config.max_weight_per_stock)))

            targets.append(PortfolioTarget(
                strategy_id=pool,
                symbol=symbol,
                target_weight=weight,
                confidence=confidence,
                calibrated=calibrated,
                raw_score=raw_score,
                decision_id=self._decision_id(date, pool),
                meta={"pool": pool, "score": raw_score},
            ))
        return targets, notes

    @staticmethod
    def _decision_id(date: str, strategy_id: str) -> str:
        return f"dl-{date}-{strategy_id}"

    @staticmethod
    def _size_weight(item: Dict[str, Any]) -> Optional[float]:
        """Explicit size suggestion as a fraction of equity, if honestly present."""
        for key in ("size", "weight"):
            v = item.get(key)
            if isinstance(v, dict):
                v = v.get("weight", v.get("pct"))
            if isinstance(v, str) and v.endswith("%"):
                try:
                    v = float(v[:-1]) / 100.0
                except ValueError:
                    v = None
            if isinstance(v, (int, float)) and 0 < float(v) <= 1.0:
                return float(v)
        return None

    # ------------------------------------------------------------------ #
    # mode resolution (Scenario G/H) + kill switch
    # ------------------------------------------------------------------ #
    def _resolve_mode(self, best_status: str) -> tuple[ExecutionMode, List[str]]:
        cfg = self.config
        try:
            mode = ExecutionMode(str(cfg.execution_mode).upper())
        except ValueError:
            return ExecutionMode.PAPER, [f"invalid_execution_mode:{cfg.execution_mode}"]

        if mode != ExecutionMode.LIVE:
            return mode, []

        from engine.compliance.program_trading import ComplianceManager, LiveGate

        manager = ComplianceManager(db_path=cfg.compliance_db_path,
                                    audit_dir=cfg.compliance_audit_dir)
        compliance_ok, compliance_reason = manager.can_enable_live()
        failed: List[str] = []
        if not compliance_ok:
            failed.append(f"compliance:{compliance_reason}")
        status_rank = _STATUS_LIFECYCLE.get(str(best_status), -1)
        gate = LiveGate.evaluate(
            strategy_status=_status_enum(best_status),
            execution_mode=mode,
            compliance_ok=compliance_ok,
            paper_passed=status_rank >= _STATUS_LIFECYCLE["shadow_live"],
            shadow_passed=status_rank >= _STATUS_LIFECYCLE["limited_live"],
            broker_ok=bool(cfg.live_adapter),
            reconcile_ok=False,   # honest default: no completed live reconcile evidence
            kill_switch_ready=True,  # OMS kill switch is armed/available
        )
        failed.extend(gate.failed_checks)
        if not cfg.live_adapter:
            failed.append("no_live_broker_adapter_configured")
        return mode, failed

    # ------------------------------------------------------------------ #
    def _kill_switch_state(self) -> str:
        for key in ("kill_switch_engaged", "manual_intervention_required", "trading_blocked"):
            try:
                with sqlite3.connect(self.config.db_path, timeout=10) as conn:
                    row = conn.execute(
                        "SELECT value FROM system_state WHERE key = ?", (key,)).fetchone()
            except sqlite3.Error:
                continue
            if row and row[0]:
                return "kill_switch" if key == "kill_switch_engaged" else key
        return ""

    # ------------------------------------------------------------------ #
    # PIT helpers
    # ------------------------------------------------------------------ #
    def _safe_quote(self, symbol: str, date: str) -> Optional[Dict]:
        try:
            return self._quote_provider(symbol, date)
        except Exception:  # noqa: BLE001 — no archive / provider error → no quote
            return None

    def _pit_adv(self, symbols: List[str]) -> Dict[str, float]:
        """symbol → decision-day amount (turnover) from the PIT archive."""
        out = {s: 0.0 for s in symbols}
        if not symbols:
            return out
        try:
            panel = self.pit_store.daily_panel(self._date, self._date,
                                               sorted(set(symbols)))
        except Exception:  # noqa: BLE001 — fail closed to 0 (liquidity floor vetoes)
            return out
        if panel is None or panel.empty:
            return out
        for code in set(panel.index.get_level_values("code")):
            row = panel.xs(code, level="code").iloc[0]
            amt = row.get("amount")
            try:
                amt = float(amt)
            except (TypeError, ValueError):
                amt = 0.0
            out[str(code)] = amt if amt == amt else 0.0
        return out

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #
    def run(self, date: Optional[str] = None) -> LoopResult:
        date = to_compact(date or self.now_fn())
        self._date = date
        cfg = self.config

        if self.adapter is not None:
            self.adapter.set_date(date)

        # 0. execution disabled → honest no-op
        if not cfg.enabled:
            return LoopResult(status="disabled", decision_date=date,
                              reason="execution.enabled=false")

        try:
            mode = ExecutionMode(str(cfg.execution_mode).upper())
        except ValueError:
            return LoopResult(status="blocked", decision_date=date,
                              reason=f"invalid_execution_mode:{cfg.execution_mode}")
        if mode == ExecutionMode.DISABLED:
            return LoopResult(status="disabled", decision_date=date,
                              reason="execution_mode=DISABLED")

        # 1. data quality gate (Scenario F)
        if cfg.require_pit_safe:
            pit_safe, dq = self._check_data_quality(date)
            if not pit_safe:
                return LoopResult(
                    status="blocked", decision_date=date, reason="data_quality",
                    execution_mode=mode.value,
                    notes=[f"pit_safe=False: {json.dumps(dq, ensure_ascii=False, default=str)[:400]}"])

        # 2. strategy authority (Scenario H): only executable strategies count
        targets, auth_notes = self._collect_targets(date)
        if not targets:
            statuses = {m.strategy_id: m.approval_state.value for m in self.registry.list()}
            return LoopResult(
                status="no_executable_strategy", decision_date=date,
                execution_mode=mode.value,
                notes=auth_notes + [
                    "no final recommendation maps to an executable strategy "
                    f"(registry statuses: {statuses})"])

        best_rank = -1
        for t in targets:
            rank = _STATUS_LIFECYCLE.get(self.registry.get(t.strategy_id).approval_state.value, -1)
            best_rank = max(best_rank, rank)
        best_status_name = next(
            (s for s, r in sorted(_STATUS_LIFECYCLE.items(), key=lambda kv: kv[1])
             if r == best_rank), "research")

        # 3. mode resolution: LIVE needs compliance + LiveGate (Scenario G/H)
        mode, failed_checks = self._resolve_mode(best_status_name)
        if mode == ExecutionMode.LIVE and failed_checks:
            return LoopResult(status="live_blocked", decision_date=date,
                              execution_mode=ExecutionMode.LIVE.value,
                              failed_checks=failed_checks,
                              notes=["LIVE requires every check to pass; nothing was placed"])

        # 4. kill switch / manual intervention / trading block
        block = self._kill_switch_state()
        if block:
            return LoopResult(status="blocked", decision_date=date,
                              reason=block, execution_mode=mode.value,
                              notes=[f"OMS trading not allowed: {block}"])

        # 5. portfolio plan
        ctor = PortfolioConstructor(PortfolioConfig(
            max_weight_per_stock=float(cfg.max_weight_per_stock),
            max_positions=int(cfg.max_positions),
        ))
        equity = float(cfg.equity)
        drawdown = self.state.current_drawdown(equity)
        plan = ctor.build(targets, PortfolioPlanContext(
            decision_date=date, equity=equity,
            current_positions={}, current_drawdown=drawdown,
            market_vol_pctile=None))
        self.state.record_plan(plan, decision_date=date)

        # 6. risk engine per allocation (Scenario I)
        pnl_rec = self.state.get_daily_pnl(date) or {}
        pnl_frac = float(pnl_rec.get("total") or 0.0) / equity
        self._oms_config["daily_pnl"] = pnl_frac  # wired into DailyLossCircuitBreaker
        adv = self._pit_adv([a["symbol"] for a in plan.allocations])
        risk_engine = RiskEngine(RiskEngineConfig(
            max_weight_per_stock=float(cfg.max_weight_per_stock),
            daily_loss_soft=float(cfg.daily_loss_soft),
            daily_loss_hard=float(cfg.daily_loss_hard),
        ))
        risk_ctx = RiskContext(daily_pnl=pnl_frac, equity=equity, drawdown=drawdown,
                               adv=adv, tradable=True, mode=mode)

        # 7. OMS stage (+submit unless SHADOW / MANUAL_CONFIRM)
        shadow_only = mode in (ExecutionMode.SHADOW, ExecutionMode.MANUAL_CONFIRM)
        orders: List[Dict[str, Any]] = []
        risk_decisions: List[Dict[str, Any]] = []
        notes: List[str] = list(auth_notes)
        staged_ids: List[str] = []
        seq = 0
        for alloc in plan.allocations:
            symbol = str(alloc["symbol"])
            strategy_id = str(alloc.get("strategy_id") or "")
            weight = float(alloc.get("weight") or 0.0)
            decision: RiskDecision = risk_engine.assess_target(
                {"symbol": symbol, "strategy_id": strategy_id, "weight": weight,
                 "side": "BUY", "notional": weight * equity},
                risk_ctx)
            self.state.record_risk_decision(decision, decision_date=date,
                                            symbol=symbol, strategy_id=strategy_id)
            risk_decisions.append({"symbol": symbol, "strategy_id": strategy_id,
                                   "approved": bool(decision.approved),
                                   "reason": decision.reason})
            if not decision.approved:
                notes.append(f"risk_veto {symbol}: {decision.reason}")
                continue

            quote = self._safe_quote(symbol, date)
            if not quote:
                notes.append(f"skip {symbol}: no PIT bar for {date} (suspended?)")
                continue
            ref_price = float(quote.get("close") or quote.get("preclose") or 0.0)
            if ref_price <= 0:
                notes.append(f"skip {symbol}: PIT bar has no usable close/preclose")
                continue

            seq += 1
            budget = weight * equity / _PRICE_HEADROOM
            qty = int(budget / ref_price // 100) * 100
            if qty <= 0:
                notes.append(f"skip {symbol}: budget {weight * equity:.0f} below one "
                             f"100-share lot at {ref_price:.2f}")
                continue

            order = self.oms.stage_order(
                strategy_id=strategy_id,
                decision_id=self._decision_id(date, strategy_id),
                symbol=symbol, side="BUY", qty=qty,
                limit_price=round(ref_price, 2),
                order_type=OrderType.LIMIT, seq=seq)
            if order.status != OrderStatus.CREATED:
                notes.append(f"skip {symbol}: order {order.client_order_id} already "
                             f"{order.status.value} (idempotent re-run)")
                continue
            self._attach_reference_price(order, ref_price)
            staged_ids.append(order.client_order_id)

            if shadow_only:
                notes.append(f"shadow: {order.client_order_id} staged (CREATED), not submitted")
                continue
            try:
                submitted = self.oms.submit(order.client_order_id)
            except NeedsReconciliation as exc:
                notes.append(f"submit blocked for {order.client_order_id}: {exc}")
                continue
            notes.append(f"order {submitted.client_order_id} → {submitted.status.value}")

        # 8. reconcile + collect final per-order outcomes
        reconcile_report = self.oms.reconcile(trade_date=date)
        for cid in staged_ids:
            try:
                o = self.oms.get_order(cid)
            except KeyError:
                continue
            orders.append({
                "client_order_id": o.client_order_id,
                "status": o.status.value,
                "symbol": o.symbol,
                "strategy_id": o.strategy_id,
                "qty": o.qty,
                "filled_qty": o.filled_qty,
                "avg_price": o.avg_price,
                "reject_reason": o.reject_reason,
            })

        return LoopResult(
            status="ok", decision_date=date, reason="", execution_mode=mode.value,
            plans=[{"decision_date": plan.decision_date,
                    "allocations": plan.allocations,
                    "cash_weight": plan.cash_weight,
                    "notes": plan.notes,
                    "no_trade_reason": plan.no_trade_reason}],
            risk_decisions=risk_decisions,
            orders=orders,
            reconcile=reconcile_report.to_dict(),
            notes=notes,
            failed_checks=failed_checks,
        )

    # ------------------------------------------------------------------ #
    def _attach_reference_price(self, order: Order, ref_price: float) -> None:
        order.meta["reference_price"] = float(ref_price)
        with sqlite3.connect(self.config.db_path, timeout=10) as conn:
            conn.execute("UPDATE orders SET json = ? WHERE client_order_id = ?",
                         (json.dumps(order.to_dict(), ensure_ascii=False),
                          order.client_order_id))


def _status_enum(status_name: str):
    from engine.contracts import StrategyStatus

    try:
        return StrategyStatus(str(status_name))
    except ValueError:
        return StrategyStatus.RESEARCH


# ---------------------------------------------------------------------------
# Top-level entry used by the scheduler / CLI
# ---------------------------------------------------------------------------

def run_daily_loop(
    date: Optional[str] = None,
    settings_path: str = "config/settings.yaml",
    pipeline_result: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> LoopResult:
    """Run one daily execution loop and persist the LoopResult JSON.

    `pipeline_result` may be injected (tests / scheduler reuse of the scan
    step output); otherwise a fresh pipeline run would be required — this
    entry point does NOT run the pipeline itself, it consumes an existing
    result (the scheduler passes today's scan output).
    """
    if cfg is None:
        from .config import load_config

        cfg = load_config(settings_path) if settings_path else {}
    compact = to_compact(date or datetime.now())
    loop = DailyExecutionLoop.from_config(cfg, pipeline_result=pipeline_result)
    result = loop.run(compact)
    out_dir = Path(loop.config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"daily_loop_{compact}.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result
