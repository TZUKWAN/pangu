"""Phase 11 产品化 API：执行(OMS/纸面撮合)/策略注册表/研究登记簿/组合状态。

设计要点（诚实失败原则）：
1. 所有子系统的落库路径支持环境变量覆盖（测试用 tmp 目录，生产用默认路径）：
   - PANGU_EXECUTION_DB            OMS + 策略注册表（默认 data/execution.db）
   - PANGU_PAPER_BROKER_DB         纸面撮合（默认 data/paper_broker.db）
   - PANGU_COMPLIANCE_DB           程序化交易合规（默认 data/compliance.db）
   - PANGU_COMPLIANCE_AUDIT_DIR    合规审计目录（默认 data/audit）
   - PANGU_PORTFOLIO_DB            组合/风控状态（默认 data/portfolio_state.db）
   - PANGU_EXPERIMENTS_REGISTRY    实验登记簿 jsonl（默认 data/experiments/registry.jsonl）
   - PANGU_FACTOR_INDEX            因子索引 jsonl（默认 data/experiments/factor_index.jsonl）
   - PANGU_FACTORS_DIR             因子报告目录（默认 data/experiments/factors）
   - PANGU_PIT_DB                  PIT 档案（默认 data/market_breadth/raw.sqlite3）
2. 执行模式 execution_mode 持久化在 OMS 库 system_state 表，缺省 PAPER。
   切到 LIVE 必须通过 compliance.LiveGate.evaluate（Scenario G/H 强制），
   任一检查失败返回 409 + failed_checks，绝不放水。
3. PaperBrokerAsAdapter：把 PaperBroker 适配成 OMS 使用的 BrokerAdapter 接缝
   （薄包装，全部委托，不改变 engine/execution 下的任何实现）。
4. 任何文件/表缺失都显式降级（graceful），不伪造数据；失败实验永远可见。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..contracts import (
    COMPLIANCE_LIVE_OK,
    DEFAULT_EXECUTION_MODE,
    EXECUTABLE_STRATEGY_STATUS,
    BrokerError,
    ComplianceState,
    ExecutionMode,
    StrategyStatus,
)
from ..compliance.program_trading import ComplianceManager, LiveGate
from ..execution.broker import BrokerAdapter
from ..execution.oms import NeedsReconciliation, OrderManagementSystem
from ..execution.paper import PaperBroker
from ..execution.risk_controls import (
    CashCheckGuard,
    DailyLossCircuitBreaker,
    DuplicateOrderGuard,
    PositionCheckGuard,
    PriceDeviationGuard,
    RateLimitGuard,
)
from ..strategies.gates import PromotionContext
from ..strategies.manifest import StrategyManifest
from ..strategies.registry import StrategyRegistry

router = APIRouter()

# ---------------------------------------------------------------------- #
# 路径解析（env 覆盖 → 默认值）
# ---------------------------------------------------------------------- #
def _env_path(env_key: str, default: str) -> str:
    return os.environ.get(env_key) or default


def execution_db() -> str:
    return _env_path("PANGU_EXECUTION_DB", "data/execution.db")


def paper_broker_db() -> str:
    return _env_path("PANGU_PAPER_BROKER_DB", "data/paper_broker.db")


def compliance_db() -> str:
    return _env_path("PANGU_COMPLIANCE_DB", "data/compliance.db")


def compliance_audit_dir() -> str:
    return _env_path("PANGU_COMPLIANCE_AUDIT_DIR", "data/audit")


def portfolio_db() -> str:
    return _env_path("PANGU_PORTFOLIO_DB", "data/portfolio_state.db")


def experiments_registry_path() -> str:
    return _env_path("PANGU_EXPERIMENTS_REGISTRY", "data/experiments/registry.jsonl")


def factor_index_path() -> str:
    return _env_path("PANGU_FACTOR_INDEX", "data/experiments/factor_index.jsonl")


def factors_dir() -> str:
    return _env_path("PANGU_FACTORS_DIR", "data/experiments/factors")


def pit_db() -> str:
    try:
        from ..data.pit_store import DEFAULT_DB
        return _env_path("PANGU_PIT_DB", str(DEFAULT_DB))
    except Exception:  # noqa: BLE001
        return _env_path("PANGU_PIT_DB", "data/market_breadth/raw.sqlite3")


def execution_cfg() -> Dict[str, Any]:
    """config 中 execution 段（缺省安全）。"""
    try:
        from ..config import load_config
        cfg = (load_config() or {}).get("execution") or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    return cfg if isinstance(cfg, dict) else {}


def rate_limits_cfg() -> Dict[str, Any]:
    rl = execution_cfg().get("rate_limits") or {}
    return rl if isinstance(rl, dict) else {}


# ---------------------------------------------------------------------- #
# 单例管理（线程安全；测试可 reset）
# ---------------------------------------------------------------------- #
_quote_provider_factory: Optional[Callable[[], Callable[[str, str], Optional[Dict]]]] = None
_singletons: Dict[str, Any] = {}
# 可重入锁：get_oms → get_adapter → get_paper_broker 会嵌套获取
_singles_lock = threading.RLock()


def reset_execution_singletons() -> None:
    """仅测试使用：清空全部惰性单例（切 tmp 库后必须调用）。"""
    with _singles_lock:
        _singletons.clear()


def _singleton(key: str, factory: Callable[[], Any]) -> Any:
    with _singles_lock:
        if key not in _singletons:
            _singletons[key] = factory()
        return _singletons[key]


# ---------------------------------------------------------------------- #
# PaperBrokerAsAdapter：PaperBroker → BrokerAdapter 薄适配
# ---------------------------------------------------------------------- #
class PaperBrokerAsAdapter(BrokerAdapter):
    """把 PaperBroker 包装成 OMS 依赖的 BrokerAdapter 接缝。

    PaperBroker 自身已实现全部委托语义；这层薄包装只负责：
    1) 显式 connect() 生命周期（OMS 拿到的一定是已连接适配器）；
    2) health() 合并纸面撮合的日期信息，供前端展示。
    """

    name = "paper_as_adapter"

    def __init__(self, broker: PaperBroker) -> None:
        super().__init__()
        self._broker = broker

    @property
    def broker(self) -> PaperBroker:
        return self._broker

    def connect(self) -> None:
        try:
            self._broker.connect()
            self._connected = True
            self._last_error = ""
        except Exception as exc:  # noqa: BLE001
            self._connected = False
            self._last_error = f"paper broker connect failed: {exc}"

    def health(self) -> Dict:
        base = super().health()
        try:
            base.update(self._broker.health())
        except Exception:  # noqa: BLE001
            pass
        return base

    # -- account ------------------------------------------------------------
    def get_balance(self):
        return self._broker.get_balance()

    def get_positions(self):
        return self._broker.get_positions()

    # -- orders -------------------------------------------------------------
    def get_orders(self, trade_date: Optional[str] = None):
        return self._broker.get_orders(trade_date)

    def get_trades(self, trade_date: Optional[str] = None):
        return self._broker.get_trades(trade_date)

    def submit_order(self, order):
        return self._broker.submit_order(order)

    def cancel_order(self, broker_order_id: str) -> bool:
        return self._broker.cancel_order(broker_order_id)

    def query_order(self, broker_order_id: str):
        return self._broker.query_order(broker_order_id)


# ---------------------------------------------------------------------- #
# 默认报价源：共享 DataLoader 全市场快照（best-effort，失败返回 None →
# PaperBroker 会以 "no quote" 诚实拒单，绝不伪造成交）
# ---------------------------------------------------------------------- #
def _default_quote_provider_factory() -> Callable[[str, str], Optional[Dict]]:
    def quote(symbol: str, date_str: str) -> Optional[Dict]:
        try:
            from ..data_loader import DataLoader, find_col, safe_float  # noqa: PLC0415
            dl = _singleton("dl", lambda: DataLoader())
            df = dl.all_spot()
            code_col = find_col(df, ["代码", "code"])
            px_col = find_col(df, ["最新价", "现价", "close", "收盘价"])
            pct_col = find_col(df, ["涨跌幅", "pct"])
            name_col = find_col(df, ["名称", "name"])
            target = str(symbol).split(".")[-1].zfill(6)
            row = None
            if code_col is not None:
                match = df[df[code_col].astype(str).str.strip().str.zfill(6) == target]
                if len(match):
                    row = match.iloc[-1]
            if row is None or px_col is None:
                return None
            close = safe_float(row.get(px_col), 0.0)
            if close <= 0:
                return None
            pct = safe_float(row.get(pct_col), 0.0) if pct_col is not None else 0.0
            denom = 1.0 + pct / 100.0
            preclose = close / denom if denom else close
            name = str(row.get(name_col, "") or "") if name_col is not None else ""
            return {
                "open": close, "high": close, "low": close, "close": close,
                "preclose": round(preclose, 3),
                "is_st": 1 if "ST" in name.upper() else 0,
                "source": "all_spot_cache",
            }
        except Exception:  # noqa: BLE001
            return None
    return quote


def _quote_factory() -> Callable[[str, str], Optional[Dict]]:
    return (_quote_provider_factory or _default_quote_provider_factory)()


# ---------------------------------------------------------------------- #
# 惰性单例访问器
# ---------------------------------------------------------------------- #
def get_paper_broker() -> PaperBroker:
    def _build() -> PaperBroker:
        cfg = execution_cfg()
        broker = PaperBroker(
            db_path=paper_broker_db(),
            quote_provider=_quote_factory(),
            initial_cash=float(cfg.get("paper_initial_cash", 1_000_000.0)),
            slippage_bps=float((cfg.get("paper") or {}).get("slippage_bps", 10.0)),
        )
        broker.connect()
        return broker
    return _singleton("paper_broker", _build)


def get_adapter() -> PaperBrokerAsAdapter:
    def _build() -> PaperBrokerAsAdapter:
        adapter = PaperBrokerAsAdapter(get_paper_broker())
        adapter.connect()  # 保证 health()/一致性语义：OMS 拿到已连接适配器
        return adapter
    return _singleton("adapter", _build)


def get_oms() -> OrderManagementSystem:
    def _build() -> OrderManagementSystem:
        rl = rate_limits_cfg()
        cfg = dict(execution_cfg())
        cfg.setdefault("equity", float(cfg.get("paper_initial_cash", 1_000_000.0)))
        checks = [
            DuplicateOrderGuard(window_minutes=int(rl.get("duplicate_window_minutes", 5))),
            PriceDeviationGuard(max_deviation_pct=float(rl.get("max_price_deviation_pct", 0.03))),
            CashCheckGuard(),
            PositionCheckGuard(),
            RateLimitGuard(
                max_per_day=int(rl.get("max_per_day", 200)),
                min_interval_seconds=int(rl.get("min_interval_seconds", 2)),
            ),
            DailyLossCircuitBreaker(
                soft_stop=float(cfg.get("daily_loss_soft_stop", -0.02)),
                hard_stop=float(cfg.get("daily_loss_hard_stop", -0.05)),
            ),
        ]
        return OrderManagementSystem(
            adapter=get_adapter(),
            checks=checks,
            db_path=execution_db(),
            ack_timeout_s=float(cfg.get("ack_timeout_s", 5.0)),
            config={"daily_pnl": float(cfg.get("daily_pnl", 0.0)), "equity": cfg["equity"]},
        )
    return _singleton("oms", _build)


def get_registry() -> StrategyRegistry:
    return _singleton("registry", lambda: StrategyRegistry(db_path=execution_db()))


def get_compliance_manager() -> ComplianceManager:
    return _singleton(
        "compliance",
        lambda: ComplianceManager(
            db_path=compliance_db(), audit_dir=compliance_audit_dir(),
        ),
    )


def get_portfolio_state():
    from ..portfolio_engine.state import PortfolioState
    return _singleton("portfolio_state", lambda: PortfolioState(db_path=portfolio_db()))


# ---------------------------------------------------------------------- #
# system_state 直接读写（只读路径不建库，避免意外的文件副作用）
# ---------------------------------------------------------------------- #
def read_system_state(db_path: str, key: str, default: str = "") -> str:
    if not Path(db_path).exists():
        return default
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            row = conn.execute("SELECT value FROM system_state WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row and row[0] is not None else default
    except sqlite3.OperationalError:
        return default


def current_execution_mode() -> str:
    raw = read_system_state(execution_db(), "execution_mode", "")
    if raw in ExecutionMode._value2member_map_:  # noqa: SLF001
        return raw
    return DEFAULT_EXECUTION_MODE.value


def _trading_allowed_from_state(db_path: str) -> bool:
    if read_system_state(db_path, "kill_switch_engaged") == "true":
        return False
    if read_system_state(db_path, "manual_intervention_required"):
        return False
    if read_system_state(db_path, "trading_blocked"):
        return False
    return True


def _persist_execution_mode(new_value: str, reason: str, operator: str) -> str:
    """写 system_state 并落一条审计（order_events, kind=execution_mode）。"""
    get_oms()  # 确保 schema 存在
    db = execution_db()
    previous = current_execution_mode()
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    detail = f"{previous} -> {new_value} by {operator}"
    if reason.strip():
        detail += f"; reason: {reason.strip()}"
    with sqlite3.connect(db, timeout=10) as conn:
        conn.execute(
            "INSERT INTO system_state (key, value) VALUES ('execution_mode', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (new_value,))
        conn.execute(
            "INSERT INTO order_events (ts, client_order_id, kind, status, detail)"
            " VALUES (?, 'execution_mode', 'execution_mode', ?, ?)",
            (ts, new_value, detail))
    return previous


def _audit_event(subject: str, status: str, detail: str) -> None:
    """向 OMS 库 order_events 追加一条系统审计（表不存在时静默跳过）。"""
    db = execution_db()
    if not Path(db).exists():
        return
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        with sqlite3.connect(db, timeout=10) as conn:
            conn.execute(
                "INSERT INTO order_events (ts, client_order_id, kind, status, detail)"
                " VALUES (?, ?, 'execution_mode', ?, ?)", (ts, subject, status, detail))
    except sqlite3.OperationalError:
        pass


def latest_reconcile_report(db_path: str) -> Optional[Dict[str, Any]]:
    if not Path(db_path).exists():
        return None
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            row = conn.execute(
                "SELECT ts, report_json FROM reconcile_runs ORDER BY id DESC LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    try:
        report = json.loads(row[1])
    except json.JSONDecodeError:
        return None
    return {"ts": row[0], **report}


# ---------------------------------------------------------------------- #
# 订单读取（新→旧，带 timeline）
# ---------------------------------------------------------------------- #
def list_order_dicts(limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
    db = execution_db()
    if not Path(db).exists():
        return []
    sql = "SELECT rowid, json FROM orders"
    params: List[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    out: List[Dict[str, Any]] = []
    with sqlite3.connect(db, timeout=10) as conn:
        rows = conn.execute(sql, params).fetchall()
    for rowid, raw in rows:
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        d["_rowid"] = rowid
        out.append(d)
    out.sort(key=lambda d: (str(d.get("created_at") or ""), int(d.get("_rowid") or 0)), reverse=True)
    for d in out:
        d.pop("_rowid", None)
    return out[: max(1, int(limit))]


def orders_index(limit: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    """symbol → [order dict 新→旧]，供今日执行链路交叉引用。"""
    idx: Dict[str, List[Dict[str, Any]]] = {}
    for o in list_order_dicts(limit=limit):
        idx.setdefault(str(o.get("symbol") or ""), []).append(o)
    return idx


def symbol_matches(order_symbol: str, code: str) -> bool:
    o, c = str(order_symbol or "").strip(), str(code or "").strip()
    if not o or not c:
        return False
    return o == c or o.split(".")[0] == c or c.split(".")[0] == o


# ---------------------------------------------------------------------- #
# LIVE 门禁（Scenario G/H 强制点）
# ---------------------------------------------------------------------- #
_PAPER_PASSED_STATUSES = {StrategyStatus.SHADOW_LIVE, StrategyStatus.LIMITED_LIVE, StrategyStatus.LIVE}
_SHADOW_PASSED_STATUSES = {StrategyStatus.LIMITED_LIVE, StrategyStatus.LIVE}
_VALIDATED_PLUS = {StrategyStatus.VALIDATED, StrategyStatus.PAPER, StrategyStatus.SHADOW_LIVE,
                   StrategyStatus.LIMITED_LIVE, StrategyStatus.LIVE}


def evaluate_live_gate() -> Dict[str, Any]:
    """用当前 registry/compliance/paper/shadow/broker/reconcile 状态评估 LiveGate。"""
    registry = get_registry()
    statuses = [m.approval_state for m in registry.list()]
    _RANK = {StrategyStatus.IDEA: 0, StrategyStatus.RESEARCH: 1, StrategyStatus.VALIDATED: 2,
             StrategyStatus.PAPER: 3, StrategyStatus.SHADOW_LIVE: 4,
             StrategyStatus.LIMITED_LIVE: 5, StrategyStatus.LIVE: 6,
             StrategyStatus.SUSPENDED: -1, StrategyStatus.RETIRED: -1}
    strategy_status = max(statuses, key=lambda s: _RANK.get(s, -1)) if statuses else StrategyStatus.RESEARCH

    compliance_state = get_compliance_manager().state()
    compliance_ok = compliance_state in COMPLIANCE_LIVE_OK
    paper_passed = any(s in _PAPER_PASSED_STATUSES for s in statuses)
    shadow_passed = any(s in _SHADOW_PASSED_STATUSES for s in statuses)

    try:
        broker_ok = bool(get_adapter().health().get("ok"))
    except Exception:  # noqa: BLE001
        broker_ok = False

    last_reconcile = latest_reconcile_report(execution_db())
    reconcile_ok = bool(last_reconcile and last_reconcile.get("reconcile_ok"))

    kill_switch_engaged = read_system_state(execution_db(), "kill_switch_engaged") == "true"
    kill_switch_ready = not kill_switch_engaged  # 已具备熔断能力且当前未触发

    result = LiveGate.evaluate(
        strategy_status=strategy_status,
        execution_mode=ExecutionMode.LIVE,
        compliance_ok=compliance_ok,
        paper_passed=paper_passed,
        shadow_passed=shadow_passed,
        broker_ok=broker_ok,
        reconcile_ok=reconcile_ok,
        kill_switch_ready=kill_switch_ready,
    )
    return {
        "approved": result.approved,
        "failed_checks": list(result.failed_checks),
        "checked_at": result.checked_at,
        "context": {
            "strategy_status": strategy_status.value,
            "compliance_state": compliance_state.value,
            "paper_passed": paper_passed,
            "shadow_passed": shadow_passed,
            "broker_ok": broker_ok,
            "reconcile_ok": reconcile_ok,
            "kill_switch_ready": kill_switch_ready,
            "registered_strategies": len(statuses),
        },
    }


# ---------------------------------------------------------------------- #
# 请求模型
# ---------------------------------------------------------------------- #
class ModeRequest(BaseModel):
    mode: str
    reason: str = ""
    operator: str = "web"


class KillSwitchRequest(BaseModel):
    action: str  # engage | disengage
    reason: str = ""
    operator: str = "web"


class PlaceOrderRequest(BaseModel):
    strategy_id: str
    symbol: str
    side: str  # BUY | SELL
    qty: int
    limit_price: float
    decision_id: str = ""


# ---------------------------------------------------------------------- #
# 执行总览 / 模式 / 熔断 / 对账
# ---------------------------------------------------------------------- #
@router.get("/api/execution/overview")
def api_execution_overview() -> Dict[str, Any]:
    oms = get_oms()
    db = execution_db()
    mode = current_execution_mode()
    compliance_state = get_compliance_manager().state()
    try:
        adapter_health = get_adapter().health()
    except Exception as exc:  # noqa: BLE001
        adapter_health = {"ok": False, "adapter": "paper_as_adapter", "reason": str(exc)}
    kill_switch_engaged = read_system_state(db, "kill_switch_engaged") == "true"
    manual_reason = read_system_state(db, "manual_intervention_required")
    return {
        "execution_mode": mode,
        "trading_allowed": oms.trading_allowed(),
        "kill_switch": kill_switch_engaged,
        "kill_switch_reason": read_system_state(db, "kill_switch_reason"),
        "manual_intervention_required": bool(manual_reason),
        "manual_intervention_reason": manual_reason or None,
        "trading_blocked": read_system_state(db, "trading_blocked") or None,
        "last_reconcile": latest_reconcile_report(db),
        "compliance_state": compliance_state.value
        if isinstance(compliance_state, ComplianceState) else str(compliance_state),
        "adapter_health": adapter_health,
        "paper_broker": {
            "db_path": paper_broker_db(),
            "date": getattr(get_paper_broker(), "current_date", None),
        },
        "order_count": len(list_order_dicts(limit=10_000)),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


@router.post("/api/execution/mode")
def api_execution_mode_switch(req: ModeRequest) -> Dict[str, Any]:
    raw = str(req.mode or "").strip().upper()
    try:
        target = ExecutionMode(raw)
    except ValueError:
        raise HTTPException(
            400,
            f"未知执行模式: {req.mode!r}；可选: {sorted(m.value for m in ExecutionMode)}",
        )
    get_oms()  # 确保 system_state/order_events schema 存在（拒绝也要留审计）
    previous = current_execution_mode()
    gate_payload: Optional[Dict[str, Any]] = None
    if target is ExecutionMode.LIVE and previous != ExecutionMode.LIVE.value:
        gate_payload = evaluate_live_gate()
        if not gate_payload["approved"]:
            # Scenario G/H：LIVE 门禁 fail-closed，拒绝并完整披露失败项（审计留痕）
            _audit_event("execution_mode", "LIVE_REFUSED",
                         f"{previous} -> LIVE refused; failed_checks: "
                         + "; ".join(gate_payload["failed_checks"]))
            raise HTTPException(409, detail={
                "message": "切换到 LIVE 被拒绝（LiveGate fail-closed）",
                "failed_checks": gate_payload["failed_checks"],
                "gate": gate_payload,
            })
    _persist_execution_mode(target.value, req.reason or "", req.operator or "web")
    return {
        "ok": True,
        "execution_mode": target.value,
        "previous": previous,
        "trading_allowed": get_oms().trading_allowed(),
        "gate": gate_payload,
        "note": "LIVE 模式受 LiveGate 门禁与熔断机制约束" if target is ExecutionMode.LIVE else None,
    }


@router.post("/api/execution/kill-switch")
def api_execution_kill_switch(req: KillSwitchRequest) -> Dict[str, Any]:
    action = str(req.action or "").strip().lower()
    if action not in ("engage", "disengage"):
        raise HTTPException(400, "action 必须是 engage 或 disengage")
    if not str(req.reason or "").strip():
        raise HTTPException(400, "必须填写原因（审计要求）")
    oms = get_oms()
    if action == "engage":
        oms.engage_kill_switch(req.reason.strip(), req.operator or "web")
    else:
        try:
            oms.disengage_kill_switch(req.operator or "web", req.reason.strip())
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    db = execution_db()
    return {
        "ok": True,
        "action": action,
        "kill_switch": read_system_state(db, "kill_switch_engaged") == "true",
        "trading_allowed": oms.trading_allowed(),
        "reason": req.reason.strip(),
        "operator": req.operator or "web",
    }


@router.post("/api/execution/reconcile")
def api_execution_reconcile() -> Dict[str, Any]:
    oms = get_oms()
    report = oms.reconcile()
    return {
        "ok": bool(report.reconcile_ok),
        "report": report.to_dict(),
        "trading_allowed": oms.trading_allowed(),
        "manual_intervention_required": bool(
            read_system_state(execution_db(), "manual_intervention_required")),
    }


# ---------------------------------------------------------------------- #
# 订单
# ---------------------------------------------------------------------- #
@router.get("/api/execution/orders")
def api_execution_orders(
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
) -> Dict[str, Any]:
    orders = list_order_dicts(limit=limit, status=status)
    return {"orders": orders, "count": len(orders), "db_path": execution_db()}


@router.post("/api/execution/orders")
def api_place_order(req: PlaceOrderRequest) -> Dict[str, Any]:
    strategy_id = str(req.strategy_id or "").strip()
    symbol = str(req.symbol or "").strip().upper()
    side = str(req.side or "").strip().upper()
    qty = int(req.qty or 0)
    limit_price = float(req.limit_price or 0.0)
    if not strategy_id:
        raise HTTPException(400, "strategy_id 不能为空")
    if not symbol:
        raise HTTPException(400, "symbol 不能为空")
    if side not in ("BUY", "SELL"):
        raise HTTPException(400, f"side 必须是 BUY/SELL，收到 {req.side!r}")
    if qty <= 0:
        raise HTTPException(400, "qty 必须大于 0")
    if limit_price <= 0:
        raise HTTPException(400, "limit_price 必须大于 0（OMS 只接受限价单）")

    # 策略治理：只有注册且处于可执行状态的策略可以下单
    registry = get_registry()
    try:
        manifest = registry.get(strategy_id)
    except KeyError:
        raise HTTPException(400, f"策略未注册: {strategy_id}；请在策略页查看注册表")
    if manifest.approval_state not in EXECUTABLE_STRATEGY_STATUS:
        raise HTTPException(
            403,
            f"策略 {strategy_id} 状态为 {manifest.approval_state.value}，不可执行下单"
            f"（可执行状态: {sorted(s.value for s in EXECUTABLE_STRATEGY_STATUS)}）",
        )

    oms = get_oms()
    mode = current_execution_mode()
    if not oms.trading_allowed():
        blocked_by = [k for k, v in (
            ("kill_switch", read_system_state(execution_db(), "kill_switch_engaged")),
            ("manual_intervention", read_system_state(execution_db(), "manual_intervention_required")),
            ("trading_blocked", read_system_state(execution_db(), "trading_blocked")),
        ) if v]
        raise HTTPException(409, detail={
            "message": "交易被阻断，禁止下单",
            "blocked_by": blocked_by,
        })

    decision_id = str(req.decision_id or "").strip() or (
        f"WEB-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}")
    order = oms.stage_order(strategy_id, decision_id, symbol, side, qty, limit_price)

    submitted = False
    note = ""
    if mode == ExecutionMode.PAPER.value:
        try:
            order = oms.submit(order.client_order_id)
            submitted = True
        except NeedsReconciliation as exc:
            raise HTTPException(409, f"该订单需要先对账（idempotent re-submit 被禁止）: {exc}")
        except BrokerError as exc:
            raise HTTPException(502, f"纸面撮合通道异常（订单可能已进入 UNKNOWN，请先对账）: {exc}")
    else:
        note = f"当前模式 {mode} 只登记计划不下单；仅 PAPER 模式会经 PaperBroker 撮合"

    return {
        "ok": True,
        "submitted": submitted,
        "message": note or None,
        "execution_mode": mode,
        "order": order.to_dict(),
    }


# ---------------------------------------------------------------------- #
# 策略注册表
# ---------------------------------------------------------------------- #
@router.get("/api/strategies")
def api_strategies() -> Dict[str, Any]:
    registry = get_registry()
    out: List[Dict[str, Any]] = []
    for m in registry.list():
        status = m.approval_state
        validated = status in _VALIDATED_PLUS
        live_capable = status in (StrategyStatus.LIVE, StrategyStatus.LIMITED_LIVE)
        if live_capable:
            advice = "可实盘（受限）" if status is StrategyStatus.LIMITED_LIVE else "可实盘"
        elif validated:
            advice = "已验证·非实盘阶段"
        else:
            advice = "未验证/研究——不构成实盘建议"
        out.append({
            "strategy_id": m.strategy_id,
            "status": status.value,
            "version": m.version,
            "ranges": {
                "train": m.train_range,
                "validation": m.validation_range,
                "test": m.test_range,
            },
            "metrics": m.metrics,
            "evidence_refs": m.evidence_refs,
            "notes": m.notes,
            "updated_at": m.updated_at,
            "executable": status in EXECUTABLE_STRATEGY_STATUS,
            "validated": validated,
            "research_only": not validated,
            "live_capable": live_capable,
            "advice": advice,
            "model": m.model,
            "data_snapshot": m.data_snapshot,
        })
    out.sort(key=lambda s: s["strategy_id"])
    return {
        "strategies": out,
        "count": len(out),
        "registry_db": execution_db(),
        "executable_statuses": sorted(s.value for s in EXECUTABLE_STRATEGY_STATUS),
    }


# ---------------------------------------------------------------------- #
# 研究登记簿 + 因子库
# ---------------------------------------------------------------------- #
@router.get("/api/research/experiments")
def api_research_experiments() -> Dict[str, Any]:
    from ..validation.experiment_registry import ExperimentRegistry

    reg_path = experiments_registry_path()
    experiments: List[Dict[str, Any]] = []
    warnings: List[str] = []
    try:
        entries = ExperimentRegistry(reg_path).query()
        experiments = entries
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"实验登记簿读取失败: {exc}")

    factors: List[Dict[str, Any]] = []
    idx_path = factor_index_path()
    if Path(idx_path).exists():
        seen: set = set()
        try:
            for line in Path(idx_path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = str(rec.get("name") or "")
                if not name or name in seen:
                    continue
                seen.add(name)
                factors.append(rec)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"因子索引读取失败: {exc}")
    else:
        warnings.append(f"因子索引不存在: {idx_path}（尚未产出因子报告）")

    failed_count = sum(1 for e in experiments if str(e.get("status")) == "failed")
    return {
        "experiments": experiments,
        "factors": factors,
        "failed_count": failed_count,
        "total": len(experiments),
        "factor_count": len(factors),
        "registry_path": reg_path,
        "registry_exists": Path(reg_path).exists(),
        "warnings": warnings,
    }


_FACTOR_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


@router.get("/api/research/factors/{name}")
def api_research_factor(name: str) -> Dict[str, Any]:
    if not _FACTOR_NAME_RE.match(name or ""):
        raise HTTPException(400, "非法因子名")
    path = Path(factors_dir()) / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"因子报告不存在: {name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"因子报告解析失败: {exc}")


# ---------------------------------------------------------------------- #
# 组合状态 / 数据质量 / 设置摘要
# ---------------------------------------------------------------------- #
@router.get("/api/portfolio/latest")
def api_portfolio_latest() -> Dict[str, Any]:
    db = portfolio_db()
    if not Path(db).exists():
        return {
            "status": "empty", "plan": None, "risk_decisions": [],
            "reason": f"组合状态库不存在: {db}（组合构造器尚未产出计划）",
        }
    try:
        with sqlite3.connect(db, timeout=10) as conn:
            plan_row = conn.execute(
                "SELECT id, decision_date, rule_hash, cash_weight, plan_json, ts"
                " FROM portfolio_plans ORDER BY id DESC LIMIT 1").fetchone()
            risk_rows = conn.execute(
                "SELECT id, decision_date, symbol, strategy_id, approved, reason, decision_json, ts"
                " FROM risk_decisions ORDER BY id DESC LIMIT 50").fetchall()
            pnl_row = conn.execute(
                "SELECT date, realized, unrealized, total FROM daily_pnl"
                " ORDER BY date DESC LIMIT 1").fetchone()
            state_rows = conn.execute("SELECT key, value FROM account_state").fetchall()
    except sqlite3.OperationalError as exc:
        return {"status": "empty", "plan": None, "risk_decisions": [],
                "reason": f"组合状态库表缺失: {exc}"}

    plan = None
    if plan_row:
        try:
            plan_payload = json.loads(plan_row[4])
        except json.JSONDecodeError:
            plan_payload = {"raw": plan_row[4]}
        plan = {
            "id": plan_row[0], "decision_date": plan_row[1], "rule_hash": plan_row[2],
            "cash_weight": plan_row[3], "ts": plan_row[5], **plan_payload,
        }
    decisions = []
    for r in risk_rows:
        try:
            payload = json.loads(r[6])
        except json.JSONDecodeError:
            payload = {}
        decisions.append({
            "id": r[0], "decision_date": r[1], "symbol": r[2], "strategy_id": r[3],
            "approved": bool(r[4]), "reason": r[5], "checks": payload.get("checks"), "ts": r[7],
        })
    account = {k: v for k, v in state_rows}
    return {
        "status": "ok" if plan or decisions else "empty",
        "plan": plan,
        "risk_decisions": decisions,
        "daily_pnl": {"date": pnl_row[0], "realized": pnl_row[1],
                      "unrealized": pnl_row[2], "total": pnl_row[3]} if pnl_row else None,
        "account": {"high_water": account.get("high_water"), "last_equity": account.get("last_equity")},
        "db_path": db,
    }


@router.get("/api/system/data-quality")
def api_system_data_quality(date: Optional[str] = Query(None)) -> Dict[str, Any]:
    try:
        from ..data.pit_store import PITStore
        from ..data.quality import DataQualityReport
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "date": date,
                "reason": f"数据质量模块不可用: {exc}"}
    db = pit_db()
    if not Path(db).exists():
        return {
            "status": "unavailable", "date": date,
            "reason": f"PIT 档案不存在: {db}（请先运行 PIT 归档）",
        }
    try:
        store = PITStore(db)
        target = date or (store.trade_dates_iso()[-1] if store.trade_dates_iso() else None)
        if not target:
            return {"status": "unavailable", "date": date, "reason": "PIT 档案无任何交易日"}
        report = DataQualityReport.build(store, target)
        report["status"] = "ok"
        report["pit_db"] = db
        return report
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "date": date, "reason": f"数据质量报告构建失败: {exc}"}


def execution_settings_summary() -> Dict[str, Any]:
    """给 /api/settings 的执行默认值只读摘要（不触发任何库文件创建）。"""
    db = execution_db()
    rl = rate_limits_cfg()
    cfg = execution_cfg()
    return {
        "execution_mode": current_execution_mode(),
        "trading_allowed": _trading_allowed_from_state(db),
        "rate_limits": {
            "max_per_day": int(rl.get("max_per_day", 200)),
            "min_interval_seconds": int(rl.get("min_interval_seconds", 2)),
        },
        "paper_initial_cash": float(cfg.get("paper_initial_cash", 1_000_000.0)),
        "editable": False,
        "note": "执行默认值当前只读展示；修改需编辑 config 的 execution 段后重启",
    }
