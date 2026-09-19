"""程序化交易合规门禁 (Phase 10).

- ComplianceManager: persisted ComplianceState machine + jsonl audit trail +
  《程序化交易信息表》(export_trading_software_info).  State transitions are
  deliberately narrow: the compliance paperwork can only mature along
  UNKNOWN → REPORT_REQUIRED/SUBMITTED/NOT_REQUIRED_CONFIRMED → CONFIRMED →
  EXPIRED_OR_CHANGED → REPORT_REQUIRED ...  Illegal jumps raise.
- LiveGate: fails closed.  LIVE execution requires ALL checks true AND a
  live-capable strategy status (LIMITED_LIVE grants limited approval only).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from engine.contracts import (
    COMPLIANCE_LIVE_OK,
    ComplianceState,
    ExecutionMode,
    StrategyStatus,
)

# legal transitions; REPORT_REQUIRED→SUBMITTED models "report filed after decision"
COMPLIANCE_TRANSITIONS: Dict[ComplianceState, set] = {
    ComplianceState.UNKNOWN: {ComplianceState.REPORT_REQUIRED, ComplianceState.SUBMITTED,
                              ComplianceState.NOT_REQUIRED_CONFIRMED},
    ComplianceState.REPORT_REQUIRED: {ComplianceState.SUBMITTED},
    ComplianceState.SUBMITTED: {ComplianceState.CONFIRMED, ComplianceState.EXPIRED_OR_CHANGED},
    ComplianceState.CONFIRMED: {ComplianceState.EXPIRED_OR_CHANGED},
    ComplianceState.EXPIRED_OR_CHANGED: {ComplianceState.REPORT_REQUIRED},
    ComplianceState.NOT_REQUIRED_CONFIRMED: set(),  # terminal until regulation changes
}


class IllegalComplianceTransition(ValueError):
    pass


class ComplianceManager:
    def __init__(self, db_path: str = "data/compliance.db",
                 audit_dir: str = "data/audit", developer: str = "Pangu 量化研究") -> None:
        import sqlite3
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._audit_dir = Path(audit_dir)
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._developer = developer
        self._sqlite3 = sqlite3
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS compliance_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state TEXT NOT NULL,
                    operator TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS compliance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    operator TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT ''
                );
            """)
            if conn.execute("SELECT 1 FROM compliance_state WHERE id = 1").fetchone() is None:
                conn.execute("INSERT INTO compliance_state (id, state, updated_at) VALUES"
                             " (1, ?, ?)", (ComplianceState.UNKNOWN.value, self._ts()))

    def _conn(self):
        return self._sqlite3.connect(self._db_path, timeout=10)

    def _ts(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _append_audit(self, record: Dict) -> None:
        with open(self._audit_dir / "compliance_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- state machine ------------------------------------------------------
    def state(self) -> ComplianceState:
        with self._conn() as conn:
            row = conn.execute("SELECT state FROM compliance_state WHERE id = 1").fetchone()
        return ComplianceState(row[0]) if row else ComplianceState.UNKNOWN

    def set_state(self, new: ComplianceState, operator: str, evidence: str) -> None:
        current = self.state()
        if new not in COMPLIANCE_TRANSITIONS.get(current, set()):
            raise IllegalComplianceTransition(
                f"illegal compliance transition {current.value} -> {new.value}")
        ts = self._ts()
        with self._conn() as conn:
            conn.execute("UPDATE compliance_state SET state = ?, operator = ?, evidence = ?,"
                         " updated_at = ? WHERE id = 1",
                         (new.value, operator, evidence, ts))
            conn.execute("INSERT INTO compliance_history (ts, from_state, to_state, operator,"
                         " evidence) VALUES (?, ?, ?, ?, ?)",
                         (ts, current.value, new.value, operator, evidence))
        self._append_audit({"ts": ts, "event": "state_change", "from": current.value,
                            "to": new.value, "operator": operator, "evidence": evidence})

    def history(self) -> List[Dict]:
        with self._conn() as conn:
            conn.row_factory = self._sqlite3.Row
            rows = conn.execute("SELECT * FROM compliance_history ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def can_enable_live(self) -> Tuple[bool, str]:
        state = self.state()
        if state in COMPLIANCE_LIVE_OK:
            return True, f"compliance state {state.value} allows live trading"
        return False, f"compliance state {state.value} does not allow live trading; " \
                      f"need one of {sorted(s.value for s in COMPLIANCE_LIVE_OK)}"

    # -- 程序化交易信息表 ----------------------------------------------------
    def export_trading_software_info(self, rate_limit_cfg: Optional[Dict] = None,
                                     kill_switch_info: Optional[Dict] = None) -> Dict:
        cfg = dict(rate_limit_cfg or {})
        ks = dict(kill_switch_info or {})
        info = {
            "软件名称": "Pangu",
            "版本": self._git_sha(),
            "开发主体": self._developer,
            "策略类型": "低中频A股现货",
            "指令生成方式": "策略信号→组合→OMS限价单",
            "最大预计下单频率": (
                f"≤{cfg.get('max_per_day', 200)} 笔/日，单账户最小下单间隔 "
                f"{cfg.get('min_interval_seconds', 2)} 秒"),
            "风控摘要": ks.get("risk_controls",
                             "盘前风控检查(重复/偏离/资金/持仓/频次/日亏熔断) + OMS 状态机 + 对账"),
            "应急方案": ks.get("emergency_plan",
                             "一键 kill switch：撤销全部未完成委托并停止交易，人工确认后方可恢复"),
            "generated_at": self._ts(),
        }
        path = self._audit_dir / "program_trading_report.json"
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return info

    @staticmethod
    def _git_sha() -> str:
        try:
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=5)
            if sha.returncode == 0 and sha.stdout.strip():
                return sha.stdout.strip()
        except Exception:
            pass
        return "unknown"


# ---------------------------------------------------------------------------
# Live gate
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    approved: bool
    failed_checks: List[str] = field(default_factory=list)
    checked_at: str = ""

    def to_dict(self) -> Dict:
        return {"approved": self.approved, "failed_checks": self.failed_checks,
                "checked_at": self.checked_at}


class LiveGate:
    """LIVE requires ALL checks true AND live-capable strategy status.  Fails closed."""

    LIVE_CAPABLE = {StrategyStatus.LIVE, StrategyStatus.LIMITED_LIVE}

    @staticmethod
    def evaluate(strategy_status: StrategyStatus, execution_mode: ExecutionMode,
                 compliance_ok: bool, paper_passed: bool, shadow_passed: bool,
                 broker_ok: bool, reconcile_ok: bool, kill_switch_ready: bool) -> GateResult:
        failed: List[str] = []
        if strategy_status not in LiveGate.LIVE_CAPABLE:
            failed.append(f"strategy_status_not_live_capable:{strategy_status.value}")
        if execution_mode != ExecutionMode.LIVE:
            failed.append(f"execution_mode_not_live:{execution_mode.value}")
        if not compliance_ok:
            failed.append("compliance_not_confirmed")
        if not paper_passed:
            failed.append("paper_not_passed")
        if not shadow_passed:
            failed.append("shadow_not_passed")
        if not broker_ok:
            failed.append("broker_not_ok")
        if not reconcile_ok:
            failed.append("reconcile_not_ok")
        if not kill_switch_ready:
            failed.append("kill_switch_not_ready")
        return GateResult(approved=not failed, failed_checks=failed,
                          checked_at=datetime.now().astimezone().isoformat(timespec="seconds"))
