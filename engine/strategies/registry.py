"""SQLite-backed strategy registry: lifecycle state machine + history.

Tables
------
strategy_registry          : current manifest (json) + status per strategy.
strategy_registry_history  : append-only audit of every status change.

Lifecycle edges (everything else raises IllegalStrategyTransition):
  idea -> research -> validated -> paper -> shadow_live -> limited_live -> live
  any non-terminal -> suspended;  suspended -> {paper, retired};  retired: none.

Transitions INTO paper / shadow_live / limited_live / live must pass
gates.evaluate_promotion with the checks evidence.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from ..contracts import StrategyStatus, EXECUTABLE_STRATEGY_STATUS
from .gates import PromotionContext, evaluate_promotion
from .manifest import ALLOWED_INITIAL_STATUS, StrategyManifest


class IllegalStrategyTransition(ValueError):
    pass


class GateFailed(IllegalStrategyTransition):
    pass


# P6-003 auto-suspend triggers.
AUTO_SUSPEND_TRIGGERS = {
    "live_deviates_from_oos",
    "drawdown_breach",
    "data_source_changed",
    "feature_schema_changed",
    "model_drift",
    "broker_execution_error",
    "compliance_lost",
    "daily_loss_limit",
}

_CHAIN = [
    StrategyStatus.IDEA,
    StrategyStatus.RESEARCH,
    StrategyStatus.VALIDATED,
    StrategyStatus.PAPER,
    StrategyStatus.SHADOW_LIVE,
    StrategyStatus.LIMITED_LIVE,
    StrategyStatus.LIVE,
]

_ALLOWED: Dict[StrategyStatus, set] = {s: set() for s in _CHAIN}
for _a, _b in zip(_CHAIN, _CHAIN[1:]):
    _ALLOWED[_a].add(_b)
for _s in _CHAIN:
    _ALLOWED[_s].add(StrategyStatus.SUSPENDED)  # any non-terminal -> suspended
_ALLOWED[StrategyStatus.SUSPENDED] = {StrategyStatus.PAPER, StrategyStatus.RETIRED}
_ALLOWED[StrategyStatus.RETIRED] = set()

_GATED_TARGETS = {
    StrategyStatus.PAPER,
    StrategyStatus.SHADOW_LIVE,
    StrategyStatus.LIMITED_LIVE,
    StrategyStatus.LIVE,
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_registry (
    strategy_id   TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    status        TEXT NOT NULL,
    version       TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS strategy_registry_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    operator    TEXT NOT NULL,
    reason      TEXT NOT NULL DEFAULT '',
    checks_json TEXT,
    ts          TEXT NOT NULL
);
"""


def _identity_json(manifest: StrategyManifest) -> str:
    """Canonical manifest identity: timestamps excluded (they are stamped
    by the registry, not by the author of the manifest)."""
    d = manifest.to_dict()
    d["created_at"] = ""
    d["updated_at"] = ""
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


class StrategyRegistry:
    def __init__(self, db_path: Union[str, Path] = "data/execution.db") -> None:
        self.db_path = str(db_path)
        parent = Path(self.db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        # Per-call connection: safe across threads, no shared handle.
        return sqlite3.connect(self.db_path, timeout=10)

    # ------------------------------------------------------------------ #
    def register(self, manifest: StrategyManifest) -> StrategyManifest:
        """Register a new strategy.  Initial status must be idea/research/
        validated.  Registering the exact same manifest again is an idempotent
        no-op; a conflicting manifest raises ValueError."""
        if manifest.approval_state not in ALLOWED_INITIAL_STATUS:
            raise ValueError(
                f"initial approval_state must be one of "
                f"{sorted(s.value for s in ALLOWED_INITIAL_STATUS)}, "
                f"got '{manifest.approval_state.value}'; register lower, promote via transition()"
            )
        existing = self._raw_row(manifest.strategy_id)
        if existing is not None:
            stored = StrategyManifest.from_json(existing["manifest_json"])
            if _identity_json(stored) == _identity_json(manifest):
                return manifest  # idempotent re-registration
            raise ValueError(
                f"strategy '{manifest.strategy_id}' already registered with a "
                f"different manifest; bump version instead of overwriting"
            )
        now = _now_iso()
        if not manifest.created_at:
            manifest.created_at = now
        manifest.updated_at = now
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO strategy_registry "
                "(strategy_id, manifest_json, status, version, updated_at) VALUES (?,?,?,?,?)",
                (manifest.strategy_id, manifest.to_json(),
                 manifest.approval_state.value, manifest.version, now),
            )
            conn.execute(
                "INSERT INTO strategy_registry_history "
                "(strategy_id, from_status, to_status, operator, reason, checks_json, ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (manifest.strategy_id, None, manifest.approval_state.value,
                 "registry:register", "initial registration", None, now),
            )
        return manifest

    # ------------------------------------------------------------------ #
    def _raw_row(self, strategy_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM strategy_registry WHERE strategy_id = ?", (strategy_id,)
            )
            return cur.fetchone()

    def get(self, strategy_id: str) -> StrategyManifest:
        row = self._raw_row(strategy_id)
        if row is None:
            raise KeyError(f"strategy '{strategy_id}' not registered")
        return StrategyManifest.from_json(row["manifest_json"])

    def list(self, status: Union[StrategyStatus, str, None] = None) -> List[StrategyManifest]:
        query = "SELECT manifest_json FROM strategy_registry"
        params: tuple = ()
        if status is not None:
            status_val = status.value if isinstance(status, StrategyStatus) else str(status)
            query += " WHERE status = ?"
            params = (status_val,)
        query += " ORDER BY strategy_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [StrategyManifest.from_json(r[0]) for r in rows]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_promotion_context(checks) -> Optional[PromotionContext]:
        if checks is None:
            return None
        if isinstance(checks, PromotionContext):
            return checks
        if isinstance(checks, dict):
            return PromotionContext.from_dict(checks)
        raise TypeError("checks must be a PromotionContext, a dict, or None")

    def transition(
        self,
        strategy_id: str,
        new_status: Union[StrategyStatus, str],
        operator: str,
        reason: str,
        checks: Union[PromotionContext, Dict, None] = None,
    ) -> StrategyManifest:
        manifest = self.get(strategy_id)
        current = manifest.approval_state
        target = new_status if isinstance(new_status, StrategyStatus) else StrategyStatus(new_status)

        if target not in _ALLOWED.get(current, set()):
            raise IllegalStrategyTransition(
                f"illegal strategy transition {current.value} -> {target.value} "
                f"for '{strategy_id}'"
            )

        if target in _GATED_TARGETS:
            ctx = self._as_promotion_context(checks)
            if ctx is None:
                raise GateFailed(
                    f"transition into '{target.value}' requires gate checks "
                    f"(PromotionContext), got none"
                )
            result = evaluate_promotion(ctx, current, target)
            if not result.approved:
                raise GateFailed(
                    f"promotion gate failed for {current.value} -> {target.value}: "
                    f"{', '.join(result.failed_checks)}"
                )

        manifest.approval_state = target
        manifest.updated_at = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE strategy_registry SET manifest_json = ?, status = ?, updated_at = ? "
                "WHERE strategy_id = ?",
                (manifest.to_json(), target.value, manifest.updated_at, strategy_id),
            )
            conn.execute(
                "INSERT INTO strategy_registry_history "
                "(strategy_id, from_status, to_status, operator, reason, checks_json, ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (strategy_id, current.value, target.value, operator, reason,
                 json.dumps(checks.to_dict(), ensure_ascii=False) if isinstance(checks, PromotionContext)
                 else (json.dumps(checks, ensure_ascii=False, default=str) if checks else None),
                 manifest.updated_at),
            )
        return manifest

    # ------------------------------------------------------------------ #
    def auto_suspend(
        self, strategy_id: str, trigger: str, detail: str = ""
    ) -> Optional[StrategyManifest]:
        """Auto-suspend on a P6-003 trigger.  Idempotent when already
        suspended (returns None); raises IllegalStrategyTransition when the
        current status cannot be left (e.g. retired)."""
        if trigger not in AUTO_SUSPEND_TRIGGERS:
            raise ValueError(
                f"unknown auto-suspend trigger '{trigger}'; "
                f"expected one of {sorted(AUTO_SUSPEND_TRIGGERS)}"
            )
        current = self.get(strategy_id).approval_state
        if current == StrategyStatus.SUSPENDED:
            return None
        reason = f"auto_suspend:{trigger}" + (f" {detail}" if detail else "")
        return self.transition(
            strategy_id, StrategyStatus.SUSPENDED,
            operator="risk_engine:auto_suspend", reason=reason,
        )

    # ------------------------------------------------------------------ #
    def executable_strategy_ids(self) -> List[str]:
        status_vals = {s.value for s in EXECUTABLE_STRATEGY_STATUS}
        marks = ",".join("?" for _ in status_vals)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT strategy_id FROM strategy_registry WHERE status IN ({marks}) "
                f"ORDER BY strategy_id",
                tuple(sorted(status_vals)),
            ).fetchall()
        return [r[0] for r in rows]

    def history(self, strategy_id: str) -> Iterable[sqlite3.Row]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                "SELECT * FROM strategy_registry_history WHERE strategy_id = ? ORDER BY id",
                (strategy_id,),
            ).fetchall()


# ---------------------------------------------------------------------- #
def seed_registry(
    registry: StrategyRegistry, manifests_dir: Union[str, Path, None] = None
) -> List[StrategyManifest]:
    """Register every manifest JSON shipped in engine/strategies/manifests/.
    Idempotent: re-seeding an identical manifest is a no-op."""
    directory = Path(manifests_dir) if manifests_dir else Path(__file__).parent / "manifests"
    seeded: List[StrategyManifest] = []
    for path in sorted(directory.glob("*.json")):
        manifest = StrategyManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        registry.register(manifest)
        seeded.append(manifest)
    return seeded
