"""StrategyManifest: the auditable identity card of a strategy.

A manifest pins together code, data, features, model, parameters, ranges,
execution assumptions and evidence.  Nothing may trade on a strategy whose
manifest is not registered and whose approval_state is executable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import StrategyStatus

ALLOWED_INITIAL_STATUS = {
    StrategyStatus.IDEA,
    StrategyStatus.RESEARCH,
    StrategyStatus.VALIDATED,
}


def code_sha_for_module(module_path: str) -> str:
    """sha256 hex digest of the strategy implementation file bytes."""
    with open(module_path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@dataclass
class StrategyManifest:
    strategy_id: str
    version: str
    code_sha: str
    data_snapshot: str
    features: List[str] = field(default_factory=list)
    model: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    train_range: Optional[str] = None
    validation_range: Optional[str] = None
    test_range: Optional[str] = None
    execution_assumptions: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    approval_state: StrategyStatus = StrategyStatus.IDEA
    created_at: str = ""
    updated_at: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["approval_state"] = self.approval_state.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyManifest":
        return cls(
            strategy_id=str(d["strategy_id"]),
            version=str(d.get("version", "0.0")),
            code_sha=str(d.get("code_sha", "")),
            data_snapshot=str(d.get("data_snapshot", "")),
            features=list(d.get("features", [])),
            model=str(d.get("model", "")),
            params=dict(d.get("params", {})),
            train_range=d.get("train_range"),
            validation_range=d.get("validation_range"),
            test_range=d.get("test_range"),
            execution_assumptions=dict(d.get("execution_assumptions", {})),
            metrics=dict(d.get("metrics", {})),
            approval_state=StrategyStatus(d.get("approval_state", "idea")),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            evidence_refs=list(d.get("evidence_refs", [])),
            notes=str(d.get("notes", "")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "StrategyManifest":
        return cls.from_dict(json.loads(raw))
