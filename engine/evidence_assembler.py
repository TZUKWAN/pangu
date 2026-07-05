"""Assemble unified CandidateEvidence from pipeline artifacts."""

from __future__ import annotations

from typing import Any

from .evidence import CandidateEvidence, decision_from_item


class EvidenceAssembler:
    def assemble(
        self,
        *,
        candidates: list[dict[str, Any]],
        final_recommendations: list[dict[str, Any]],
        watchlist: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        strategy_signals: dict[str, list[dict[str, Any]]] | None = None,
        source_status: dict[str, Any] | None = None,
        news_evidence: dict[str, dict[str, Any]] | None = None,
        market_phase: dict[str, Any] | None = None,
        data_quality: str = "unknown",
    ) -> dict[str, dict[str, Any]]:
        source_status = source_status or {}
        news_evidence = news_evidence or {}
        strategy_by_code = self._strategy_lookup(strategy_signals or {})
        decision_items: dict[str, dict[str, Any]] = {}
        for status, items in (("final", final_recommendations), ("watch", watchlist), ("rejected", rejected)):
            for item in items:
                code = str(item.get("code") or "")
                if code and code not in decision_items:
                    merged = dict(item)
                    merged.setdefault("gate_status", status)
                    decision_items[code] = merged

        evidence: dict[str, dict[str, Any]] = {}
        all_items = []
        seen: set[str] = set()
        for group in (final_recommendations, watchlist, rejected, candidates):
            for item in group:
                code = str(item.get("code") or "")
                if code and code not in seen:
                    all_items.append(item)
                    seen.add(code)

        for item in all_items:
            code = str(item.get("code") or "")
            if not code:
                continue
            decision_item = decision_items.get(code, item)
            ev = CandidateEvidence(
                code=code,
                name=str(item.get("name") or decision_item.get("name") or ""),
                strategy=self._strategy_evidence(item, strategy_by_code.get(code), market_phase or {}),
                data_quality=self._data_quality_evidence(source_status, data_quality),
                price_action=self._price_action(item),
                volume_audit=dict(item.get("volume_audit") or {}),
                liquidity=self._liquidity(item),
                news_evidence=dict(item.get("news_evidence") or news_evidence.get(code) or {}),
                anti_chase=dict(item.get("anti_chase") or {}),
                entry_plan=dict(item.get("entry_plan") or {}),
                decision=decision_from_item(decision_item),
                raw={
                    "score": item.get("score"),
                    "rps": item.get("rps"),
                    "rps_mode": item.get("rps_mode"),
                    "fund_flow_status": item.get("fund_flow_status"),
                    "xuanwu": item.get("xuanwu"),
                },
            )
            evidence[code] = ev.to_dict()
        return evidence

    def _strategy_lookup(self, strategy_signals: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for strategy_name, signals in strategy_signals.items():
            for sig in signals:
                code = str(sig.get("code") or "")
                if not code:
                    continue
                if code not in out or float(sig.get("score") or 0) > float(out[code].get("score") or 0):
                    out[code] = {"strategy_name": strategy_name, **sig}
        return out

    def _strategy_evidence(self, item: dict[str, Any], signal: dict[str, Any] | None, market_phase: dict[str, Any]) -> dict[str, Any]:
        signal = signal or {}
        return {
            "strategy_name": item.get("strategy_name") or signal.get("strategy_name") or item.get("strategy"),
            "theme": item.get("theme") or signal.get("theme") or item.get("board"),
            "board": item.get("board") or signal.get("board"),
            "role": item.get("role") or signal.get("role"),
            "score": item.get("score") or signal.get("score"),
            "market_phase": market_phase.get("market_phase"),
            "is_trend_only": (item.get("strategy_name") == "trend_supplement"),
        }

    def _data_quality_evidence(self, source_status: dict[str, Any], data_quality: str) -> dict[str, Any]:
        missing_fields: dict[str, Any] = {}
        for name, status in source_status.items():
            if isinstance(status, dict) and status.get("field_quality"):
                missing_fields[name] = {
                    k: v for k, v in status["field_quality"].items() if v != "ok"
                }
        return {
            "overall": data_quality,
            "sources": source_status,
            "missing_fields": missing_fields,
        }

    def _price_action(self, item: dict[str, Any]) -> dict[str, Any]:
        technical = item.get("technical") or {}
        return {
            "close": item.get("close"),
            "pct_change": item.get("pct_change"),
            "rps": item.get("rps"),
            "rps_mode": item.get("rps_mode"),
            "ma": technical.get("ma") or {},
            "trend_windows": technical.get("trend_windows") or {},
        }

    def _liquidity(self, item: dict[str, Any]) -> dict[str, Any]:
        turnover_status = item.get("turnover_status")
        turnover_missing = item.get("turnover_missing")
        if turnover_status is None:
            turnover_status = "missing" if turnover_missing else "ok"
        return {
            "turnover_rate": item.get("turnover_rate"),
            "turnover_status": turnover_status,
            "turnover_missing": bool(turnover_missing) if turnover_missing is not None else turnover_status == "missing",
            "circ_mv_yi": item.get("circ_mv_yi"),
            "amount": item.get("amount") or item.get("成交额"),
            "fund_flow_status": item.get("fund_flow_status"),
            "fund_flow_net": item.get("fund_flow_net"),
        }
