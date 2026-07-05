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
        # 一等输入（可选）：让证据层直接消费 trend / guard / technical / entry_exit / source_quality / fund_flow
        trend_candidates: list[dict[str, Any]] | None = None,
        guarded: dict[str, Any] | None = None,
        technical_snapshot: dict[str, dict[str, Any]] | None = None,
        entry_exit: dict[str, dict[str, Any]] | None = None,
        source_quality: dict[str, Any] | None = None,
        fund_flow: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        source_status = source_status or {}
        news_evidence = news_evidence or {}
        technical_snapshot = technical_snapshot or {}
        entry_exit = entry_exit or {}
        strategy_by_code = self._strategy_lookup(strategy_signals or {})

        # trend_candidates → 按 code 索引（标记是否为 trend-only 补充）
        trend_by_code: dict[str, dict[str, Any]] = {}
        for tc in (trend_candidates or []):
            code = str(tc.get("code") or "")
            if code:
                trend_by_code[code] = tc

        # guarded → kept/watch/rejected code 集合
        guarded = guarded or {}
        guard_kept = {str(c) for c in guarded.get("kept_codes", [])}
        guard_watch = {str(c) for c in guarded.get("watch_codes", [])}
        guard_rejected = {str(c) for c in guarded.get("rejected_codes", [])}

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
                strategy=self._strategy_evidence(item, strategy_by_code.get(code), market_phase or {}, trend_by_code.get(code)),
                data_quality=self._data_quality_evidence(source_status, data_quality, source_quality),
                price_action=self._price_action(item, technical_snapshot.get(code)),
                volume_audit=dict(item.get("volume_audit") or {}),
                liquidity=self._liquidity(item, fund_flow),
                news_evidence=dict(item.get("news_evidence") or news_evidence.get(code) or {}),
                anti_chase=dict(item.get("anti_chase") or {}),
                entry_plan=dict(item.get("entry_plan") or entry_exit.get(code) or {}),
                decision=decision_from_item(decision_item),
                raw=self._raw(
                    item, decision_item,
                    trend_by_code.get(code),
                    guarded_state=self._guarded_state(code, guard_kept, guard_watch, guard_rejected),
                    entry_exit_snapshot=entry_exit.get(code),
                    fund_flow=fund_flow,
                ),
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

    def _strategy_evidence(
        self,
        item: dict[str, Any],
        signal: dict[str, Any] | None,
        market_phase: dict[str, Any],
        trend_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        signal = signal or {}
        trend_candidate = trend_candidate or {}
        is_trend_only = (item.get("strategy_name") == "trend_supplement") or (not signal and bool(trend_candidate))
        return {
            "strategy_name": item.get("strategy_name") or signal.get("strategy_name") or item.get("strategy"),
            "theme": item.get("theme") or signal.get("theme") or item.get("board") or trend_candidate.get("board"),
            "board": item.get("board") or signal.get("board") or trend_candidate.get("board"),
            "role": item.get("role") or signal.get("role") or trend_candidate.get("role"),
            "score": item.get("score") or signal.get("score") or trend_candidate.get("score"),
            "market_phase": market_phase.get("market_phase"),
            "is_trend_only": is_trend_only,
            "trigger_reason": signal.get("trigger_reason") or trend_candidate.get("trigger_reason"),
        }

    def _data_quality_evidence(
        self,
        source_status: dict[str, Any],
        data_quality: str,
        source_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        missing_fields: dict[str, Any] = {}
        for name, status in source_status.items():
            if isinstance(status, dict) and status.get("field_quality"):
                missing_fields[name] = {
                    k: v for k, v in status["field_quality"].items() if v != "ok"
                }
        out = {
            "overall": data_quality,
            "sources": source_status,
            "missing_fields": missing_fields,
        }
        if source_quality:
            out["source_quality"] = source_quality
        return out

    def _price_action(
        self,
        item: dict[str, Any],
        technical_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        technical = item.get("technical") or technical_snapshot or {}
        return {
            "close": item.get("close"),
            "pct_change": item.get("pct_change"),
            "rps": item.get("rps"),
            "rps_mode": item.get("rps_mode"),
            "ma": technical.get("ma") or {},
            "trend_windows": technical.get("trend_windows") or {},
            "structure": technical.get("structure") or {},
        }

    def _liquidity(
        self,
        item: dict[str, Any],
        fund_flow: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        turnover_status = item.get("turnover_status")
        turnover_missing = item.get("turnover_missing")
        if turnover_status is None:
            turnover_status = "missing" if turnover_missing else "ok"
        out = {
            "turnover_rate": item.get("turnover_rate"),
            "turnover_status": turnover_status,
            "turnover_missing": bool(turnover_missing) if turnover_missing is not None else turnover_status == "missing",
            "circ_mv_yi": item.get("circ_mv_yi"),
            "amount": item.get("amount") or item.get("成交额"),
            "fund_flow_status": item.get("fund_flow_status"),
            "fund_flow_net": item.get("fund_flow_net"),
        }
        if fund_flow:
            # 全局资金流状态（如 available/snapshot_only/unavailable）
            out["fund_flow_global"] = fund_flow
        return out

    def _guarded_state(self, code: str, kept: set[str], watch: set[str], rejected: set[str]) -> str:
        if code in rejected:
            return "rejected"
        if code in watch:
            return "watch"
        if code in kept:
            return "kept"
        return "unknown"

    def _raw(
        self,
        item: dict[str, Any],
        decision_item: dict[str, Any],
        trend_candidate: dict[str, Any] | None,
        guarded_state: str,
        entry_exit_snapshot: dict[str, Any] | None,
        fund_flow: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raw = {
            "score": item.get("score"),
            "rps": item.get("rps"),
            "rps_mode": item.get("rps_mode"),
            "fund_flow_status": item.get("fund_flow_status"),
            "xuanwu": item.get("xuanwu"),
            "guarded_state": guarded_state,
        }
        if trend_candidate:
            raw["trend_candidate"] = {
                "code": trend_candidate.get("code"),
                "board": trend_candidate.get("board"),
                "trigger_reason": trend_candidate.get("trigger_reason"),
                "score": trend_candidate.get("score"),
            }
        if entry_exit_snapshot:
            raw["entry_exit_snapshot"] = {
                k: entry_exit_snapshot.get(k)
                for k in ("entry_style", "stop_loss", "risk_reward_ratio")
                if k in entry_exit_snapshot
            }
        if fund_flow:
            raw["fund_flow_global"] = fund_flow
        return raw
