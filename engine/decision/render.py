"""极简用户输出（Phase 10）。

默认视图只展示：市场结论 + Top20 表。详细证据按需展开（/pangu why）。
禁止第一屏输出全部原始因子/source_status/策略池/OMS 状态机。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from engine.decision.contracts import DecisionAction, DecisionRun, StockDecision


def render_top20(run: DecisionRun) -> str:
    """/pangu 默认输出（任务书 §24 格式）。"""
    tset = run.recommendations
    lines = [
        f"分析时点：{run.asof_timestamp}",
        f"目标交易日：{run.execution_date}",
    ]
    if tset:
        lines.append(f"市场状态：{tset.market_conclusion or tset.market_status.value}")
        lines.append(f"数据状态：{'正常' if tset.data_status == 'ok' else tset.data_status.upper()}")
        lines.append("")
        lines.append(f"Top {len(tset.decisions)}"
                     + (f"（{tset.note}）" if tset.note else ""))
        lines.append(_table(tset.decisions))
        for w in run.warnings[:3]:
            lines.append(f"⚠ {w}")
    else:
        lines.append("数据状态：DEGRADED")
        lines.append("本次无决策候选：" + "；".join(run.blocked_reasons) if
                     run.blocked_reasons else "本次无决策候选")
    return "\n".join(lines)


def _table(decisions: List[StockDecision]) -> str:
    headers = ["#", "股票", "决策", "入场", "持有", "止损", "核心理由", "最大风险"]
    rows = []
    for d in decisions:
        entry = (f"{d.entry_zone[0]:.2f}-{d.entry_zone[1]:.2f}"
                 if d.entry_zone else (f"{d.entry_condition[:12]}…" if d.entry_condition else "--"))
        hold = (f"{d.expected_holding_days}d"
                if d.expected_holding_days else "--")
        stop = f"{d.stop_loss:.2f}" if d.stop_loss else "--"
        reason = (d.reasons[0] if d.reasons else
                  (d.primary_strategy or "--"))
        risk = d.risks[0] if d.risks else _top_risk(d)
        rows.append([str(d.rank), f"{d.code} {d.name}", d.decision.value,
                     entry, hold, stop, reason, risk])
    widths = [max(len(r[i]) if i < len(r) else 0 for r in [headers] + rows)
              for i in range(len(headers))]
    out = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
           "-+-".join("-" * w for w in widths)]
    for r in rows:
        out.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def _top_risk(d: StockDecision) -> str:
    neg = [e for e in d.event_evidence if "negative" in e or "risk" in e]
    if neg:
        return "存在风险事件证据"
    if d.score_breakdown.get("vol_penalty"):
        return "波动偏高"
    return "--"


def render_single(run: DecisionRun, code: str) -> str:
    """/pangu 代码：单票 7 要素。"""
    code6 = str(code).split(".")[-1].zfill(6)
    d = next((x for x in (run.recommendations.decisions if run.recommendations else [])
              if x.code == code6), None)
    if d is None:
        return f"{code}：本次运行未覆盖该票（未进入 Top{run.recommendations.counts()} 决策集）"
    lines = [
        f"{d.code} {d.name}  决策：{d.decision.value}  分数：{d.score}",
        f"下一交易日计划：{d.execution_date} 开盘进入入场区 {d.entry_zone}；失效：{d.invalid_condition}",
        f"建议持有：{d.expected_holding_days} 个交易日（区间 {d.holding_range}）",
        "最重要理由：",
        *[f"  {i+1}. {r}" for i, r in enumerate(d.reasons[:3])],
        "最大风险：",
        *[f"  {i+1}. {r}" for i, r in enumerate(d.risks[:2])],
        f"止损：{d.stop_loss}  目标：{d.target_zone}",
        "退出条件：" + "；".join(d.exit_conditions[:4]),
    ]
    return "\n".join(lines)


def render_why(run_dict: Dict[str, Any], decision_dict: Dict[str, Any]) -> str:
    """/pangu why：完整证据链（按需展开）。"""
    d = decision_dict
    lines = [
        f"运行 {run_dict.get('run_id')}（asof={run_dict.get('asof_timestamp')}，"
        f"模型={run_dict.get('model_version')}，证据版本={run_dict.get('evidence_version')}）",
        f"数据新鲜度：{json_s(run_dict.get('data_freshness'))}",
        f"来源健康：{json_s(run_dict.get('source_health'))}",
        f"分数分解：{json_s(d.get('score_breakdown'))}",
        f"因子证据：{json_s(d.get('factor_evidence'))}",
        f"事件证据：{json_s(d.get('event_evidence'))}",
        f"市场证据：{json_s(d.get('market_evidence'))}",
        f"流动性证据：{json_s(d.get('liquidity_evidence'))}",
        f"全部退出条件：{json_s(d.get('exit_conditions'))}",
        f"置信：{d.get('confidence')}（{d.get('confidence_type')}）",
    ]
    return "\n".join(lines)


def render_status(status: Dict[str, Any]) -> str:
    lines = [
        f"行情/新闻/公告更新时间与状态（asof={status.get('asof')}）：",
    ]
    for src, meta in (status.get("sources") or {}).items():
        lines.append(f"  {src}: fetched={meta.get('fetched_at')} "
                     f"age={meta.get('age_seconds')}s stale={meta.get('stale')} "
                     f"quality={meta.get('quality')}")
    lines.append(f"PIT 档案最新日：{(status.get('pit_archive') or {}).get('max_date')}")
    lines.append(f"MCP：{(status.get('mcp') or {}).get('transport')}")
    return "\n".join(lines)


def json_s(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
