"""结构化因子（P0）纯文本格式化：GUI 面板与 Markdown 报告共用，保证两处口径一致、绝不伪造。

数据来源：
- 个股因子值：``candidate.structured_factors``（由 ``engine.p0_factors.P0FactorCollector`` 注入）。
- 降级原因：``source_state.structured_data[源名].{status, warnings}``（``engine.pipeline`` 原始状态）。

设计原则：有真实值才展示字段；缺失按真实 source_state 降级说明；后端未采集则显式提示。
不硬编码任何"源可用/不可用"结论——源可用性是环境特异的，必须随实时 source_state 呈现。
"""

from __future__ import annotations

from typing import Any


# (candidate.structured_factors 键, source_state.structured_data 键, 中文名)
STRUCTURED_FACTOR_META: list[tuple[str, str, str]] = [
    ("dragon_tiger", "dragon_tiger_daily", "龙虎榜"),
    ("hot_rank", "hot_rank", "热榜人气"),
    ("capital_flow_120d", "capital_flow_120d", "资金流"),
    ("lockup", "lockup", "解禁"),
    ("research", "research", "研报"),
    ("announcements", "announcements", "公告"),
    ("irm", "irm", "互动易"),
    ("northbound", "northbound", "北向资金"),
]

# 个股维度未覆盖、但需可见的全市场/事件因子（值仅在 source_state，无个股挂载）
MARKET_FACTOR_SOURCES: list[tuple[str, str]] = [
    ("limit_up_sentiment", "涨停情绪（全市场）"),
    ("dividend", "分红"),
]

# 已移除因子（东财独占，无免费替代源）
REMOVED_FACTORS: list[str] = ["margin", "block_trade", "holder_num"]

# 状态→展示语义（GUI 用于着色，报告用于标注）
STATUS_LABEL: dict[str, str] = {
    "ok": "已采集",
    "degraded": "部分降级",
    "unavailable": "源不可用",
    "skipped": "已跳过",
    "empty": "无数据",
}


def num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def degrade_text(status: str, warnings: list[str]) -> str:
    """按真实 source_state 状态/告警生成降级说明，不臆测原因。"""
    first = str(warnings[0]) if warnings else ""
    if status == "ok":
        # 源整体正常但本只候选无值（如未上榜/无公告/无解禁）
        return "该股无数据"
    if status == "skipped":
        return first or "已跳过（预算/限流/未启用）"
    if status == "unavailable":
        return f"源不可用{('：' + first) if first else ''}"
    if status == "empty":
        return "该股无数据"
    if status == "degraded":
        return f"部分降级{('：' + first) if first else ''}"
    return first or "未采集"


def generic_value_text(v: dict[str, Any]) -> str:
    """对未专门格式化的因子，按标量字段做简短摘要，绝不臆造字段名。"""
    parts: list[str] = []
    for key, val in v.items():
        if key in {"recent", "latest", "upcoming"} or isinstance(val, (list, dict)):
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            parts.append(f"{key} {val}")
        elif isinstance(val, str) and val.strip() and len(val) <= 30:
            parts.append(f"{key} {val.strip()}")
        if len(parts) >= 4:
            break
    return "；".join(parts) if parts else "已采集"


def factor_value_text(factor_key: str, v: dict[str, Any]) -> str:
    """逐因子的人类可读摘要（字段名严格对齐 engine.p0_factors 实际产出）。"""
    if factor_key == "dragon_tiger":
        reason = f" · {v.get('reason')}" if v.get("reason") else ""
        return (f"净买入 {num(v.get('net_buy_wan'))}万（买 {num(v.get('buy_wan'))}万/"
                f"卖 {num(v.get('sell_wan'))}万），换手 {num(v.get('turnover_pct'))}%{reason}")
    if factor_key == "hot_rank":
        return f"排名 #{v.get('rank')}（来源 {v.get('source', '-')}）"
    if factor_key == "capital_flow_120d":
        sum20 = num(v.get("sum_20d_main_net"))
        return (f"20日主力净流入 {round(sum20 / 1e8, 2)}亿，"
                f"20日正流入 {int(num(v.get('positive_days_20')))} 天（样本 {int(num(v.get('days')))} 日）")
    if factor_key == "lockup":
        return (f"90日内 {int(num(v.get('upcoming_count_90d')))} 批解禁，"
                f"合计占流通比 {num(v.get('upcoming_ratio_sum'))}%")
    if factor_key == "research":
        latest = v.get("latest") or {}
        return (f"共 {int(num(v.get('count')))} 篇，最新 {latest.get('org', '-')} "
                f"{latest.get('rating', '-')} {latest.get('date', '-')} · {latest.get('title', '-')}")
    if factor_key == "announcements":
        latest = v.get("latest") or {}
        risk_txt = f"，含风险公告 {int(num(v.get('risk_count')))} 条" if num(v.get("risk_count")) else ""
        return (f"共 {int(num(v.get('count')))} 条{risk_txt}，"
                f"最新 {latest.get('date', '-')} · {latest.get('title', '-')}")
    if factor_key == "irm":
        return f"共 {int(num(v.get('count')))} 条，已回复 {int(num(v.get('answered_count')))} 条"
    if factor_key == "northbound":
        # 同花顺市场级北向（东财 per-code 已移除）。
        # latest.net_buy 已是亿（元需 /1e8 为亿，但 ths_hexin 返回的已是亿元量级）。
        latest = v.get("latest") if isinstance(v.get("latest"), dict) else {}
        parts: list[str] = []
        net_yi = num(v.get("net_buy_yi") or v.get("net_yi"))
        if not net_yi and num(latest.get("net_buy")):
            net_yi = num(latest.get("net_buy"))
        if net_yi:
            parts.append(f"净流入 {round(net_yi, 2)}亿")
        if v.get("note"):
            parts.append(str(v.get("note"))[:60])
        return "、".join(parts) if parts else "市场级北向数据"
    return generic_value_text(v)


def format_per_code_factor(
    factor_key: str,
    source_key: str,
    cn: str,
    value: Any,
    state: dict[str, Any],
) -> tuple[str, str] | None:
    """返回 (展示文本, 状态)；返回 None 表示该源整体未运行、无需展示。"""
    if isinstance(value, dict) and value:
        return (f"{cn}：{factor_value_text(factor_key, value)}", "ok")
    if not state:
        return None
    status = str(state.get("status") or "").lower()
    warnings = state.get("warnings") or []
    return (f"{cn}：{degrade_text(status, warnings)}", status or "empty")


def format_market_factor(cn: str, src_state: dict[str, Any] | None) -> tuple[str, str] | None:
    """全市场/事件因子：无个股值，仅展示 source_state 状态。"""
    if not src_state:
        return None
    status = str((src_state or {}).get("status") or "").lower()
    warnings = (src_state or {}).get("warnings") or []
    text = f"{cn}：{degrade_text(status or 'skipped', warnings) if status != 'ok' else '已采集'}"
    return (text, status or "empty")


def structured_factor_lines(
    candidate: dict[str, Any],
    source_state: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    """返回结构化因子展示行 [(文本, 状态)]，顺序：全市场因子 → 个股因子 → 通用回退。

    不含 reasons/risk_notes/评分贡献，由调用方按需追加（GUI 与 MD 各自处理）。
    """
    sf = candidate.get("structured_factors") or {}
    if not isinstance(sf, dict):
        sf = {}
    state = source_state if isinstance(source_state, dict) else {}

    out: list[tuple[str, str]] = []
    for source_key, cn in MARKET_FACTOR_SOURCES:
        row = format_market_factor(cn, state.get(source_key))
        if row:
            out.append(row)

    shown: set[str] = set()
    for factor_key, source_key, cn in STRUCTURED_FACTOR_META:
        row = format_per_code_factor(factor_key, source_key, cn, sf.get(factor_key), state.get(source_key) or {})
        if row is None:
            continue
        shown.add(factor_key)
        out.append(row)

    for extra_key, extra_val in sf.items():
        if extra_key in shown or extra_key in {"source_coverage", "reasons", "risk_notes"}:
            continue
        if isinstance(extra_val, dict) and extra_val:
            out.append((f"{extra_key}：{generic_value_text(extra_val)}", "ok"))
    return out


def has_any_structured_signal(candidate: dict[str, Any]) -> bool:
    """该候选是否存在任何结构化因子值（用于报告决定是否输出该小节）。"""
    sf = candidate.get("structured_factors") or {}
    if not isinstance(sf, dict):
        return False
    return any(
        isinstance(v, dict) and v
        for k, v in sf.items()
        if k not in {"source_coverage", "reasons", "risk_notes"}
    )
