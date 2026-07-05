"""把 PipelineResult 渲染成 Markdown 简报。

这份简报会被 LLM Agent 读作上下文，做最终的综合分析；
也可以直接给人看（命令行 / 后续前端展示）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .pipeline import PipelineResult

logger = logging.getLogger("pangu.report")
from .structured_format import structured_factor_lines, has_any_structured_signal


# 结构化因子状态 → Markdown 前缀标记（与 GUI 徽标语义一致，绝不把降级渲染成可用）
_STRUCTURED_TAG: dict[str, str] = {
    "ok": "✅", "degraded": "⚠️", "unavailable": "✖️", "skipped": "⏭", "empty": "·",
}


def render_markdown(result: PipelineResult) -> str:
    """渲染 Markdown 简报。"""
    s = result.sentiment
    # P0 结构化源原始状态（pipeline.source_status.structured_data，dict 形态）；逐候选降级原因由此呈现
    src_state = getattr(result, "source_status", None) or getattr(result, "source_state", None) or {}
    structured_src_state = (src_state or {}).get("structured_data") or {}
    if not isinstance(structured_src_state, dict):
        structured_src_state = {}
    components = s.get("components", {})
    lines: list[str] = [
        f"# 盘古选股简报 · {result.date}",
        "",
    ]

    # 数据质量显式标注
    if result.data_quality in ("failed", "degraded"):
        lines += [
            "> 🚨 **数据降级诊断报告**：不作为正式选股依据。",
            f"> 质量等级：`{result.data_quality}`",
        ]
        if result.block_reasons:
            lines.append(f"> 阻断原因：{'；'.join(result.block_reasons)}")
        lines.append("")

    lines += [
        f"> 数据更新于 {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> ⚠️ 本简报为决策辅助，不构成投资建议，盈亏自负。",
        "",
        "## 一、情绪温度计",
        "",
        f"**温度 {s.get('temperature')} / 100 — {s.get('posture')}**",
        "",
        f"{s.get('advice')}",
        "",
        "| 分项 | 数值 | 得分 |",
        "|------|------|------|",
        f"| 涨停家数 | {components.get('limit_up_count', 0)} | {components.get('limit_up_score', 0)} |",
        f"| 最高连板 | {components.get('consecutive_height', 0)}板 | {components.get('consecutive_score', 0)} |",
        f"| 炸板率 | {components.get('broke_rate', 0)*100:.1f}% | {components.get('broke_rate_score', 0)} |",
        f"| 跌停家数 | {components.get('limit_down_count', 0)} | {components.get('limit_down_score', 0)} |",
        f"| 涨/跌家数 | {components.get('advance', 0)}/{components.get('decline', 0)} | {components.get('advance_decline_score', 0)} |",
        "",
    ]

    # 二、热门板块
    if result.boards:
        lines += ["## 二、热门概念板块（涨幅+资金双因子排名）", "",
                  "| 板块 | 涨跌幅% | 主力净流入(万) | 综合分 |", "|------|---------|---------------|--------|"]
        for b in result.boards[:8]:
            lines.append(f"| {b['name']} | {b['pct']} | {b['fund_net_wan']} | {b['score']} |")
        lines.append("")

    # 三、候选股
    if result.candidates:
        lines += ["## 三、候选股池（情绪+趋势筛选，已过量化护栏）", "",
                  "| 代码 | 名称 | 板块 | 现价 | 涨跌% | RPS | 资金连流 | 推荐度 | 等级 | 上涨概率 | 预测涨幅 | 理由 |",
                  "|------|------|------|------|-------|-----|---------|--------|------|---------|---------|------|"]
        for c in result.candidates:
            reasons = "；".join(c.get("reasons", []))
            rec = c.get("recommend", {})
            grade = rec.get("grade", "-")
            score = rec.get("recommend_score", 0)
            up_prob = rec.get("up_prob", 0)
            target = rec.get("target_pct", [0, 0])
            calibrated = "✓" if rec.get("calibrated") else ""
            lines.append(
                f"| {c['code']} | {c['name']} | {c['board']} | {c['close']} | "
                f"{c['pct_change']} | {c['rps']:.0f} | {c['fund_inflow_days']}日 | "
                f"{score:.1f} | {grade} | {up_prob:.0f}%{calibrated} | {target[0]}-{target[1]}% | {rec.get('tag', reasons)} |"
            )
        lines += ["",
                  "### 个股入选理由明细", ""]
        # P0 结构化源整体可用性（source_state.structured_data 聚合）：不渲染失败源为可用，逐源列出降级
        if structured_src_state:
            _src_states = {
                k: (v.get("status") if isinstance(v, dict) else None)
                for k, v in structured_src_state.items() if k != "summary"
            }
            if _src_states:
                _ok_n = sum(1 for st in _src_states.values() if st == "ok")
                _not_ok = {k: st for k, st in _src_states.items() if st and st != "ok"}
                lines.append(
                    f"- 结构化源（P0）：{_ok_n}/{len(_src_states)} 源正常"
                    + (f"；降级/不可用 → {'，'.join(f'{k}={st}' for k, st in _not_ok.items())}" if _not_ok else "（全部正常）")
                )
                lines.append("")
        for c in result.candidates:
            rec = c.get("recommend", {})
            lines.append(f"**{c['name']}（{c['code']}）** — {c['board']}，流通市值 {c['circ_mv_yi']:.0f}亿，换手 {c['turnover_rate']}%")
            if rec:
                lines.append(f"- 推荐度：{rec.get('recommend_score', 0):.1f}  等级：{rec.get('grade', '-')}  上涨概率：{rec.get('up_prob', 0):.0f}%")
                lines.append(f"- 预测涨幅：{rec.get('target_pct', [0,0])[0]}-{rec.get('target_pct', [0,0])[1]}%  盈亏比：{rec.get('risk_reward_ratio', 0):.1f}")
                lines.append(f"- 简短理由：{rec.get('tag', '')}")
            for r in c.get("reasons", []):
                lines.append(f"- {r}")
            # 买卖点
            ee = c.get("entry_exit", {})
            if ee:
                bp = ee.get("buy_points", [])
                primary = next((b for b in bp if b.get("is_primary")), bp[0] if bp else None)
                if primary:
                    lines.append(f"- 主买点：{primary['price']:.2f}（{primary['type']}）{primary.get('condition', '')}")
                sl = ee.get("stop_loss", {})
                if sl:
                    lines.append(f"- 止损：{sl.get('price', '-')}（{sl.get('method', '-')}）")
                tps = ee.get("take_profit", [])
                if tps:
                    lines.append(f"- 止盈：{tps[0]['price']:.2f}（{tps[0]['method']}）")
                pos = ee.get("position", {})
                if pos:
                    lines.append(f"- 仓位建议：{pos.get('shares', 0)}股  风险{pos.get('risk_pct', 0):.2f}%")
            # 结构化因子（P0）：逐项真实字段 + source_state 降级原因，绝不伪造；无值则不输出该小节
            _sf_rows = structured_factor_lines(c, structured_src_state)
            _sf = c.get("structured_factors") or {}
            _sf_reasons = [str(r) for r in (_sf.get("reasons") or []) if str(r).strip()]
            _sf_risks = [str(r) for r in (_sf.get("risk_notes") or []) if str(r).strip()]
            if _sf_rows or _sf_reasons or _sf_risks or has_any_structured_signal(c):
                lines.append("- 结构化因子（P0）：")
                for _txt, _st in _sf_rows:
                    lines.append(f"  - {_STRUCTURED_TAG.get(_st, '·')} {_txt}")
                if _sf_reasons:
                    lines.append(f"  - 加分因子：{'；'.join(_sf_reasons)}")
                if _sf_risks:
                    lines.append(f"  - 结构化风险：{'；'.join(_sf_risks)}")
                _bd = (c.get("recommend") or {}).get("score_breakdown") or {}
                _sig = _bd.get("structured_signal")
                if _sig is not None:
                    try:
                        lines.append(
                            f"  - 评分贡献：结构化信号 {round(float(_sig), 1)}"
                            f"（加分 {round(float(_bd.get('structured_bonus') or 0.0), 1)}"
                            f" / 扣分 {round(float(_bd.get('structured_penalty') or 0.0), 1)}）"
                        )
                    except (TypeError, ValueError):
                        pass
            lines.append("")
    else:
        lines += ["## 三、候选股池", "", "*当前扫描无候选股（情绪冰点或无符合趋势的标的）*", ""]

    # 四、新闻多空证据
    evidence = (result.news or {}).get("evidence")
    if evidence:
        lines += ["## 四、新闻多空证据（板块驱动）", ""]
        top_bullish = evidence.get("top_bullish_themes") or []
        top_bearish = evidence.get("top_bearish_themes") or []
        risks = evidence.get("risk_events") or []
        if top_bullish:
            lines += ["### 利多题材", ""]
            for t in top_bullish[:5]:
                snippets = "；".join(t.get("bullish_snippets", [])[:2])
                lines.append(f"- **{t['theme']}**（得分 {t.get('score', '-')}）：{snippets}")
            lines.append("")
        if top_bearish:
            lines += ["### 利空题材", ""]
            for t in top_bearish[:5]:
                snippets = "；".join(t.get("bearish_snippets", [])[:2])
                lines.append(f"- **{t['theme']}**（得分 {t.get('score', '-')}）：{snippets}")
            lines.append("")
        if risks:
            lines += ["### 风险事件", ""]
            for r in risks[:5]:
                lines.append(f"- {r}")
            lines.append("")
        cand_evidence = evidence.get("candidate_evidence") or {}
        if cand_evidence:
            lines += ["### 个股新闻审计", ""]
            for c in result.candidates:
                code = c.get("code")
                ev = cand_evidence.get(code)
                if not ev:
                    continue
                label = ev.get("sentiment_label", "neutral")
                lines.append(
                    f"- **{c.get('name', code)}（{code}）** 新闻情绪 `{label}`："
                    f"{ev.get('verdict_reason', '')}"
                )
            lines.append("")

    # 五、被剔除的
    if result.rejected:
        lines += ["## 五、被护栏剔除（参考）", "",
                  "| 代码 | 名称 | 剔除原因 |", "|------|------|---------|"]
        for r in result.rejected[:15]:
            lines.append(f"| {r['code']} | {r['name']} | {r['reason']} |")
        lines.append("")

    # 六、姿态总结
    lines += ["## 六、操作建议", "", f"{result.posture_advice}", ""]
    if result.warnings:
        lines += ["### 提示", ""]
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines += ["---", "*盘古 Pangu · A股短线情绪+趋势选股引擎*"]
    return "\n".join(lines)


def save_report(
    result: PipelineResult,
    report_dir: str = "data/reports",
    *,
    force_degraded: bool = False,
) -> Path:
    """渲染并保存报告。

    - 数据质量 ok 且非 force_degraded：写入 data/reports/YYYYMMDD.{md,json}，
      并更新 data/reports/latest_ok.json 指针。
    - 否则：写入 data/reports/degraded/YYYYMMDD.{md,json}，不会覆盖主报告指针。
    """
    import math
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_degraded = force_degraded or result.data_quality in ("failed", "degraded")
    if is_degraded:
        out_dir = out_dir / "degraded"
        out_dir.mkdir(parents=True, exist_ok=True)

    md = render_markdown(result)
    md_path = out_dir / f"{result.date}.md"
    md_path.write_text(md, encoding="utf-8")

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    json_path = out_dir / f"{result.date}.json"
    try:
        payload = json.dumps(_sanitize(result.to_dict()), ensure_ascii=False, indent=2)
        json_path.write_text(payload, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("JSON 报告保存失败: %s", e)

    if not is_degraded:
        try:
            latest_ok = {
                "date": result.date,
                "json_path": str(json_path),
                "md_path": str(md_path),
                "generated_at": datetime.now().isoformat(),
                "data_quality": result.data_quality,
                "tradable": result.tradable,
            }
            latest_ok_path = Path(report_dir) / "latest_ok.json"
            tmp_path = latest_ok_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(latest_ok, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(latest_ok_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("latest_ok.json 更新失败: %s", e)

    return md_path
