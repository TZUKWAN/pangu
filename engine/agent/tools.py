"""盘古独立 Agent 的工具集合。

所有工具直接调用 engine 的 Python API，无外部 agent 框架依赖。
工具定义采用 OpenAI function calling 格式。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from ..config import build_data_loader, load_config
from ..data_loader import DataLoader, safe_float, find_col as _find_col
from ..pipeline import Pipeline
from ..sentiment_meter import SentimentMeter

logger = logging.getLogger("pangu.agent.tools")


class _PanguJSONEncoder(json.JSONEncoder):
    """处理 pandas/numpy 原生类型，避免工具返回序列化失败。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        # numpy 标量
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:  # noqa: BLE001
                pass
        return super().default(obj)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, cls=_PanguJSONEncoder)


# ---------------------------------------------------------------------- #
# Tool 协议
# ---------------------------------------------------------------------- #
@dataclass
class Tool:
    """一个可调用工具。"""

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：未知工具 {name}"
        try:
            return tool.execute(arguments)
        except Exception as e:  # noqa: BLE001
            logger.exception("工具 %s 执行失败", name)
            return f"工具 {name} 执行失败：{e}"


# ---------------------------------------------------------------------- #
# 工具实现
# ---------------------------------------------------------------------- #
class EngineToolKit:
    """封装 engine 能力，供 Agent 调用。"""

    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        self.cfg = cfg or load_config()
        self.dl = build_data_loader(self.cfg)
        self.pipeline = Pipeline(
            dl=self.dl,
            sentiment_cfg=self.cfg.get("sentiment", {}),
            trend_cfg=self.cfg.get("trend", {}),
            guard_cfg=self.cfg.get("guard", {}),
            entry_exit_cfg=self.cfg.get("entry_exit", self.cfg),
            pick_count=self.cfg.get("output", {}).get("pick_count", 5),
            db_path=self.cfg.get("output", {}).get("db_path", "data/pangu.db"),
            full_cfg=self.cfg,
        )
        self.sentiment_meter = SentimentMeter(self.dl, self.cfg.get("sentiment", {}))
        # LLM 多空辩论器（可用时优先，失败降级到规则版 _generate_debate）
        try:
            from .debate import StockDebater
            self.debater = StockDebater(cfg=self.cfg)
        except Exception as e:  # noqa: BLE001
            logger.debug("LLM 辩论器不可用，将用规则版辩论：%s", e)
            self.debater = None

    # -------------------------------------------------------------- #
    def get_sentiment(self, date: Optional[str] = None) -> dict[str, Any]:
        """情绪温度计。"""
        try:
            from ..market_structure import MarketStructureAnalyzer
            ms = MarketStructureAnalyzer(self.dl, {})
            result = ms.analyze(date)
            return result.to_legacy_sentiment()
        except Exception as e:  # noqa: BLE001
            logger.warning("增强情绪分析失败，回退基础版：%s", e)
            return self.sentiment_meter.measure(date).to_dict()

    def scan_trend(self, date: Optional[str] = None) -> dict[str, Any]:
        """完整选股链路。"""
        result = self.pipeline.run(date)
        return result.to_dict()

    def get_news_briefing(self) -> str:
        """读最新财经简报。"""
        report_dir = self.cfg.get("output", {}).get("report_dir", "data/reports")
        d = Path(report_dir)
        if not d.exists():
            return "（暂无新闻简报）"
        files = sorted([f for f in d.iterdir() if f.suffix == ".md"], reverse=True)
        if not files:
            return "（暂无新闻简报）"
        try:
            return files[0].read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            return f"（读取简报失败：{e}）"

    def analyze_stock(self, code: str) -> dict[str, Any]:
        """个股深度分析。"""
        code = str(code).strip()
        k = self.dl.daily_kline(code, 60)
        f = self.dl.individual_fund_flow(code)
        s = self.dl.all_spot()
        spot_row = {}
        if len(s) > 0:
            code_col = _find_col(s, ["代码"]) or s.columns[1]
            rows = s[s[code_col].astype(str).str.strip() == code]
            if len(rows) > 0:
                spot_row = rows.iloc[0].to_dict()

        return {
            "code": code,
            "kline_tail": k.tail(10).to_dict(orient="records") if len(k) > 0 else [],
            "fund_flow_tail": f.tail(10).to_dict(orient="records") if len(f) > 0 else [],
            "spot": spot_row,
        }

    def debate_stock(self, code: str) -> dict[str, Any]:
        """对单只票进行多空辩论。优先用 LLM 真辩论，失败降级规则版。"""
        code = str(code).strip()
        data = self.analyze_stock(code)
        name = str(data.get("spot", {}).get("名称", code))
        # 优先 LLM 三方辩论（看多/看空/裁决）
        if self.debater is not None:
            try:
                return self.debater.debate(code, name, data)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s LLM 辩论失败，降级规则版: %s", code, e)
        # 降级：规则化打分
        result = _generate_debate(data)
        result["mode"] = "rule"
        return result

    def deep_pick(self, date: Optional[str] = None) -> dict[str, Any]:
        """一键综合选股：情绪→趋势→新闻→辩论→报告。"""
        result = self.pipeline.run(date)
        news = self.get_news_briefing()
        candidates = result.to_dict().get("candidates", [])
        debates: dict[str, Any] = {}
        # 优先用 LLM 批量辩论（并发，更准），降级规则版
        if self.debater is not None and candidates:
            try:
                debates = self.debater.debate_batch(candidates, max_n=5)
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM 批量辩论失败，逐只规则辩论: %s", e)
                debates = {}
        for c in candidates[:5]:
            code = c.get("code")
            if code and code not in debates:
                try:
                    debates[code] = self.debate_stock(code)
                except Exception as e:  # noqa: BLE001
                    debates[code] = {"verdict": "观望", "error": str(e)}
        return {
            "pipeline": result.to_dict(),
            "news_briefing": news,
            "debates": debates,
        }


# ---------------------------------------------------------------------- #
# 多空辩论规则化评分（与 agent/tools.ts 的 generateDebate 对齐）
# ---------------------------------------------------------------------- #
def _generate_debate(data: dict[str, Any]) -> dict[str, Any]:
    """基于个股数据生成多空观点与裁决。"""
    kline = data.get("kline_tail", [])
    spot = data.get("spot", {})
    fund = data.get("fund_flow_tail", [])

    closes = [safe_float(r.get("收盘", r.get("close"))) for r in kline]
    closes = [c for c in closes if c > 0]

    ma_bull = False
    breakout = False
    volume_ratio = 1.0
    fund_inflow_days = 0
    if len(closes) >= 20:
        ma5 = sum(closes[-5:]) / 5
        ma10 = sum(closes[-10:]) / 10
        ma20 = sum(closes[-20:]) / 20
        ma_bull = ma5 > ma10 > ma20
        recent_high = max(closes[-21:-1]) if len(closes) >= 22 else max(closes[:-1])
        breakout = closes[-1] > recent_high
    if len(closes) >= 6:
        vols = [safe_float(r.get("成交量", r.get("volume"))) for r in kline]
        vols = [v for v in vols if v > 0]
        if len(vols) >= 6 and sum(vols[-6:-1]) / 5 > 0:
            volume_ratio = vols[-1] / (sum(vols[-6:-1]) / 5)

    if fund:
        nets = [safe_float(r.get("主力净流入-净额", 0)) for r in fund[-5:]]
        nets.reverse()
        for v in nets:
            if v > 0:
                fund_inflow_days += 1
            else:
                break

    pe = safe_float(spot.get("市盈率-动态"))
    pb = safe_float(spot.get("市净率"))
    debt = safe_float(spot.get("资产负债率"))
    pct = safe_float(spot.get("涨跌幅"))

    bull_points: list[str] = []
    bear_points: list[str] = []
    score = 50

    if ma_bull:
        bull_points.append("均线多头排列（MA5>MA10>MA20）")
        score += 10
    else:
        bear_points.append("均线未形成多头排列")
        score -= 8

    if breakout:
        bull_points.append(f"突破近期平台/新高（量比约{volume_ratio:.1f}）")
        score += 10
    else:
        bear_points.append("尚未突破近期平台，存在震荡风险")
        score -= 5

    if volume_ratio >= 1.5:
        bull_points.append(f"放量（量比{volume_ratio:.1f}）")
        score += 6
    else:
        bear_points.append("量能不够突出")
        score -= 3

    if fund_inflow_days >= 2:
        bull_points.append(f"主力连续{fund_inflow_days}日净流入")
        score += 8
    else:
        bear_points.append("主力资金未形成连续流入")
        score -= 4

    if not pd.isna(pe) and pe > 0:
        if pe > 100:
            bear_points.append(f"动态PE {pe:.1f} 偏高，注意估值风险")
            score -= 6
        elif pe < 50:
            bull_points.append(f"估值相对合理（PE {pe:.1f}）")
            score += 3

    if pct > 7:
        bear_points.append(f"当日涨幅已达 {pct:.1f}%，追高接力风险大")
        score -= 8
    elif pct > 0:
        bull_points.append(f"当日上涨 {pct:.1f}%，有资金关注")
        score += 3

    score = max(0, min(100, score))
    if score >= 65:
        verdict = "推荐"
    elif score >= 40:
        verdict = "观望"
    else:
        verdict = "回避"

    return {
        "code": data.get("code"),
        "bull": "；".join(bull_points) if bull_points else "暂无明显看多信号",
        "bear": "；".join(bear_points) if bear_points else "暂无明显看空信号",
        "neutral": "个股趋势与资金一般，建议结合自身风险偏好决定仓位。",
        "verdict": verdict,
        "confidence": score,
        "reasoning": f"综合评分 {score}/100，{'多头因素占优' if score >= 60 else '多空交织或空头占优'}。",
        "bull_points": bull_points,
        "bear_points": bear_points,
    }


# ---------------------------------------------------------------------- #
# 注册表工厂
# ---------------------------------------------------------------------- #
def build_tool_registry(cfg: Optional[dict[str, Any]] = None) -> ToolRegistry:
    """构造完整的 Agent 工具注册表。"""
    kit = EngineToolKit(cfg)
    registry = ToolRegistry()

    registry.register(Tool(
        name="get_sentiment",
        description="查询当日 A 股市场情绪温度（0-100）及攻防姿态（冰点/正常/亢奋）。"
                    "包含涨停家数、最高连板、炸板率、跌停数、涨跌家数等分项。"
                    "用于判断明日是否值得出手。先调用此工具定调，再决定是否选股。",
        parameters={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "日期 YYYYMMDD，默认今天",
                },
            },
        },
        execute=lambda args: _json_dumps(kit.get_sentiment(args.get("date"))),
    ))

    registry.register(Tool(
        name="scan_trend",
        description="执行完整选股链路（情绪→趋势→护栏→买卖点），返回明日候选股池。"
                    "每只候选含代码、名称、所属热门板块、入选理由、entry_exit 交易计划。"
                    "情绪冰点时返回空候选池并建议观望。",
        parameters={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "日期 YYYYMMDD，默认今天",
                },
            },
        },
        execute=lambda args: _json_dumps(kit.scan_trend(args.get("date"))),
    ))

    registry.register(Tool(
        name="get_news_briefing",
        description="读取最新一期财经新闻简报（由 capitalise-finnews 或 report 命令生成的 Markdown）。"
                    "包含今日看点、A股异动、全球要闻、概念科普、资金动向等。"
                    "选股综合分析时必须结合新闻面。",
        parameters={"type": "object", "properties": {}},
        execute=lambda args: kit.get_news_briefing(),
    ))

    registry.register(Tool(
        name="analyze_stock",
        description="对单只 A 股做深度检查：近 60 日 K 线摘要、主力资金流、实时行情。"
                    "用于对候选股做最后确认，或回答用户「这只票怎么样」。",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "6 位股票代码，如 000001",
                },
            },
            "required": ["code"],
        },
        execute=lambda args: _json_dumps(kit.analyze_stock(args.get("code", ""))),
    ))

    registry.register(Tool(
        name="debate_stock",
        description="对单只候选股进行多空辩论，输出看多/看空/中性观点与最终裁决。"
                    "只有 verdict 为「推荐」的票才纳入最终推荐池。",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "6 位股票代码，如 000001",
                },
            },
            "required": ["code"],
        },
        execute=lambda args: _json_dumps(kit.debate_stock(args.get("code", ""))),
    ))

    registry.register(Tool(
        name="deep_pick",
        description="一键执行：读情绪 → 跑选股链路 → 读新闻简报 → 对候选股做多空辩论。"
                    "适合用户直接说「帮我选股」「明天关注什么」时调用。",
        parameters={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "日期 YYYYMMDD，默认今天",
                },
            },
        },
        execute=lambda args: _json_dumps(kit.deep_pick(args.get("date"))),
    ))

    return registry
