"""多智能体多空辩论：基于 TradingAgents-CN 思想的增强实现。

多轮辩论流程：
1. 看多分析师 vs 看空分析师 → 最多 N 轮交替辩论
2. 裁决官综合双方观点给出 verdict
3. 风险辩论层（可选）：激进/保守/中性三方评估仓位与风险

防降级：LLM 心跳检测 + 单次失败重试，仅连续失败才降级规则模式。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .llm import OpenAICompatibleClient, build_llm_client

logger = logging.getLogger("pangu.debate")

# ── 看多/看空/裁决 prompts ──────────────────────────────────────

_BULL_PROMPT = """你是一名坚定的【看多分析师】。请基于以下{stock_name}（{code}）的客观数据，
全力论证「为什么现在应该买入」。你必须站在多头立场，从数据中找出所有支持上涨的证据。

## 个股数据
{data_json}

{opponent_view}

## 要求
1. 只用提供的数据论证，不要编造数据
2. 从趋势形态、资金面、题材催化、相对强度、买卖点性价比 5 个维度找做多理由
3. 输出 3-5 条核心做多论点，每条 20-40 字，具体引用数据
4. 格式：纯文本，每条论点一行，开头加「+」

看多论点："""

_BEAR_PROMPT = """你是一名敏锐的【看空分析师】。请基于以下{stock_name}（{code}）的客观数据，
全力论证「为什么现在不该买/风险很大」。你必须站在空头立场，找出所有风险和隐患。

## 个股数据
{data_json}

{opponent_view}

## 要求
1. 只用提供的数据论证，不要编造数据
2. 从估值泡沫、技术破位风险、资金撤退迹象、追高风险、板块退潮 5 个维度找做空理由
3. 输出 3-5 条核心风险点，每条 20-40 字，具体引用数据
4. 格式：纯文本，每条论点一行，开头加「-」

风险点："""

_JUDGE_PROMPT = """你是【裁决官】。两名分析师对 {stock_name}（{code}）进行了多轮辩论：

## 看多分析师观点
{bull_view}

## 看空分析师观点
{bear_view}

## 个股客观数据
{data_json}

## 辩论历史
{debate_history}

## 你的任务
综合双方观点 + 客观数据，给出最终裁决。要求：
1. 不偏袒任何一方，只看数据和逻辑强度
2. 给出 verdict：推荐 / 观望 / 回避（三选一）
3. 给出 confidence：0-100（裁决信心）
4. 给出 50-100 字的综合理由

严格按以下 JSON 格式输出（不要其他内容）：
{{"verdict": "推荐|观望|回避", "confidence": 75, "reason": "综合理由"}}"""

# ── 风险辩论 prompts ────────────────────────────────────────────

_RISK_PROMPT_TEMPLATE = """你是【{role}风险分析师】。交易员对 {stock_name}（{code}）提出了以下投资计划：

## 投资计划
{plan_summary}

## 原始辩论裁决
{verdict_text}

## 个股数据
{data_json}

{opponent_view}

## 你的立场
{stance_description}

请输出你的风险评估（JSON格式）：
{{"risk_level": "低|中|高", "position_advice": "建议仓位比例", "stop_loss_advice": "止损建议", "key_concern": "核心关切"}}"""

_RISK_JUDGE_PROMPT = """你是【风险裁决官】。三位风险分析师（激进/保守/中性）对 {stock_name}（{code}）发表了意见：

{risk_views}

## 原始投资计划
{plan_summary}

请给出最终风险裁定（JSON格式）：
{{"verdict": "通过|谨慎通过|否决", "max_position_pct": 仓位上限百分比数字, "risk_confidence": 0-100, "reason": "综合理由"}}"""


# ── 状态常量 ─────────────────────────────────────────────────────

DEFAULT_AGENT_PROMPTS = {
    "bull": _BULL_PROMPT,
    "bear": _BEAR_PROMPT,
    "judge": _JUDGE_PROMPT,
    "risk": _RISK_PROMPT_TEMPLATE,
    "risk_judge": _RISK_JUDGE_PROMPT,
}


def get_agent_prompts(cfg: Optional[dict] = None) -> dict[str, str]:
    """Return debate prompt templates after applying user strategy overrides."""
    prompts = dict(DEFAULT_AGENT_PROMPTS)
    overrides = ((cfg or {}).get("agent_prompts") or {}) if isinstance(cfg, dict) else {}
    if isinstance(overrides, dict):
        for key in prompts:
            value = overrides.get(key)
            if isinstance(value, str) and value.strip():
                prompts[key] = value
    return prompts


_RISK_ROLES = [
    {
        "role": "激进型",
        "prompt_field": "risky",
        "stance": "你偏好高风险高收益，愿意承受较大回撤以换取更高回报。找理由支持加大仓位。",
    },
    {
        "role": "保守型",
        "prompt_field": "safe",
        "stance": "你极度厌恶风险，首要目标是保本。找一切理由降低仓位和风险敞口。",
    },
    {
        "role": "中性型",
        "prompt_field": "neutral",
        "stance": "你追求风险收益平衡，不偏激。根据数据和市场状态客观评估合理仓位。",
    },
]


class StockDebater:
    """多智能体多空辩论器（基于 LLM）。

    特性：
    - 多轮辩论（max_rounds 控制）
    - LLM 心跳检测，避免误降级
    - 失败自动重试，连续失败才降级
    - 裁决JSON容错解析
    """

    def __init__(
        self,
        llm: Optional[OpenAICompatibleClient] = None,
        deep_llm: Optional[OpenAICompatibleClient] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        self.cfg = cfg or {}
        self.llm = llm
        self.deep_llm = deep_llm  # 可选：裁决官/风险分析使用更强的模型
        self._llm_available: Optional[bool] = None  # 缓存心跳结果
        self._consecutive_failures = 0
        self.max_failures_before_degrade = 3
        xuanwu_cfg = (self.cfg.get("xuanwu_pool") or {}) if isinstance(self.cfg, dict) else {}
        self.debate_rounds = int(xuanwu_cfg.get("debate_rounds", 1) or 1)
        self.debate_max_tokens = int(xuanwu_cfg.get("debate_max_tokens", 1600) or 1600)
        self.judge_max_tokens = int(xuanwu_cfg.get("judge_max_tokens", 1200) or 1200)
        self.prompts = get_agent_prompts(self.cfg)

    def _format_prompt(self, key: str, **kwargs: Any) -> str:
        template = self.prompts.get(key) or DEFAULT_AGENT_PROMPTS[key]
        try:
            return template.format(**kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning("自定义提示词 %s 格式化失败，回退默认模板: %s", key, e)
            return DEFAULT_AGENT_PROMPTS[key].format(**kwargs)

    def _get_llm(self) -> Optional[OpenAICompatibleClient]:
        if self.llm is None:
            try:
                from ..config import load_config
                self.llm = build_llm_client(load_config())
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM 未配置，辩论将降级为规则模式: %s", e)
                self.llm = None
        return self.llm

    def _get_deep_llm(self) -> Optional[OpenAICompatibleClient]:
        """返回深度思考模型（裁决官优先使用），无则回退主 llm。"""
        if self.deep_llm is not None:
            return self.deep_llm
        return self._get_llm()

    def reset_llm(self) -> None:
        """清空 LLM 缓存并强制下次辩论重新构造客户端。"""
        self.llm = None
        self.deep_llm = None
        self._llm_available = None
        self._consecutive_failures = 0
        try:
            from .llm import invalidate_llm_cache
            invalidate_llm_cache()
            logger.info("StockDebater LLM 已重置")
        except Exception as e:  # noqa: BLE001
            logger.warning("重置 LLM 缓存失败: %s", e)

    def _check_llm_alive(self) -> bool:
        """LLM 心跳检测：发轻量请求确认 LLM 真实可用。"""
        if self._llm_available is not None:
            return self._llm_available
        llm = self._get_llm()
        if llm is None:
            self._llm_available = False
            return False
        try:
            # 轻量 ping：只请求 1 个 token
            resp = llm.simple_chat(
                [{"role": "user", "content": "ping"}],
                temperature=0.0, max_tokens=1,
            )
            self._llm_available = bool(resp)
            if self._llm_available:
                self._consecutive_failures = 0
            return self._llm_available
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 心跳失败: %s", e)
            self._llm_available = False
            return False

    def _try_llm_call(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 400,
        retry: int = 1,
    ) -> Optional[str]:
        """带重试的 LLM 调用。单次失败降 temperature 重试，仍失败返回 None。"""
        llm = self._get_llm()
        if llm is None:
            return None
        for attempt in range(retry + 1):
            t = temperature * (0.7 ** attempt)  # 每次重试降低 temperature
            try:
                return llm.simple_chat(messages, temperature=t, max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM 调用失败 (attempt %d/%d): %s", attempt + 1, retry + 1, e)
                if attempt >= retry:
                    self._consecutive_failures += 1
                    return None
        return None

    def debate(
        self,
        code: str,
        name: str,
        stock_data: dict[str, Any],
        news_sentiment: Optional[dict[str, Any]] = None,
        hot_themes: Optional[list[Any]] = None,
        max_rounds: int | None = None,
    ) -> dict[str, Any]:
        """对单只股票做多空辩论。

        Args:
            code: 股票代码
            name: 股票名称
            stock_data: 个股数据
            news_sentiment: 新闻情绪
            hot_themes: 热门题材
            max_rounds: 最大辩论轮次（默认2轮，每轮bull+bear各一次）
        Returns:
            {verdict, confidence, reason, bull_points, bear_points, bull_history, bear_history, mode, debate_rounds}
        """
        max_rounds = max_rounds if max_rounds is not None else self.debate_rounds
        data_brief = self._brief_data(stock_data, news_sentiment, hot_themes)

        # 心跳检测
        if not self._check_llm_alive():
            if self._consecutive_failures >= self.max_failures_before_degrade:
                logger.warning("LLM 连续 %d 次不可用，降级规则模式", self._consecutive_failures)
                return self._rule_fallback(code, name, stock_data, news_sentiment, hot_themes)
            # 尝试重置后再试
            self.reset_llm()
            if not self._check_llm_alive():
                return self._rule_fallback(code, name, stock_data, news_sentiment, hot_themes)

        try:
            # ── 多轮辩论 ──
            bull_history: list[str] = []
            bear_history: list[str] = []
            bull_view = ""
            bear_view = ""

            for rnd in range(max_rounds):
                # 构建对手观点（第2轮起）
                opponent = ""
                if rnd > 0:
                    opponent = f"\n## 对手上一轮观点\n{bear_view if rnd > 0 else ''}"

                # 看多
                bull_msg = [{"role": "user", "content": self._format_prompt("bull",
                    stock_name=name, code=code, data_json=data_brief,
                    opponent_view=f"\n## 对手上轮观点（请反驳）\n{bear_view}" if rnd > 0 else "",
                )}]
                bull_view = self._try_llm_call(bull_msg, temperature=0.7, max_tokens=self.debate_max_tokens)
                if bull_view:
                    bull_history.append(f"[第{rnd+1}轮] {bull_view[:200]}")
                else:
                    break  # LLM调用失败，停止辩论

                # 看空
                bear_msg = [{"role": "user", "content": self._format_prompt("bear",
                    stock_name=name, code=code, data_json=data_brief,
                    opponent_view=f"\n## 对手上轮观点（请反驳）\n{bull_view}" if rnd > 0 else "",
                )}]
                bear_view = self._try_llm_call(bear_msg, temperature=0.7, max_tokens=self.debate_max_tokens)
                if bear_view:
                    bear_history.append(f"[第{rnd+1}轮] {bear_view[:200]}")
                else:
                    break

            if not bull_view and not bear_view:
                return self._rule_fallback(code, name, stock_data, news_sentiment, hot_themes)

            # ── 裁决官（使用深度模型） ──
            deep_llm = self._get_deep_llm()
            debate_history_text = "\n".join(bull_history[-3:] + bear_history[-3:])
            judge_msg = [{"role": "user", "content": self._format_prompt("judge",
                stock_name=name, code=code,
                bull_view=bull_view or "(未生成)", bear_view=bear_view or "(未生成)",
                data_json=data_brief, debate_history=debate_history_text,
            )}]

            # 裁决官优先用 deep_llm
            if deep_llm and deep_llm != self._get_llm():
                try:
                    judge_text = deep_llm.simple_chat(judge_msg, temperature=0.3, max_tokens=self.judge_max_tokens)
                except Exception:  # noqa: BLE001
                    judge_text = self._try_llm_call(judge_msg, temperature=0.3, max_tokens=self.judge_max_tokens)
            else:
                judge_text = self._try_llm_call(judge_msg, temperature=0.3, max_tokens=self.judge_max_tokens)

            verdict_data = self._parse_judge(judge_text or "")

            # 如果置信度异常低（<30），重试一次裁决
            if verdict_data.get("confidence", 50) < 30:
                judge_text2 = self._try_llm_call(judge_msg, temperature=0.2, max_tokens=self.judge_max_tokens)
                if judge_text2:
                    verdict_data2 = self._parse_judge(judge_text2)
                    if verdict_data2.get("confidence", 0) > verdict_data.get("confidence", 0):
                        verdict_data = verdict_data2

            def _split_points(text: str) -> list[str]:
                pts = [line.strip().lstrip("+").lstrip("-").strip()
                       for line in (text or "").splitlines()]
                return [p for p in pts if p][:5]

            # 重置连续失败计数
            self._consecutive_failures = 0
            self._llm_available = True

            llm = self._get_llm()
            return {
                "verdict": verdict_data.get("verdict", "观望"),
                "confidence": verdict_data.get("confidence", 50),
                "reason": verdict_data.get("reason", (judge_text or "")[:100]),
                "bull_points": _split_points(bull_view),
                "bear_points": _split_points(bear_view),
                "bull_history": bull_history,
                "bear_history": bear_history,
                "mode": "llm",
                "debate_mode": "llm",
                "llm_called": True,
                "provider": getattr(llm, "provider_name", None) if llm else None,
                "model": getattr(llm, "model", None) if llm else None,
                "rule_degraded": False,
                "debate_rounds": max_rounds,
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("%s LLM 辩论失败，降级规则: %s", code, e)
            return self._rule_fallback(code, name, stock_data, news_sentiment, hot_themes)

    def debate_batch(
        self,
        candidates: list[dict[str, Any]],
        max_n: int = 5,
        news_sentiment: Optional[dict[str, dict[str, Any]]] = None,
        hot_themes: Optional[list[Any]] = None,
        ) -> dict[str, dict[str, Any]]:
        """批量辩论候选股。每只票独立并发。"""
        to_debate = candidates[:max_n]
        results: dict[str, dict[str, Any]] = {}

        def _one(c):
            code = str(c.get("code", ""))
            name = str(c.get("name", code))
            if not code:
                return code, {
                    "verdict": "观望", "mode": "skip", "reason": "缺少股票代码",
                    "bull_points": [], "bear_points": [],
                }
            ns = (news_sentiment or {}).get(code)
            try:
                r = self.debate(code, name, c, news_sentiment=ns, hot_themes=hot_themes)
                logger.info("辩论 %s(%s): %s conf=%s", name, code,
                            r.get("verdict"), r.get("confidence"))
                return code, r
            except Exception as e:  # noqa: BLE001
                return code, {
                    "verdict": "观望",
                    "reason": f"辩论执行失败，已跳过：{e}",
                    "mode": "error",
                    "debate_mode": "not_run",
                    "llm_called": False,
                    "provider": None,
                    "model": None,
                    "rule_degraded": True,
                    "bull_points": [], "bear_points": [],
                }

        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(5, len(to_debate) or 1)) as pool:
                for code, r in pool.map(_one, to_debate):
                    if code:
                        results[code] = r
        except Exception:  # noqa: BLE001
            for c in to_debate:
                code, r = _one(c)
                if code:
                    results[code] = r
        return results

    def rule_validate(
        self,
        code: str,
        name: str,
        stock_data: dict[str, Any],
        news_sentiment: Optional[dict[str, Any]] = None,
        hot_themes: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """不调用 LLM 的候选完整性验证。

        这个方法用于保证候选池里的每只票都有可审计的多空论证结构。
        它不会冒充 LLM 深度辩论，返回值会明确标注 rule_degraded。
        """
        result = self._rule_fallback(code, name, stock_data, news_sentiment, hot_themes)
        result["mode"] = "rule_validation"
        result["debate_mode"] = "rule_validation"
        result["llm_called"] = False
        result["provider"] = None
        result["model"] = None
        result["rule_degraded"] = True
        result["reason"] = (
            "候选池完整性验证：未调用 LLM，使用规则多空代理完成基础论证；"
            + str(result.get("reason") or "")
        )
        return result

    # ── 风险辩论 ─────────────────────────────────────────────────
    def risk_debate(
        self,
        code: str,
        name: str,
        stock_data: dict[str, Any],
        debate_result: dict[str, Any],
        news_sentiment: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """风险辩论层：激进/保守/中性三方辩论 + 风险裁决官。

        在投资辩论后调用。输入是原始辩论结果 + 股票数据。
        """
        if not self._check_llm_alive():
            return {
                "verdict": "通过", "max_position_pct": 20,
                "risk_confidence": 30, "reason": "LLM 不可用，风险辩论跳过（默认仓位上限 20%）",
                "mode": "rule_skip",
            }

        data_brief = self._brief_data(stock_data, news_sentiment, None)
        plan_summary = (
            f"裁决: {debate_result.get('verdict', '观望')} "
            f"信心: {debate_result.get('confidence', 50)} "
            f"理由: {debate_result.get('reason', '-')[:100]}"
        )
        verdict_text = plan_summary

        risk_views: list[str] = []
        prev_view = ""

        for role_info in _RISK_ROLES:
            opponent_view = f"\n## 前一位分析师观点\n{prev_view}" if prev_view else ""
            prompt = self._format_prompt("risk",
                role=role_info["role"],
                stock_name=name, code=code,
                plan_summary=plan_summary,
                verdict_text=verdict_text,
                data_json=data_brief,
                opponent_view=opponent_view,
                stance_description=role_info["stance"],
            )
            resp = self._try_llm_call(
                [{"role": "user", "content": prompt}],
                temperature=0.5, max_tokens=300,
            )
            view = resp or f'{{"risk_level":"中","position_advice":"无法评估","stop_loss_advice":"无法评估","key_concern":"LLM调用失败"}}'
            risk_views.append(f"**{role_info['role']}**: {view[:200]}")
            prev_view = view

        judge_prompt = self._format_prompt("risk_judge",
            stock_name=name, code=code,
            risk_views="\n\n".join(risk_views),
            plan_summary=plan_summary,
        )
        judge_raw = self._try_llm_call(
            [{"role": "user", "content": judge_prompt}],
            temperature=0.2, max_tokens=300,
        )
        judge_data = self._parse_judge(judge_raw or "")

        return {
            "verdict": judge_data.get("verdict", "通过"),
            "max_position_pct": judge_data.get("max_position_pct", 20),
            "risk_confidence": judge_data.get("risk_confidence", 50),
            "reason": judge_data.get("reason", (judge_raw or "")[:100]),
            "risk_views": risk_views,
            "mode": "llm_risk",
        }

    # ── 内部方法 ─────────────────────────────────────────────────
    def _brief_data(
        self, stock_data: dict[str, Any],
        news_sentiment: Optional[dict[str, Any]] = None,
        hot_themes: Optional[list[Any]] = None,
    ) -> str:
        parts = []
        parts.append(f"代码:{stock_data.get('code')} 名称:{stock_data.get('name')} "
                     f"板块:{stock_data.get('board','-')} 现价:{stock_data.get('close','-')}")
        parts.append(f"涨跌幅:{stock_data.get('pct_change','-')}% RPS:{stock_data.get('rps','-')} "
                     f"换手:{stock_data.get('turnover_rate','-')}% 流通市值:{stock_data.get('circ_mv_yi','-')}亿")
        parts.append(f"主力连续净流入:{stock_data.get('fund_inflow_days',0)}日")
        reasons = stock_data.get("reasons", [])
        if reasons:
            parts.append("入选理由:" + ";".join(reasons[:4]))
        ee = stock_data.get("entry_exit", {}) or {}
        rec = stock_data.get("recommend", {}) or {}
        if rec:
            parts.append(f"推荐度:{rec.get('recommend_score','-')}/100 等级:{rec.get('grade','-')} "
                         f"上涨概率:{rec.get('up_prob','-')}%")
        if ee:
            bp = ee.get("buy_points", [])
            primary = next((b for b in bp if b.get("is_primary")), bp[0] if bp else None)
            if primary:
                parts.append(f"主买点:{primary.get('price')} 止损:{(ee.get('stop_loss') or {}).get('price','-')} "
                             f"盈亏比:{ee.get('risk_reward_ratio','-')}")
        stock_news = stock_data.get("stock_news", [])
        if stock_news:
            titles = [n.get("title", "") for n in stock_news[:3] if n.get("title")]
            parts.append("个股新闻:" + " | ".join(titles))
        if news_sentiment:
            explain = news_sentiment.get("explain", "")
            parts.append(
                f"新闻情绪:{news_sentiment.get('sentiment_score','-')}/100 "
                f"({news_sentiment.get('sentiment_label','-')}) "
                f"题材:{', '.join(str(t) for t in news_sentiment.get('themes', [])[:3])}"
            )
            if explain:
                parts.append(f"情绪说明:{explain[:160]}")
        if hot_themes:
            themes = [str(t[0] if isinstance(t, (list, tuple)) else t) for t in hot_themes[:5]]
            parts.append("热门题材:" + ", ".join(themes))
        return "\n".join(parts)

    def _parse_judge(self, text: str) -> dict[str, Any]:
        """Enhanced JSON extraction from judge output."""
        text = text.strip()
        # Handle markdown code blocks
        for marker in ("```json", "```"):
            if marker in text:
                parts = text.split(marker)
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # inside code block
                        try:
                            return json.loads(part.strip())
                        except json.JSONDecodeError:
                            continue

        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
            # Try multiple parse approaches
            for attempt in [candidate, candidate.replace("'", '"')]:
                try:
                    result = json.loads(attempt)
                    # Normalize keys
                    normalized = {}
                    for k, v in result.items():
                        key = str(k).lower()
                        if key in ("verdict", "裁决"):
                            normalized["verdict"] = str(v)
                        elif key in ("confidence", "信心", "置信度"):
                            normalized["confidence"] = int(float(str(v)))
                        elif key in ("reason", "理由", "原因"):
                            normalized["reason"] = str(v)
                        elif key in ("risk_level", "风险等级"):
                            normalized["risk_level"] = str(v)
                        elif key in ("position_advice", "仓位建议"):
                            normalized["position_advice"] = str(v)
                        elif key in ("stop_loss_advice", "止损建议"):
                            normalized["stop_loss_advice"] = str(v)
                        elif key in ("key_concern", "核心关切"):
                            normalized["key_concern"] = str(v)
                        elif key in ("max_position_pct", "最大仓位"):
                            normalized["max_position_pct"] = int(float(str(v)))
                        elif key in ("risk_confidence", "风险信心"):
                            normalized["risk_confidence"] = int(float(str(v)))
                        else:
                            normalized[k] = v
                    if "verdict" in normalized or "risk_level" in normalized:
                        return normalized
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

            # Fallback: regex extract individual fields
            import re
            result: dict[str, Any] = {}
            v_match = re.search(r'(?:verdict|裁决)["\']?\s*[:：]\s*["\']?([^"\'，,\n}]+)', candidate)
            if v_match:
                result["verdict"] = v_match.group(1).strip().strip("'\"")
            c_match = re.search(r'(?:confidence|信心|置信度)["\']?\s*[:：]\s*(\d+)', candidate)
            if c_match:
                result["confidence"] = int(c_match.group(1))
            r_match = re.search(r'(?:reason|理由)["\']?\s*[:：]\s*["\']?([^"]+)', candidate)
            if r_match:
                result["reason"] = r_match.group(1).strip().strip("'\"")
            if "verdict" in result:
                return result

        return {"verdict": "观望", "confidence": 50, "reason": text[:100]}

    def _rule_fallback(
        self, code: str, name: str, stock_data: dict[str, Any],
        news_sentiment: Optional[dict[str, Any]] = None,
        hot_themes: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """LLM 不可用时的规则降级。"""
        from .tools import _generate_debate
        kline = stock_data.get("kline_tail", [])
        spot = {"最新价": stock_data.get("close"), "涨跌幅": stock_data.get("pct_change"),
                "换手率": stock_data.get("turnover_rate")}
        rec = stock_data.get("recommend", {}) or {}
        data = {
            "kline_tail": kline, "spot": spot, "fund_flow_tail": [],
            "recommend_score": rec.get("recommend_score", 50),
            "reasons": stock_data.get("reasons", []),
        }
        rule_result = _generate_debate(data)
        rule_result["mode"] = "rule"
        rule_result["confidence"] = min(rule_result.get("confidence", 50), 40)
        rule_result["bull_points"] = rule_result.get("bull_points", [])
        rule_result["bear_points"] = rule_result.get("bear_points", [])
        rule_result["debate_rounds"] = 1
        reasoning = rule_result.get("reasoning", "")
        rule_note = "LLM 不可用或连续失败，已降级为规则多空打分；结果为基于结构化数据的规则推断，非 AI 深度辩论。"
        rule_result["reason"] = f"{rule_note} {reasoning}".strip()
        rule_result["mode"] = "rule"
        rule_result["debate_mode"] = "rule_fallback"
        rule_result["llm_called"] = False
        rule_result["provider"] = None
        rule_result["model"] = None
        rule_result["rule_degraded"] = True
        ns_themes = []
        if news_sentiment:
            ns_themes = [str(t) for t in news_sentiment.get("themes", [])[:2]]
        if hot_themes and not ns_themes:
            ns_themes = [str(t[0] if isinstance(t, (list, tuple)) else t) for t in hot_themes[:2]]
        if ns_themes:
            rule_result["reason"] += f"（题材参考：{', '.join(ns_themes)}）"
        return rule_result
