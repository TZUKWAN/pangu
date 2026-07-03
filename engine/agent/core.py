"""盘古独立 Agent 核心：工具调用循环。"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .llm import ChatResponse, OpenAICompatibleClient, build_llm_client
from .prompts import SYSTEM_PROMPT, render_deep_pick_prompt
from .tools import ToolRegistry, build_tool_registry

logger = logging.getLogger("pangu.agent.core")


class PanguAgent:
    """盘古 Agent：规则选股 + LLM 解读 / 复核。

    无外部 agent 框架依赖。选股决策由 engine 规则链路完成；LLM 仅用于解释、
    复核候选池质量、并在数据不足时拒绝推荐。不声称"LLM 自主选股"。

    Args:
        llm_client: LLM 客户端
        tool_registry: 工具注册表
        system_prompt: system prompt
        max_rounds: 最大工具调用轮数
    """

    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        tool_registry: ToolRegistry,
        system_prompt: str = SYSTEM_PROMPT,
        max_rounds: int = 12,
    ) -> None:
        self.llm = llm_client
        self.tools = tool_registry
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PanguAgent":
        """从 settings.yaml 构造 Agent。"""
        llm = build_llm_client(cfg)
        registry = build_tool_registry(cfg)
        return cls(llm_client=llm, tool_registry=registry)

    # ------------------------------------------------------------------ #
    def run(self, question: str, date: Optional[str] = None) -> str:
        """回答用户问题。选股类问题走规则选股 + LLM 解读流程，非选股问题走工具调用循环。"""
        # 选股/推荐类关键词触发规则选股 + LLM 解读一键流程
        q = question.strip().lower()
        if any(k in q for k in ("选股", "买什么", "推荐", "今天买", "明天", "明日", "选几只", "关注")):
            return self._deep_pick(date)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        return self._loop(messages)

    # ------------------------------------------------------------------ #
    def _loop(self, messages: list[dict[str, Any]]) -> str:
        """工具调用主循环（含卡死检测）。"""
        last_call: tuple[str, str] | None = None  # (工具名, 参数摘要) 用于卡死检测
        repeat_count = 0
        for round_idx in range(self.max_rounds):
            logger.info("Agent 第 %d 轮调用", round_idx + 1)
            resp = self.llm.chat(
                messages=messages,
                tools=self.tools.schemas(),
                tool_choice="auto",
            )
            if resp.error:
                return f"⚠️ LLM 调用失败：{resp.error}"

            messages.append(self._assistant_message(resp))

            if not resp.tool_calls:
                return resp.content or "（模型未返回结论）"

            for tc in resp.tool_calls:
                logger.info("工具调用 %s(%s)", tc.name, tc.arguments)
                # 卡死检测：连续 2 轮调用相同工具+相同参数 → 注入换思路提示
                call_sig = (tc.name, json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True))
                if last_call == call_sig:
                    repeat_count += 1
                else:
                    repeat_count = 0
                last_call = call_sig
                if repeat_count >= 2:
                    logger.warning("检测到工具 %s 重复调用，注入换思路提示", tc.name)
                    messages.append({
                        "role": "user",
                        "content": "你刚才已多次调用相同工具且参数不变。请换个思路："
                                   "若数据已足够，直接给出结论；若缺数据，换一个工具或说明限制。",
                    })
                    repeat_count = 0
                    break

                result = self.tools.execute(tc.name, tc.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result,
                })

        return "⚠️ 达到最大调用轮数，未获得最终结论。"

    # ------------------------------------------------------------------ #
    def _deep_pick(self, date: Optional[str] = None) -> str:
        """规则选股 + LLM 解读：先由 engine 规则链路产出候选池，再由 LLM 生成解读报告。

        LLM 会复核候选池质量；若数据不完整或候选质量不足，LLM 应明确拒绝推荐。
        """
        tool = self.tools.get("deep_pick")
        if tool is None:
            return "⚠️ 未找到 deep_pick 工具"

        logger.info("执行 deep_pick")
        result = tool.execute({"date": date} if date else {})
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return f"⚠️ deep_pick 结果解析失败：{result[:500]}"

        pipeline = data.get("pipeline", {})
        candidates = pipeline.get("candidates", [])
        debates = data.get("debates", {})

        # 防御性兜底：候选池为空时直接劝退，避免 LLM 编造股票
        if not candidates:
            posture = pipeline.get("posture_advice", "建议观望")
            warnings = pipeline.get("warnings", [])
            warn_text = "\n".join(f"- {w}" for w in warnings) if warnings else "- 趋势扫描未筛选出符合均线多头+突破+放量+RPS 条件的个股。"
            return (
                "## 📊 盘古选股报告\n\n"
                f"**日期**：{pipeline.get('date', date or '今日')}\n\n"
                f"**结论**：{posture}\n\n"
                "**明日无符合选股条件的候选股**，系统未生成具体标的。\n\n"
                "**原因**：\n"
                f"{warn_text}\n\n"
                "**操作建议**：\n"
                "- 情绪冰点或趋势不满足时，空仓等待是更优策略。\n"
                "- 可盘后运行 `python -m engine.cli rps-build` 更新真实 RPS 表，提升选股覆盖度。\n\n"
                "⚠️ 以上为系统辅助分析，非投资建议，盈亏自负。"
            )

        pipeline_json = json.dumps(pipeline, ensure_ascii=False, indent=2)
        debates_json = json.dumps(debates, ensure_ascii=False, indent=2)
        news = data.get("news_briefing", "（无新闻简报）")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_deep_pick_prompt(pipeline_json, news, debates_json)},
        ]
        # 一键流程通常不需要再调工具，直接生成报告（非流式，CLI 场景）
        resp = self.llm.chat(messages=messages, tools=None)
        if resp.error:
            return f"⚠️ 报告生成失败：{resp.error}"
        return resp.content or "（模型未返回报告）"

    # ------------------------------------------------------------------ #
    def agent_review(self, pipeline_result: dict[str, Any]) -> dict[str, Any]:
        """LLM 复核候选池质量，可拒绝推荐。

        检查项：数据源健康、候选池完整、观察池未混入、RPS 可用、买卖点已计算、
        无风险票、市场状态允许推荐。
        """
        review_prompt = self._build_review_prompt(pipeline_result)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": review_prompt},
        ]
        try:
            resp = self.llm.chat(messages=messages, tools=None)
            content = resp.content or ""
            approved = "拒绝" not in content and "不推荐" not in content
            return {
                "approved": approved,
                "review_text": content,
                "llm_called": not bool(resp.error),
                "error": resp.error,
            }
        except Exception as e:  # noqa: BLE001
            return {"approved": False, "review_text": f"复核调用失败：{e}", "llm_called": False, "error": str(e)}

    def _build_review_prompt(self, pipeline_result: dict[str, Any]) -> str:
        source_status = pipeline_result.get("source_status", {})
        candidates = pipeline_result.get("candidates", [])
        rejected = pipeline_result.get("rejected", [])
        watchlist = pipeline_result.get("watchlist", [])
        rec_allowed = pipeline_result.get("recommendation_allowed", False)

        checks = []
        checks.append(f"数据源状态：{json.dumps(source_status, ensure_ascii=False, indent=2)}")
        checks.append(f"推荐是否允许：{'是' if rec_allowed else '否'}")
        checks.append(f"严格候选数量：{len(candidates)}，观察池数量：{len(watchlist)}，硬剔除数量：{len(rejected)}")

        rps_ok = source_status.get("rps", {}).get("status") == "ok"
        checks.append(f"真实 RPS 是否可用：{'是' if rps_ok else '否'}")

        watch_in_final = [c for c in candidates if c.get("is_watchlist")]
        checks.append(f"观察池是否混入候选：{'是' if watch_in_final else '否'}")

        missing_ee = [c.get("code") for c in candidates if not (c.get("entry_exit") or {}).get("stop_loss")]
        checks.append(f"缺少买卖点的候选：{missing_ee[:5]}")

        risk_in_final = [c.get("code") for c in candidates if (c.get("risk_flags") or [])]
        checks.append(f"带 risk_flags 的候选：{risk_in_final[:5]}")

        prompt_lines = [
            "你是一名投资复核员。请根据以下系统选股链路输出，判断是否可以向用户给出正式推荐。",
            "规则：",
            "1. 若 recommendation_allowed=False、真实 RPS 不可用、观察池混入候选、或关键数据源失败，必须拒绝推荐。",
            '2. 若候选数量过少或质量不足，应说明"今日无符合条件的正式推荐"。',
            "3. 不要从烂候选中硬挑股票。",
            "",
        ]
        prompt_lines.extend(checks)
        prompt_lines.extend(["", "请用 100 字以内给出复核结论：是否允许推荐，并说明理由。"])
        return "\\n".join(prompt_lines)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _assistant_message(resp: ChatResponse) -> dict[str, Any]:
        """把 ChatResponse 转成 messages 中的 assistant 消息。"""
        msg: dict[str, Any] = {"role": "assistant", "content": resp.content}
        if resp.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in resp.tool_calls
            ]
        return msg
