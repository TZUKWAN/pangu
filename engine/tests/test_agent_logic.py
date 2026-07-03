"""独立 Agent 逻辑测试（mock LLM，不依赖网络）。"""

from __future__ import annotations

import pytest

from engine.agent.core import PanguAgent
from engine.agent.debate import StockDebater
from engine.agent.llm import ChatResponse, OpenAICompatibleClient, ToolCall
from engine.agent.tools import Tool, ToolRegistry


def test_build_llm_client_fallback_uses_backup(monkeypatch):
    """Primary provider failure should fall back to the backup provider."""
    from engine.agent import llm as llm_mod

    def fake_chat(self, *args, **kwargs):
        if self.provider_name == "primary":
            return ChatResponse(error="primary failed")
        return ChatResponse(content="backup ok")

    monkeypatch.setattr(llm_mod.OpenAICompatibleClient, "chat", fake_chat)
    client = llm_mod.build_llm_client({
        "llm": {
            "providers": [
                {"name": "primary", "api_key": "primary-token-for-test", "base_url": "https://p.example", "model": "m1"},
                {"name": "backup", "api_key": "backup-token-for-test", "base_url": "https://b.example", "model": "m2"},
            ]
        }
    })
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.error is None
    assert resp.content == "backup ok"
    assert client.last_provider == "backup"


def test_llm_fallback_errors_do_not_leak_keys(monkeypatch):
    """Provider failure messages must be sanitized before returning to users."""
    from engine.agent import llm as llm_mod

    secret1 = "primary-token-secret-123456"
    secret2 = "backup-token-secret-654321"

    def fake_chat(self, *args, **kwargs):
        return ChatResponse(error=f"bad auth {self.api_key} Bearer {self.api_key}")

    monkeypatch.setattr(llm_mod.OpenAICompatibleClient, "chat", fake_chat)
    client = llm_mod.build_llm_client({
        "llm": {
            "providers": [
                {"name": "primary", "api_key": secret1, "base_url": "https://p.example", "model": "m1"},
                {"name": "backup", "api_key": secret2, "base_url": "https://b.example", "model": "m2"},
            ]
        }
    })
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.error
    assert secret1 not in resp.error
    assert secret2 not in resp.error
    assert "Bearer sk-" not in resp.error


def test_llm_governance_status_has_no_secret(monkeypatch):
    """Governance status exposes key source names, never key values."""
    from engine.agent.llm import llm_governance_status

    monkeypatch.setenv("PANGU_TEST_LLM_KEY", "real-token-hidden")
    status = llm_governance_status({
        "llm": {
            "providers": [
                {"name": "primary", "api_key_env": "PANGU_TEST_LLM_KEY", "base_url": "https://p.example", "model": "m1"},
            ]
        }
    })
    text = str(status)
    assert status["status"] == "ok"
    assert status["providers"][0]["key_source"] == "env:PANGU_TEST_LLM_CREDENTIAL"
    assert "real-token-hidden" not in text


def test_stock_debater_uses_custom_prompt_overrides():
    debater = StockDebater(cfg={"agent_prompts": {"bull": "自定义看多 {stock_name} {code}"}})

    assert debater.prompts["bull"] == "自定义看多 {stock_name} {code}"
    assert "bear" in debater.prompts
    assert debater._format_prompt("bull", stock_name="测试股", code="000001") == "自定义看多 测试股 000001"


class FakeLLMClient:
    """用于测试的 mock LLM 客户端。"""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def chat(self, **kwargs):
        resp = self.responses[self.calls]
        self.calls += 1
        return resp


class FakeToolKit:
    """用于测试的 mock 工具。"""

    def __init__(self):
        self.calls = []

    def echo(self, args: dict) -> str:
        self.calls.append(args)
        return f"echo: {args.get('msg', '')}"


def test_tool_registry_execute():
    """工具注册表能正确注册和执行工具。"""
    registry = ToolRegistry()
    toolkit = FakeToolKit()
    registry.register(Tool(
        name="echo",
        description="回显",
        parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
        execute=toolkit.echo,
    ))

    assert len(registry.schemas()) == 1
    result = registry.execute("echo", {"msg": "hello"})
    assert result == "echo: hello"
    assert toolkit.calls == [{"msg": "hello"}]


def test_agent_responds_without_tools():
    """Agent 在模型直接返回文本时不进入工具循环。"""
    llm = FakeLLMClient([ChatResponse(content="今日情绪冰点，建议观望。")])
    registry = ToolRegistry()
    agent = PanguAgent(llm_client=llm, tool_registry=registry)  # type: ignore[arg-type]
    answer = agent.run("分析一下今天市场")
    assert "观望" in answer
    assert llm.calls == 1


def test_agent_tool_loop():
    """Agent 能执行模型返回的工具调用并继续对话。"""
    llm = FakeLLMClient([
        ChatResponse(
            content="",
            tool_calls=[ToolCall(id="call_1", name="echo", arguments={"msg": "hi"})],
        ),
        ChatResponse(content="收到结果，建议观望。"),
    ])

    registry = ToolRegistry()
    toolkit = FakeToolKit()
    registry.register(Tool(
        name="echo",
        description="回显",
        parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
        execute=toolkit.echo,
    ))

    agent = PanguAgent(llm_client=llm, tool_registry=registry)  # type: ignore[arg-type]
    answer = agent.run("测试")
    assert "收到结果" in answer
    assert llm.calls == 2
    assert len(toolkit.calls) == 1


def test_agent_max_rounds():
    """Agent 达到最大轮数后返回提示。"""
    # 模型每次都返回工具调用，触发最大轮数
    llm = FakeLLMClient([
        ChatResponse(
            content="",
            tool_calls=[ToolCall(id=f"call_{i}", name="echo", arguments={"msg": str(i)})],
        )
        for i in range(15)
    ])
    registry = ToolRegistry()
    registry.register(Tool(
        name="echo",
        description="回显",
        parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
        execute=lambda x: f"echo: {x.get('msg')}",
    ))

    agent = PanguAgent(llm_client=llm, tool_registry=registry, max_rounds=3)  # type: ignore[arg-type]
    answer = agent.run("测试")
    assert "最大调用轮数" in answer


def test_build_llm_client_requires_api_key(monkeypatch):
    """缺少 api_key 时 build_llm_client 抛 ValueError。"""
    from engine.agent.llm import build_llm_client
    monkeypatch.delenv("PANGU_LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="缺少 LLM API key"):
        build_llm_client({"llm": {"base_url": "https://x", "model": "m"}})


def test_build_llm_client_requires_base_url_and_model():
    """缺少 base_url 或 model 时抛 ValueError。"""
    from engine.agent.llm import build_llm_client
    with pytest.raises(ValueError, match="缺少 LLM base_url"):
        build_llm_client({"llm": {"api_key": "sk", "model": "m"}})
    with pytest.raises(ValueError, match="缺少 LLM model"):
        build_llm_client({"llm": {"api_key": "sk", "base_url": "https://x"}})


def test_deep_pick_empty_candidates_returns_cash():
    """deep_pick 返回空候选池时，Agent 不应调用 LLM 编造股票。"""
    import json

    llm = FakeLLMClient([])  # 如果调用了 LLM，会因 responses 为空而抛 IndexError
    registry = ToolRegistry()
    registry.register(Tool(
        name="deep_pick",
        description="选股",
        parameters={"type": "object", "properties": {}},
        execute=lambda x: json.dumps({
            "pipeline": {
                "date": "20260101",
                "candidates": [],
                "posture_advice": "情绪冰点，建议观望",
                "warnings": ["趋势扫描无候选股"],
            },
            "news_briefing": "（无新闻简报）",
            "debates": {},
        }, ensure_ascii=False),
    ))

    agent = PanguAgent(llm_client=llm, tool_registry=registry)  # type: ignore[arg-type]
    answer = agent.run("帮我选股")
    assert "无符合选股条件" in answer
    assert "建议观望" in answer
    assert llm.calls == 0  # 未调用 LLM



def test_debate_brief_injects_stock_news_and_sentiment():
    """debate _brief_data 必须包含 stock_news、news_sentiment、hot_themes。"""
    from engine.agent.debate import StockDebater

    debater = StockDebater(cfg={})
    stock_data = {
        "code": "000001",
        "name": "平安银行",
        "board": "银行",
        "close": 12.5,
        "pct_change": 2.1,
        "turnover_rate": 3.5,
        "circ_mv_yi": 2000,
        "rps": 88,
        "fund_inflow_days": 3,
        "reasons": ["均线多头", "突破平台"],
        "stock_news": [{"title": "平安银行业绩预增", "content": "", "source": "财联社"}],
    }
    news_sentiment = {
        "sentiment_score": 78.0,
        "sentiment_label": "positive",
        "themes": ["银行", "业绩预增"],
        "explain": "来源：财联社；正向词 2 个 > 负向词 0 个",
    }
    hot_themes = [("银行", 12), ("AI", 45)]
    brief = debater._brief_data(stock_data, news_sentiment=news_sentiment, hot_themes=hot_themes)
    assert "000001" in brief
    assert "平安银行" in brief
    assert "平安银行业绩预增" in brief
    assert "78.0" in brief or "78" in brief
    assert "positive" in brief
    assert "业绩预增" in brief
    assert "银行" in brief
    assert "AI" in brief


def test_debate_rule_fallback_returns_points_and_reason(monkeypatch):
    """LLM 不可用时规则降级必须给出 bull_points/bear_points 数组和显式 reason。"""
    from engine.agent.debate import StockDebater

    debater = StockDebater(cfg={})
    monkeypatch.setattr(debater, "_check_llm_alive", lambda: False)
    debater._consecutive_failures = debater.max_failures_before_degrade
    stock_data = {
        "code": "000001",
        "name": "测试银行",
        "close": 10.0,
        "pct_change": 1.5,
        "turnover_rate": 5.0,
        "reasons": ["均线多头排列"],
        "recommend": {"recommend_score": 50},
    }
    r = debater.debate("000001", "测试银行", stock_data)
    assert r["mode"] == "rule"
    assert "bull_points" in r and isinstance(r["bull_points"], list)
    assert "bear_points" in r and isinstance(r["bear_points"], list)
    assert r.get("reason") and "已降级为规则多空打分" in r["reason"]
    assert r.get("rule_degraded") is True
