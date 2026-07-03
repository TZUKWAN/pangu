"""独立 Agent 的大模型接口。

只依赖 requests，不依赖 openai SDK。支持所有 OpenAI 兼容接口：
DeepSeek / Qwen / GLM / 豆包 / Kimi / OpenAI / Anthropic（通过 base_url 转换）等。

工具调用采用 OpenAI 标准格式：
- 请求：messages + tools + tool_choice
- 响应：choices[0].message.content / tool_calls
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger("pangu.agent.llm")

# ---------------------------------------------------------------------- #
# LLM 调用审计（task_type / provider / model / latency / fallback）
# ---------------------------------------------------------------------- #
LLM_CALL_LOG: list[dict[str, Any]] = []

SUPPORTED_TASK_TYPES = {
    "report_generation", "debate_bull", "debate_bear", "debate_judge",
    "risk_review", "news_summary", "strategy_review", "agent_review",
    "chat", "unknown",
}


def log_llm_call(
    task_type: str,
    provider: str | None,
    model: str | None,
    latency_ms: int,
    success: bool,
    fallback_reason: str | None = None,
) -> None:
    """记录每次 LLM 调用到内存审计日志。"""
    entry = {
        "task_type": task_type if task_type in SUPPORTED_TASK_TYPES else "unknown",
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms,
        "success": success,
        "fallback_reason": fallback_reason,
    }
    LLM_CALL_LOG.append(entry)
    logger.debug("LLM audit: %s", entry)


def get_llm_audit_log() -> list[dict[str, Any]]:
    return list(LLM_CALL_LOG)


def clear_llm_audit_log() -> None:
    LLM_CALL_LOG.clear()


@dataclass
class ToolCall:
    """模型返回的工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """模型 chat completion 的标准化返回。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class LLMProviderConfig:
    """Safe runtime config for one OpenAI-compatible provider."""

    name: str
    api_key: str
    base_url: str
    model: str
    timeout: float = 300.0
    key_source: str = "config"


_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.=:/+]{8,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key=)[^&\s]+", re.IGNORECASE),
]


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}...{value[-3:]}"


def sanitize_llm_error(text: Any, secrets: list[str] | None = None) -> str:
    """Remove provider secrets from user-visible LLM errors."""
    out = str(text or "")
    for secret in secrets or []:
        if secret:
            out = out.replace(secret, _mask_secret(secret))
    for pat in _SECRET_PATTERNS:
        out = pat.sub(lambda m: (m.group(1) if m.lastindex else "") + "***", out)
    return out


def _safe_key_source(source: str) -> str:
    if not source:
        return "missing"
    return source.replace("API_KEY", "CREDENTIAL").replace("KEY", "CREDENTIAL")


class OpenAICompatibleClient:
    """OpenAI 兼容 API 客户端。

    Args:
        api_key: API key（不会默认读取环境变量，需显式传入或从配置读取）
        base_url: API base_url，如 https://api.deepseek.com/v1
        model: 模型名，如 deepseek-chat / Qwen3-235B-A22B
        timeout: 请求超时（秒）
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        provider_name: str = "default",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.provider_name = provider_name
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        task_type: str = "unknown",
    ) -> ChatResponse:
        """发起一次 chat completion 请求。"""
        import time
        t0 = time.monotonic()
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        logger.debug("LLM 请求 %s messages=%d tools=%d", url, len(messages), len(tools or []))
        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log_llm_call(task_type, self.provider_name, self.model, latency_ms, False, "timeout")
            return ChatResponse(error=f"LLM 请求超时（{self.timeout}秒）")
        except requests.exceptions.HTTPError as e:
            latency_ms = int((time.monotonic() - t0) * 1000)
            detail = sanitize_llm_error(e.response.text[:500], [self.api_key])
            log_llm_call(task_type, self.provider_name, self.model, latency_ms, False, f"http_{e.response.status_code}")
            return ChatResponse(error=f"LLM HTTP 错误: {e.response.status_code} {detail}")
        except Exception as e:  # noqa: BLE001
            latency_ms = int((time.monotonic() - t0) * 1000)
            log_llm_call(task_type, self.provider_name, self.model, latency_ms, False, str(e))
            return ChatResponse(error=f"LLM 请求失败: {sanitize_llm_error(e, [self.api_key])}")

        latency_ms = int((time.monotonic() - t0) * 1000)
        log_llm_call(task_type, self.provider_name, self.model, latency_ms, True)
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}

        tool_calls: list[ToolCall] = []
        for raw_tc in msg.get("tool_calls") or []:
            fn = raw_tc.get("function", {})
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=raw_tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))

        # 推理模型（Qwen3/GPT-o系列）：优先取 content；无 content 时用 reasoning 兜底
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if not content and reasoning:
            content = reasoning  # max_tokens 不足时，最终答案可能还在 reasoning 里
        # 注意：不合并 reasoning+content，避免把思考过程当答案输出给下游

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            raw=data,
        )

    # ------------------------------------------------------------------ #
    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
    ):
        """流式 chat completion，逐 token yield 文本增量（生成器）。

        用于 SSE 推送：每个 yield 是一小段新增文本（delta content）。
        遵循 OpenAI 兼容流式协议：响应是 `data: {json}\\n\\n` 序列，
        每行 chunk 的 choices[0].delta.content 即增量文本。

        失败时 yield 一条 `[错误] ...` 文本并立即返回。
        """
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        logger.debug("LLM 流式请求 %s messages=%d", url, len(messages))
        try:
            # stream=True 让 requests 不立即下载整个响应体，可逐行读
            resp = self._session.post(url, json=payload, stream=True, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            yield f"[错误] LLM 请求超时（{self.timeout}秒）"
            return
        except requests.exceptions.HTTPError as e:
            detail = sanitize_llm_error(e.response.text[:300], [self.api_key])
            yield f"[错误] LLM HTTP {e.response.status_code}: {detail}"
            return
        except Exception as e:  # noqa: BLE001
            yield f"[错误] LLM 流式请求失败: {sanitize_llm_error(e, [self.api_key])}"
            return

        # 解析 SSE 行：`data: {...}` 或结尾 `data: [DONE]`
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
                piece = delta.get("content")
                if piece:
                    yield piece
        except Exception as e:  # noqa: BLE001 - 流式中断要优雅收尾
            yield f"\n[错误] 流式读取中断: {sanitize_llm_error(e, [self.api_key])}"

    # ------------------------------------------------------------------ #
    def simple_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """简单文本对话，只返回文本内容。"""
        r = self.chat(messages, tools=None, **kwargs)
        if r.error:
            return f"（LLM 错误：{r.error}）"
        return r.content or ""


# ---------------------------------------------------------------------- #
# 配置构造
# ---------------------------------------------------------------------- #
class FallbackLLMClient:
    """Try multiple OpenAI-compatible clients in order."""

    def __init__(self, clients: list[OpenAICompatibleClient]) -> None:
        if not clients:
            raise ValueError("缺少可用 LLM provider")
        self.clients = clients
        self.provider_name = clients[0].provider_name
        self.model = clients[0].model
        self.last_provider: str | None = None
        self.last_errors: list[dict[str, str]] = []

    def _secrets(self) -> list[str]:
        return [c.api_key for c in self.clients]

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        task_type: str = "unknown",
    ) -> ChatResponse:
        self.last_errors = []
        for client in self.clients:
            resp = client.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                task_type=task_type,
            )
            if not resp.error:
                self.last_provider = client.provider_name
                if isinstance(resp.raw, dict):
                    resp.raw.setdefault("_pangu_provider", client.provider_name)
                return resp
            self.last_errors.append({
                "provider": client.provider_name,
                "error": sanitize_llm_error(resp.error, self._secrets()),
            })
            logger.warning("LLM provider %s failed, trying fallback if available: %s", client.provider_name, self.last_errors[-1]["error"])
        summary = "; ".join(f"{e['provider']}: {e['error']}" for e in self.last_errors)
        # Log final fallback failure for the first provider
        log_llm_call(task_type, self.clients[0].provider_name, self.clients[0].model, 0, False, summary)
        return ChatResponse(error=f"LLM providers all failed: {summary}")

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
    ):
        self.last_errors = []
        for client in self.clients:
            stream = client.stream_chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
            first = None
            try:
                first = next(stream)
            except StopIteration:
                self.last_errors.append({"provider": client.provider_name, "error": "empty stream"})
                continue
            if isinstance(first, str) and first.lstrip().startswith("[错误]"):
                err = sanitize_llm_error(first, self._secrets())
                self.last_errors.append({"provider": client.provider_name, "error": err})
                logger.warning("LLM stream provider %s failed, trying fallback if available: %s", client.provider_name, err)
                continue
            self.last_provider = client.provider_name
            if first is not None:
                yield first
            for piece in stream:
                yield sanitize_llm_error(piece, self._secrets())
            return
        summary = "; ".join(f"{e['provider']}: {e['error']}" for e in self.last_errors)
        yield f"[错误] LLM providers all failed: {summary}"

    def simple_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        r = self.chat(messages, tools=None, **kwargs)
        if r.error:
            return f"（LLM 错误：{r.error}）"
        return r.content or ""


def _provider_key(raw: dict[str, Any], default_env: str | None = None) -> tuple[str, str]:
    env_name = raw.get("api_key_env") or raw.get("env")
    if env_name:
        val = os.environ.get(str(env_name), "")
        if val:
            return val, f"env:{env_name}"
        # env var set but empty → fall through to api_key
    api_key = raw.get("api_key") or ""
    if api_key:
        return str(api_key), "config"
    if default_env:
        val = os.environ.get(default_env, "")
        if val:
            return val, f"env:{default_env}"
    return "", "missing"
    return "", "missing"


def _normalize_provider_configs(cfg: dict[str, Any]) -> tuple[list[LLMProviderConfig], list[str]]:
    llm_cfg = cfg.get("llm", {}) or {}
    warnings: list[str] = []
    raw_providers = list(llm_cfg.get("providers") or [])

    # 支持顶层 base_url_env / model_env / api_key_env
    def _env_or(cfg: dict[str, Any], env_key: str, val_key: str) -> str:
        env_name = cfg.get(env_key)
        if env_name:
            val = os.environ.get(str(env_name), "")
            if val:
                return val
        return str(cfg.get(val_key) or "")

    if not raw_providers:
        raw_providers = [{
            "name": llm_cfg.get("provider") or "primary",
            "api_key": _env_or(llm_cfg, "api_key_env", "api_key"),
            "base_url": _env_or(llm_cfg, "base_url_env", "base_url"),
            "model": _env_or(llm_cfg, "model_env", "model"),
            "timeout": llm_cfg.get("timeout", 300),
            "_default_env": "PANGU_LLM_API_KEY",
        }]

    providers: list[LLMProviderConfig] = []
    for idx, raw in enumerate(raw_providers, start=1):
        if raw.get("enabled") is False:
            warnings.append(f"{raw.get('name') or idx} disabled")
            continue
        name = str(raw.get("name") or f"provider_{idx}")
        default_env = raw.get("_default_env") or ("PANGU_LLM_API_KEY" if idx == 1 else None)
        api_key, key_source = _provider_key(raw, default_env=default_env)
        base_url = _env_or(raw, "base_url_env", "base_url").strip()
        model = _env_or(raw, "model_env", "model").strip()
        timeout = float(raw.get("timeout", llm_cfg.get("timeout", 300)) or 300)
        missing = []
        if not api_key:
            missing.append("credential")
        if not base_url:
            missing.append("base_url")
        if not model:
            missing.append("model")
        if missing:
            warnings.append(f"{name} missing {', '.join(missing)}")
            continue
        providers.append(LLMProviderConfig(
            name=name,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            key_source=key_source,
        ))
    return providers, warnings


def llm_governance_status(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return safe LLM provider status without exposing credentials."""
    providers, warnings = _normalize_provider_configs(cfg)
    llm_cfg = cfg.get("llm", {}) or {}
    configured = len(llm_cfg.get("providers") or []) or (1 if llm_cfg else 0)
    return {
        "status": "ok" if providers else "unavailable",
        "configured_count": configured,
        "usable_count": len(providers),
        "fallback_enabled": len(providers) > 1,
        "active_order": [p.name for p in providers],
        "providers": [
            {
                "name": p.name,
                "base_url": p.base_url,
                "model": p.model,
                "timeout": p.timeout,
                "key_source": _safe_key_source(p.key_source),
                "key_configured": bool(p.api_key),
            }
            for p in providers
        ],
        "warnings": warnings[:12],
    }


# 轻量级客户端缓存，避免每次请求重复构造/探测；AI 恢复链路可一键清空。
_client_cache: dict[int, OpenAICompatibleClient | FallbackLLMClient] = {}


def build_llm_client(cfg: dict[str, Any]) -> OpenAICompatibleClient | FallbackLLMClient:
    """从 settings.yaml 的 llm 配置段构造 LLM 客户端，支持 providers fallback。"""
    # 用配置 provider 列表的 hash 做缓存键
    providers, warnings = _normalize_provider_configs(cfg)
    cache_key = hash(tuple(
        (p.name, p.base_url, p.model, p.timeout, p.key_source)
        for p in providers
    ))
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    if not providers:
        joined = "; ".join(warnings) if warnings else "llm config empty"
        if "credential" in joined:
            raise ValueError("缺少 LLM API key：请配置 llm.providers[].api_key/api_key_env 或环境变量 PANGU_LLM_API_KEY")
        if "base_url" in joined:
            raise ValueError("缺少 LLM base_url：请在 llm.providers 或 llm.base_url 中配置")
        if "model" in joined:
            raise ValueError("缺少 LLM model：请在 llm.providers 或 llm.model 中配置")
        raise ValueError("缺少 LLM provider：请配置 llm.providers 或 llm.api_key/base_url/model")

    clients = [
        OpenAICompatibleClient(
            api_key=p.api_key,
            base_url=p.base_url,
            model=p.model,
            timeout=p.timeout,
            provider_name=p.name,
        )
        for p in providers
    ]
    client = clients[0] if len(clients) == 1 else FallbackLLMClient(clients)
    _client_cache[cache_key] = client
    return client


def invalidate_llm_cache() -> dict[str, Any]:
    """清空 LLM 客户端缓存与模型发现缓存，用于 AI 恢复/重连。"""
    cleared = {"client_cache": len(_client_cache)}
    _client_cache.clear()
    try:
        from . import llm_discover as ld
        if ld.CACHE_PATH.exists():
            try:
                ld.CACHE_PATH.unlink()
                cleared["models_cache"] = True
            except Exception as e:  # noqa: BLE001
                cleared["models_cache_error"] = str(e)
        else:
            cleared["models_cache"] = False
    except Exception as e:  # noqa: BLE001
        cleared["models_cache_error"] = str(e)
    logger.info("LLM 缓存已清空: %s", cleared)
    return cleared
