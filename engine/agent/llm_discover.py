"""LLM provider 模型自动发现与连接测试。

只依赖 requests；探测 OpenAI 兼容的 /models 端点。
结果缓存到 data/llm_models_cache.json，默认 TTL 24h。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from .llm import _provider_key, sanitize_llm_error, invalidate_llm_cache

logger = logging.getLogger("pangu.agent.llm_discover")

CACHE_PATH = Path("data/llm_models_cache.json")
DEFAULT_TTL_SECONDS = 24 * 3600


def _load_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("读取 LLM 模型缓存失败: %s", e)
        return {}


def _save_cache(name: str, models: list[str], endpoint: str) -> None:
    cache = _load_cache()
    cache[name] = {
        "models": models,
        "endpoint": endpoint,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ttl_until": time.time() + DEFAULT_TTL_SECONDS,
    }
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("保存 LLM 模型缓存失败: %s", e)


def discover_models(
    raw_provider: dict[str, Any],
    *,
    force_refresh: bool = False,
    timeout: int = 15,
) -> dict[str, Any]:
    """探测 provider 可用模型列表。

    返回：
        status: ok / auth_failed / timeout / not_supported
        models: list[str]
        endpoint, cached, count, hint
    """
    name = raw_provider.get("name") or "default"
    if not force_refresh:
        cache = _load_cache()
        entry = cache.get(str(name))
        if entry and entry.get("ttl_until", 0) > time.time():
            return {
                "status": "ok",
                "models": entry.get("models", []),
                "endpoint": entry.get("endpoint"),
                "cached": True,
                "count": len(entry.get("models", [])),
                "hint": "缓存命中",
            }

    api_key, key_source = _provider_key(raw_provider)
    base_url = str(raw_provider.get("base_url") or "").rstrip("/")
    if not api_key:
        return {
            "status": "auth_failed",
            "models": [],
            "endpoint": None,
            "cached": False,
            "count": 0,
            "hint": "未配置 API key（对应环境变量未设置）",
        }
    if not base_url:
        return {
            "status": "not_supported",
            "models": [],
            "endpoint": None,
            "cached": False,
            "count": 0,
            "hint": "base_url 为空，无法探测模型",
        }

    candidates = [
        f"{base_url}/models",
        f"{base_url}/v1/models",
    ]
    last_err = ""
    for url in candidates:
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
        except requests.Timeout:
            return {
                "status": "timeout",
                "models": [],
                "endpoint": url,
                "cached": False,
                "count": 0,
                "hint": "请求超时，请检查 base_url 或网络",
            }
        except requests.RequestException as e:
            last_err = sanitize_llm_error(str(e), [api_key])
            continue

        if resp.status_code == 401:
            return {
                "status": "auth_failed",
                "models": [],
                "endpoint": url,
                "cached": False,
                "count": 0,
                "hint": "API key 无效或无权限",
            }
        if resp.status_code == 404:
            last_err = "404"
            continue
        if resp.status_code == 200:
            try:
                data = resp.json() or {}
            except Exception as e:  # noqa: BLE001
                last_err = sanitize_llm_error(str(e), [api_key])
                continue
            models = sorted({m.get("id") for m in data.get("data", []) if m.get("id")})
            _save_cache(str(name), models, url)
            return {
                "status": "ok",
                "models": models,
                "endpoint": url,
                "cached": False,
                "count": len(models),
                "hint": f"发现 {len(models)} 个模型",
            }
        last_err = f"HTTP {resp.status_code}"

    return {
        "status": "not_supported",
        "models": [],
        "endpoint": None,
        "cached": False,
        "count": 0,
        "hint": f"该 provider 不支持模型自动发现（{last_err}），请手填 model 名",
    }


def probe_capabilities(
    raw_provider: dict[str, Any], *, timeout: int = 25
) -> dict[str, Any]:
    """探测 provider 能力：流式输出、工具调用、上下文长度。

    返回：
        {
          "streaming": {"supported": bool, "source": "probe"|"unsupported"|"error", ...},
          "tools":     {"supported": bool, "source": "probe"|"unsupported"|"error", ...},
          "context_length": {"value": int, "source": "provider"|"heuristic"|"error"},
          "latency_ms": int,
          "model": str,
        }
    """
    api_key, _ = _provider_key(raw_provider)
    base_url = str(raw_provider.get("base_url") or "").rstrip("/")
    model = str(raw_provider.get("model") or "").strip()
    if not api_key or not base_url or not model:
        return {
            "streaming": {"supported": False, "source": "error", "hint": "缺少 api_key/base_url/model"},
            "tools": {"supported": False, "source": "error", "hint": "缺少 api_key/base_url/model"},
            "context_length": {"value": 4096, "source": "error", "hint": "缺少 api_key/base_url/model"},
            "latency_ms": 0,
            "model": model,
        }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url}/chat/completions"

    # ---------- 1. 流式能力 ----------
    streaming: dict[str, Any] = {"supported": False, "source": "unsupported", "hint": ""}
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": True,
        }
        start = time.time()
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout)
        if resp.status_code == 200:
            # 真正读取第一块，确认是 SSE 格式
            first = ""
            try:
                for chunk in resp.iter_lines(decode_unicode=True):
                    if chunk:
                        first = chunk.strip()
                        break
            except Exception:  # noqa: BLE001
                pass
            if first.startswith("data:"):
                streaming = {"supported": True, "source": "probe", "hint": "SSE 流式可用"}
            else:
                streaming = {"supported": False, "source": "unsupported", "hint": "接口 200 但非 SSE 流式返回"}
        else:
            detail = sanitize_llm_error(resp.text[:200], [api_key])
            streaming = {"supported": False, "source": "error", "hint": f"HTTP {resp.status_code}: {detail}"}
        resp.close()
    except requests.Timeout:
        streaming = {"supported": False, "source": "error", "hint": "流式探测超时"}
    except Exception as e:  # noqa: BLE001
        streaming = {"supported": False, "source": "error", "hint": sanitize_llm_error(str(e), [api_key])}

    # ---------- 2. 工具调用能力 ----------
    tools: dict[str, Any] = {"supported": False, "source": "unsupported", "hint": ""}
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "上海今天天气"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取城市天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "tool_choice": "auto",
            "max_tokens": 20,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            tools = {"supported": True, "source": "probe", "hint": "工具调用可用"}
        elif resp.status_code in (400, 422):
            detail = sanitize_llm_error(resp.text[:200], [api_key])
            tools = {"supported": False, "source": "unsupported", "hint": f"模型/服务不支持工具调用 ({resp.status_code}): {detail}"}
        else:
            detail = sanitize_llm_error(resp.text[:200], [api_key])
            tools = {"supported": False, "source": "error", "hint": f"HTTP {resp.status_code}: {detail}"}
    except requests.Timeout:
        tools = {"supported": False, "source": "error", "hint": "工具探测超时"}
    except Exception as e:  # noqa: BLE001
        tools = {"supported": False, "source": "error", "hint": sanitize_llm_error(str(e), [api_key])}

    # ---------- 3. 上下文长度 ----------
    context_length: dict[str, Any] = {"value": _heuristic_context_length(model), "source": "heuristic", "hint": f"按模型名 {model} 启发式估算"}
    for endpoint in [f"{base_url}/v1/models/{model}", f"{base_url}/models/{model}"]:
        try:
            r = requests.get(endpoint, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
            if r.status_code == 200:
                data = r.json() or {}
                info = data.get("data", data)
                if isinstance(info, dict):
                    ctx = info.get("context_window") or info.get("max_context") or info.get("max_tokens")
                    if ctx:
                        context_length = {"value": int(ctx), "source": "provider", "hint": f"从 {endpoint} 读取"}
                        break
        except Exception:  # noqa: BLE001
            continue

    return {
        "streaming": streaming,
        "tools": tools,
        "context_length": context_length,
        "latency_ms": 0,
        "model": model,
    }


def _heuristic_context_length(model: str) -> int:
    """根据常见模型名估算上下文长度。"""
    m = model.lower()
    if "32k" in m or "qwen3-235b" in m or "gpt-4-32k" in m:
        return 32768
    if "128k" in m or "gpt-4o" in m or "gpt-4-turbo" in m or "claude-3" in m or "qwen3-30b-a3b" in m:
        return 128000
    if "deepseek-r1" in m or "deepseek-v3" in m or "deepseek-chat" in m:
        return 64000
    if "glm-4" in m or "glm4" in m:
        return 128000
    if "doubao" in m or "skylark" in m:
        return 32000
    return 4096


def probe_provider(raw_provider: dict[str, Any], *, timeout: int = 15) -> dict[str, Any]:
    """测试 provider 连通性。优先 /models 探测，回退 1 token chat。

    返回：
        status: ok / auth_failed / timeout / request_failed
        latency_ms, hint
    """
    start = time.time()
    disc = discover_models(raw_provider, force_refresh=True, timeout=timeout)
    latency_ms = int((time.time() - start) * 1000)
    caps = probe_capabilities(raw_provider, timeout=timeout)
    caps["latency_ms"] = latency_ms
    if disc["status"] == "ok":
        return {"status": "ok", "latency_ms": latency_ms, "hint": disc["hint"], "capabilities": caps}
    if disc["status"] == "auth_failed":
        return {"status": "auth_failed", "latency_ms": latency_ms, "hint": disc["hint"], "capabilities": caps}
    if disc["status"] == "timeout":
        return {"status": "timeout", "latency_ms": latency_ms, "hint": disc["hint"], "capabilities": caps}

    # 回退 chat 探测
    api_key, _ = _provider_key(raw_provider)
    base_url = str(raw_provider.get("base_url") or "").rstrip("/")
    model = str(raw_provider.get("model") or "").strip()
    if not api_key or not base_url or not model:
        return {
            "status": "request_failed",
            "latency_ms": latency_ms,
            "hint": "缺少 api_key/base_url/model，无法测试",
            "capabilities": caps,
        }
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    start = time.time()
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    except requests.Timeout:
        return {"status": "timeout", "latency_ms": int((time.time() - start) * 1000), "hint": "chat 探测超时", "capabilities": caps}
    except Exception as e:  # noqa: BLE001
        return {
            "status": "request_failed",
            "latency_ms": int((time.time() - start) * 1000),
            "hint": sanitize_llm_error(str(e), [api_key]),
            "capabilities": caps,
        }
    latency_ms = int((time.time() - start) * 1000)
    caps["latency_ms"] = latency_ms
    if resp.status_code == 401:
        return {"status": "auth_failed", "latency_ms": latency_ms, "hint": "API key 无效", "capabilities": caps}
    if resp.status_code != 200:
        detail = sanitize_llm_error(resp.text[:200], [api_key])
        return {"status": "request_failed", "latency_ms": latency_ms, "hint": f"HTTP {resp.status_code}: {detail}", "capabilities": caps}
    return {"status": "ok", "latency_ms": latency_ms, "hint": "chat 探测成功", "capabilities": caps}
