"""LLM 配置中心测试：发现、测试、持久化、API 契约、脱敏。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from engine.agent.llm_discover import discover_models, probe_provider
from engine.config import save_llm_providers, load_config


def test_discover_models_ok_and_cache(tmp_path, monkeypatch):
    monkeypatch.setitem(os.environ, "TEST_KEY", "sk-test123")
    cache = tmp_path / "cache.json"
    monkeypatch.setattr("engine.agent.llm_discover.CACHE_PATH", cache)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"data": [{"id": "m1"}, {"id": "m2"}]}

    with patch("engine.agent.llm_discover.requests.get", return_value=fake_resp) as mock_get:
        result = discover_models({
            "name": "p1", "base_url": "https://api.example.com", "api_key_env": "TEST_KEY"
        })
    assert result["status"] == "ok"
    assert result["models"] == ["m1", "m2"]
    assert result["cached"] is False

    # 第二次应命中缓存
    result2 = discover_models({
        "name": "p1", "base_url": "https://api.example.com", "api_key_env": "TEST_KEY"
    })
    assert result2["cached"] is True
    # 缓存命中不发起网络请求
    assert mock_get.call_count == 1


def test_discover_models_auth_failed(monkeypatch):
    monkeypatch.setitem(os.environ, "TEST_KEY", "sk-bad")
    fake_resp = MagicMock()
    fake_resp.status_code = 401
    with patch("engine.agent.llm_discover.requests.get", return_value=fake_resp):
        result = discover_models({
            "name": "p1", "base_url": "https://api.example.com", "api_key_env": "TEST_KEY"
        })
    assert result["status"] == "auth_failed"
    assert "sk-bad" not in str(result)


def test_probe_provider_chat_fallback(monkeypatch):
    monkeypatch.setitem(os.environ, "TEST_KEY", "sk-test")
    # /models 返回 404 -> 回退 chat
    model_resp = MagicMock()
    model_resp.status_code = 404
    chat_resp = MagicMock()
    chat_resp.status_code = 200

    def fake_get(url, **kwargs):
        return model_resp

    def fake_post(url, **kwargs):
        return chat_resp

    with patch("engine.agent.llm_discover.requests.get", side_effect=fake_get), \
         patch("engine.agent.llm_discover.requests.post", side_effect=fake_post):
        result = probe_provider({
            "name": "p1", "base_url": "https://api.example.com", "api_key_env": "TEST_KEY", "model": "m1"
        })
    assert result["status"] == "ok"
    assert result["hint"] == "chat 探测成功"


def test_save_llm_providers_rejects_plain_key_and_atomically_writes(tmp_path, monkeypatch):
    override = tmp_path / "llm_providers.json"
    monkeypatch.setattr("engine.config.LLM_OVERRIDE", override)
    providers = [
        {"name": "deepseek", "base_url": "https://api.deepseek.com", "api_key_env": "DS_KEY", "api_key": "sk-secret"},
    ]
    save_llm_providers(providers)
    assert override.exists()
    data = json.loads(override.read_text(encoding="utf-8"))
    saved = data["providers"][0]
    assert saved["api_key_env"] == "DS_KEY"
    assert "api_key" not in saved
    assert "sk-secret" not in override.read_text(encoding="utf-8")


def test_load_config_overrides_llm_providers(tmp_path, monkeypatch):
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text("llm:\n  providers: []\n", encoding="utf-8")
    override = tmp_path / "llm_providers.json"
    override.write_text(json.dumps({"providers": [{"name": "x", "base_url": "http://x", "api_key_env": "X"}]}), encoding="utf-8")
    monkeypatch.setattr("engine.config.LLM_OVERRIDE", override)
    cfg = load_config(str(yaml_path))
    assert cfg["llm"]["providers"][0]["name"] == "x"


def test_api_llm_providers_no_key_leak():
    """FastAPI 端点：列出 provider 时不应返回明文 key。"""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from engine.web.server import app

    with tempfile.TemporaryDirectory() as td:
        override = Path(td) / "llm_providers.json"
        override.write_text(json.dumps({
            "providers": [{
                "name": "deepseek",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "PANGU_LLM_API_KEY",
                "model": "deepseek-chat",
                "timeout": 300,
                "enabled": True,
            }]
        }), encoding="utf-8")
        import engine.config as config_mod
        orig_override = config_mod.LLM_OVERRIDE
        config_mod.LLM_OVERRIDE = override
        try:
            client = TestClient(app)
            resp = client.get("/api/llm/providers")
            assert resp.status_code == 200
            payload = resp.json()
            text = json.dumps(payload)
            assert '"api_key":' not in text
            assert "deepseek-chat" in text
        finally:
            config_mod.LLM_OVERRIDE = orig_override


def test_api_llm_providers_reject_plain_key_on_create():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from engine.web.server import app

    client = TestClient(app)
    resp = client.post("/api/llm/providers", json={
        "name": "evil", "base_url": "http://x", "api_key": "sk-secret", "model": "m"
    })
    assert resp.status_code == 400
    assert "明文" in resp.text or "api_key" in resp.text
