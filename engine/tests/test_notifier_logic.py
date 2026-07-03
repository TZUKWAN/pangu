"""通知推送逻辑测试：验证缺 env 时安全跳过、fake endpoint 时正确发送。"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from engine.notifier import Notifier, NotifyConfig


def test_notifier_disabled_without_webhook(monkeypatch):
    """未配置 PANGU_NOTIFY_WEBHOOK 时 not enabled，发送安全跳过。"""
    monkeypatch.delenv("PANGU_NOTIFY_WEBHOOK", raising=False)
    notifier = Notifier.from_env()
    assert notifier.enabled is False
    result = notifier.send({"date": "20260101"})
    assert result["sent"] is False
    assert "未配置" in result["reason"]


def test_notifier_enabled_with_webhook(monkeypatch):
    """配置 PANGU_NOTIFY_WEBHOOK 后启用。"""
    monkeypatch.setenv("PANGU_NOTIFY_WEBHOOK", "https://example.com/hook")
    notifier = Notifier.from_env()
    assert notifier.enabled is True


def test_notifier_send_to_fake_endpoint(monkeypatch):
    """构造 fake endpoint 发送通知，验证请求体包含关键摘要。"""
    monkeypatch.setenv("PANGU_NOTIFY_WEBHOOK", "https://example.com/hook")
    notifier = Notifier.from_env()

    summary = {
        "date": "20260101",
        "temperature": 72.5,
        "posture": "正常",
        "candidate_count": 3,
        "top_candidates": [
            {"code": "000001", "name": "平安银行", "grade": "A", "score": 85},
        ],
        "report_path": "data/reports/20260101.md",
        "warnings": [],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "ok"

    with patch("requests.request", return_value=mock_resp) as mock_req:
        result = notifier.send(summary)

    assert result["sent"] is True
    assert result["status_code"] == 200
    call_kwargs = mock_req.call_args.kwargs
    assert call_kwargs["url"] == "https://example.com/hook"
    assert call_kwargs["method"] == "POST"
    body = call_kwargs["json"]
    assert "盘古盘后分析" in body["title"]
    assert "000001" in body["text"]
    assert "平安银行" in body["text"]
    assert body["summary"]["candidate_count"] == 3


def test_notifier_handles_request_exception(monkeypatch):
    """请求异常时返回 sent=False 与原因，不抛异常。"""
    monkeypatch.setenv("PANGU_NOTIFY_WEBHOOK", "https://example.com/hook")
    notifier = Notifier.from_env()

    with patch("requests.request", side_effect=requests.exceptions.ConnectTimeout("timeout")):
        result = notifier.send({"date": "20260101"})

    assert result["sent"] is False
    assert "超时" in result["reason"] or "失败" in result["reason"]


def test_notifier_custom_template(monkeypatch):
    """自定义模板字符串替换。"""
    monkeypatch.setenv("PANGU_NOTIFY_WEBHOOK", "https://example.com/hook")
    monkeypatch.setenv("PANGU_NOTIFY_TEMPLATE", "日期：{date}，候选：{candidate_count}")
    notifier = Notifier.from_env()
    body = notifier._render_body({"date": "20260101", "candidate_count": 5})
    assert body["text"] == "日期：20260101，候选：5"


def test_notifier_custom_headers(monkeypatch):
    """PANGU_NOTIFY_HEADERS 被正确解析并合并。"""
    monkeypatch.setenv("PANGU_NOTIFY_WEBHOOK", "https://example.com/hook")
    monkeypatch.setenv("PANGU_NOTIFY_HEADERS", json.dumps({"Authorization": "Bearer token123"}))
    notifier = Notifier.from_env()
    assert notifier.cfg.headers["Authorization"] == "Bearer token123"
    assert notifier.cfg.headers["Content-Type"] == "application/json"
