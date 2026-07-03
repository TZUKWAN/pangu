"""通用 HTTP 通知推送模块。

支持任意 webhook（Bark / Server酱 / 飞书 webhook / 企业微信 / 钉钉 等）。
默认通过环境变量配置，未配置时安全跳过，不会误发。

环境变量：
    PANGU_NOTIFY_WEBHOOK     推送 webhook URL
    PANGU_NOTIFY_METHOD      HTTP 方法，默认 POST
    PANGU_NOTIFY_HEADERS     额外请求头 JSON，默认 {"Content-Type": "application/json"}
    PANGU_NOTIFY_TEMPLATE    可选 jinja2/json 模板字符串；留空则用内置摘要模板
    PANGU_NOTIFY_TIMEOUT     请求超时秒数，默认 15
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger("pangu.notifier")


@dataclass
class NotifyConfig:
    webhook: str | None = None
    method: str = "POST"
    headers: dict[str, str] | None = None
    template: str | None = None
    timeout: int = 15


class Notifier:
    """通用 webhook 通知器。"""

    def __init__(self, cfg: NotifyConfig) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.webhook)

    @classmethod
    def from_env(cls) -> "Notifier":
        """从环境变量构造通知器。"""
        headers_str = os.environ.get("PANGU_NOTIFY_HEADERS", "")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if headers_str:
            try:
                headers.update(json.loads(headers_str))
            except json.JSONDecodeError:
                logger.warning("PANGU_NOTIFY_HEADERS 不是合法 JSON，使用默认请求头")
        return cls(NotifyConfig(
            webhook=os.environ.get("PANGU_NOTIFY_WEBHOOK") or None,
            method=os.environ.get("PANGU_NOTIFY_METHOD", "POST").upper() or "POST",
            headers=headers,
            template=os.environ.get("PANGU_NOTIFY_TEMPLATE") or None,
            timeout=int(os.environ.get("PANGU_NOTIFY_TIMEOUT", "15") or "15"),
        ))

    def _render_body(self, summary: dict[str, Any]) -> dict[str, Any]:
        """渲染通知体。未配置模板时返回结构化摘要。"""
        if self.cfg.template:
            try:
                # 简单字符串模板：用 summary 的字段做 {field} 替换
                text = self.cfg.template
                for key, val in summary.items():
                    placeholder = "{" + key + "}"
                    if placeholder in text:
                        text = text.replace(placeholder, self._fmt(val))
                return {"text": text}
            except Exception as e:  # noqa: BLE001
                logger.warning("通知模板渲染失败，使用默认摘要: %s", e)
        return self._default_body(summary)

    @staticmethod
    def _fmt(val: Any) -> str:
        if isinstance(val, list):
            return "\n".join(str(x) for x in val)
        return str(val)

    @staticmethod
    def _default_body(summary: dict[str, Any]) -> dict[str, Any]:
        top = summary.get("top_candidates") or []
        lines = [
            f"盘古盘后分析 · {summary.get('date', '-')}",
            f"情绪温度 {summary.get('temperature', '-')} / {summary.get('posture', '-')}",
            f"候选股 {summary.get('candidate_count', 0)} 只",
        ]
        if top:
            lines.append("Top 候选:")
            for c in top:
                lines.append(f"  {c.get('code')} {c.get('name')} {c.get('grade')}级 {c.get('score')}分")
        report_path = summary.get("report_path")
        if report_path:
            lines.append(f"报告: {report_path}")
        warnings = summary.get("warnings") or []
        if warnings:
            lines.append("提醒: " + "; ".join(str(w) for w in warnings))
        return {
            "title": f"盘古盘后分析 {summary.get('date', '-')}",
            "text": "\n".join(lines),
            "summary": summary,
        }

    def send(self, summary: dict[str, Any]) -> dict[str, Any]:
        """发送通知。未启用时安全跳过。"""
        if not self.enabled or not self.cfg.webhook:
            return {"sent": False, "reason": "PANGU_NOTIFY_WEBHOOK 未配置，通知已跳过"}

        body = self._render_body(summary)
        method = self.cfg.method
        headers = self.cfg.headers or {"Content-Type": "application/json"}
        try:
            logger.info("[notifier] 发送通知到 %s", self.cfg.webhook)
            resp = requests.request(
                method=method,
                url=self.cfg.webhook,
                json=body,
                headers=headers,
                timeout=self.cfg.timeout,
            )
            resp.raise_for_status()
            return {
                "sent": True,
                "status_code": resp.status_code,
                "response_preview": resp.text[:200],
            }
        except requests.exceptions.Timeout:
            logger.warning("[notifier] 通知超时")
            return {"sent": False, "reason": "通知请求超时"}
        except requests.exceptions.RequestException as e:
            logger.warning("[notifier] 通知失败: %s", e)
            return {"sent": False, "reason": f"通知请求失败: {type(e).__name__}: {e}"}
        except Exception as e:  # noqa: BLE001
            logger.warning("[notifier] 通知异常: %s", e)
            return {"sent": False, "reason": f"通知异常: {type(e).__name__}: {e}"}
