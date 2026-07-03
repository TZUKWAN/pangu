"""盘古 Web UI：FastAPI 后端 + 单页暗色 HTML 前端。

启动方式：
    python -m engine.web            # 默认 localhost:8000
    python -m engine.web --port 9000 --host 0.0.0.0

提供：
- GET  /            单页 HTML（暗色专业版选股看板）
- GET  /api/latest  最新选股结果（JSON）
- POST /api/scan    触发后台扫描（异步）
- GET  /api/scan/{task_id}/status  查扫描进度
- GET  /api/reports 历史报告列表
- GET  /api/llm/summary  SSE 流式 LLM 盘面解读

不修改 engine 任何核心逻辑，仅以只读方式 import Pipeline。
"""

from __future__ import annotations

from .server import app, run

__all__ = ["app", "run"]
