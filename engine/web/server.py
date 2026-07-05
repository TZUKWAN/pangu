"""FastAPI 应用：选股引擎的 HTTP/SSE 网关。

设计要点：
1. Pipeline 是同步阻塞的（akshare 取数要 1-3 分钟），用 threading.Thread 跑扫描，
   主线程立即返回 task_id，前端轮询 status。
2. LLM 流式用 SSE：调 OpenAICompatibleClient.stream_chat() 生成器，
   包成 text/event-stream 响应，前端 EventSource 逐字接收。
3. 全局单例 Pipeline 复用 DataLoader 缓存，避免每次重建。
4. 线程安全：扫描任务有全局锁，同一时刻只跑一个（akshare 限流，并发会炸）。

启动：python -m engine.web
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
import asyncio
import copy
import importlib.util
from datetime import date as date_cls, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import build_data_loader, load_config, save_llm_providers, save_strategy_settings
from ..agent.debate import get_agent_prompts
from ..data_loader import find_col, safe_float
from ..market_phase import MarketPhaseAnalyzer
from ..pipeline import Pipeline
from ..strategy_pools import run_all_pools
from ..scheduler import DailyScheduler

logger = logging.getLogger("pangu.web")

# ---------------------------------------------------------------------- #
# 全局状态：单例 Pipeline + 扫描任务表
# ---------------------------------------------------------------------- #
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_REPORT_DIR = Path("data/reports")

# 扫描任务状态
class _TaskState:
    """一次后台扫描的状态。"""

    def __init__(self, task_id: str, date: Optional[str]) -> None:
        self.task_id = task_id
        self.date = date
        self.status: str = "pending"  # pending / running / done / failed
        self.logs: list[str] = []
        self.result: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.started_at = time.time()

    def log(self, msg: str) -> None:
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        # 只保留最近 50 条，防爆内存
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]


_tasks: dict[str, _TaskState] = {}
_latest_result: Optional[dict[str, Any]] = None
_trade_calendar_cache: Optional[tuple[float, set[str], str]] = None

# Pipeline 单例（惰性构造）
_pipeline: Optional[Pipeline] = None
_pipeline_lock = threading.Lock()

# DataLoader 全局单例（共享内存缓存，避免每个请求重建+重新拉数据）
_dl: Any = None
_dl_lock = threading.Lock()


def _get_dl() -> Any:
    """获取全局 DataLoader 单例（线程安全）。"""
    global _dl
    if _dl is None:
        with _dl_lock:
            if _dl is None:
                from ..data_loader import DataLoader
                _dl = DataLoader()
    return _dl


_warmed_up = False
_warmup_lock = threading.Lock()


def _ensure_warmup():
    """后台线程预热：首次调用时触发，后续调用立即返回。"""
    global _warmed_up
    if _warmed_up:
        return
    with _warmup_lock:
        if _warmed_up:
            return
        try:
            _get_dl().all_spot()  # 拉取 all_spot 到内存缓存
            _warmed_up = True
            logger.info("all_spot 预热完成")
        except Exception as e:
            logger.warning("预热失败: %s", e)


def _beijing_now() -> datetime:
    """返回 Asia/Shanghai 时区的当前时间。"""
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def _parse_schedule_time(cfg: dict[str, Any]) -> Optional[dt_time]:
    """解析配置中的 schedule.scan_time/time，默认 15:05。"""
    schedule_cfg = cfg.get("schedule") or {}
    raw = schedule_cfg.get("scan_time") or schedule_cfg.get("time") or "15:05"
    if not raw:
        return None
    try:
        parts = raw.split(":")
        return dt_time(int(parts[0]), int(parts[1]))
    except Exception:
        logger.warning("schedule.scan_time 格式错误: %s，使用默认 15:05", raw)
        return dt_time(15, 5)


def _next_run_time(schedule_time: dt_time, now: Optional[datetime] = None) -> datetime:
    """计算下一个调度时刻（Asia/Shanghai）。"""
    now = now or _beijing_now()
    target = now.replace(hour=schedule_time.hour, minute=schedule_time.minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def _is_trading_day(date_str: str) -> bool:
    """粗略判断是否为 A 股交易日（周一到周五）。"""
    try:
        d = datetime.strptime(date_str, "%Y%m%d").date()
        return d.weekday() < 5
    except Exception:
        return True


def _run_scheduled_daily() -> None:
    """后台调度线程：每日收盘后自动跑盘后链路并刷新内存缓存。"""
    cfg = load_config()
    schedule_cfg = cfg.get("schedule") or {}
    if not schedule_cfg.get("enabled", False):
        logger.info("自动调度已禁用（schedule.enabled=false）")
        return

    schedule_time = _parse_schedule_time(cfg)
    if schedule_time is None:
        return

    logger.info("自动调度已启用，每日 %s (Asia/Shanghai) 执行盘后链路", schedule_time.strftime("%H:%M"))
    while True:
        now = _beijing_now()
        target = _next_run_time(schedule_time, now)
        sleep_seconds = (target - now).total_seconds()
        logger.info("下次自动调度: %s，约 %.0f 秒后", target.isoformat(), sleep_seconds)
        time.sleep(max(1.0, sleep_seconds))

        now = _beijing_now()
        date_str = now.strftime("%Y%m%d")
        if not _is_trading_day(date_str):
            logger.info("%s 非 A 股交易日，跳过今日自动调度", date_str)
            continue

        # 避免和手动扫描冲突：复用全局任务锁
        running = next((t for t in _tasks.values() if t.status in ("pending", "running")), None)
        if running is not None:
            logger.warning("已有扫描任务 %s 在运行，自动调度跳过", running.task_id)
            continue

        task_id = uuid.uuid4().hex[:12]
        state = _TaskState(task_id, date_str)
        _tasks[task_id] = state
        state.status = "running"
        state.log(f"自动调度开始 date={date_str}")
        try:
            scheduler = DailyScheduler(cfg=cfg, date=date_str, workers=10)
            summary = scheduler.run()
            state.status = "done"
            state.log(f"自动调度完成: {summary.get('overall_status')}，候选 {summary.get('candidate_count', 0)} 只")
            # 刷新内存缓存：加载最新报告
            latest = _find_latest_report()
            if latest is not None:
                global _latest_result
                _latest_result = _enrich_response(latest)
                logger.info("自动调度已刷新内存缓存: %s", latest.get("date"))
        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            state.log(f"自动调度失败: {e}")
            logger.exception("自动调度失败")


def _start_scheduler() -> None:
    """启动后台自动调度线程。"""
    t = threading.Thread(target=_run_scheduled_daily, daemon=True, name="pangu-scheduler")
    t.start()


def _build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    """从配置构造 Pipeline（与 cli._build_pipeline 保持一致）。"""
    return Pipeline(
        dl=build_data_loader(cfg),
        sentiment_cfg=cfg.get("sentiment", {}),
        trend_cfg=cfg.get("trend", {}),
        guard_cfg=cfg.get("guard", {}),
        entry_exit_cfg=cfg.get("entry_exit", cfg),  # 期望整个 cfg
        pick_count=cfg.get("output", {}).get("pick_count", 5),
        db_path=cfg.get("output", {}).get("db_path", "data/pangu.db"),
        full_cfg=cfg,
    )


def get_pipeline() -> Pipeline:
    """获取全局 Pipeline 单例（线程安全）。

    扫描入口和新闻刷新入口都复用这一个实例，避免同一服务进程内反复构造
    DataLoader、重复预热缓存或产生配置不一致。
    """
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = _build_pipeline(load_config())
    return _pipeline


def _beijing_now() -> datetime:
    """返回北京时间，精确到秒。"""
    return datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=0)


def _dates_from_akshare_calendar_json() -> set[str]:
    """Read akshare's bundled calendar.json without importing akshare itself."""
    spec = importlib.util.find_spec("akshare")
    roots = list(spec.submodule_search_locations or []) if spec else []
    for root in roots:
        path = Path(root) / "file_fold" / "calendar.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        dates = {str(v).replace("-", "")[:8] for v in data if str(v).strip()}
        dates = {v for v in dates if len(v) == 8 and v.isdigit()}
        if dates:
            return dates
    return set()


# 公开 HTTP 明文交易日历兜底源（不依赖 akshare/py_mini_racer）
_CALENDAR_HTTP_URLS: list[str] = [
    # akshare 仓库内明文 calendar.json（YYYYMMDD 字符串数组），格式与本地一致
    "https://raw.githubusercontent.com/akfamily/akshare/main/akshare/file_fold/calendar.json",
]


def _dates_from_http_calendar() -> set[str]:
    """公开 HTTP 明文交易日历兜底（不依赖 akshare/py_mini_racer），超时/失败返回空集。

    在 akshare 主源与本地 calendar.json 均不可用时启用；对任意返回（JSON 数组或纯
    文本）正则提取 8 位日期，数量足够（>=200）才采信，避免误把零散数字当日历。
    """
    import urllib.parse
    import urllib.request
    for url in _CALENDAR_HTTP_URLS:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 PanguCalendar"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                text = resp.read().decode("utf-8", errors="ignore")
            dates = {s for s in re.findall(r"\d{8}", text) if "19900101" <= s <= "20991231"}
            if len(dates) >= 200:
                return dates
        except Exception:  # noqa: BLE001
            continue
    return set()


def _load_trade_calendar() -> tuple[set[str], list[str], str]:
    """加载 A 股交易日历；失败时仅按周末降级。"""
    global _trade_calendar_cache
    now_ts = time.time()
    if _trade_calendar_cache and now_ts - _trade_calendar_cache[0] < 6 * 3600:
        return _trade_calendar_cache[1], [], _trade_calendar_cache[2]
    warnings: list[str] = []
    dates: set[str] = set()
    try:
        import akshare as ak  # type: ignore
        df = ak.tool_trade_date_hist_sina()
        if df is not None and len(df) > 0:
            col = "trade_date" if "trade_date" in df.columns else df.columns[0]
            for v in df[col].dropna().astype(str):
                dates.add(v.replace("-", "")[:8])
    except Exception as e:  # noqa: BLE001
        warnings.append(f"交易日历主源 akshare 不可用：{e}")
    if dates:
        _trade_calendar_cache = (now_ts, dates, "akshare.tool_trade_date_hist_sina")
        return dates, warnings, "akshare.tool_trade_date_hist_sina"
    try:
        dates = _dates_from_akshare_calendar_json()
    except Exception as e:  # noqa: BLE001
        warnings.append(f"交易日历内置 calendar.json 兜底不可用：{e}")
        dates = set()
    if dates:
        _trade_calendar_cache = (now_ts, dates, "akshare.file_fold.calendar_json")
        return dates, warnings, "akshare.file_fold.calendar_json"
    # 公开 HTTP 明文兜底（不依赖 akshare/py_mini_racer）
    try:
        dates = _dates_from_http_calendar()
    except Exception as e:  # noqa: BLE001
        warnings.append(f"交易日历 HTTP 明文兜底不可用：{e}")
        dates = set()
    if dates:
        _trade_calendar_cache = (now_ts, dates, "http.public_calendar")
        return dates, warnings, "http.public_calendar"
    if not warnings:
        warnings.append("交易日历依赖缺失，当前仅按周六周日判断交易日，节假日不可完整覆盖")
    else:
        warnings.append("交易日历已降级为仅识别周末，节假日不可完整覆盖")
    return set(), warnings, "weekday_fallback"


def _next_weekday(d: date_cls) -> date_cls:
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _next_trade_date_from_calendar(today: date_cls, calendar: set[str]) -> str:
    d = today + timedelta(days=1)
    for _ in range(370):
        key = d.strftime("%Y%m%d")
        if key in calendar:
            return key
        d += timedelta(days=1)
    return _next_weekday(today).strftime("%Y%m%d")


def _runtime_context() -> dict[str, Any]:
    """构建 GUI 每次刷新都应展示的真实运行时上下文。"""
    now = _beijing_now()
    today_key = now.strftime("%Y%m%d")
    loaded_calendar = _load_trade_calendar()
    if len(loaded_calendar) == 2:
        calendar, warnings = loaded_calendar  # type: ignore[misc]
        calendar_source = "akshare.tool_trade_date_hist_sina" if calendar else "weekday_fallback"
    else:
        calendar, warnings, calendar_source = loaded_calendar
    if calendar:
        is_trade_day = today_key in calendar
        target = today_key if is_trade_day and now.time() < dt_time(15, 0) else _next_trade_date_from_calendar(now.date(), calendar)
        if calendar_source == "akshare.tool_trade_date_hist_sina":
            coverage = "含交易所节假日历"
        elif calendar_source == "http.public_calendar":
            coverage = "含公开 HTTP 交易日历兜底"
        else:
            coverage = "含 akshare 内置交易日历，覆盖到内置文件末端"
    else:
        is_trade_day = now.weekday() < 5
        calendar_source = "weekday_fallback"
        target = today_key if is_trade_day and now.time() < dt_time(15, 0) else _next_weekday(now.date()).strftime("%Y%m%d")
        coverage = "仅周末降级，不保证节假日覆盖"
        if not warnings:
            warnings.append("交易日历依赖缺失，当前仅按周六周日判断交易日，节假日不可完整覆盖")
    session = "盘中" if is_trade_day and dt_time(9, 30) <= now.time() < dt_time(15, 0) else ("盘后" if is_trade_day and now.time() >= dt_time(15, 0) else "非交易时段")
    return {
        "now": now.isoformat(),
        "now_text": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "Asia/Shanghai",
        "is_trade_day": is_trade_day,
        "market_session": session,
        "recommended_trade_date": target,
        "recommendation_rule": "15:00 前推荐当天；15:00 后或非交易日推荐下一交易日",
        "calendar_source": calendar_source,
        "calendar_coverage": coverage,
        "warnings": warnings,
    }


def _source_status(data: dict[str, Any]) -> dict[str, Any]:
    warnings = list(data.get("warnings") or [])
    warnings += list((data.get("sentiment") or {}).get("warnings") or [])
    news_warnings = (data.get("news") or {}).get("warnings") or []
    warnings += list(news_warnings)
    structured = data.get("source_state") or {}
    structured_status = "unknown"
    structured_warnings: list[str] = []
    structured_payload = structured.get("structured_data") if isinstance(structured, dict) else None
    if isinstance(structured_payload, dict):
        states = [
            v.get("status")
            for k, v in structured_payload.items()
            if isinstance(v, dict) and k != "summary"
        ]
        if states:
            if any(s == "ok" for s in states):
                structured_status = "degraded" if any(s in {"degraded", "unavailable"} for s in states) else "ok"
            else:
                structured_status = "unavailable" if any(s == "unavailable" for s in states) else "empty"
        for key, value in structured_payload.items():
            if isinstance(value, dict):
                for w in value.get("warnings") or []:
                    structured_warnings.append(f"{key}: {w}")
    warnings += structured_warnings
    margin = (((data.get("sentiment") or {}).get("market_structure") or {}).get("margin") or {})
    if margin.get("warning"):
        warnings.append(str(margin["warning"]))
    return {
        "market_data": "degraded" if warnings else "ok",
        "news": "degraded" if news_warnings else ("ok" if data.get("news") else "unknown"),
        "structured_data": structured_status,
        "margin": "unavailable" if margin.get("balance_yi") is None else "ok",
        "warnings": warnings[:12],
    }


def _report_status(data: dict[str, Any]) -> dict[str, Any]:
    report_date = str(data.get("date") or "")
    # 优先检查 P0 JSON，再回退普通 JSON
    json_path = None
    md_path = _REPORT_DIR / f"{report_date}.md" if report_date else None
    if report_date:
        for suffix in (f"{report_date}_p0.json", f"{report_date}.json"):
            candidate = _REPORT_DIR / suffix
            if candidate.exists():
                json_path = candidate
                break
    latest_json = None
    files = _list_report_paths()
    latest_json = files[0] if files else None
    latest_date = latest_json.stem if latest_json else None
    generated_at = None
    if json_path and json_path.exists():
        generated_at = datetime.fromtimestamp(json_path.stat().st_mtime, tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    now = _beijing_now()
    now_date = now.strftime("%Y%m%d")
    is_stale = bool(report_date and report_date < now_date)

    data_quality = data.get("data_quality", "unknown")
    tradable = bool(data.get("tradable"))
    no_trade_reason = data.get("no_trade_reason") or None
    block_reasons = list(data.get("block_reasons") or [])
    final_count = len(data.get("final_recommendations") or [])
    watch_count = len(data.get("watchlist") or [])
    raw_candidate_count = len(data.get("candidates") or [])

    if data_quality in ("failed", "degraded"):
        freshness_status = "degraded"
        freshness_note = f"数据降级（{data_quality}），未生成可信正式推荐"
    elif is_stale:
        freshness_status = "historical"
        freshness_note = f"当前展示的是 {report_date} 历史报告，不是 {now_date} 实时新扫描结果；点击刷新扫描才会运行主链路"
    elif report_date:
        freshness_status = "fresh"
        freshness_note = f"当前报告日期为 {report_date}"
    else:
        freshness_status = "empty"
        freshness_note = "暂无报告"

    return {
        "date": report_date or None,
        "current_date": now_date,
        "has_json": bool(json_path and json_path.exists()),
        "has_md": bool(md_path and md_path.exists()),
        "json_path": str(json_path) if json_path and json_path.exists() else None,
        "md_path": str(md_path) if md_path and md_path.exists() else None,
        "latest_report_date": latest_date,
        "generated_at": generated_at,
        "is_stale": is_stale,
        "freshness_status": freshness_status,
        "freshness_note": freshness_note,
        "status": "ok" if data and not data.get("empty") else "empty",
        "data_quality": data_quality,
        "tradable": tradable,
        "no_trade_reason": no_trade_reason,
        "block_reasons": block_reasons,
        "final_count": final_count,
        "watch_count": watch_count,
        "raw_candidate_count": raw_candidate_count,
    }


def _sentiment_report(data: dict[str, Any]) -> dict[str, Any]:
    """Build a transparent multi-role sentiment synthesis for the UI.

    This is deterministic report synthesis unless candidate-level LLM debates are
    already present in the report. It never claims a live LLM call happened.
    """
    sentiment = data.get("sentiment") or {}
    source = data.get("source_status") or {}
    news = data.get("news") or {}
    boards = data.get("boards") or []
    xsum = ((data.get("xuanwu_pool") or {}).get("summary") or {})
    debate = ((data.get("daily_loop") or {}).get("debate") or _debate_status(data.get("candidates") or []))
    report = data.get("report_status") or _report_status(data)

    temp = safe_float(sentiment.get("temperature"), 50.0)
    coverage = safe_float(debate.get("coverage_pct"), 0.0)
    confidence = 42.0
    confidence += min(25.0, coverage * 0.25)
    confidence += 9.0 if source.get("news") == "ok" else -6.0
    confidence += 8.0 if source.get("structured_data") == "ok" else -6.0
    confidence += 7.0 if source.get("market_data") == "ok" else -6.0
    if report.get("is_stale"):
        confidence -= 12.0
    if xsum.get("xuanwu_count"):
        confidence += min(8.0, float(xsum.get("xuanwu_count") or 0) * 1.5)
    confidence = round(max(5.0, min(95.0, confidence)), 1)

    top_boards = boards[:3]
    themes = news.get("hot_themes") or []
    flashes = news.get("flashes") or []
    blockers = xsum.get("top_blockers") or []
    board_names = [str(b.get("name") or "") for b in top_boards if isinstance(b, dict)]
    theme_names = []
    for t in themes[:4]:
        theme_names.append(str(t[0] if isinstance(t, (list, tuple)) and t else (t.get("name") or t.get("theme") if isinstance(t, dict) else t)))

    if temp >= 70:
        market_view = "情绪偏热，优先等分歧后的回踩确认"
    elif temp >= 55:
        market_view = "情绪可交易，但只接受板块、量价和计划共振"
    elif temp >= 40:
        market_view = "情绪中性偏弱，仓位应轻，等待确定性"
    else:
        market_view = "情绪低温，玄武池应以防守和复盘为主"

    agents = [
        {
            "role": "情绪温度员",
            "stance": market_view,
            "confidence": round(max(10.0, min(92.0, temp)), 1),
            "evidence": [f"情绪温度 {round(temp, 1)}", str(sentiment.get("posture") or "未知状态")],
        },
        {
            "role": "板块轮动员",
            "stance": "优先验证 " + "、".join(board_names[:3]) if board_names else "板块映射不足，不能把候选视为主线票",
            "confidence": 72 if board_names else 35,
            "evidence": board_names[:3] or ["缺少可靠板块到个股映射"],
        },
        {
            "role": "新闻舆情员",
            "stance": "新闻催化集中在 " + "、".join(theme_names[:3]) if theme_names else "新闻主题不足，催化证据偏弱",
            "confidence": 76 if (theme_names or flashes) else 38,
            "evidence": theme_names[:3] or [f"快讯 {len(flashes)} 条"],
        },
        {
            "role": "风险审查员",
            "stance": "玄武池暂不放票" if not xsum.get("xuanwu_count") else f"玄武池通过 {xsum.get('xuanwu_count')} 只，仍需按买点执行",
            "confidence": 82 if blockers or not xsum.get("xuanwu_count") else 70,
            "evidence": [str(b[0]) for b in blockers[:3]] or ["暂无主要阻断项"],
        },
        {
            "role": "多智能体协调员",
            "stance": str(debate.get("message") or "候选级多空辩论未覆盖，不能直接进入推荐池"),
            "confidence": round(max(10.0, min(92.0, coverage)), 1),
            "evidence": [f"覆盖率 {coverage:.1f}%", f"缺失 {debate.get('missing_count', 0)} 只"],
        },
    ]
    if not xsum.get("xuanwu_count") and coverage >= 95:
        position_text = "候选已完成规则多空验证，但没有股票通过玄武硬闸门；默认轻仓或空仓等待回踩、量价转强和数据源恢复。"
    elif not xsum.get("xuanwu_count"):
        position_text = "当前没有充分验证的玄武推荐，默认轻仓或空仓等待；只把观察池当作次日预案。"
    else:
        position_text = "存在少量玄武候选，但仍只在回踩买点和止损条件成立时执行。"

    conclusions = [
        {
            "title": "仓位结论",
            "level": "defensive" if confidence < 50 or not xsum.get("xuanwu_count") else "selective",
            "text": position_text,
        },
        {
            "title": "主线结论",
            "level": "watch" if board_names else "risk",
            "text": "优先观察 " + "、".join(board_names[:3]) if board_names else "板块到个股映射不足，主线判断不能落到具体股票。",
        },
        {
            "title": "风险结论",
            "level": "risk" if source.get("market_data") != "ok" or report.get("is_stale") else "watch",
            "text": (source.get("warnings") or [report.get("freshness_note") or "暂无额外风险提示"])[0],
        },
    ]
    return {
        "mode": "rule_multi_role_synthesis",
        "confidence": confidence,
        "coverage_pct": coverage,
        "freshness_status": report.get("freshness_status"),
        "summary": conclusions[0]["text"],
        "agents": agents,
        "conclusions": conclusions,
        "warnings": (source.get("warnings") or [])[:6] + (debate.get("warnings") or [])[:4],
    }


def _trend_windows_from_kline(kline: list[dict[str, Any]]) -> dict[str, Any]:
    """Backfill 1w/2w/1m trend windows from existing K-line rows."""
    if not kline:
        return {}
    closes = [safe_float(row.get("close"), 0.0) for row in kline if isinstance(row, dict)]
    if not closes or closes[-1] <= 0:
        return {}
    close = closes[-1]
    out: dict[str, Any] = {}
    for key, window in (("week_1", 5), ("week_2", 10), ("month_1", 20)):
        if len(kline) <= window:
            continue
        ref = safe_float(kline[-window - 1].get("close"), 0.0)
        tail = kline[-window:]
        highs = [safe_float(row.get("high", row.get("close")), 0.0) for row in tail if isinstance(row, dict)]
        lows = [safe_float(row.get("low", row.get("close")), 0.0) for row in tail if isinstance(row, dict)]
        recent_high = max(highs) if highs else close
        positive_lows = [v for v in lows if v > 0]
        recent_low = min(positive_lows) if positive_lows else close
        ret_pct = (close / ref - 1) * 100 if ref > 0 else 0.0
        pullback_pct = (close / recent_high - 1) * 100 if recent_high > 0 else 0.0
        rebound_pct = (close / recent_low - 1) * 100 if recent_low > 0 else 0.0
        if ret_pct > 8 and pullback_pct > -2:
            state = "加速上涨，避免追高"
        elif ret_pct > 0 and pullback_pct <= -3:
            state = "趋势内回踩"
        elif ret_pct < -5:
            state = "走弱修复"
        else:
            state = "震荡观察"
        out[key] = {
            "days": window,
            "return_pct": round(ret_pct, 2),
            "pullback_from_high_pct": round(pullback_pct, 2),
            "rebound_from_low_pct": round(rebound_pct, 2),
            "state": state,
        }
    return out


def _sanitize_floats(obj: Any) -> Any:
    """递归把 nan/inf 替换为 None，保证 JSON 序列化合法。"""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _backfill_candidate_trend_windows(data: dict[str, Any]) -> None:
    """Ensure old reports expose explicit 1w/2w/1m windows without refetching data."""
    for bucket in ("candidates", "final_recommendations", "watchlist"):
        for c in data.get(bucket) or []:
            technical = c.get("technical")
            if not isinstance(technical, dict):
                continue
            if technical.get("trend_windows"):
                continue
            windows = _trend_windows_from_kline(technical.get("kline") or [])
            if windows:
                technical["trend_windows"] = windows


def _backfill_candidate_debates(data: dict[str, Any]) -> bool:
    """Ensure every candidate has an auditable debate structure.

    Existing LLM/rule debates are preserved. Missing rows get deterministic
    rule-validation debates so the UI never suggests an item was simply
    inserted without being challenged.
    """
    all_items: list[dict[str, Any]] = []
    for bucket in ("candidates", "final_recommendations", "watchlist"):
        all_items.extend(data.get(bucket) or [])
    if not all_items:
        return False
    try:
        from ..agent.debate import StockDebater
        debater = StockDebater(cfg=load_config())
    except Exception:
        return False
    hot_themes = ((data.get("news") or {}).get("hot_themes") or [])
    changed = False
    for c in all_items:
        debate = c.get("debate") or {}
        if debate.get("verdict"):
            continue
        code = str(c.get("code") or "")
        if not code:
            continue
        c["debate"] = debater.rule_validate(
            code,
            str(c.get("name") or code),
            c,
            news_sentiment=None,
            hot_themes=hot_themes,
        )
        changed = True
    return changed


def _cache_status() -> dict[str, Any]:
    try:
        cfg = load_config()
        cache_dir = Path((cfg.get("data") or {}).get("cache_dir", "data/cache"))
        if not cache_dir.exists():
            return {"status": "missing", "cache_dir": str(cache_dir), "files": 0}
        files = [p for p in cache_dir.glob("*") if p.is_file()]
        latest = max((p.stat().st_mtime for p in files), default=None)
        latest_text = (
            datetime.fromtimestamp(latest, tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
            if latest else None
        )
        return {"status": "ok" if files else "empty", "cache_dir": str(cache_dir), "files": len(files), "latest_cache_time": latest_text}
    except Exception as e:  # noqa: BLE001
        return {"status": "unknown", "warning": f"缓存状态读取失败: {e}"}


def _debate_status(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(candidates or [])
    with_debate = 0
    modes: dict[str, int] = {}
    verdicts: dict[str, int] = {}
    missing: list[str] = []
    warnings: list[str] = []
    for c in candidates or []:
        debate = c.get("debate") or {}
        code = str(c.get("code") or "")
        if not debate or not debate.get("verdict"):
            if code:
                missing.append(code)
            continue
        with_debate += 1
        mode = str(debate.get("mode") or "unknown")
        verdict = str(debate.get("verdict") or "unknown")
        modes[mode] = modes.get(mode, 0) + 1
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        if not isinstance(debate.get("bull_points"), list) or not isinstance(debate.get("bear_points"), list):
            warnings.append(f"{code} 多空论点不是数组，前端已兼容但后端契约需复核")
    missing_count = total - with_debate
    if total and missing_count:
        warnings.append(f"{missing_count} 只候选股缺少多空辩论结果")
    status = "ok" if total and missing_count == 0 and not warnings else ("degraded" if total else "empty")
    message = None
    if total:
        if missing_count:
            message = f"多空辩论降级：仅 {with_debate}/{total} 候选有覆盖（{round(with_debate / total * 100, 1)}%），{missing_count} 只缺失"
        else:
            message = f"多空辩论覆盖：{with_debate}/{total} 候选有覆盖（{round(with_debate / total * 100, 1)}%）"
    return {
        "status": status,
        "candidate_count": total,
        "with_debate": with_debate,
        "coverage_pct": round(with_debate / total * 100, 2) if total else 0.0,
        "missing_count": missing_count,
        "message": message,
        "modes": modes,
        "verdicts": verdicts,
        "missing_codes": missing[:20],
        "warnings": warnings[:8],
    }


def _news_status(data: dict[str, Any]) -> dict[str, Any]:
    news = data.get("news") or {}
    flashes = news.get("flashes") or []
    themes = news.get("hot_themes") or []
    candidate_news = 0
    sources: set[str] = set()
    for c in data.get("candidates") or []:
        for item in c.get("stock_news") or []:
            candidate_news += 1
            if item.get("source"):
                sources.add(str(item["source"]))
    for item in flashes:
        if isinstance(item, dict) and item.get("source"):
            sources.add(str(item["source"]))
    warnings = list(news.get("warnings") or [])
    status = "ok" if flashes or themes or candidate_news else ("degraded" if warnings else "empty")
    return {
        "status": status,
        "date": news.get("date"),
        "flash_count": len(flashes),
        "theme_count": len(themes),
        "candidate_news_count": candidate_news,
        "sources": sorted(sources)[:12],
        "warnings": warnings[:8],
    }


def _portfolio_status() -> dict[str, Any]:
    try:
        cfg = load_config()
        db_path = Path((cfg.get("output") or {}).get("db_path", "data/pangu.db"))
        if not db_path.exists():
            return {"status": "empty", "db_path": str(db_path), "reason": "组合数据库不存在，尚未记录真实买卖流水"}
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            holding_count = int(conn.execute("SELECT COUNT(*) FROM holdings").fetchone()[0])
            tx_count = int(conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0])
            invested = float(conn.execute("SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE action='buy'").fetchone()[0] or 0.0)
            realized = float(conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM transactions WHERE action='sell'").fetchone()[0] or 0.0)
        return {
            "status": "ok" if tx_count else "empty",
            "db_path": str(db_path),
            "holding_count": holding_count,
            "transaction_count": tx_count,
            "total_invested": round(invested, 2),
            "realized_pnl": round(realized, 2),
            "note": "此处为本地交易流水摘要，避免在仪表盘刷新时触发实时行情拉取",
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "reason": f"组合摘要读取失败: {e}"}


def _backtest_status(data: dict[str, Any]) -> dict[str, Any]:
    artifacts = []
    for root in (Path("data"), Path("audit")):
        if root.exists():
            artifacts.extend([p for p in root.rglob("*backtest*") if p.is_file()])
    artifacts = sorted(artifacts, key=lambda p: p.stat().st_mtime, reverse=True)
    latest = artifacts[0] if artifacts else None
    return {
        "status": "available" if latest else "not_run",
        "module": "engine.backtest.Backtester",
        "latest_artifact": str(latest) if latest else None,
        "latest_artifact_time": (
            datetime.fromtimestamp(latest.stat().st_mtime, tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
            if latest else None
        ),
        "watchlist_count": len(data.get("candidates") or []),
        "reason": None if latest else "未发现本日报告关联的回测产物；可用 engine.backtest 做离线验证，当前 GUI 不自动触发重型回测",
    }


def _task_status_summary() -> dict[str, Any]:
    latest = max(_tasks.values(), key=lambda t: t.started_at, default=None)
    counts: dict[str, int] = {}
    for t in _tasks.values():
        counts[t.status] = counts.get(t.status, 0) + 1
    return {
        "counts": counts,
        "latest": {
            "task_id": latest.task_id,
            "date": latest.date,
            "status": latest.status,
            "error": latest.error,
        } if latest else None,
    }


def _llm_status() -> dict[str, Any]:
    try:
        from ..agent.llm import llm_governance_status
        return llm_governance_status(load_config())
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "warnings": [f"LLM 状态读取失败: {e}"]}


def _credential_status() -> dict[str, Any]:
    cfg = load_config()
    llm_cfg = cfg.get("llm", {}) or {}
    providers = llm_cfg.get("providers") or []
    llm_envs: list[str] = []
    if providers:
        for p in providers:
            env_name = p.get("api_key_env") or p.get("env")
            if env_name:
                llm_envs.append(str(env_name))
    else:
        llm_envs.append("PANGU_LLM_API_KEY")
    def _safe_env_label(name: str) -> str:
        return name.replace("API_KEY", "CREDENTIAL").replace("KEY", "CREDENTIAL")

    llm_env_status = {_safe_env_label(name): bool(os.environ.get(name)) for name in llm_envs}
    return {
        "status": "ok" if any(llm_env_status.values()) else "degraded",
        "llm_env_keys": llm_env_status,
        "notify_webhook_configured": bool(os.environ.get("PANGU_NOTIFY_WEBHOOK")),
        "note": "仅展示是否配置，不返回任何凭据值、token 或 webhook URL",
    }


def _notifier_status() -> dict[str, Any]:
    try:
        from ..notifier import Notifier
        notifier = Notifier.from_env()
        return {
            "status": "enabled" if notifier.enabled else "disabled",
            "enabled": notifier.enabled,
            "method": notifier.cfg.method,
            "timeout": notifier.cfg.timeout,
            "headers_configured": sorted((notifier.cfg.headers or {}).keys()),
            "reason": None if notifier.enabled else "PANGU_NOTIFY_WEBHOOK 未配置，通知会安全跳过",
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "reason": f"通知状态读取失败: {e}"}


def _scheduler_status() -> dict[str, Any]:
    status_dir = Path("data/scheduler")
    cfg = load_config()
    schedule_cfg = cfg.get("schedule") or {}
    enabled = bool(schedule_cfg.get("enabled", False))
    schedule_time = _parse_schedule_time(cfg)
    next_run: Optional[str] = None
    if enabled and schedule_time is not None:
        next_run = _next_run_time(schedule_time).isoformat()

    result: dict[str, Any] = {
        "enabled": enabled,
        "schedule_time": schedule_time.strftime("%H:%M") if schedule_time else None,
        "next_run": next_run,
        "status": "not_run",
        "status_dir": str(status_dir),
    }

    if not status_dir.exists():
        result["reason"] = "尚未发现每日调度状态目录"
        return result
    files = sorted(status_dir.glob("*_status.json"), reverse=True)
    if not files:
        result["reason"] = "尚未发现每日调度状态文件"
        return result
    latest = files[0]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        result.update({
            "status": data.get("overall_status") or "unknown",
            "date": data.get("date"),
            "run_at": data.get("run_at"),
            "dry_run": data.get("dry_run"),
            "latest_status_file": str(latest),
            "steps": [
                {"name": s.get("name"), "status": s.get("status"), "error": s.get("error")}
                for s in (data.get("steps") or [])
            ],
        })
    except Exception as e:  # noqa: BLE001
        result.update({"status": "unavailable", "latest_status_file": str(latest), "reason": f"调度状态读取失败: {e}"})
    return result


def _daily_loop(data: dict[str, Any], runtime: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    candidates = data.get("candidates") or []
    warnings = []
    if source.get("warnings"):
        warnings.append("存在数据源降级或不可用提示，需结合 source_status 查看")
    if not candidates:
        warnings.append("当前报告无候选股，推荐闭环仅展示状态不提供买入目标")
    return {
        "target_trade_date": runtime.get("recommended_trade_date"),
        "report": _report_status(data),
        "source": source,
        "cache": _cache_status(),
        "debate": _debate_status(candidates),
        "news": _news_status(data),
        "backtest": _backtest_status(data),
        "portfolio": _portfolio_status(),
        "llm": _llm_status(),
        "credentials": _credential_status(),
        "scheduler": _scheduler_status(),
        "notifier": _notifier_status(),
        "scan_tasks": _task_status_summary(),
        "risk_notes": warnings[:8],
    }


def _enrich_response(data: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(data)
    _backfill_candidate_trend_windows(enriched)
    if not enriched.get("market_modules"):
        report_date = enriched.get("date") or "-"
        enriched["market_modules"] = {
            "short_line": {
                "title": "短线连板",
                "description": "旧报告未包含独立短线连板模块；请点击刷新/生成明日报告获取真实涨停池拆分。",
                "items": [],
                "warnings": [f"{report_date} 报告缺少短线连板模块，需要重新扫描生成"],
            },
            "yesterday_performance": {
                "title": "昨日表现",
                "description": "旧报告未包含昨日涨停/二板以上/一字板承接统计；请重新扫描生成。",
                "items": [],
                "warnings": [f"{report_date} 报告缺少昨日表现模块，需要重新扫描生成"],
            },
        }
    if not enriched.get("news"):
        enriched["news"] = {
            "date": enriched.get("date") or _beijing_now().strftime("%Y%m%d"),
            "flashes": [],
            "stock_news": {},
            "hot_themes": [],
            "warnings": ["当前报告未包含实时新闻快照，请点击刷新/生成明日报告或等待新闻接口刷新"],
        }
    runtime = _runtime_context()
    enriched["runtime"] = runtime
    enriched["data_update_time"] = runtime["now_text"]
    enriched["source_status"] = _source_status(enriched)
    debates_backfilled = _backfill_candidate_debates(enriched)
    # 若报告已含推荐闸门结果，不再用旧玄武 builder 覆盖 gate 决策
    has_gate_output = bool(enriched.get("final_recommendations"))
    if not has_gate_output and (debates_backfilled or not enriched.get("xuanwu_pool")) and enriched.get("candidates"):
        try:
            from ..xuanwu_pool import XuanwuPoolBuilder
            xuanwu_pool = XuanwuPoolBuilder(load_config()).build(
                sentiment=enriched.get("sentiment") or {},
                boards=enriched.get("boards") or [],
                candidates=enriched.get("candidates") or [],
                news=enriched.get("news") or {},
                source_status=enriched.get("source_status") or {},
            )
            decisions = xuanwu_pool.get("all_decisions") or {}
            for c in enriched.get("candidates") or []:
                code = str(c.get("code") or "")
                if code in decisions:
                    c["xuanwu"] = decisions[code]
            enriched["xuanwu_pool"] = xuanwu_pool
        except Exception as e:  # noqa: BLE001
            warnings = list(enriched.get("warnings") or [])
            warnings.append(f"玄武池补算失败: {e}")
            enriched["warnings"] = warnings
    enriched["report_status"] = _report_status(enriched)
    # 顶层暴露关键交易状态字段，方便前端直接取用
    enriched["data_quality"] = enriched.get("data_quality") or enriched["report_status"].get("data_quality", "unknown")
    enriched["tradable"] = bool(enriched.get("tradable"))
    enriched["no_trade_reason"] = enriched.get("no_trade_reason") or enriched["report_status"].get("no_trade_reason")
    enriched["block_reasons"] = list(enriched.get("block_reasons") or enriched["report_status"].get("block_reasons") or [])
    enriched["final_count"] = len(enriched.get("final_recommendations") or [])
    enriched["watch_count"] = len(enriched.get("watchlist") or [])
    enriched["raw_candidate_count"] = len(enriched.get("candidates") or [])
    enriched["daily_loop"] = _daily_loop(enriched, runtime, enriched["source_status"])
    enriched["sentiment_report"] = _sentiment_report(enriched)
    enriched["latest_report_date"] = enriched["report_status"].get("latest_report_date")
    extra_warnings = runtime.get("warnings") or []
    if extra_warnings:
        existing = list(enriched.get("warnings") or [])
        for w in extra_warnings:
            if w not in existing:
                existing.append(w)
        enriched["warnings"] = existing
    return _sanitize_floats(enriched)


# ---------------------------------------------------------------------- #
# FastAPI 应用
# ---------------------------------------------------------------------- #
app = FastAPI(title="盘古 Pangu 选股看板", docs_url="/docs", redoc_url=None)


@app.get("/")
async def index():
    """返回单页 HTML。"""
    idx = _STATIC_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(500, f"前端文件不存在: {idx}")
    return FileResponse(idx, media_type="text/html; charset=utf-8")


# 静态资源（Chart.js 等如需本地化可放此目录）
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/api/latest")
async def api_latest(date: Optional[str] = Query(None, description="YYYYMMDD，缺省取内存缓存或最新报告")):
    """获取最新选股结果。

    优先级：
    1. 指定 date → 读 data/reports/{date}.json（若存在）
    2. 内存缓存（_latest_result）
    3. 返回提示「无数据，请点「生成明日报告」开始分析」
    """
    global _latest_result
    if date:
        # 历史报告：优先 json，无则提示
        for ext in (".json",):
            p = _REPORT_DIR / f"{date}{ext}"
            if p.exists():
                try:
                    return _enrich_response(json.loads(p.read_text(encoding="utf-8")))
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(500, f"报告解析失败: {e}")
        raise HTTPException(404, f"无 {date} 的历史报告，请先扫描")

    if _latest_result is not None:
        return _enrich_response(_latest_result)

    # 尝试加载最近一份历史报告
    latest = _find_latest_report()
    if latest is not None:
        return _enrich_response(latest)

    return JSONResponse(
        _enrich_response({"empty": True, "message": "暂无选股数据，请点击「生成明日报告」开始分析"}),
        status_code=200,
    )


@app.get("/api/governance/status")
async def api_governance_status(date: Optional[str] = Query(None, description="YYYYMMDD, defaults to latest report")):
    """Return daily governance and recommendation-loop status for the GUI."""
    if date:
        p = _REPORT_DIR / f"{date}.json"
        if not p.exists():
            raise HTTPException(404, f"report {date} not found")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"report parse failed: {e}")
    else:
        data = _latest_result or _find_latest_report() or {"empty": True}
    enriched = _enrich_response(data)
    return {
        "runtime": enriched.get("runtime"),
        "data_update_time": enriched.get("data_update_time"),
        "source_status": enriched.get("source_status"),
        "daily_loop": enriched.get("daily_loop"),
    }


@app.get("/api/recommendations/performance")
async def api_recommendation_performance(
    days: int = Query(30, ge=1, le=365),
    refresh: bool = Query(False),
    only_recommended: bool = Query(True),
):
    """Return forward performance of recorded daily recommendations."""
    cfg = load_config()
    from ..recommendation_journal import RecommendationJournal
    journal = RecommendationJournal(
        cfg.get("output", {}).get("db_path", "data/pangu.db"),
        data_loader=build_data_loader(cfg),
    )
    eval_result = journal.evaluate(only_recommended=only_recommended) if refresh else None
    return {"ok": True, "evaluation": eval_result, "performance": journal.summary(days=days, only_recommended=only_recommended)}


@app.post("/api/recommendations/record-latest")
async def api_recommendation_record_latest():
    """Record the latest report into the recommendation journal."""
    cfg = load_config()
    data = _latest_result or _find_latest_report()
    if not data:
        raise HTTPException(404, "暂无最新报告可记录")
    from ..recommendation_journal import RecommendationJournal
    journal = RecommendationJournal(
        cfg.get("output", {}).get("db_path", "data/pangu.db"),
        data_loader=build_data_loader(cfg),
    )
    recorded = journal.record_pipeline_result(_enrich_response(data))
    return {"ok": True, "recorded": recorded}


def _safe_strategy_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "xuanwu_pool": cfg.get("xuanwu_pool", {}),
        "agent_prompts": get_agent_prompts(cfg),
        "trend": {
            "board": (cfg.get("trend") or {}).get("board", {}),
            "stock": (cfg.get("trend") or {}).get("stock", {}),
        },
        "structured_data": {
            "deep_candidate_limit": (cfg.get("structured_data") or {}).get("deep_candidate_limit", 60),
        },
        "output": {
            "pick_count": (cfg.get("output") or {}).get("pick_count", 5),
            "debate_top_n": (cfg.get("output") or {}).get("debate_top_n"),
        },
    }


@app.get("/api/settings")
async def api_settings_get():
    """Return editable strategy and LLM settings without exposing secrets."""
    cfg = load_config()
    providers = _llm_provider_dicts(cfg)
    safe_providers = []
    for p in providers:
        api_key_env = str(p.get("api_key_env") or "")
        safe_providers.append({
            "name": p.get("name"),
            "enabled": p.get("enabled", True),
            "base_url": p.get("base_url"),
            "model": p.get("model"),
            "api_key_env": api_key_env,
            "api_key_configured": bool(api_key_env and os.environ.get(api_key_env)),
            "timeout": p.get("timeout"),
        })
    return {
        "ok": True,
        "strategy": _safe_strategy_settings(cfg),
        "llm": {"providers": safe_providers},
    }


@app.post("/api/settings")
async def api_settings_save(payload: dict[str, Any]):
    """Save allowed local settings. API keys are only put in process env."""
    cfg = load_config()
    llm_payload = payload.get("llm") or {}
    strategy_payload = payload.get("strategy") or {}
    providers = llm_payload.get("providers")
    saved_providers = 0
    saved_strategy: dict[str, Any] | None = None
    if isinstance(providers, list):
        cleaned = []
        for p in providers:
            if not isinstance(p, dict):
                continue
            api_key_env = str(p.get("api_key_env") or "").strip()
            api_key = str(p.get("api_key") or "").strip()
            if api_key and api_key_env:
                os.environ[api_key_env] = api_key
            cleaned.append({
                "name": str(p.get("name") or "custom").strip(),
                "enabled": bool(p.get("enabled", True)),
                "base_url": str(p.get("base_url") or "").strip(),
                "model": str(p.get("model") or "").strip(),
                "api_key_env": api_key_env,
                "timeout": float(p.get("timeout") or 60),
            })
        save_llm_providers(cleaned)
        saved_providers = len(cleaned)
    if isinstance(strategy_payload, dict) and strategy_payload:
        saved_strategy = save_strategy_settings(strategy_payload)
    return {
        "ok": True,
        "saved_providers": saved_providers,
        "saved_strategy": saved_strategy,
        "settings": (await api_settings_get()),
    }


@app.post("/api/scan")
async def api_scan(date: Optional[str] = Query(None)):
    """触发后台扫描（异步）。立即返回 task_id，前端轮询 status。"""
    # 同一时刻只允许一个扫描（用状态字段判断，简单可靠）
    running = next((t for t in _tasks.values() if t.status in ("pending", "running")), None)
    if running is not None:
        return {"task_id": running.task_id, "message": "已有扫描在进行中"}

    task_id = uuid.uuid4().hex[:12]
    state = _TaskState(task_id, date)
    _tasks[task_id] = state

    def _worker():
        global _latest_result
        state.status = "running"
        state.log(f"开始扫描 date={date or '今天'}")
        try:
            pipe = get_pipeline()
            state.log("调用 Pipeline.run()，取数+选股中（约 1-3 分钟）...")
            result = pipe.run(date)
            state.log("Pipeline 完成，序列化结果")
            data = json.loads(result.to_json())
            data = _enrich_response(data)
            state.result = data
            state.status = "done"
            state.log(f"完成：候选 {len(data.get('candidates', []))} 只")
            # 更新内存缓存
            _latest_result = data
            # 存盘：P0 完整报告作为默认产物，同时保留 {date}.json 兼容旧路径
            try:
                _REPORT_DIR.mkdir(parents=True, exist_ok=True)
                payload = json.dumps(data, ensure_ascii=False, indent=2)
                (_REPORT_DIR / f"{result.date}_p0.json").write_text(payload, encoding="utf-8")
                (_REPORT_DIR / f"{result.date}.json").write_text(payload, encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                logger.warning("报告存盘失败: %s", e)
            try:
                from ..recommendation_journal import RecommendationJournal
                cfg = load_config()
                journal = RecommendationJournal(
                    cfg.get("output", {}).get("db_path", "data/pangu.db"),
                    data_loader=build_data_loader(cfg),
                )
                journal_result = journal.record_pipeline_result(data)
                state.log(
                    f"推荐日志已记录：{journal_result.get('recorded', 0)} 条，"
                    f"玄武 {journal_result.get('recommended', 0)} 条"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("推荐日志写入失败: %s", e)
                state.log(f"推荐日志写入失败: {e}")
        except Exception as e:  # noqa: BLE001
            state.status = "failed"
            state.error = str(e)
            state.log(f"扫描失败: {e}")
            logger.exception("扫描任务失败")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return {"task_id": task_id, "message": "扫描已启动，请轮询 status"}


@app.get("/api/scan/{task_id}/status")
async def api_scan_status(task_id: str):
    """查扫描进度。"""
    state = _tasks.get(task_id)
    if state is None:
        raise HTTPException(404, "任务不存在")
    return {
        "task_id": task_id,
        "status": state.status,
        "logs": state.logs[-10:],  # 最近 10 条日志
        "error": state.error,
        "result": _sanitize_floats(state.result) if state.status == "done" else None,
    }


# ── 排行榜 API ────────────────────────────────────────────────────
@app.get("/api/rankings/{rank_type}")
async def api_rankings(rank_type: str = "gainers", top_n: int = 30, exclude_st: bool = True):
    from ..ranking import MarketRanking, RANK_TYPES
    if rank_type not in RANK_TYPES:
        raise HTTPException(400, f"未知排行榜类型: {rank_type}，可用: {', '.join(RANK_TYPES)}")
    mr = MarketRanking(dl=_get_dl())  # 复用全局DL缓存
    result = mr.get_rank(rank_type, top_n=min(top_n, 100), exclude_st=exclude_st)
    return {"rank_type": rank_type, "label": result.label, "updated": result.updated,
            "total_stocks": result.total_stocks, "top_n": result.top_n, "rows": result.rows}


@app.get("/api/rankings")
async def api_all_rankings(top_n: int = 20):
    from ..ranking import MarketRanking
    mr = MarketRanking(dl=_get_dl())
    return {"breadth": mr.get_market_breadth(),
            "rankings": {k: {"label": v.label, "updated": v.updated,
                             "total_stocks": v.total_stocks, "rows": v.rows}
                         for k, v in mr.get_all_ranks(top_n=min(top_n, 50)).items()}}


@app.get("/api/market/overview")
async def api_market_overview():
    """市场总览（单次请求返回breadth+三大排行榜）。启动时异步预热all_spot。"""
    _ensure_warmup()  # 首次调用触发预热，后续立即返回
    from ..ranking import MarketRanking
    mr = MarketRanking(dl=_get_dl())
    breadth = mr.get_market_breadth()
    gainers = mr.get_rank("gainers", top_n=20)
    volume = mr.get_rank("volume", top_n=20)
    inflow = mr.get_rank("net_inflow", top_n=20)
    return {"breadth": breadth,
            "gainers": {"label": gainers.label, "rows": gainers.rows},
            "volume": {"label": volume.label, "rows": volume.rows},
            "inflow": {"label": inflow.label, "rows": inflow.rows}}


@app.get("/api/market/boards")
async def api_market_boards(limit: int = Query(1000, ge=1, le=3000)):
    """Return the available A-share concept board universe from the current data source."""
    try:
        df = _get_dl().concept_boards()
    except Exception as e:  # noqa: BLE001
        logger.warning("concept boards fetch failed: %s", e)
        return {
            "runtime": _runtime_context(),
            "count": 0,
            "boards": [],
            "warnings": [f"板块列表拉取失败: {e}"],
        }
    if df is None or len(df) == 0:
        return {
            "runtime": _runtime_context(),
            "count": 0,
            "boards": [],
            "warnings": ["当前数据源未返回概念板块列表"],
        }
    name_col = find_col(df, ["板块名称", "概念名称", "名称", "name"])
    code_col = find_col(df, ["板块代码", "概念代码", "代码", "code"])
    pct_col = find_col(df, ["涨跌幅", "涨幅", "涨跌幅%", "pct"])
    boards: list[dict[str, Any]] = []
    for _, row in df.head(limit).iterrows():
        name = str(row.get(name_col, "") if name_col else "").strip()
        if not name or name.lower() == "nan":
            continue
        pct = safe_float(row.get(pct_col)) if pct_col else 0.0
        boards.append({
            "name": name,
            "code": str(row.get(code_col, "") if code_col else "").strip(),
            "pct": pct,
            "score": abs(pct),
            "count": 0,
            "source": "ths_concept",
        })
    return {
        "runtime": _runtime_context(),
        "count": len(boards),
        "boards": boards,
        "warnings": [],
    }


@app.get("/api/market/breadth")
async def api_market_breadth():
    from ..ranking import MarketRanking
    return MarketRanking(dl=_get_dl()).get_market_breadth()


@app.get("/api/market/phase")
async def api_market_phase(date: Optional[str] = Query(None, description="YYYYMMDD，默认今天")):
    """返回当前市场阶段/情绪周期（独立接口，不触发扫描）。"""
    try:
        phase = MarketPhaseAnalyzer(_get_dl(), load_config()).analyze(date)
        return phase.to_dict()
    except Exception as e:  # noqa: BLE001
        logger.warning("市场阶段识别失败: %s", e)
        raise HTTPException(500, f"市场阶段识别失败: {e}")


@app.get("/api/market/pools")
async def api_market_pools(date: Optional[str] = Query(None, description="YYYYMMDD，默认今天")):
    """返回七大策略池原始信号（独立接口，不触发扫描，可能较慢）。"""
    try:
        results = run_all_pools(_get_dl(), load_config(), date)
        return {name: [s.to_dict() for s in sigs] for name, sigs in results.items()}
    except Exception as e:  # noqa: BLE001
        logger.warning("策略池运行失败: %s", e)
        raise HTTPException(500, f"策略池运行失败: {e}")


@app.get("/api/reports")
async def api_reports():
    """列出历史报告（data/reports/*.json + *.md）。P0 JSON 优先作为代表。"""
    if not _REPORT_DIR.exists():
        return {"reports": []}
    reports = []
    seen_dates: set[str] = set()
    for p in _list_report_paths():
        date = p.stem
        # 统一去掉 _p0 后缀得到展示日期
        display_date = date[:-3] if date.endswith("_p0") else date
        if display_date in seen_dates:
            continue
        seen_dates.add(display_date)
        reports.append({
            "date": display_date,
            "has_json": ((_REPORT_DIR / f"{display_date}_p0.json").exists() or
                        (_REPORT_DIR / f"{display_date}.json").exists()),
            "has_md": (_REPORT_DIR / f"{display_date}.md").exists(),
        })
    # 再补充只有 md 没有 json 的日期
    for p in sorted(_REPORT_DIR.glob("*.md"), reverse=True):
        display_date = p.stem
        if display_date in seen_dates:
            continue
        seen_dates.add(display_date)
        reports.append({
            "date": display_date,
            "has_json": False,
            "has_md": True,
        })
    return {"reports": reports[:60]}  # 最近 60 天


@app.get("/api/news/latest")
async def api_news_latest():
    """获取最新新闻快照。优先返回内存报告里的新闻；缺失时实时拉取并明确降级。"""
    data = _latest_result or _find_latest_report() or {}
    news = data.get("news") if isinstance(data, dict) else None
    if news and (news.get("flashes") or news.get("hot_themes") or news.get("warnings")):
        return {"runtime": _runtime_context(), "news": news}
    try:
        from ..news_fetcher import NewsFetcher
        fetcher = NewsFetcher(get_pipeline().dl, load_config())
        news_result = fetcher.fetch_today(date=_beijing_now().strftime("%Y%m%d"))
        return {"runtime": _runtime_context(), "news": news_result.to_dict()}
    except Exception as e:  # noqa: BLE001
        return {
            "runtime": _runtime_context(),
            "news": {"date": _beijing_now().strftime("%Y%m%d"), "flashes": [], "hot_themes": [], "warnings": [f"新闻实时刷新失败: {e}"]},
        }


@app.get("/api/news/stream")
async def api_news_stream(interval_seconds: int = Query(30, ge=5, le=300), limit: Optional[int] = Query(None, ge=1, le=20)):
    """SSE 推送新闻快照；浏览器保持连接，测试可用 limit=1 读取一次。"""
    def _stream():
        sent = 0
        yield _sse("start", "")
        while True:
            try:
                data = _latest_result or _find_latest_report() or {}
                news = data.get("news") if isinstance(data, dict) else None
                if not news:
                    news = {"date": _beijing_now().strftime("%Y%m%d"), "flashes": [], "hot_themes": [], "warnings": ["暂无内存新闻快照，请先生成明日报告或调用 /api/news/latest"]}
                payload = {"runtime": _runtime_context(), "news": news}
                yield _sse("news", json.dumps(payload, ensure_ascii=False))
            except Exception as e:  # noqa: BLE001
                yield _sse("error", f"新闻推送失败: {e}")
            sent += 1
            if limit is not None and sent >= limit:
                yield _sse("done", "")
                return
            time.sleep(interval_seconds)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/report/{date}.md")
async def api_report_md(date: str):
    """取某日 Markdown 报告原文。"""
    p = _REPORT_DIR / f"{date}.md"
    if not p.exists():
        raise HTTPException(404, "报告不存在")
    return FileResponse(p, media_type="text/markdown; charset=utf-8")


# ---------------------------------------------------------------------- #
# LLM 流式解读（SSE）
# ---------------------------------------------------------------------- #
@app.get("/api/llm/summary")
async def api_llm_summary(date: Optional[str] = Query(None)):
    """SSE 流式输出 AI 摘要/解读（非决策）。

    该接口仅基于已有选股结果生成 200-400 字展望，不重新选股，也不修改推荐。
    前端用 EventSource 监听，data 字段是逐段文本增量。
    失败（无 key / 无数据）会推一条 [error] 事件。
    """
    global _latest_result

    def _build_messages() -> tuple[list[dict[str, str]], str, dict[str, Any]]:
        # 1. 取选股数据
        data = _latest_result
        if data is None:
            data = _find_latest_report()
        if data is None:
            raise RuntimeError("暂无选股数据，请先点「生成明日报告」开始分析")

        # 2. 组装 prompt（精简版，避免 token 过长）
        from ..agent.prompts import SYSTEM_PROMPT
        pipeline_json = json.dumps(data, ensure_ascii=False)[:6000]  # 截断防爆
        user_prompt = (
            f"下面是盘古选股引擎基于今日收盘盘面分析的结果（JSON）。"
            f"请为用户写一段「AI 摘要/明日解读」，200-400 字，口语化。"
            f"这是总结性摘要，不是新的选股决策。重点说明：\n"
            f"1. 今日情绪温度和姿态对明日的指引、明日是否值得出手\n"
            f"2. 如果有候选股，挑明日最值得关注的 2-3 只简评（推荐度/买点/风险）；"
            f"如果没有候选股，说明为什么建议明日观望\n"
            f"3. 一句话风险提示\n\n"
            f"选股结果：\n```json\n{pipeline_json}\n```"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        meta = {
            "summary_only": True,
            "decision_mode": "summary",
            "report_date": data.get("date", ""),
            "recommendation_allowed": data.get("recommendation_allowed", False),
        }
        return messages, data.get("date", ""), meta

    def _event_stream():
        """生成 SSE 事件流。"""
        try:
            messages, _, meta = _build_messages()
        except RuntimeError as e:
            yield _sse("error", str(e))
            return
        except Exception as e:  # noqa: BLE001
            yield _sse("error", f"准备数据失败: {e}")
            return

        # 构造 LLM 客户端
        try:
            from ..agent.llm import build_llm_client
            cfg = load_config()
            client = build_llm_client(cfg)
        except ValueError as e:
            yield _sse("error", f"LLM 未配置：{e}（请在 config/settings.yaml 配 llm 段或设置环境变量）")
            return
        except Exception as e:  # noqa: BLE001
            yield _sse("error", f"LLM 客户端构造失败: {e}")
            return

        # 流式推送
        try:
            yield _sse("start", json.dumps(meta, ensure_ascii=False))
            for piece in client.stream_chat(messages, temperature=0.6):
                yield _sse("delta", piece)
            yield _sse("done", json.dumps(meta, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            yield _sse("error", f"流式生成中断: {e}")

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


@app.post("/api/llm/review")
async def api_llm_review():
    """LLM 复核最新选股结果，检查候选池质量和推荐合法性。

    返回是否允许推荐、复核理由、以及检测到的主要问题。
    """
    global _latest_result
    data = _latest_result or _find_latest_report()
    if data is None:
        raise HTTPException(404, "暂无选股数据")

    try:
        from ..agent.core import PanguAgent
        cfg = load_config()
        agent = PanguAgent.from_config(cfg)
        review = agent.agent_review(data)
        return {
            "approved": review.get("approved", False),
            "review_text": review.get("review_text", ""),
            "llm_called": review.get("llm_called", False),
            "error": review.get("error"),
            "report_date": data.get("date"),
            "recommendation_allowed": data.get("recommendation_allowed", False),
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"复核失败: {e}")



# ---------------------------------------------------------------------- #
# LLM 配置中心 API
# ---------------------------------------------------------------------- #
def _llm_provider_dicts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """从配置（含配置中心覆盖）读取 provider 原始列表，兼容旧版单 provider。"""
    llm_cfg = cfg.get("llm") or {}
    providers = list(llm_cfg.get("providers") or [])
    if not providers and llm_cfg.get("base_url"):
        providers = [{
            "name": llm_cfg.get("provider") or "primary",
            "base_url": llm_cfg.get("base_url", ""),
            "api_key_env": "PANGU_LLM_API_KEY",
            "model": llm_cfg.get("model", ""),
            "timeout": llm_cfg.get("timeout", 300),
            "enabled": True,
        }]
    return providers


def _safe_provider_view(p: dict[str, Any]) -> dict[str, Any]:
    """返回给前端的脱敏 provider 视图（无 key 值）。"""
    from ..agent.llm import _provider_key
    api_key, source = _provider_key(p)
    return {
        "name": p.get("name", ""),
        "base_url": p.get("base_url", ""),
        "model": p.get("model", ""),
        "timeout": p.get("timeout", 300),
        "enabled": bool(p.get("enabled", True)),
        "api_key_env": p.get("api_key_env") or p.get("env") or "",
        "key_source": source,
        "key_configured": bool(api_key),
    }


def _find_provider_by_name(providers: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for p in providers:
        if p.get("name") == name:
            return p
    return None


@app.get("/api/llm/providers")
async def api_llm_providers():
    """列出所有 LLM provider（脱敏）。"""
    cfg = load_config()
    providers = _llm_provider_dicts(cfg)
    return {
        "providers": [_safe_provider_view(p) for p in providers],
        "fallback_enabled": len([p for p in providers if p.get("enabled", True)]) > 1,
    }


@app.post("/api/llm/providers")
async def api_llm_providers_create(req: Request):
    """新增 provider。请求体只接受 api_key_env，拒绝明文 api_key。"""
    body = await req.json()
    if "api_key" in body:
        raise HTTPException(status_code=400, detail="禁止提交明文 api_key，请使用 api_key_env")
    new_provider = {
        "name": str(body.get("name", "")).strip(),
        "base_url": str(body.get("base_url", "")).strip(),
        "api_key_env": str(body.get("api_key_env") or body.get("env") or "").strip(),
        "model": str(body.get("model", "")).strip(),
        "timeout": float(body.get("timeout", 300) or 300),
        "enabled": bool(body.get("enabled", True)),
    }
    if not new_provider["name"]:
        raise HTTPException(status_code=400, detail="name 不能为空")
    cfg = load_config()
    providers = _llm_provider_dicts(cfg)
    if _find_provider_by_name(providers, new_provider["name"]):
        raise HTTPException(status_code=409, detail="provider 已存在")
    providers.append(new_provider)
    save_llm_providers(providers)
    return {"ok": True, "name": new_provider["name"]}


@app.put("/api/llm/providers/{name}")
async def api_llm_providers_update(name: str, req: Request):
    """编辑 provider。"""
    body = await req.json()
    if "api_key" in body:
        raise HTTPException(status_code=400, detail="禁止提交明文 api_key，请使用 api_key_env")
    cfg = load_config()
    providers = _llm_provider_dicts(cfg)
    p = _find_provider_by_name(providers, name)
    if not p:
        raise HTTPException(status_code=404, detail="provider 不存在")
    p["base_url"] = str(body.get("base_url", p.get("base_url", ""))).strip()
    p["api_key_env"] = str(body.get("api_key_env") or body.get("env") or p.get("api_key_env") or "").strip()
    p["model"] = str(body.get("model", p.get("model", ""))).strip()
    p["timeout"] = float(body.get("timeout", p.get("timeout", 300)) or 300)
    p["enabled"] = bool(body.get("enabled", p.get("enabled", True)))
    save_llm_providers(providers)
    return {"ok": True}


@app.delete("/api/llm/providers/{name}")
async def api_llm_providers_delete(name: str):
    cfg = load_config()
    providers = _llm_provider_dicts(cfg)
    new_providers = [p for p in providers if p.get("name") != name]
    if len(new_providers) == len(providers):
        raise HTTPException(status_code=404, detail="provider 不存在")
    save_llm_providers(new_providers)
    return {"ok": True}


@app.post("/api/llm/providers/reorder")
async def api_llm_providers_reorder(req: Request):
    body = await req.json()
    order = list(body.get("order") or [])
    cfg = load_config()
    providers = _llm_provider_dicts(cfg)
    by_name = {p.get("name"): p for p in providers if p.get("name")}
    missing = [n for n in order if n not in by_name]
    if missing:
        raise HTTPException(status_code=400, detail=f"未知 provider: {missing}")
    reordered = [by_name[n] for n in order]
    # 保留未排序的 provider 在末尾
    for p in providers:
        if p.get("name") not in order:
            reordered.append(p)
    save_llm_providers(reordered)
    return {"ok": True, "active_order": [p.get("name") for p in reordered if p.get("enabled", True)]}


@app.post("/api/llm/providers/{name}/test")
async def api_llm_providers_test(name: str):
    """测试 provider 连通性（后端用 env 读 key，不从前端接收 key）。"""
    cfg = load_config()
    providers = _llm_provider_dicts(cfg)
    p = _find_provider_by_name(providers, name)
    if not p:
        raise HTTPException(status_code=404, detail="provider 不存在")
    from ..agent.llm_discover import probe_provider
    return probe_provider(p, timeout=min(int(p.get("timeout", 300) or 300), 60))


@app.get("/api/llm/providers/{name}/models")
async def api_llm_provider_models(name: str, force: int = Query(0, ge=0, le=1)):
    """自动发现 provider 可用模型。"""
    cfg = load_config()
    providers = _llm_provider_dicts(cfg)
    p = _find_provider_by_name(providers, name)
    if not p:
        raise HTTPException(status_code=404, detail="provider 不存在")
    from ..agent.llm_discover import discover_models
    return discover_models(p, force_refresh=bool(force))


@app.get("/api/llm/presets")
async def api_llm_presets():
    """读取 provider 预设库。"""
    presets_path = Path(__file__).resolve().parent.parent.parent / "config" / "llm_presets.json"
    try:
        data = json.loads(presets_path.read_text(encoding="utf-8"))
        return {"presets": data.get("presets", [])}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取预设库失败: {e}")


@app.get("/api/llm/providers/{name}/probe")
async def api_llm_provider_probe(name: str):
    """探测指定 provider 的能力（流式/工具/上下文长度）。"""
    cfg = load_config()
    providers = _llm_provider_dicts(cfg)
    p = _find_provider_by_name(providers, name)
    if not p:
        raise HTTPException(status_code=404, detail="provider 不存在")
    from ..agent.llm_discover import probe_capabilities
    return probe_capabilities(p, timeout=min(int(p.get("timeout", 300) or 300), 60))


@app.get("/api/llm/probe")
async def api_llm_probe():
    """探测第一个启用的 provider 能力（前端快速自检）。"""
    cfg = load_config()
    providers = _llm_provider_dicts(cfg)
    enabled = [p for p in providers if p.get("enabled", True)]
    if not enabled:
        raise HTTPException(status_code=503, detail="无可用 LLM provider")
    from ..agent.llm_discover import probe_capabilities
    return probe_capabilities(enabled[0], timeout=60)


@app.post("/api/llm/chat")
async def api_llm_chat(payload: dict[str, Any]):
    """简单 LLM 对话：发消息，返回文本。不依赖扫描数据。"""
    from ..agent.llm import build_llm_client
    from ..config import load_config
    msg = str(payload.get("message", "")).strip()
    if not msg:
        raise HTTPException(400, "message 不能为空")
    sys_prompt = str(payload.get("system", "你是A股短线分析助手，用中文回答，简洁专业。"))
    try:
        cfg = copy.deepcopy(load_config())
        llm_cfg = cfg.get("llm", {}) or {}
        llm_cfg["timeout"] = min(float(llm_cfg.get("timeout", 45) or 45), 45.0)
        for provider in llm_cfg.get("providers") or []:
            provider["timeout"] = min(float(provider.get("timeout", llm_cfg["timeout"]) or llm_cfg["timeout"]), 45.0)
        client = build_llm_client(cfg)
        resp = await asyncio.to_thread(
            client.simple_chat,
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": msg},
            ],
            temperature=0.3,
            max_tokens=1600,
        )
        content = resp if resp else ""
        if not content or content.startswith("（LLM 错误"):
            raise RuntimeError(content or "LLM 返回空")
        return {"ok": True, "content": content}
    except Exception as e:
        raise HTTPException(500, f"LLM 调用失败: {e}")


# ── 舆情引擎端点 ────────────────────────────────────────────────
@app.get("/api/sentiment/fetch")
async def api_sentiment_fetch(count: int = Query(200, ge=50, le=500)):
    """拉取今日多源新闻（目标200条），返回原始新闻列表。"""
    from ..sentiment_engine import SentimentEngine
    engine = SentimentEngine()
    news = engine.fetch_today_news(target_count=count)
    return {"ok": True, "count": len(news), "news": news}


@app.post("/api/sentiment/analyze")
async def api_sentiment_analyze():
    """拉取新闻并调用LLM深度分析，返回完整舆情报告。"""
    from ..sentiment_engine import SentimentEngine
    from ..agent.llm import build_llm_client
    from ..config import load_config
    engine = SentimentEngine()
    try:
        cfg = load_config()
        llm = build_llm_client(cfg)
    except Exception:
        llm = None
    report = engine.get_or_fetch(llm_client=llm)
    return {"ok": True, "report": report.to_dict()}


@app.get("/api/sentiment/backtest")
async def api_sentiment_backtest(days: int = Query(7, ge=1, le=30)):
    """舆情回测：对过去N天的舆情缓存进行汇总分析。"""
    from ..sentiment_engine import SentimentEngine
    engine = SentimentEngine()
    result = engine.backtest(lookback_days=days)
    return {"ok": True, "backtest": result}


@app.get("/api/sentiment/evolution")
async def api_sentiment_evolution(days: int = Query(7, ge=1, le=30)):
    """舆情演化：过去N天的情感指数趋势+分布变化。"""
    from ..sentiment_model import SentimentTracker
    tracker = SentimentTracker()
    return {"ok": True, "evolution": tracker.evolution(lookback_days=days)}


@app.post("/api/sentiment/daily")
async def api_sentiment_daily():
    """对今日新闻做完整情感分析（ML模型5级分类+分布）。"""
    from ..sentiment_engine import SentimentEngine
    from ..sentiment_model import SentimentTracker
    engine = SentimentEngine()
    news = engine.fetch_today_news(target_count=200)
    tracker = SentimentTracker()
    report = tracker.daily_report(news)
    return {"ok": True, "count": len(news), "report": report.to_dict()}


@app.get("/api/news/archive")
async def api_news_archive(date: str = Query(...)):
    """按日期获取历史新闻（从本地缓存/reports中查找）。"""
    import glob as _glob
    from pathlib import Path as _Path
    report_dir = _Path("data/reports")
    pattern = str(report_dir / f"{date}*.json")
    files = _glob.glob(pattern)
    news_data = {"date": date, "has_data": len(files) > 0, "news_count": 0, "flashes": []}
    if files:
        try:
            with open(files[0], encoding="utf-8") as f:
                data = json.load(f)
            news = data.get("news", {})
            flashes = news.get("flashes", [])
            news_data["news_count"] = len(flashes)
            news_data["flashes"] = flashes[:100]
        except Exception as e:
            news_data["error"] = str(e)
    return news_data


@app.post("/api/llm/invalidate")
async def api_llm_invalidate():
    """AI 恢复链路：清空 LLM 客户端缓存与模型发现缓存。"""
    from ..agent.llm import invalidate_llm_cache
    from ..agent.llm_discover import CACHE_PATH
    cleared = invalidate_llm_cache()
    # 同时尝试清空全局 debate 缓存（如果后端持有）
    try:
        import engine.agent.debate as debate_mod
        for obj in getattr(debate_mod, "_debater_instances", []):
            if hasattr(obj, "reset_llm"):
                obj.reset_llm()
        cleared["debate_reset"] = True
    except Exception as e:  # noqa: BLE001
        cleared["debate_reset_error"] = str(e)
    return {"ok": True, "cleared": cleared, "models_cache_path": str(CACHE_PATH)}


def _sse(event: str, data: str) -> str:
    """格式化一个 SSE 事件帧。data 中的换行要拆成多行 data:。"""
    # SSE 规范：data 字段内换行需拆成多个 data: 行
    safe = data.replace("\r\n", "\n").replace("\r", "\n")
    lines = safe.split("\n")
    payload = "\n".join(f"data: {l}" for l in lines)
    return f"event: {event}\n{payload}\n\n"


def _report_is_complete(data: Any) -> bool:
    """判断报告是否为受控完整产物（非外部/中间残件）。

    校验：候选非空且多数含 ``recommend.recommend_score``；存在结构化数据状态
    （``source_status.structured_data`` 或 ``source_state.structured_data``）。
    用于跳过外部手写/旧的 ``{date}_p0.json`` 劫持更新的正式报告。
    """
    if not isinstance(data, dict):
        return False
    cands = data.get("candidates")
    if not isinstance(cands, list) or not cands:
        return False
    scored = sum(
        1 for c in cands
        if isinstance(c, dict) and isinstance((c.get("recommend") or {}).get("recommend_score"), (int, float))
    )
    if scored < max(1, len(cands) // 2):
        return False
    src_status = data.get("source_status")
    src_state = data.get("source_state")
    has_struct = (
        isinstance(src_status, dict) and "structured_data" in src_status
    ) or (
        isinstance(src_state, dict) and isinstance(src_state.get("structured_data"), dict)
    )
    return has_struct


def _report_sort_key(p: Path) -> tuple:
    """报告排序：先按报告日期(新>旧)、再按修改时间(新>旧)；``_p0`` 仅作同分 tiebreaker。

    不再无条件让 ``*_p0.json`` 优先——避免外部/旧的 ``_p0`` 文件劫持更新的正式报告。
    完整性校验由 :func:`_find_latest_report` 加载时执行。
    """
    stem = p.stem
    is_p0 = stem.endswith("_p0")
    prefix = stem[:8]
    date_str = prefix if prefix.isdigit() and len(prefix) == 8 else "00000000"
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (date_str, mtime, is_p0)


def _list_report_paths() -> list[Path]:
    """列出 data/reports 下所有 json 报告：按日期+修改时间倒序，``_p0`` 仅作 tiebreaker。

    注意：不进入 ``degraded/`` 子目录，也不把 ``latest_ok.json`` 当普通报告。
    """
    if not _REPORT_DIR.exists():
        return []
    return sorted(
        (p for p in _REPORT_DIR.glob("*.json") if p.stem != "latest_ok"),
        key=_report_sort_key,
        reverse=True,
    )


def _find_latest_report() -> Optional[dict[str, Any]]:
    """加载最新且完整的报告。

    优先读取 ``latest_ok.json`` 指针；若指针失效，再回退到主目录扫描。
    旧/外部手写的不完整 ``_p0.json`` 会被 :func:`_report_is_complete` 拦下。
    """
    latest_ok_path = _REPORT_DIR / "latest_ok.json"
    if latest_ok_path.exists():
        try:
            ptr = json.loads(latest_ok_path.read_text(encoding="utf-8"))
            json_path = Path(ptr.get("json_path") or "")
            if not json_path.is_absolute():
                json_path = _REPORT_DIR / json_path.name
            if json_path.exists():
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if _report_is_complete(data):
                    return data
        except Exception:  # noqa: BLE001
            pass

    for p in _list_report_paths():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if _report_is_complete(data):
            return data
    return None


# ---------------------------------------------------------------------- #
# 启动入口：python -m engine.web
# ---------------------------------------------------------------------- #
def run() -> None:
    """命令行启动入口。"""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m engine.web", description="盘古选股 Web 看板")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    import uvicorn, threading
    print(f"\n  盘古 Pangu 选股看板启动中...")
    # 后台预热：提前拉取全市场数据到内存缓存
    threading.Thread(target=_ensure_warmup, daemon=True).start()
    # 后台自动调度：每日收盘后跑盘后链路
    _start_scheduler()
    print(f"  → http://{args.host}:{args.port}  (后台预热中，约10秒后首请求秒开)\n")
    uvicorn.run(
        "engine.web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    run()
