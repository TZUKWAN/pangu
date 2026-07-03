"""配置加载：从 config/settings.yaml 读配置，转成各模块需要的 dict。

把 YAML 解析集中在这里，engine 各模块只接收纯 dict，便于测试和替换。
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("pangu.config")

# 默认配置文件位置（相对项目根 / 绝对路径均可）
DEFAULT_CONFIG_PATH = os.environ.get(
    "PANGU_CONFIG", str(Path(__file__).resolve().parent.parent / "config" / "settings.yaml")
)

# 配置中心覆盖文件（独立，避免改写用户手改的 settings.yaml）
LLM_OVERRIDE = Path("data/llm_providers.json")
STRATEGY_OVERRIDE = Path("data/strategy_settings.json")

_ENV_PLACEHOLDER = re.compile(r"\$\{ENV:([^}]+)\}")


def _resolve_env(obj: Any) -> Any:
    """递归解析字符串中的 ${ENV:VAR_NAME} 占位符。"""
    if isinstance(obj, str):
        def _repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), "")
        return _ENV_PLACEHOLDER.sub(_repl, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    return obj


def load_config(path: str | None = None) -> dict[str, Any]:
    """加载 YAML 配置。失败返回空 dict（模块会用各自默认值）。"""
    p = Path(path or DEFAULT_CONFIG_PATH)
    if not p.exists():
        logger.warning("配置文件不存在: %s，使用默认值", p)
        cfg = {}
    else:
        try:
            with open(p, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error("配置解析失败 %s: %s", p, e)
            return {}

    cfg = _resolve_env(cfg)
    # 允许配置中心覆盖 llm.providers
    if LLM_OVERRIDE.exists():
        try:
            ov = json.loads(LLM_OVERRIDE.read_text(encoding="utf-8"))
            if ov.get("providers"):
                cfg.setdefault("llm", {})["providers"] = ov["providers"]
                logger.debug("LLM provider 配置由 %s 覆盖", LLM_OVERRIDE)
        except Exception as e:  # noqa: BLE001
            logger.warning("读取 LLM 配置覆盖文件 %s 失败: %s", LLM_OVERRIDE, e)
    if STRATEGY_OVERRIDE.exists():
        try:
            ov = json.loads(STRATEGY_OVERRIDE.read_text(encoding="utf-8"))
            strategy = ov.get("strategy") if isinstance(ov, dict) else None
            if isinstance(strategy, dict):
                _deep_merge(cfg, strategy)
                logger.debug("策略配置由 %s 覆盖", STRATEGY_OVERRIDE)
        except Exception as e:  # noqa: BLE001
            logger.warning("读取策略配置覆盖文件 %s 失败: %s", STRATEGY_OVERRIDE, e)
    logger.debug("配置加载自 %s", p)
    return cfg


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def save_llm_providers(providers: list[dict[str, Any]]) -> None:
    """原子写 LLM provider 覆盖配置。仅保存 api_key_env，不保存明文 key。"""
    LLM_OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
    # 校验：拒绝明文 api_key 入盘
    cleaned: list[dict[str, Any]] = []
    for p in providers:
        cp = {k: v for k, v in p.items() if k != "api_key"}
        cleaned.append(cp)

    fd, tmp = tempfile.mkstemp(dir=str(LLM_OVERRIDE.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"providers": cleaned}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, LLM_OVERRIDE)
        logger.info("LLM provider 配置已保存至 %s", LLM_OVERRIDE)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_strategy_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """原子写策略覆盖配置，只允许保存可解释的数值阈值。"""
    cleaned = _sanitize_strategy_settings(settings)
    STRATEGY_OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(STRATEGY_OVERRIDE.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"strategy": cleaned}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STRATEGY_OVERRIDE)
        logger.info("策略覆盖配置已保存至 %s", STRATEGY_OVERRIDE)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return cleaned


def _sanitize_strategy_settings(settings: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}

    def put(path: tuple[str, ...], value: Any, *, kind: str = "float", min_v: float = 0.0, max_v: float = 10_000.0) -> None:
        if value in (None, ""):
            return
        try:
            n = int(value) if kind == "int" else float(value)
        except (TypeError, ValueError):
            return
        if n < min_v or n > max_v:
            return
        cur = cleaned
        for p in path[:-1]:
            cur = cur.setdefault(p, {})
        cur[path[-1]] = n

    def put_text(path: tuple[str, ...], value: Any, *, max_len: int = 20_000) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text:
            return
        cur = cleaned
        for p in path[:-1]:
            cur = cur.setdefault(p, {})
        cur[path[-1]] = text[:max_len]

    xw = settings.get("xuanwu_pool") or {}
    put(("xuanwu_pool", "max_size"), xw.get("max_size"), kind="int", min_v=1, max_v=50)
    put(("xuanwu_pool", "watch_size"), xw.get("watch_size"), kind="int", min_v=1, max_v=500)
    put(("xuanwu_pool", "min_total_score"), xw.get("min_total_score"), min_v=0, max_v=100)
    put(("xuanwu_pool", "rule_min_total_score"), xw.get("rule_min_total_score"), min_v=0, max_v=100)
    put(("xuanwu_pool", "min_rps"), xw.get("min_rps"), min_v=0, max_v=100)
    put(("xuanwu_pool", "min_debate_confidence"), xw.get("min_debate_confidence"), min_v=0, max_v=100)
    put(("xuanwu_pool", "min_risk_reward"), xw.get("min_risk_reward"), min_v=0, max_v=20)
    put(("xuanwu_pool", "turnover_min"), xw.get("turnover_min"), min_v=0, max_v=100)
    put(("xuanwu_pool", "turnover_max"), xw.get("turnover_max"), min_v=0, max_v=100)
    put(("xuanwu_pool", "debate_top_n"), xw.get("debate_top_n"), kind="int", min_v=1, max_v=300)
    put(("xuanwu_pool", "debate_rounds"), xw.get("debate_rounds"), kind="int", min_v=1, max_v=3)
    put(("xuanwu_pool", "debate_max_tokens"), xw.get("debate_max_tokens"), kind="int", min_v=300, max_v=8000)
    put(("xuanwu_pool", "judge_max_tokens"), xw.get("judge_max_tokens"), kind="int", min_v=300, max_v=8000)

    stock = ((settings.get("trend") or {}).get("stock") or {})
    put(("trend", "stock", "fallback_top_n"), stock.get("fallback_top_n"), kind="int", min_v=50, max_v=5000)
    put(("trend", "stock", "broad_pool_target"), stock.get("broad_pool_target"), kind="int", min_v=10, max_v=1000)
    put(("trend", "stock", "rps_min"), stock.get("rps_min"), min_v=0, max_v=100)
    put(("trend", "stock", "volume_ratio_min"), stock.get("volume_ratio_min"), min_v=0, max_v=20)

    structured = settings.get("structured_data") or {}
    put(("structured_data", "deep_candidate_limit"), structured.get("deep_candidate_limit"), kind="int", min_v=1, max_v=1000)

    output = settings.get("output") or {}
    put(("output", "pick_count"), output.get("pick_count"), kind="int", min_v=1, max_v=200)
    put(("output", "debate_top_n"), output.get("debate_top_n"), kind="int", min_v=1, max_v=300)

    prompts = settings.get("agent_prompts") or settings.get("prompts") or {}
    if isinstance(prompts, dict):
        for key in ("bull", "bear", "judge", "risk", "risk_judge"):
            put_text(("agent_prompts", key), prompts.get(key))
    return cleaned


def build_data_loader(cfg: dict[str, Any]):
    """根据 data 配置构造 DataLoader / MultiSourceDataLoader。"""
    from .data_loader import DataLoader, MultiSourceDataLoader
    d = cfg.get("data", {})
    retry = d.get("retry", {})
    common = dict(
        cache_dir=d.get("cache_dir", "data/cache"),
        cache_ttl_minutes=d.get("cache_ttl_minutes", 30),
        retry_times=retry.get("times", 3),
        backoff_seconds=retry.get("backoff_seconds", 2.0),
    )
    if d.get("multi_source", False):
        return MultiSourceDataLoader(
            snapshot_dir=d.get("snapshot_dir", "data/snapshots"),
            **common,
        )
    return DataLoader(**common)
