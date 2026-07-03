"""选股主链路：串联 ①情绪 → ②趋势 → ③护栏 → 候选池。

这是 engine 的「总指挥」，把 sentiment_meter / trend_scanner / quant_guard
串成一个端到端流程，输出结构化 JSON。

流程：
    情绪温度计  →  姿态判定
        ├ 冰点(<40): 直接返回，建议观望（不浪费趋势扫描的耗时）
        ├ 正常/亢奋: 继续
    趋势扫描    →  候选池
    量化护栏    →  排雷后的最终候选池

输出 schema（给 LLM 综合用 + 前端展示用）：
    {
      "date": "...",
      "sentiment": {...},      # 情绪温度 + 姿态
      "boards": [...],         # 热门板块
      "candidates": [...],     # 最终候选股（含入选理由）
      "rejected": [...],       # 被护栏剔除的
      "posture_advice": "...", # 姿态建议
      "warnings": [...]
    }
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar

import pandas as pd

from .data_loader import DataLoader, safe_float
from .data_loader import find_col as _find_col
from .sentiment_meter import SentimentMeter
from .trend_scanner import TrendScanner, TrendResult
from .quant_guard import QuantGuard, GuardResult
from .entry_exit import EntryExitEngine
from .news_sentiment import NewsSentimentScorer
from .xuanwu_pool import XuanwuPoolBuilder
from .market_phase import MarketPhaseAnalyzer
from .strategy_pools import run_all_pools
from .recommendation_gate import RecommendationGate

logger = logging.getLogger("pangu.pipeline")

T = TypeVar("T")


@dataclass
class PipelineResult:
    date: str
    sentiment: dict[str, Any]
    boards: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    posture_advice: str
    warnings: list[str] = field(default_factory=list)
    news: dict[str, Any] = field(default_factory=dict)  # 今日新闻聚合（财联社电报+题材热度+个股新闻）
    market_modules: dict[str, Any] = field(default_factory=dict)
    source_status: dict[str, Any] = field(default_factory=dict)
    xuanwu_pool: dict[str, Any] = field(default_factory=dict)
    recommendation_allowed: bool = False
    historical_mode: str = "live"  # live / historical / incomplete
    watchlist: list[dict[str, Any]] = field(default_factory=list)
    final_recommendations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "sentiment": self.sentiment,
            "boards": self.boards,
            "candidates": self.candidates,
            "rejected": self.rejected,
            "posture_advice": self.posture_advice,
            "warnings": self.warnings,
            "news": self.news,
            "market_modules": self.market_modules,
            "source_status": self.source_status,
            "xuanwu_pool": self.xuanwu_pool,
            "recommendation_allowed": self.recommendation_allowed,
            "historical_mode": self.historical_mode,
            "watchlist": self.watchlist,
            "final_recommendations": self.final_recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class Pipeline:
    """主链路。"""

    def __init__(
        self,
        dl: Optional[DataLoader] = None,
        sentiment_cfg: Optional[dict] = None,
        trend_cfg: Optional[dict] = None,
        guard_cfg: Optional[dict] = None,
        entry_exit_cfg: Optional[dict] = None,
        pick_count: Optional[int] = None,
        db_path: str = "data/pangu.db",
        full_cfg: Optional[dict] = None,
    ) -> None:
        self.full_cfg = full_cfg or {}
        self.dl = dl or DataLoader()
        self.meter = SentimentMeter(self.dl, sentiment_cfg or self.full_cfg.get("sentiment", {}))
        self.scanner = TrendScanner(self.dl, trend_cfg or self.full_cfg.get("trend", {}))
        self.guard = QuantGuard(self.dl, guard_cfg or self.full_cfg.get("guard", {}))
        # EntryExitEngine 接收扁平 entry_exit 子配置
        self.entry_exit = EntryExitEngine(self.dl, entry_exit_cfg or self.full_cfg.get("entry_exit") or {})
        self.pick_count = pick_count if pick_count is not None else self.full_cfg.get("output", {}).get("pick_count", 5)
        self.db_path = db_path
        # 控制需要深度计算（买卖点/技术快照/P0 因子）的候选数量，避免 100+ 候选时超时
        self.deep_candidate_limit = int(
            (self.full_cfg.get("structured_data") or {}).get("deep_candidate_limit", 60)
        )
        xuanwu_cfg = self.full_cfg.get("xuanwu_pool", {}) or {}
        configured_debate_limit = int(
            xuanwu_cfg.get(
                "debate_top_n",
                self.full_cfg.get("output", {}).get(
                    "debate_top_n",
                    min(5, self.deep_candidate_limit),
                ),
            )
        )
        # debate_candidate_limit is the expensive true LLM debate budget.
        # The rest of the candidate pool is still covered by rule_validate
        # later in the pipeline, so this must not be inflated by watch_size.
        self.debate_candidate_limit = max(0, configured_debate_limit)

    # ------------------------------------------------------------------ #
    def _stage(
        self,
        name: str,
        fn: Callable[[], T],
        timeout: float,
        default: T,
    ) -> T:
        """阶段级包装：超时/异常均返回 default，绝不阻塞主链路。

        使用 daemon 线程运行阶段函数；超时后主线程立即返回 default，daemon 线程
        随进程退出被强制终止，避免 ThreadPoolExecutor 上下文等待导致进程挂起。
        """
        import threading
        t0 = time.monotonic()
        result_container: list[Any] = [None]
        exception_container: list[Any] = [None]

        def _target() -> None:
            try:
                result_container[0] = fn()
            except Exception as e:  # noqa: BLE001
                exception_container[0] = e

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning("[阶段] %s 超时 %.1fs，跳过", name, timeout)
            return default
        if exception_container[0] is not None:
            logger.warning("[阶段] %s 失败: %s", name, exception_container[0])
            return default
        logger.info("[阶段] %s 完成，耗时 %.2fs", name, time.monotonic() - t0)
        return result_container[0]

    # ------------------------------------------------------------------ #
    def run(self, date: Optional[str] = None) -> PipelineResult:
        """跑完整链路，返回结构化结果。

        核心变更：
        - 所有关键数据源必须记录 source_status。
        - 关键数据源失败或真实 RPS 缺失时，recommendation_allowed=False。
        - 观察池（watchlist）与严格候选池彻底分离，不进入最终推荐。
        - 历史日期模式下，若缺少历史关键数据，historical_mode='incomplete'。
        """
        date = date or datetime.now().strftime("%Y%m%d")
        logger.info("==== 盘古选股 %s 开始 ====", date)
        overall_t0 = time.monotonic()

        source_status: dict[str, Any] = {}
        recommendation_allowed = True
        block_reasons: list[str] = []
        historical_mode = "live" if date == datetime.now().strftime("%Y%m%d") else "historical"

        def _update_status(name: str, status: str, reason: str | None = None, **extra) -> None:
            source_status[name] = {"status": status, "date": date, "reason": reason or "", **extra}
            if status == "failed":
                nonlocal recommendation_allowed
                recommendation_allowed = False
                block_reasons.append(f"{name}: {reason or 'failed'}")

        # 0. 全市场快照状态（关键源）
        try:
            spot = self.dl.all_spot()
            if len(spot) > 0:
                _update_status("all_spot", "ok", rows=len(spot))
            else:
                _update_status("all_spot", "failed", "返回空数据")
        except Exception as e:  # noqa: BLE001
            _update_status("all_spot", "failed", str(e))
            spot = pd.DataFrame()

        # 注入预计算的真实 RPS 表（关键源）。
        rps_available = False

        def _load_rps() -> dict[str, Any]:
            nonlocal rps_available
            try:
                from . import rps as rps_mod
                rps_map = rps_mod.load_rps_map(date, self.db_path)
                if rps_map:
                    self.scanner.set_rps_map(rps_map, date=date)
                    rps_available = True
                    logger.info("已加载真实 RPS 表：%d 只", len(rps_map))
                    return {"status": "ok", "mode": "real", "count": len(rps_map)}
                else:
                    logger.warning("无预计算 RPS 表，建议跑 `python -m engine.cli rps-build`")
                    return {"status": "failed", "mode": "unavailable", "count": 0}
            except Exception as e:  # noqa: BLE001
                logger.warning("RPS 表加载失败：%s", e)
                return {"status": "failed", "mode": "unavailable", "error": str(e)}

        rps_state = self._stage("RPS预加载", _load_rps, timeout=5.0, default={"status": "failed"})
        rps_status = rps_state.get("status", "failed") if isinstance(rps_state, dict) else "failed"
        if rps_status == "ok":
            _update_status("rps", "ok", mode="real", count=rps_state.get("count", 0))
        else:
            _update_status("rps", "failed", mode="unavailable", reason="无预计算 RPS 表")

        # RPS 硬前置：默认 require_real，缺失则阻断正式推荐。
        trend_cfg = self.full_cfg.get("trend", {})
        rps_cfg = trend_cfg.get("rps", {})
        require_real_rps = rps_cfg.get("require_real", True)
        allow_approx_rps = rps_cfg.get("allow_approx", False)
        if require_real_rps and not rps_available:
            recommendation_allowed = False
            block_reasons.append("真实 RPS 表缺失（require_real_rps=True）")

        # ① 情绪温度计
        sentiment: dict[str, Any]
        temp = 50.0
        advice = ""
        posture = "正常"
        ms_warnings: list[str] = []

        def _sentiment_stage() -> dict[str, Any]:
            nonlocal temp, advice, posture, ms_warnings
            bd = self.meter.measure(date)
            sentiment = bd.to_dict()
            temp = bd.temperature
            advice = bd.advice
            posture = bd.posture
            ms_warnings = bd.warnings
            logger.info("情绪温度 %.1f (%s) [基础版]", temp, posture)
            return sentiment

        sentiment = self._stage("情绪温度计", _sentiment_stage, timeout=120.0, default={"temperature": temp, "posture": posture, "advice": advice})
        if "temperature" in sentiment:
            _update_status("sentiment", "ok", posture=posture, temperature=temp)
        else:
            _update_status("sentiment", "failed", "情绪温度计阶段失败")

        # 冰点：直接观望，跳过趋势扫描
        if temp < 40:
            logger.info("情绪冰点，建议观望，跳过趋势扫描")
            return PipelineResult(
                date=date,
                sentiment=sentiment,
                boards=[],
                candidates=[],
                rejected=[],
                posture_advice=advice,
                warnings=(ms_warnings or []) + ["情绪冰点，未执行趋势扫描"],
                market_modules=self._build_market_modules(date),
                source_status=source_status,
                recommendation_allowed=False,
                historical_mode=historical_mode,
            )

        # ② 趋势扫描（历史日期透传）
        def _trend_stage() -> TrendResult:
            return self.scanner.scan(date=date)
        trend: TrendResult = self._stage("趋势扫描", _trend_stage, timeout=300.0, default=TrendResult(boards=[], candidates=[], warnings=["趋势扫描阶段超时或失败"]))
        if trend.candidates:
            _update_status("trend_scan", "ok", candidates=len(trend.candidates))
        else:
            _update_status("trend_scan", "failed", reason="无候选股" if not trend.warnings else trend.warnings[0])
            return PipelineResult(
                date=date,
                sentiment=sentiment,
                boards=trend.boards,
                candidates=[],
                rejected=[],
                posture_advice=advice,
                warnings=(ms_warnings or []) + trend.warnings + ["趋势扫描无候选股"],
                market_modules=self._build_market_modules(date),
                source_status=source_status,
                recommendation_allowed=False,
                historical_mode=historical_mode,
            )

        # ③ 量化护栏（历史日期一并透传）
        def _guard_stage() -> GuardResult:
            return self.guard.filter(trend.candidates, date=date)
        guarded: GuardResult = self._stage("量化护栏", _guard_stage, timeout=120.0, default=GuardResult(kept=trend.candidates, watch=[], rejected=[], warnings=["护栏阶段超时，原池通过"]))
        kept = guarded.kept
        watch_from_guard = guarded.watch
        if guarded.rejected:
            _update_status("quant_guard", "degraded", reason=f"硬剔除 {len(guarded.rejected)} 只", rejected_count=len(guarded.rejected))
        else:
            _update_status("quant_guard", "ok", kept_count=len(kept))

        # ④ 买卖点/技术快照：仅对 deep candidate 并发做完整计算
        deep_candidates = kept[: self.deep_candidate_limit]
        broad_candidates = kept[self.deep_candidate_limit :]

        def _entry_exit_technical_stage() -> list[dict[str, Any]]:
            def _one(cand: Any) -> dict[str, Any]:
                d = cand.to_dict()
                ee = self.entry_exit.compute(cand, temperature=temp, account_size=None, date=date)
                d["entry_exit"] = ee.to_dict()
                d["technical"] = self._technical_snapshot(cand.code, date)
                return d
            results: list[dict[str, Any]] = [c.to_dict() for c in deep_candidates]
            try:
                with ThreadPoolExecutor(max_workers=3) as pool:
                    results = list(pool.map(_one, deep_candidates))
            except Exception as e:  # noqa: BLE001
                logger.warning("买卖点+技术快照并发失败，回退串行: %s", e)
                results = [_one(c) for c in deep_candidates]
            return results
        deep_full: list[dict[str, Any]] = self._stage(
            "买卖点+技术快照", _entry_exit_technical_stage, timeout=300.0,
            default=[c.to_dict() for c in deep_candidates]
        )
        entry_exit_ok = all(d.get("entry_exit") and not (d["entry_exit"].get("warnings") or [])
                         for d in deep_full) if deep_full else False
        if entry_exit_ok or not deep_full:
            _update_status("entry_exit", "ok", computed=len(deep_full))
        else:
            _update_status("entry_exit", "degraded", reason="部分候选买卖点计算失败", computed=len(deep_full))

        strict_candidates: list[dict[str, Any]] = []
        watchlist: list[dict[str, Any]] = []
        for d in deep_full:
            d = dict(d)
            if "entry_exit" not in d:
                d["entry_exit"] = {
                    "code": d.get("code"), "name": d.get("name"), "close": round(safe_float(d.get("close")), 2),
                    "buy_points": [], "stop_loss": None, "take_profit": [],
                    "trailing_stop": None, "position": None,
                    "risk_reward_ratio": 0.0, "warnings": ["观察池：未计算买卖点"],
                }
            if "technical" not in d:
                d["technical"] = {
                    "ma": {}, "macd": {}, "volume": {}, "trend_windows": {}, "kline": [],
                    "hints": [], "warnings": ["观察池：未计算技术指标"],
                }
            strict_candidates.append(d)

        # 观察池 = guard.watch + 超出 deep limit 的 kept + 原始观察池候选
        for cand in broad_candidates:
            d = cand.to_dict()
            d["entry_exit"] = {
                "code": cand.code, "name": cand.name, "close": round(cand.close, 2),
                "buy_points": [], "stop_loss": None, "take_profit": [],
                "trailing_stop": None, "position": None,
                "risk_reward_ratio": 0.0, "warnings": ["观察池：未计算买卖点"],
                "entry_exit_status": "not_computed", "tradable": False,
            }
            d["technical"] = {
                "ma": {}, "macd": {}, "volume": {}, "trend_windows": {}, "kline": [],
                "hints": [], "warnings": ["观察池：未计算技术指标"],
            }
            d["is_watchlist"] = True
            watchlist.append(d)
        for cand in watch_from_guard:
            d = cand.to_dict()
            d["entry_exit"] = {
                "code": cand.code, "name": cand.name, "close": round(cand.close, 2),
                "buy_points": [], "stop_loss": None, "take_profit": [],
                "trailing_stop": None, "position": None,
                "risk_reward_ratio": 0.0, "warnings": ["观察池：护栏轻微风险"],
                "entry_exit_status": "not_computed", "tradable": False,
            }
            d["technical"] = {
                "ma": {}, "macd": {}, "volume": {}, "trend_windows": {}, "kline": [],
                "hints": [], "warnings": ["观察池：未计算技术指标"],
            }
            d["is_watchlist"] = True
            watchlist.append(d)

        # 原始 trend 扫描里的宽松观察池（is_watchlist=True）也从严格候选中移除
        for cand in trend.candidates:
            if cand.is_watchlist:
                d = cand.to_dict()
                d["entry_exit"] = {
                    "code": cand.code, "name": cand.name, "close": round(cand.close, 2),
                    "buy_points": [], "stop_loss": None, "take_profit": [],
                    "trailing_stop": None, "position": None,
                    "risk_reward_ratio": 0.0, "warnings": ["观察池：未计算买卖点"],
                    "entry_exit_status": "not_computed", "tradable": False,
                }
                d["technical"] = {
                    "ma": {}, "macd": {}, "volume": {}, "trend_windows": {}, "kline": [],
                    "hints": [], "warnings": ["观察池：未计算技术指标"],
                }
                d["is_watchlist"] = True
                if d not in watchlist:
                    watchlist.append(d)

        if watchlist:
            logger.info("%d 只进入观察池，不进入最终推荐", len(watchlist))

        # ③⑤ 策略框架：市场阶段 + 7 大策略池 + 最终推荐闸门
        strategy_framework_enabled = self.full_cfg.get("strategy_framework", {}).get("enabled", True)
        gate_result = None
        market_phase_dict: dict[str, Any] = {}
        if strategy_framework_enabled:
            def _strategy_framework_stage() -> tuple[dict[str, Any], Any]:
                analyzer = MarketPhaseAnalyzer(self.dl, self.full_cfg)
                phase = analyzer.analyze(date)
                phase_dict = phase.to_dict()
                phase_dict["recommendation_allowed"] = recommendation_allowed
                pooled = run_all_pools(self.dl, self.full_cfg, date)

                # 策略池候选是主入口：收集所有信号 code，补齐不在旧 kept/watch 中的候选
                signal_codes = {s.code for sigs in pooled.values() for s in sigs}
                # 1. 优先用 trend 扫描已产生的候选
                candidate_map: dict[str, StockCandidate] = {c.code: c for c in trend.candidates}
                # 2. 对策略池特有、但 trend 未覆盖的 code，现场构建候选并过护栏
                missing_codes = signal_codes - set(candidate_map.keys())
                if missing_codes:
                    built = [self._build_candidate_for_code(code, date) for code in missing_codes]
                    built = [c for c in built if c is not None]
                    if built:
                        extra_guarded = self.guard.filter(built, date=date)
                        # 合并 guard 结果
                        guarded.kept.extend(extra_guarded.kept)
                        guarded.watch.extend(extra_guarded.watch)
                        guarded.rejected.extend(extra_guarded.rejected)
                        guarded.warnings.extend(extra_guarded.warnings)
                        for c in extra_guarded.kept + extra_guarded.watch:
                            candidate_map[c.code] = c

                gate = RecommendationGate(
                    self.dl, guarded, phase_dict, self.full_cfg,
                    recommendation_allowed=recommendation_allowed,
                    date=date,
                    temperature=temp,
                )
                # LLM review 预留：目前不阻塞，避免无 LLM 时全部降级
                gate_result = gate.pass_gate(pooled, candidate_map, llm_review_map={})
                return phase_dict, gate_result
            fw_result = self._stage("策略框架", _strategy_framework_stage, timeout=300.0, default=({}, None))
            if isinstance(fw_result, tuple) and len(fw_result) == 2:
                market_phase_dict, gate_result = fw_result
                _update_status("strategy_framework", "ok", pools=list((gate_result.to_dict() if gate_result else {}).keys()) if gate_result else [])
            else:
                _update_status("strategy_framework", "failed", reason="策略框架阶段未返回结果")

        # 历史模式：若 all_spot 不是历史数据，则标记 incomplete
        if historical_mode == "historical":
            if source_status.get("all_spot", {}).get("status") == "ok":
                historical_mode = "incomplete"
                recommendation_allowed = False
                block_reasons.append("历史模式缺少历史 all_spot 数据")

        candidates = strict_candidates  # 后续流程只对严格候选继续

        # ⑤ 新闻聚合 + 题材情绪（非关键）
        news_cfg = self.full_cfg.get("news_sentiment", {})
        report_dir = news_cfg.get(
            "report_dir",
            self.full_cfg.get("output", {}).get("report_dir", "data/reports"),
        )
        news_data: dict[str, Any] = {}
        news_sentiment: dict[str, dict[str, Any]] = {}
        news_result = None

        def _news_stage() -> tuple[dict[str, Any], dict[str, dict[str, Any]], Any]:
            from .news_fetcher import NewsFetcher
            fetcher = NewsFetcher(self.dl, self.full_cfg)
            nr = fetcher.fetch_today(candidates=candidates[: self.deep_candidate_limit], date=date)
            nd = nr.to_dict()
            ns: dict[str, dict[str, Any]] = {}
            for c in candidates[: self.deep_candidate_limit]:
                code = c.get("code", "")
                if code in nr.stock_news:
                    c["stock_news"] = [n.to_dict() for n in nr.stock_news[code]]
            if nr.flashes or nr.hot_themes or nr.stock_news:
                brief_text = nr.to_markdown()
                news_scorer = NewsSentimentScorer(news_cfg, report_dir=report_dir)
                stock_news_dict = {
                    code: [n.to_dict() for n in news_list]
                    for code, news_list in nr.stock_news.items()
                }
                ns = news_scorer.score_from_text(brief_text, source="realtime", stock_news=stock_news_dict)
            return nd, ns, nr

        news_stage_result = self._stage("新闻聚合", _news_stage, timeout=120.0, default=({}, {}, None))
        if isinstance(news_stage_result, tuple) and len(news_stage_result) == 3:
            news_data, news_sentiment, news_result = news_stage_result
            if news_data.get("source_state"):
                source_status["news"] = news_data["source_state"]
            else:
                source_status["news"] = {"status": "ok"}
        else:
            news_data = {"warnings": ["新闻聚合阶段超时或失败"]}
            source_status["news"] = {"status": "degraded", "warnings": ["新闻聚合阶段超时或失败"]}

        # ⑥ P0 结构化因子（非关键）
        def _p0_stage() -> dict[str, Any]:
            from .p0_factors import P0FactorCollector
            p0_state, market_extra = P0FactorCollector(self.full_cfg, dl=self.dl).collect(
                date, candidates[: self.deep_candidate_limit]
            )
            for c in candidates[self.deep_candidate_limit :]:
                c["structured_factors"] = {
                    "source_coverage": {"_note": "观察池：未采集结构化因子"},
                    "reasons": [], "risk_notes": [],
                }
            return {"p0_state": p0_state, "market_extra": market_extra}
        p0_result = self._stage("P0结构化因子", _p0_stage, timeout=600.0, default={})
        market_modules_extra: dict[str, Any] = {}
        if p0_result and isinstance(p0_result, dict) and p0_result.get("p0_state"):
            source_status["structured_data"] = p0_result["p0_state"]
            market_modules_extra = p0_result.get("market_extra") or {}
        else:
            source_status["structured_data"] = {"status": "degraded", "warnings": ["P0 结构化因子阶段超时或失败"]}

        # ⑦ 推荐排序
        def _recommend_stage() -> list[dict[str, Any]]:
            from .recommender import Recommender
            recommender = Recommender(self.full_cfg)
            recs = recommender.rank(candidates, news_sentiment=news_sentiment)
            out: list[dict[str, Any]] = []
            for rec in recs:
                base = next((c for c in candidates if c["code"] == rec.code), {})
                base = dict(base)
                rec_dict = rec.to_dict()
                base["recommend"] = rec_dict
                for k in ("recommend_score", "grade", "confidence_score", "up_prob", "target_pct", "tag", "buy_point", "stop_loss", "take_profit", "risk_reward_ratio", "score_breakdown", "calibrated", "not_statistical_probability"):
                    if k not in base:
                        base[k] = rec_dict.get(k)
                out.append(base)
            return out
        ranked: list[dict[str, Any]] = self._stage(
            "推荐评分", _recommend_stage, timeout=120.0, default=candidates
        )

        # ⑧ 多空辩论
        def _debate_stage() -> dict[str, Any]:
            from .agent.debate import StockDebater
            debater = StockDebater(cfg=self.full_cfg)
            results = debater.debate_batch(
                ranked,
                max_n=self.debate_candidate_limit,
                news_sentiment=news_sentiment,
                hot_themes=news_result.hot_themes if news_result else None,
            )
            for item in ranked:
                code = str(item.get("code") or "")
                if not code or code in results:
                    continue
                ns = (news_sentiment or {}).get(code) if isinstance(news_sentiment, dict) else None
                results[code] = debater.rule_validate(
                    code,
                    str(item.get("name") or code),
                    item,
                    news_sentiment=ns,
                    hot_themes=news_result.hot_themes if news_result else None,
                )
            return results
        debates: dict[str, Any] = self._stage("多空辩论", _debate_stage, timeout=300.0, default={})
        for c in ranked:
            code = c.get("code", "")
            if code in debates:
                c["debate"] = debates[code]
        # LLM 状态记录
        llm_ok = any((c.get("debate") or {}).get("llm_called") for c in ranked)
        if llm_ok:
            source_status["llm"] = {"status": "ok"}
        else:
            source_status["llm"] = {"status": "degraded", "mode": "rule", "warnings": ["未调用真实 LLM，使用规则验证"]}

        verdict_order = {"推荐": 0, "观望": 1, "回避": 2}
        ranked.sort(key=lambda c: (
            -safe_float(c.get("recommend_score"), 0.0),
            verdict_order.get((c.get("debate") or {}).get("verdict", "观望"), 1),
        ))

        xuanwu_pool = XuanwuPoolBuilder(self.full_cfg).build(
            sentiment=sentiment,
            boards=trend.boards,
            candidates=ranked,
            news=news_data,
            source_status=self._xuanwu_source_status(source_status),
        )
        decisions = xuanwu_pool.get("all_decisions") or {}
        for c in ranked:
            code = str(c.get("code") or "")
            if code in decisions:
                c["xuanwu"] = decisions[code]

        # 最终推荐闸门：优先使用策略框架产出；未启用框架时回退到 xuanwu 决策
        final_recommendations: list[dict[str, Any]] = []
        if gate_result and recommendation_allowed:
            final_recommendations = gate_result.final_recommendations[: self.pick_count]
        elif recommendation_allowed:
            for c in ranked:
                if c.get("is_watchlist"):
                    continue
                xw = c.get("xuanwu") or {}
                if xw.get("status") == "xuanwu":
                    final_recommendations.append(c)
                    if len(final_recommendations) >= self.pick_count:
                        break

        if not recommendation_allowed:
            final_advice = advice + " 当前数据条件不足，系统未生成可信正式推荐。"
        else:
            final_advice = advice
            if posture == "亢奋":
                final_advice += " 当前情绪亢奋，候选股注意追高风险，轻仓试错。"

        # 清理：把 final_recommendations 之外的严格候选也保留在 candidates 里，但报告需明确区分
        # 这里 candidates 包含 ranked（严格候选），rejected 包含被 guard 硬剔除的
        logger.info("==== 盘古选股 %s 完成：严格候选 %d，观察池 %d，最终推荐 %d，耗时 %.1fs ====",
                    date, len(ranked), len(watchlist), len(final_recommendations), time.monotonic() - overall_t0)
        market_modules = self._build_market_modules(date)
        market_modules.update(market_modules_extra)
        if market_phase_dict:
            market_modules["market_phase"] = market_phase_dict

        # 策略框架观察池与既有 watchlist 合并
        if gate_result:
            gate_watch_codes = {w["code"] for w in gate_result.watchlist}
            watchlist = [w for w in watchlist if w.get("code") not in gate_watch_codes]
            watchlist.extend(gate_result.watchlist)

        result = PipelineResult(
            date=date,
            sentiment=sentiment,
            boards=trend.boards,
            candidates=ranked,
            rejected=guarded.rejected,
            posture_advice=final_advice,
            warnings=(ms_warnings or []) + trend.warnings + guarded.warnings + block_reasons,
            news=news_data,
            market_modules=market_modules,
            source_status=source_status,
            xuanwu_pool=xuanwu_pool,
            recommendation_allowed=recommendation_allowed and len(final_recommendations) > 0,
            historical_mode=historical_mode,
        )
        # 额外挂载 watchlist / final_recommendations 供报告使用
        result.watchlist = watchlist
        result.final_recommendations = final_recommendations
        return result

    def _xuanwu_source_status(self, source_status: dict[str, Any]) -> dict[str, Any]:
        """Compress pipeline source_status into gate-level statuses for Xuanwu scoring."""
        warnings: list[str] = []

        def status_of(key: str, default: str = "ok") -> str:
            state = source_status.get(key)
            if isinstance(state, dict):
                status = str(state.get("status") or state.get("overall_status") or default)
                for w in state.get("warnings") or []:
                    warnings.append(str(w))
                return status
            return default

        # 关键源任一 failed → market_data 视为 failed
        critical_keys = ["all_spot", "daily_kline", "rps", "fund_flow", "entry_exit", "quant_guard"]
        critical_failed = [k for k in critical_keys if source_status.get(k, {}).get("status") == "failed"]
        if critical_failed:
            market_status = "failed"
            reasons = []
            for k in critical_failed:
                reason = source_status[k].get("reason") or "failed"
                reasons.append(f"{k}: {reason}")
            warnings.append("关键数据源失败: " + "; ".join(reasons))
        elif any(source_status.get(k, {}).get("status") == "degraded" for k in critical_keys):
            market_status = "degraded"
        else:
            market_status = "ok"

        news_status = status_of("news", "ok") if source_status.get("news") else "degraded"
        structured_status = status_of("structured_data", "degraded") if source_status.get("structured_data") else "degraded"
        return {
            "market_data": market_status,
            "news": news_status,
            "structured_data": structured_status,
            "warnings": warnings[:20],
        }

    def _build_market_modules(self, date: str) -> dict[str, Any]:
        """拆出短线连板/昨日表现，避免混入热门板块。"""
        modules: dict[str, Any] = {
            "short_line": {
                "title": "短线连板",
                "description": "涨停、连板高度和连板梯队反映短线接力强度，不等同于行业/概念板块。",
                "items": [],
                "warnings": [],
            },
            "yesterday_performance": {
                "title": "昨日涨停/连板今日表现",
                "description": "观察昨日涨停股、二板以上及一字板次日承接，用于判断短线情绪延续或分歧。",
                "items": [],
                "warnings": [],
            },
        }
        try:
            zt = self.dl.limit_up_pool(date)
            if zt is None or zt.empty:
                modules["short_line"]["warnings"].append("今日涨停池为空，短线连板模块不可用")
            else:
                consec_col = _find_col(zt, ["连板数", "涨停统计"])
                name_col = _find_col(zt, ["名称"])
                code_col = _find_col(zt, ["代码"])
                first_col = _find_col(zt, ["首次封板时间"])
                if consec_col:
                    nums = pd.to_numeric(zt[consec_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(1).astype(int)
                    dist = nums.value_counts().sort_index().to_dict()
                    modules["short_line"]["items"] = [
                        {"label": f"{int(level)}板", "value": int(count)}
                        for level, count in sorted(dist.items())
                    ]
                    modules["short_line"]["highest"] = int(nums.max()) if len(nums) else 0
                    modules["short_line"]["two_plus_count"] = int((nums >= 2).sum())
                    if first_col:
                        first_times = zt[first_col].astype(str)
                        modules["short_line"]["one_word_count"] = int(first_times.str.contains(r"^0?9:?2[0-9]|^925|^09:25", regex=True).sum())
                    if name_col and code_col:
                        leaders = zt.assign(_consec=nums).sort_values("_consec", ascending=False).head(8)
                        modules["short_line"]["leaders"] = [
                            {"code": str(r[code_col]), "name": str(r[name_col]), "height": int(r["_consec"])}
                            for _, r in leaders.iterrows()
                        ]
                else:
                    modules["short_line"]["warnings"].append("涨停池缺少连板数字段")
        except Exception as e:  # noqa: BLE001
            modules["short_line"]["warnings"].append(f"短线连板模块取数失败: {e}")

        try:
            prev = _previous_weekday(date)
            prev_zt = self.dl.limit_up_pool(prev)
            spot = self.dl.all_spot()
            if prev_zt is None or prev_zt.empty:
                modules["yesterday_performance"]["warnings"].append(f"{prev} 涨停池为空，昨日表现不可用")
            elif spot is None or spot.empty:
                modules["yesterday_performance"]["warnings"].append("全市场行情为空，无法计算昨日涨停今日表现")
            else:
                code_col_prev = _find_col(prev_zt, ["代码"])
                consec_col = _find_col(prev_zt, ["连板数", "涨停统计"])
                first_col = _find_col(prev_zt, ["首次封板时间"])
                code_col_spot = _find_col(spot, ["代码", "股票代码"])
                pct_col = _find_col(spot, ["涨跌幅"])
                if not code_col_prev or not code_col_spot or not pct_col:
                    modules["yesterday_performance"]["warnings"].append("昨日涨停池或实时行情缺代码/涨跌幅字段")
                else:
                    prev_codes = set(prev_zt[code_col_prev].astype(str).str.zfill(6))
                    s = spot.copy()
                    s["_code"] = s[code_col_spot].astype(str).str.zfill(6)
                    s["_pct"] = pd.to_numeric(s[pct_col], errors="coerce")
                    hit = s[s["_code"].isin(prev_codes)]
                    nums = pd.to_numeric(prev_zt[consec_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(1) if consec_col else pd.Series([1] * len(prev_zt))
                    one_word = 0
                    if first_col:
                        one_word = int(prev_zt[first_col].astype(str).str.contains(r"^0?9:?2[0-9]|^925|^09:25", regex=True).sum())
                    modules["yesterday_performance"].update({
                        "prev_trade_date": prev,
                        "yesterday_limit_up_count": int(len(prev_zt)),
                        "yesterday_two_plus_count": int((nums >= 2).sum()),
                        "yesterday_one_word_count": one_word,
                        "today_avg_pct": round(float(hit["_pct"].mean()), 2) if not hit.empty else None,
                        "today_up_count": int((hit["_pct"] > 0).sum()) if not hit.empty else 0,
                        "today_down_count": int((hit["_pct"] < 0).sum()) if not hit.empty else 0,
                    })
        except Exception as e:  # noqa: BLE001
            modules["yesterday_performance"]["warnings"].append(f"昨日表现模块计算失败: {e}")
        return modules

    def _build_candidate_for_code(self, code: str, date: Optional[str] = None) -> Optional[StockCandidate]:
        """为策略池信号中、旧 trend 路径未覆盖的 code 现场构建 StockCandidate。

        仅做数据补齐（RPS/资金流/市值），不做趋势形态过滤。
        """
        code = str(code or "").strip().zfill(6)
        try:
            spot = self.dl.all_spot()
        except Exception:  # noqa: BLE001
            return None
        if spot is None or spot.empty:
            return None
        code_col = _find_col(spot, ["代码"])
        if code_col is None:
            return None
        s = spot[spot[code_col].astype(str).str.zfill(6) == code]
        if s.empty:
            return None
        row = s.iloc[0]
        name_col = _find_col(spot, ["名称"])
        close_col = _find_col(spot, ["最新价"])
        pct_col = _find_col(spot, ["涨跌幅"])
        turnover_col = _find_col(spot, ["换手率"])
        mv_col = _find_col(spot, ["流通市值", "总市值"])
        name = str(row.get(name_col, code)) if name_col else code
        close = safe_float(row.get(close_col)) or 0.0
        pct = safe_float(row.get(pct_col)) or 0.0
        turnover = safe_float(row.get(turnover_col)) or 0.0
        mv = safe_float(row.get(mv_col)) or 0.0
        circ_mv_yi = mv / 1e8 if mv > 0 else 0.0
        board = "科创板" if code.startswith("68") else "创业板" if code.startswith("30") else "北交所" if code.startswith(("8", "4")) else "深市主板" if code.startswith("0") else "沪市主板" if code.startswith("60") else "其他"

        # RPS
        rps, rps_mode = 0.0, "unavailable"
        try:
            from . import rps as rps_mod
            rps_map = rps_mod.load_rps_map(date, self.db_path)
            if rps_map and code in rps_map:
                rps = float(rps_map[code])
                rps_mode = "real"
        except Exception:  # noqa: BLE001
            pass
        if rps_mode != "real":
            try:
                k = self.dl.daily_kline(code, days=30, date=date)
                closes = pd.to_numeric(k["close"], errors="coerce").dropna() if not k.empty else pd.Series(dtype=float)
                if len(closes) >= 21:
                    from .trend_scanner import _rps
                    rps = _rps(code, closes, spot, None)
                    rps_mode = "approximate"
            except Exception:  # noqa: BLE001
                pass

        # 资金流
        fund_inflow_days, fund_flow_status, fund_flow_date, fund_flow_net = 0, "unavailable", None, None
        try:
            ff = self.dl.all_fund_flow_snapshot(fast=True)
            if ff is not None and not ff.empty:
                ff_code_col = _find_col(ff, ["股票代码", "代码"])
                ff_net_col = _find_col(ff, ["主力净流入-净额", "净额"])
                if ff_code_col and ff_net_col:
                    ff_row = ff[ff[ff_code_col].astype(str).str.zfill(6) == code]
                    if not ff_row.empty:
                        fund_flow_net = safe_float(ff_row.iloc[0].get(ff_net_col))
                        fund_flow_date = datetime.now().strftime("%Y%m%d")
                        fund_flow_status = "snapshot_only" if fund_flow_net is None or fund_flow_net <= 0 else "available"
        except Exception:  # noqa: BLE001
            pass

        return StockCandidate(
            code=code, name=name, board=board, close=close, pct_change=pct,
            turnover_rate=turnover, circ_mv_yi=circ_mv_yi,
            rps=rps, rps_mode=rps_mode,
            fund_inflow_days=fund_inflow_days, fund_flow_status=fund_flow_status,
            fund_flow_date=fund_flow_date, fund_flow_net=fund_flow_net,
            is_watchlist=False,
        )

    def _technical_snapshot(self, code: str, date: str) -> dict[str, Any]:
        """为推荐股票补齐 K 线、均线、MACD、成交量和量比提示。"""
        out: dict[str, Any] = {
            "ma": {},
            "macd": {},
            "volume": {},
            "trend_windows": {},
            "kline": [],
            "hints": [],
            "warnings": [],
        }
        try:
            k = self.dl.daily_kline(code, days=150, date=date)
        except Exception as e:  # noqa: BLE001
            out["warnings"].append(f"K线取数失败: {e}")
            return out
        if k is None or len(k) < 35:
            out["warnings"].append("K线数据不足，无法完整计算 MA/MACD/量比")
            return out
        date_col = _find_col(k, ["日期", "date"])
        open_col = _find_col(k, ["开盘", "open"])
        high_col = _find_col(k, ["最高", "high"])
        low_col = _find_col(k, ["最低", "low"])
        close_col = _find_col(k, ["收盘", "close"])
        vol_col = _find_col(k, ["成交量", "volume", "vol"])
        if close_col is None:
            out["warnings"].append("K线缺少收盘价字段")
            return out
        closes = pd.to_numeric(k[close_col], errors="coerce").dropna()
        if len(closes) < 35:
            out["warnings"].append("有效收盘价不足")
            return out
        for n in (5, 10, 30, 60, 120):
            if len(closes) >= n:
                out["ma"][f"ma{n}"] = round(float(closes.tail(n).mean()), 2)
            else:
                out["ma"][f"ma{n}"] = None
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = (dif - dea) * 2
        golden_cross = bool(len(dif) >= 2 and dif.iloc[-2] <= dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1])
        out["macd"] = {
            "dif": round(float(dif.iloc[-1]), 4),
            "dea": round(float(dea.iloc[-1]), 4),
            "hist": round(float(hist.iloc[-1]), 4),
            "golden_cross": golden_cross,
            "hint": "MACD金叉，动能转强" if golden_cross else "MACD未现金叉或已过金叉点",
        }
        if vol_col:
            vols = pd.to_numeric(k[vol_col], errors="coerce").dropna()
            if len(vols) >= 6:
                prev5 = vols.iloc[-6:-1].mean()
                ratio = float(vols.iloc[-1] / prev5) if prev5 > 0 else None
                out["volume"] = {
                    "latest": float(vols.iloc[-1]),
                    "avg5": round(float(prev5), 2),
                    "volume_ratio": round(ratio, 2) if ratio is not None else None,
                    "hint": "明显放量" if ratio and ratio >= 1.5 else ("缩量或量能不足" if ratio and ratio < 0.8 else "量能平稳"),
                }
        rows = k.tail(60)
        for _, r in rows.iterrows():
            item: dict[str, Any] = {}
            if date_col:
                item["date"] = str(r[date_col])[:10]
            for src, dst in ((open_col, "open"), (high_col, "high"), (low_col, "low"), (close_col, "close"), (vol_col, "volume")):
                if src:
                    val = pd.to_numeric(pd.Series([r[src]]), errors="coerce").iloc[0]
                    item[dst] = None if pd.isna(val) else round(float(val), 2)
            out["kline"].append(item)
        close = float(closes.iloc[-1])
        highs = pd.to_numeric(k[high_col], errors="coerce").dropna() if high_col else pd.Series(dtype=float)
        lows = pd.to_numeric(k[low_col], errors="coerce").dropna() if low_col else pd.Series(dtype=float)
        for label, window in (("week_1", 5), ("week_2", 10), ("month_1", 20)):
            if len(closes) > window:
                ref = float(closes.iloc[-window - 1])
                ret_pct = (close / ref - 1) * 100 if ref > 0 else 0.0
                recent_high = float(highs.tail(window).max()) if len(highs) >= window else float(closes.tail(window).max())
                recent_low = float(lows.tail(window).min()) if len(lows) >= window else float(closes.tail(window).min())
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
                out["trend_windows"][label] = {
                    "days": window,
                    "return_pct": round(ret_pct, 2),
                    "pullback_from_high_pct": round(pullback_pct, 2),
                    "rebound_from_low_pct": round(rebound_pct, 2),
                    "state": state,
                }
        ma5 = out["ma"].get("ma5")
        ma10 = out["ma"].get("ma10")
        ma30 = out["ma"].get("ma30")
        ma60 = out["ma"].get("ma60")
        if ma5 and ma10 and ma30 and ma5 > ma10 > ma30:
            out["hints"].append("短中期均线多头排列（MA5>MA10>MA30）")
        if ma60 and close > ma60:
            out["hints"].append("收盘价站上60日线，趋势中期偏强")
        if out["macd"].get("golden_cross"):
            out["hints"].append("MACD出现金叉")
        if out["volume"].get("volume_ratio"):
            out["hints"].append(f"量比 {out['volume']['volume_ratio']}：{out['volume'].get('hint', '')}")
        month_state = (out.get("trend_windows") or {}).get("month_1", {}).get("state")
        if month_state:
            out["hints"].append(f"一个月窗口：{month_state}")
        return out


def _previous_weekday(date: str) -> str:
    d = datetime.strptime(date, "%Y%m%d")
    from datetime import timedelta
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d.strftime("%Y%m%d")
