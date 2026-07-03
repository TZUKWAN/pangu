"""趋势扫描器：在「正常/亢奋」情绪下，选出趋势形态好的候选池。

三层递进（自上而下）：
1. 板块层：概念板块按近 N 日 RPS（相对强度）+ 当日资金净流入 排名 → 取 Top 板块
2. 个股层：每个 Top 板块的成分股，按趋势形态筛选
       - 均线多头排列（MA5 > MA10 > MA20）
       - 突破近 N 日平台/新高
       - 放量（量比 > 阈值 或 当日量 > N 日均量×倍数）
       - 个股 20 日 RPS 达标（相对强度，参考 O'Neil）
       - 流通市值在区间内（剔除超小盘流动性风险 & 超大盘弹性不足）
3. 资金确认：主力资金近 M 日连续净流入

设计取舍：
- 这里用纯 pandas 计算（MA/RPS/突破/放量），不依赖 qlib。
  你要的「情绪+趋势主导、量化辅助」，这些规则化指标足够且可解释。
- 慢操作（取每只成分股的日 K）有数量上限保护，避免扫全市场 5000 只。
- 每只候选股带「入选理由」标签，方便 LLM 综合时引用、也方便你在前端看。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader, safe_float, find_col as _find_col
from .market_structure import HistoryKeeper

logger = logging.getLogger("pangu.trend")


def _is_a_share(code: str) -> bool:
    """粗略判断是否为 A 股主板/创业板/科创板代码（排除新三板/北交所/基金等 8/9 开头）。"""
    if not code or not code.isdigit() or len(code) != 6:
        return False
    return bool(
        code.startswith(("600", "601", "603", "605", "688", "000", "001", "002", "003", "300", "301"))
    )


@dataclass
class StockCandidate:
    """单只候选股。"""

    code: str
    name: str
    board: str               # 所属热门概念板块
    close: float
    pct_change: float        # 当日涨跌幅 %
    turnover_rate: float     # 换手率 %
    circ_mv_yi: float        # 流通市值（亿元）
    rps: float               # 20 日相对强度百分位
    rps_mode: str = "unavailable"  # real / approximate / unavailable
    reasons: list[str] = field(default_factory=list)  # 入选理由
    fund_inflow_days: int = 0  # 主力连续净流入天数
    fund_flow_status: str = "unavailable"  # available / snapshot_only / unavailable
    fund_flow_date: Optional[str] = None   # 资金流数据日期
    fund_flow_net: Optional[float] = None  # 当日主力净流入（万元）
    score: float = 0.0       # 综合趋势得分（用于排序）
    risk_flags: list[str] = field(default_factory=list)  # 护栏风险标记（不剔除，仅降权）
    is_watchlist: bool = False  # 是否来自宽松观察池

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "board": self.board,
            "close": round(self.close, 2),
            "pct_change": round(self.pct_change, 2),
            "turnover_rate": round(self.turnover_rate, 2),
            "circ_mv_yi": round(self.circ_mv_yi, 2),
            "rps": round(self.rps, 1),
            "rps_mode": self.rps_mode,
            "fund_inflow_days": self.fund_inflow_days,
            "fund_flow_status": self.fund_flow_status,
            "fund_flow_date": self.fund_flow_date,
            "fund_flow_net": round(self.fund_flow_net, 2) if self.fund_flow_net is not None else None,
            "reasons": self.reasons,
            "score": round(self.score, 2),
            "risk_flags": self.risk_flags,
            "is_watchlist": self.is_watchlist,
        }


@dataclass
class TrendResult:
    """趋势扫描结果。"""

    boards: list[dict[str, Any]]          # 热门板块 Top（含 RPS/资金）
    candidates: list[StockCandidate]      # 候选股池
    scanned_count: int = 0                # 实际扫描的成分股数
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "boards": self.boards,
            "candidates": [c.to_dict() for c in self.candidates],
            "scanned_count": self.scanned_count,
            "warnings": self.warnings,
        }


class TrendScanner:
    """趋势扫描器。"""

    def __init__(self, dl: DataLoader, cfg: dict[str, Any]) -> None:
        self.dl = dl
        self.cfg = cfg or {}
        bcfg = self.cfg.get("board", {})
        self.top_n = bcfg.get("top_n", 8)
        self.lookback = bcfg.get("lookback_days", 20)
        self.min_board_turnover = bcfg.get("min_avg_turnover_yi", 5)

        scfg = self.cfg.get("stock", {})
        self.ma_periods = scfg.get("ma_periods", [5, 10, 20])
        self.breakout_lookback = scfg.get("breakout_lookback", 20)
        self.vol_ratio_min = scfg.get("volume_ratio_min", 1.5)
        self.rps_min = scfg.get("rps_min", 80)
        self.rps_hard_min = scfg.get("rps_hard_min", 45)
        self.mv_min = scfg.get("min_circ_mv_yi", 30)
        self.mv_max = scfg.get("max_circ_mv_yi", 2000)
        self.max_per_board = scfg.get("max_per_board", 80)
        self.fallback_top_n = scfg.get("fallback_top_n", 400)
        self.broad_pool_target = scfg.get("broad_pool_target", 120)
        self.broad_pct_min = scfg.get("broad_pct_min", -2.0)
        self.broad_rps_hard_min = scfg.get("broad_rps_hard_min", 35)

        fcfg = self.cfg.get("fund_flow", {})
        self.fund_days = fcfg.get("consecutive_days", 3)
        self.fund_min_inflow = fcfg.get("min_daily_inflow_wan", 500)

        # 板块轮动持续性过滤
        rcfg = self.cfg.get("sector_rotation", {})
        self.rotation_persist_threshold = rcfg.get("persistence_threshold", 0.25)
        self.rotation_persist_penalty = rcfg.get("persistence_penalty", 1.0)
        self.sector_rotation_persistence_top_n = rcfg.get("persistence_top_n", 5)

        # 板块历史（用于轮动持续性）
        self.history = HistoryKeeper(self.cfg.get("history_dir", "data/cache"))

        # RPS 模式配置（由 pipeline/settings 注入）
        rps_cfg = self.cfg.get("rps", {})
        self.require_real_rps: bool = rps_cfg.get("require_real", True)
        self.allow_approx_rps: bool = rps_cfg.get("allow_approx", False)
        self.rps_date: Optional[str] = None
        # 真实 RPS 查表（由 pipeline 注入，rps.compute_all_rps 预计算）。
        self.rps_map: dict[str, float] = {}

        # 资金流历史快照目录：用于计算连续净流入天数
        self.fund_flow_dir = Path(self.cfg.get("fund_flow_dir", "data/fund_flow"))
        self.fund_flow_dir.mkdir(parents=True, exist_ok=True)
        # 个股资金流全市场排名缓存：避免 scan 阶段每只票都发起一次网络请求
        self._ff_cache: Optional[pd.DataFrame] = None
        # 同花顺人气榜概念映射：_rank_boards 构建，供全市场扫描候选回填板块。
        self._code_board_map: dict[str, str] = {}
        self._code_board_tags: dict[str, list[str]] = {}

    def set_rps_map(self, rps_map: dict[str, float], date: Optional[str] = None) -> None:
        """注入预计算的真实 RPS 表（code -> rps 0-100）。"""
        self.rps_map = rps_map or {}
        self.rps_date = date

    # ------------------------------------------------------------------ #
    def scan(self, date: Optional[str] = None) -> TrendResult:
        res = TrendResult(boards=[], candidates=[])
        scan_date = date or datetime.now().strftime("%Y%m%d")
        # 注意：all_spot 仅支持实时快照，没有历史日期参数。
        # 历史模式下如果只有实时快照，必须标记 historical_mode=incomplete。
        spot = self.dl.all_spot()
        if len(spot) == 0:
            res.warnings.append("实时行情为空，无法扫描趋势")
            return res

        # RPS 硬前置：真实 RPS 表缺失时，默认阻断正式候选池。
        real_rps_available = bool(self.rps_map)
        if self.require_real_rps and not real_rps_available:
            res.warnings.append(
                "未检测到真实 RPS 表，严格候选池已阻断。建议先运行："
                "python -m engine.cli rps-build --workers 10"
            )
            logger.warning("真实 RPS 缺失，严格候选池阻断（require_real_rps=True）")
            # 仅保留观察池逻辑，不生成正式候选
            boards = self._rank_boards(spot, date=date)
            res.boards = boards
            # 继续执行以填充观察池
        else:
            boards = self._rank_boards(spot, date=date)
            res.boards = boards

        # 保存当日资金流快照，供未来计算连续净流入天数
        self.save_fund_flow_snapshot(scan_date)

        seen: set[str] = set()
        # ── 全市场扫描作为主路径（不依赖不稳定的板块成分股接口）──
        # 从全市场行情按涨幅筛 Top N，逐只拉日K做完整趋势评估（均线/突破/放量/RPS）。
        # 刘总明确要求：慢也要等，确保成功。
        market_cands = self._scan_whole_market(spot, date=date)
        for cand in market_cands:
            if cand.code not in seen:
                seen.add(cand.code)
                res.candidates.append(cand)
        res.scanned_count += len(market_cands)
        strict_count = len([c for c in res.candidates if c.score >= 30])
        logger.info("全市场扫描主路径：%d 只入选（%d 只通过严格趋势）",
                    len(market_cands), strict_count)

        # ── 补充观察池：确保候选池数量充足 ──
        if len(res.candidates) < self.broad_pool_target:
            extra = self._scan_broad_pool(spot, seen, date=date)
            for cand in extra:
                if cand.code not in seen:
                    seen.add(cand.code)
                    res.candidates.append(cand)
            res.scanned_count += len(extra)
            if extra:
                res.warnings.append(
                    f"候选池仅 {strict_count} 只通过严格趋势，已补 {len(extra)} 只观察池标的"
                )

        # 综合排序：RPS + 资金连续性 + 趋势形态分（龙虎榜已下线，权重并入 RPS/形态）
        res.candidates.sort(
            key=lambda c: (
                c.rps * 0.55
                + min(c.fund_inflow_days, 5) * 6
                + c.score * 0.35
                + (20 if c.score >= 30 else 0)
            ),
            reverse=True,
        )
        logger.info("趋势扫描完成：热门板块 %d，候选 %d（扫描 %d 只）",
                    len(boards), len(res.candidates), res.scanned_count)
        return res

    # ------------------------------------------------------------------ #
    def _rank_boards(
        self,
        spot: pd.DataFrame,
        date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        # Compute hot board rankings from spot data using THS hot list concept tags.
        # Aggregates top gainers by concept: avg_pct, total_amt, stock_count.
        if spot is None or len(spot) == 0:
            return self._rank_traditional_boards(date=date)

        # 尝试从 THS 人气榜 / hot themes 获取概念→股票映射
        try:
            from engine.p0_factors import P0FactorCollector
            collector = P0FactorCollector(dl=self.dl)
            hot_list = collector._ths_hot_list()
        except Exception:
            hot_list = []

        # 取涨幅前200的股票代码
        pct_col = _find_col(spot, ["涨跌幅"])
        code_col = _find_col(spot, ["代码"])
        name_col = _find_col(spot, ["名称"])
        amt_col = _find_col(spot, ["成交额"])
        inflow_col = _find_col(spot, ["主力净流入-净额"])

        if pct_col is None or code_col is None:
            return []

        top = spot.copy()
        top["_pct"] = pd.to_numeric(top[pct_col], errors="coerce")
        top["_code"] = top[code_col].astype(str).str.strip()
        top = top.dropna(subset=["_pct"]).sort_values("_pct", ascending=False).head(200)

        # 基于 news hot_themes 做概念聚合
        from collections import Counter
        concept_stocks: dict[str, list[dict[str, Any]]] = {}
        concept_counter: Counter = Counter()

        # 如果有 hot_list 数据，将热门股票的标签作为概念分类
        hot_code_map = {}
        for item in hot_list:
            code = str(item.get("code", ""))
            concepts = item.get("concepts") or []
            hot_code_map[code] = [str(c) for c in concepts if str(c).strip()]

        for _, row in top.iterrows():
            code = str(row.get("_code", ""))
            pct = float(row["_pct"])
            amt = 0.0
            inflow = 0.0
            if amt_col:
                try: amt = float(row[amt_col])
                except (ValueError, TypeError): pass
            if inflow_col:
                try: inflow = float(row[inflow_col])
                except (ValueError, TypeError): pass
            name = str(row.get(name_col, "")) if name_col else ""

            tags = hot_code_map.get(code, [])
            if not tags:
                tags = ["其他"]
            for tag in tags:
                if tag not in concept_stocks:
                    concept_stocks[tag] = []
                concept_stocks[tag].append({"code": code, "name": name, "pct": pct, "amt": amt, "inflow": inflow})
                concept_counter[tag] += 1

        # 过滤掉"其他"和太小的概念
        min_stocks = 2
        board_list = []
        for concept, stocks in concept_stocks.items():
            if concept == "其他" or len(stocks) < min_stocks:
                continue
            avg_pct = sum(s["pct"] for s in stocks) / len(stocks)
            total_amt = sum(s["amt"] for s in stocks)
            total_inflow = sum(s["inflow"] for s in stocks)
            # 综合分：平均涨幅越大越好 + 上榜越多越好
            score = avg_pct * 0.5 + len(stocks) * 0.5
            board_list.append({
                "name": concept,
                "pct": round(avg_pct, 2),
                "fund_net_wan": round(total_inflow / 1e4, 1) if total_inflow else 0,
                "score": round(score, 2),
                "count": len(stocks),
                "total_amt_yi": round(total_amt / 1e8, 1),
            })

        board_list.sort(key=lambda x: x["score"], reverse=True)
        top_boards = board_list[:self.top_n]
        if not top_boards:
            top_boards = self._rank_traditional_boards(date=date)
            code_board_map = self._top_board_constituent_map(top_boards)
            code_board_tags = {code: [board] for code, board in code_board_map.items()}
            for code, board in self._pool_board_map(date).items():
                tags = code_board_tags.setdefault(code, [])
                if board not in tags:
                    tags.append(board)
                code_board_map.setdefault(code, board)
            self._code_board_map = code_board_map
            self._code_board_tags = code_board_tags
            return top_boards
        board_scores = {str(b["name"]): float(b.get("score") or 0.0) for b in board_list}
        code_board_map: dict[str, str] = {}
        code_board_tags: dict[str, list[str]] = {}
        for concept, stocks in concept_stocks.items():
            if concept == "其他":
                continue
            for stock in stocks:
                code = str(stock.get("code") or "").strip().zfill(6)
                if not code:
                    continue
                tags = code_board_tags.setdefault(code, [])
                if concept not in tags:
                    tags.append(concept)
                current = code_board_map.get(code)
                if current is None or board_scores.get(concept, 0.0) > board_scores.get(current, 0.0):
                    code_board_map[code] = concept
        for code, board in self._top_board_constituent_map(top_boards).items():
            tags = code_board_tags.setdefault(code, [])
            if board not in tags:
                tags.insert(0, board)
            code_board_map[code] = board
        for code, board in self._pool_board_map(date).items():
            tags = code_board_tags.setdefault(code, [])
            if board not in tags:
                tags.append(board)
            code_board_map.setdefault(code, board)
        self._code_board_map = code_board_map
        self._code_board_tags = code_board_tags

        # 也包含传统 concept_boards 中的板块名（仅用于展示名称，涨跌幅来自实际聚合）
        try:
            trad_boards = self.dl.concept_boards()
            if len(trad_boards) > 0:
                trad_names = set(b["name"] for b in top_boards)
                tb_name_col = _find_col(trad_boards, ["板块名称", "名称"])
                if tb_name_col:
                    for _, row in trad_boards.iterrows():
                        n = str(row[tb_name_col])
                        if n not in trad_names and not any(p in n for p in ("昨日","连板","首板","二板","三板","高标","打板","涨停","炸板","一字板","强势股","ST板块")):
                            # 用该板块的平均涨跌幅（从 all_spot 无法关联成分股，填 0 但保留名称）
                            pass
        except Exception:
            pass

        return top_boards

    def _top_board_constituent_map(self, top_boards: list[dict[str, Any]]) -> dict[str, str]:
        """Map stock code -> hot board using THS concept constituents via DataLoader.

        This is a non-Eastmoney path: ``concept_boards`` is THS and
        ``concept_constituents`` uses adata's THS constituent endpoint when available.
        Failures are ignored because hot-list tags already provide a lighter mapping.
        """
        if not top_boards:
            return {}
        try:
            boards_df = self.dl.concept_boards()
        except Exception:
            boards_df = pd.DataFrame()
        name_col = _find_col(boards_df, ["板块名称", "名称"]) if boards_df is not None and len(boards_df) else None
        symbol_col = _find_col(boards_df, ["板块代码", "代码", "symbol"]) if boards_df is not None and len(boards_df) else None
        concept_symbols: dict[str, str] = {}
        if name_col and symbol_col:
            for _, row in boards_df.iterrows():
                name = str(row.get(name_col) or "")
                symbol = str(row.get(symbol_col) or "")
                if name and symbol:
                    concept_symbols[name] = symbol

        out: dict[str, str] = {}
        for board in top_boards[: self.top_n]:
            board_name = str(board.get("name") or "")
            symbol = str(board.get("symbol") or "")
            if not symbol:
                symbol = concept_symbols.get(board_name, "")
            if not symbol and concept_symbols:
                for name, candidate_symbol in concept_symbols.items():
                    if board_name and (board_name in name or name in board_name):
                        symbol = candidate_symbol
                        break
            if not symbol:
                continue
            board["symbol"] = symbol
            try:
                cons = self.dl.concept_constituents(symbol, board_name)
            except Exception:
                continue
            if cons is None or len(cons) == 0:
                continue
            code_col = _find_col(cons, ["代码", "股票代码", "stock_code", "code"])
            if code_col is None:
                continue
            count = 0
            for raw_code in cons[code_col].tolist():
                code = str(raw_code or "").strip().zfill(6)[-6:]
                if code.isdigit():
                    out[code] = board_name
                    count += 1
            if count:
                board["constituent_count"] = count
        return out

    def _pool_board_map(self, date: Optional[str]) -> dict[str, str]:
        """Map stock code -> theme from THS limit-up/strong pools."""
        frames: list[pd.DataFrame] = []
        fetchers = [
            getattr(self.dl, "limit_up_pool", None),
            getattr(self.dl, "strong_pool", None),
        ]
        for fetcher in fetchers:
            if fetcher is None:
                continue
            try:
                df = fetcher(date)
                if df is not None and len(df) > 0:
                    frames.append(df)
            except Exception:
                continue
        out: dict[str, str] = {}
        for df in frames:
            code_col = _find_col(df, ["代码", "股票代码", "code"])
            theme_col = _find_col(df, ["所属行业", "涨停原因", "原因", "题材", "reason_type", "reason"])
            if code_col is None or theme_col is None:
                continue
            for _, row in df.iterrows():
                code = str(row.get(code_col) or "").strip().zfill(6)[-6:]
                theme = self._normalize_theme(row.get(theme_col))
                if code.isdigit() and theme:
                    out.setdefault(code, theme)
        return out

    @staticmethod
    def _normalize_theme(value: Any) -> str:
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "none", "null", "-"}:
            return ""
        for sep in ("+", "、", ",", "，", ";", "；", "/", "|"):
            if sep in text:
                text = text.split(sep)[0].strip()
                break
        return text[:18]

    def _rank_traditional_boards(self, date: Optional[str] = None) -> list[dict[str, Any]]:
        """Fallback board ranking from THS concept board list when spot/hot-list is absent."""
        try:
            boards = self.dl.concept_boards()
        except Exception:
            return []
        if boards is None or len(boards) == 0:
            return []
        name_col = _find_col(boards, ["板块名称", "名称"])
        code_col = _find_col(boards, ["板块代码", "代码", "symbol"])
        pct_col = _find_col(boards, ["涨跌幅", "今日涨跌幅"])
        if name_col is None:
            return []
        df = boards.copy()
        df["_pct"] = pd.to_numeric(df[pct_col], errors="coerce").fillna(0.0) if pct_col else 0.0
        df["_z"] = _zscore(df["_pct"])
        persist = self._sector_persistence_score(date)
        penalty = self.rotation_persist_penalty if persist is not None and persist < self.rotation_persist_threshold else 0.0
        out: list[dict[str, Any]] = []
        for _, row in df.sort_values("_pct", ascending=False).head(self.top_n).iterrows():
            name = str(row.get(name_col) or "")
            if not name:
                continue
            out.append({
                "name": name,
                "symbol": str(row.get(code_col) or "") if code_col else "",
                "pct": round(safe_float(row.get("_pct")), 2),
                "fund_net_wan": 0.0,
                "score": round(safe_float(row.get("_z")) * 0.6 - penalty, 2),
                "count": 0,
                "total_amt_yi": 0.0,
                "persistence_score": persist,
            })
        return out

    def _board_for_code(self, code: str, default: str) -> str:
        """Return the best non-Eastmoney board/theme mapping known for a stock."""
        norm = str(code or "").strip().zfill(6)
        return self._code_board_map.get(norm) or default

    # ------------------------------------------------------------------ #
    def _sector_persistence_score(self, date: Optional[str]) -> Optional[float]:
        """计算板块轮动持续性得分（0-1），用于 _rank_boards 过滤。

        今日领涨前 N 与昨日/前 3 日领涨板块的 Jaccard 重合度。
        无历史数据时返回 None（不扣分）。
        """
        if date is None:
            return None
        try:
            boards = self.dl.concept_boards()
            if boards is None or len(boards) == 0:
                return None
            name_col = _find_col(boards, ["板块名称", "名称"])
            if name_col is None:
                return None
            top_n = self.sector_rotation_persistence_top_n
            today = set(boards[name_col].head(top_n).astype(str).tolist())

            window = self.history.prev_leaders_window(date, window=3)
            yesterday = window.get("yesterday")
            last_n = window.get("last_n")
            if yesterday is None and last_n is None:
                return None

            def _jaccard(a: set, b: set) -> float:
                union = len(a | b)
                return len(a & b) / union if union > 0 else 0.0

            score_1d = _jaccard(today, set(yesterday)) if yesterday is not None else 0.0
            score_3d = _jaccard(today, set(last_n)) if last_n is not None else 0.0
            # 综合：1 日权重 60%，3 日权重 40%
            return score_1d * 0.6 + score_3d * 0.4
        except Exception as e:  # noqa: BLE001
            logger.debug("板块轮动持续性计算失败：%s", e)
            return None

    # ------------------------------------------------------------------ #
    def _evaluate(
        self,
        code: str,
        cons_row: pd.Series,
        board_name: str,
        spot: pd.DataFrame,
        date: Optional[str] = None,
    ) -> Optional[StockCandidate]:
        """评估单只成分股是否入选。"""
        name = str(cons_row.get("名称", code))
        close = safe_float(cons_row.get("最新价"))
        if close != close:
            close = 0.0
        pct = safe_float(cons_row.get("涨跌幅"))
        if pct != pct:
            pct = 0.0
        turnover = safe_float(cons_row.get("换手率"))
        if turnover != turnover:
            turnover = 0.0

        # 市值过滤（流通市值，单位：元 → 亿元）。
        # 注意：concept_constituents 接口不返回市值列，需从 all_spot 查补。
        circ_mv = safe_float(cons_row.get("流通市值"))
        if circ_mv != circ_mv or circ_mv <= 0:  # NaN 或非正 → 从全市场快照补
            code_col = _find_col(spot, ["代码"]) or spot.columns[1]
            spot_row = spot[spot[code_col].astype(str).str.strip() == code]
            if len(spot_row) > 0:
                mv_col = _find_col(spot, ["流通市值"])
                if mv_col:
                    circ_mv = safe_float(spot_row.iloc[0].get(mv_col))
        circ_mv_yi = circ_mv / 1e8 if (circ_mv == circ_mv and circ_mv > 0) else 0
        # 市值过滤：仅在有效市值时过滤，缺失（=0）时跳过避免误杀（降级模式数据可能不全）
        if circ_mv_yi > 0 and not (self.mv_min <= circ_mv_yi <= self.mv_max):
            return None

        # 取日 K 算趋势形态（历史回看时把 date 作为终点）
        k = self.dl.daily_kline(
            code,
            days=max(self.breakout_lookback, self.lookback) + 5,
            date=date,
        )
        if len(k) < max(self.ma_periods) + 2:
            return None

        reasons: list[str] = []
        score = 0.0

        # 1. 均线多头排列
        ma5 = _ma(k, 5)
        ma10 = _ma(k, 10)
        ma20 = _ma(k, 20)
        if ma5 and ma10 and ma20 and ma5 > ma10 > ma20:
            reasons.append(f"均线多头排列 MA5>MA10>MA20（{ma5:.2f}/{ma10:.2f}/{ma20:.2f}）")
            score += 25

        # 2. 突破近 N 日平台（当日收盘 > 前 N 日最高）
        closes = k["收盘"] if "收盘" in k else k.iloc[:, 4]
        closes = pd.to_numeric(closes, errors="coerce").dropna()
        if len(closes) >= self.breakout_lookback + 1:
            recent_high = closes.iloc[-self.breakout_lookback - 1:-1].max()
            if closes.iloc[-1] > recent_high:
                reasons.append(f"突破近 {self.breakout_lookback} 日平台（{recent_high:.2f}）")
                score += 25

        # 3. 放量（当日量 > 近 5 日均量 × 倍数）
        vols = k["成交量"] if "成交量" in k else k.iloc[:, 5]
        vols = pd.to_numeric(vols, errors="coerce").dropna()
        if len(vols) >= 6 and vols.iloc[-5:].mean() > 0:
            ratio = vols.iloc[-1] / vols.iloc[-5:].mean()
            if ratio >= self.vol_ratio_min:
                reasons.append(f"放量 量比 {ratio:.1f}×")
                score += 20

        # 4. RPS（20 日累计涨幅在全市场的百分位）
        rps, rps_mode = self._lookup_rps(code, closes, spot)
        if rps < self.rps_hard_min:
            # 绝对弱势，直接淘汰
            return None
        if rps_mode == "approximate" and not self.allow_approx_rps:
            # 严格模式下，近似 RPS 不能进入严格候选池
            return None
        if rps >= self.rps_min:
            reasons.append(f"20日 RPS {rps:.0f}（相对强势）")
            score += 20
        elif rps >= self.rps_hard_min:
            reasons.append(f"20日 RPS {rps:.0f}（中等强度）")
            score += 10

        if not reasons:
            return None

        # 5. 资金确认（连续净流入）
        ff_info = self._fund_flow_info(code)
        inflow_days = ff_info.get("inflow_days", 0)
        fund_flow_status = ff_info.get("status", "unavailable")
        fund_flow_date = ff_info.get("date")
        fund_flow_net = ff_info.get("net")
        if inflow_days >= self.fund_days:
            reasons.append(f"主力连续 {inflow_days} 日净流入")
            score += 10
        elif fund_flow_status == "snapshot_only" and fund_flow_net is not None and fund_flow_net > self.fund_min_inflow:
            reasons.append(f"当日主力净流入 {fund_flow_net:.0f} 万元")
        # 资金不达标不淘汰，但加分少（避免错过刚启动的票）

        return StockCandidate(
            code=code, name=name, board=board_name,
            close=close, pct_change=pct, turnover_rate=turnover,
            circ_mv_yi=circ_mv_yi, rps=rps, rps_mode=rps_mode,
            reasons=reasons,
            fund_inflow_days=inflow_days, fund_flow_status=fund_flow_status,
            fund_flow_date=fund_flow_date, fund_flow_net=fund_flow_net,
            score=score,
        )

    # ------------------------------------------------------------------ #
    def _fund_flow_info(self, code: str) -> dict[str, Any]:
        """返回个股资金流信息：{inflow_days, status, date, net}。

        status:
            available: 有当日快照 + 至少前一日的历史快照，可计算连续天数。
            snapshot_only: 只有当日快照，无法计算连续天数（不宣称连续净流入）。
            unavailable: 当日快照缺失。
        """
        code = str(code or "").strip().zfill(6)
        # 1. 取当日全市场资金流快照
        if self._ff_cache is None:
            try:
                self._ff_cache = self.dl.all_fund_flow_snapshot(fast=True)
            except Exception as e:  # noqa: BLE001
                logger.debug("all_fund_flow_snapshot cache fill failed: %s", e)
                self._ff_cache = pd.DataFrame()
        ff = self._ff_cache
        if ff is None or len(ff) == 0:
            return {"inflow_days": 0, "status": "unavailable", "date": None, "net": None}

        code_col = _find_col(ff, ["股票代码", "代码"])
        if code_col is None:
            return {"inflow_days": 0, "status": "unavailable", "date": None, "net": None}
        row = ff[ff[code_col].astype(str).str.strip().str.zfill(6) == code]
        if len(row) == 0:
            return {"inflow_days": 0, "status": "unavailable", "date": None, "net": None}

        net_col = _find_col(row, ["主力净流入-净额", "净额"])
        if net_col is None:
            return {"inflow_days": 0, "status": "unavailable", "date": None, "net": None}
        net = _num(row.iloc[0].get(net_col))
        today = datetime.now().strftime("%Y%m%d")
        date_col = _find_col(row, ["日期", "date"])
        if date_col:
            today = str(row.iloc[0].get(date_col, today)).replace("-", "")

        # 只有当日快照，先返回 snapshot_only
        if net <= self.fund_min_inflow:
            return {"inflow_days": 0, "status": "snapshot_only", "date": today, "net": net}

        # 2. 尝试读取历史快照计算连续净流入天数
        historical = self._load_historical_fund_flow(code, as_of_date=today)
        if not historical:
            return {"inflow_days": 1, "status": "snapshot_only", "date": today, "net": net}

        consecutive = 1
        for d in sorted(historical.keys(), reverse=True):
            if d == today:
                continue
            if historical[d] > self.fund_min_inflow:
                consecutive += 1
            else:
                break
        return {
            "inflow_days": consecutive,
            "status": "available" if consecutive >= 2 else "snapshot_only",
            "date": today,
            "net": net,
        }

    def _load_historical_fund_flow(
        self, code: str, as_of_date: Optional[str] = None
    ) -> dict[str, float]:
        """读取历史资金流快照（data/fund_flow/YYYYMMDD.parquet）。

        返回 {date_str: net_inflow}。只读取 as_of_date 之前 N 个交易日。
        """
        code = str(code or "").strip().zfill(6)
        as_of = as_of_date or datetime.now().strftime("%Y%m%d")
        out: dict[str, float] = {}
        for p in sorted(self.fund_flow_dir.glob("*.parquet"), reverse=True):
            date_str = p.stem
            if not date_str.isdigit() or len(date_str) != 8 or date_str >= as_of:
                continue
            try:
                df = pd.read_parquet(p)
            except Exception:  # noqa: BLE001
                continue
            code_col = _find_col(df, ["股票代码", "代码"])
            net_col = _find_col(df, ["主力净流入-净额", "净额"])
            if code_col is None or net_col is None:
                continue
            row = df[df[code_col].astype(str).str.strip().str.zfill(6) == code]
            if len(row) > 0:
                out[date_str] = _num(row.iloc[0].get(net_col))
            if len(out) >= self.fund_days + 3:
                break
        return out

    def save_fund_flow_snapshot(self, date: Optional[str] = None) -> dict[str, Any]:
        """保存当日全市场资金流快照到 data/fund_flow/YYYYMMDD.parquet。"""
        date = date or datetime.now().strftime("%Y%m%d")
        try:
            df = self.dl.all_fund_flow_snapshot(fast=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("保存资金流快照失败：%s", e)
            return {"date": date, "saved": False, "rows": 0, "error": str(e)}
        if df is None or len(df) == 0:
            return {"date": date, "saved": False, "rows": 0, "error": "empty snapshot"}
        path = self.fund_flow_dir / f"{date}.parquet"
        try:
            df.to_parquet(path, index=False)
            return {"date": date, "saved": True, "rows": len(df), "path": str(path)}
        except Exception as e:  # noqa: BLE001
            logger.warning("写入资金流快照失败：%s", e)
            return {"date": date, "saved": False, "rows": 0, "error": str(e)}

    def _lookup_rps(self, code: str, hist_closes: pd.Series, spot: pd.DataFrame) -> tuple[float, str]:
        """查询 RPS，返回 (rps_value, mode)。"""
        if self.rps_map and code in self.rps_map:
            v = self.rps_map[code]
            if v is not None and v == v:
                return float(v), "real"
        if not self.allow_approx_rps:
            return 0.0, "unavailable"
        approx = _rps(code, hist_closes, spot, None)
        return approx, "approximate"

    def _sort_by_change(self, cons: pd.DataFrame) -> pd.DataFrame:
        col = _find_col(cons, ["涨跌幅"])
        if col is None:
            return cons
        cons = cons.copy()
        cons["_pct"] = pd.to_numeric(cons[col], errors="coerce")
        return cons.sort_values("_pct", ascending=False).drop(columns=["_pct"])

    def _scan_whole_market(
        self, spot: pd.DataFrame, date: Optional[str] = None,
    ) -> list[StockCandidate]:
        """全市场降级扫描：成分股不可用时，直接从全市场行情按涨幅+市值筛选。

        策略：从全市场行情取当日涨幅 Top N（剔除 ST/次新/超低市值），
        逐只用 _evaluate 做趋势形态评估（均线/突破/放量/RPS），
        通过的作为候选。不依赖东财板块成分股。
        """
        if spot is None or len(spot) == 0:
            return []
        pct_col = _find_col(spot, ["涨跌幅"])
        code_col = _find_col(spot, ["代码"]) or (spot.columns[1] if len(spot.columns) > 1 else None)
        name_col = _find_col(spot, ["名称"])
        mv_col = _find_col(spot, ["流通市值"])
        if pct_col is None or code_col is None:
            return []

        df = spot.copy()
        df["_pct"] = pd.to_numeric(df[pct_col], errors="coerce")
        df["_code"] = df[code_col].astype(str).str.strip()
        # 剔除 ST / 退市
        if name_col:
            df = df[~df[name_col].astype(str).str.contains("ST|退", na=False)]
        # 涨幅过滤：涨停板买不到、纯跌的不扫（观察池会兜底）
        df = df[(df["_pct"] >= 0.5) & (df["_pct"] <= 9.8)]
        # 市值过滤（仅在市值有效时过滤，同花顺源缺市值时跳过避免误杀）
        if mv_col:
            df["_mv"] = pd.to_numeric(df[mv_col], errors="coerce")
            valid_mv = df[df["_mv"] > 0]
            if len(valid_mv) > len(df) * 0.5:  # 多数票有市值才过滤
                df = df[(df["_mv"] >= self.mv_min * 1e8) & (df["_mv"] <= self.mv_max * 1e8)]
        # 按涨幅降序取 Top N（控制耗时）
        df = df.sort_values("_pct", ascending=False).head(self.fallback_top_n)
        df = self._enrich_spot_with_tencent(df, code_col="_code")

        cands: list[StockCandidate] = []
        # 全市场扫描：使用配置的 RPS 阈值（已放宽），不做临时清零
        # 准备扫描任务列表
        scan_rows = [(row["_code"], row) for _, row in df.iterrows()
                     if _is_a_share(str(row.get("_code", "")).strip())]

        def _eval_one(args):
            code, row = args
            try:
                board = self._board_for_code(code, "") or self._infer_board_from_name(row.get(name_col) if name_col else "")
                return self._evaluate(code, row, board or "全市场强势", spot, date=date)
            except Exception:  # noqa: BLE001
                return None

        # 并发取日K+评估：慢也要等，确保成功（刘总要求）
        market_deadline = time.monotonic() + 180.0  # 全市场扫描预算 180s
        try:
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {pool.submit(_eval_one, task): task for task in scan_rows}
                try:
                    for future in as_completed(futures, timeout=180.0):
                        cand = future.result()
                        if cand is not None:
                            cands.append(cand)
                        if time.monotonic() > market_deadline:
                            remaining = len([f for f in futures if not f.done()])
                            logger.warning("全市场扫描达到时间预算，剩余 %d 只未扫描，已获 %d 只候选",
                                         remaining, len(cands))
                            for pending in futures:
                                pending.cancel()
                            break
                except FuturesTimeoutError:
                    logger.warning("全市场扫描超时，已获 %d 只候选", len(cands))
                    for pending in futures:
                        pending.cancel()
        except Exception as e:  # noqa: BLE001 并发失败回退串行
            logger.warning("并发扫描失败，回退串行: %s", e)
            for code, row in scan_rows:
                try:
                    board = self._board_for_code(code, "") or self._infer_board_from_name(row.get(name_col) if name_col else "")
                    cand = self._evaluate(code, row, board or "全市场强势", spot, date=date)
                    if cand is not None:
                        cands.append(cand)
                except Exception:  # noqa: BLE001
                    pass
        logger.info("全市场扫描：%d 只入选（扫描 %d 只），预算 180s",
                    len(cands), len(scan_rows))
        return cands

    def _scan_broad_pool(
        self, spot: pd.DataFrame, seen: set[str], date: Optional[str] = None,
    ) -> list[StockCandidate]:
        """宽松观察池：仅基于全市场快照，不拉取单只日 K，保证数量与速度。

        入选条件（spot-only）：
        - 非 ST、非退市、北交所/新三板
        - 流通市值在合理区间
        - 涨幅 >= broad_pct_min（默认 -2%，避免单边下跌日无候选）
        - 剔除已在严格候选池的股票
        观察池候选的 score=0，reasons 明确标注「观察池」，由后续评分引擎自然低排序。
        """
        if spot is None or len(spot) == 0:
            return []
        pct_col = _find_col(spot, ["涨跌幅"])
        code_col = _find_col(spot, ["代码"]) or (spot.columns[1] if len(spot.columns) > 1 else None)
        name_col = _find_col(spot, ["名称"])
        mv_col = _find_col(spot, ["流通市值"])
        turnover_col = _find_col(spot, ["换手率"])
        if pct_col is None or code_col is None:
            return []

        df = spot.copy()
        df["_code"] = df[code_col].astype(str).str.strip()
        df["_pct"] = pd.to_numeric(df[pct_col], errors="coerce")
        if name_col:
            df = df[~df[name_col].astype(str).str.contains("ST|退", na=False)]
        df = df[df["_pct"] >= self.broad_pct_min]
        if mv_col:
            df["_mv"] = pd.to_numeric(df[mv_col], errors="coerce")
            valid_mv = df[df["_mv"] > 0]
            if len(valid_mv) > len(df) * 0.5:
                df = df[(df["_mv"] >= self.mv_min * 1e8) & (df["_mv"] <= self.mv_max * 1e8)]
        # 按涨幅降序取足够数量
        need = max(0, self.broad_pool_target - len(seen))
        df = df.sort_values("_pct", ascending=False).head(need + len(seen) + 50)

        cands: list[StockCandidate] = []
        for _, row in df.iterrows():
            code = str(row.get("_code", "")).strip()
            if not code or code in seen or not _is_a_share(code):
                continue
            name = str(row.get(name_col, code)) if name_col else code
            close = safe_float(row.get(_find_col(spot, ["最新价"])))
            if close != close:  # NaN
                close = 0.0
            pct = safe_float(row.get(pct_col))
            if pct != pct:
                pct = 0.0
            turnover = safe_float(row.get(turnover_col)) if turnover_col else 0.0
            if turnover != turnover:
                turnover = 0.0
            mv = safe_float(row.get(mv_col)) if mv_col else 0.0
            mv_yi = mv / 1e8 if mv > 0 else 0.0
            # 优先使用注入的真实 RPS 表（回测等场景），否则用当日涨幅近似代理
            proxy_rps = self.rps_map.get(code)
            if proxy_rps is not None and proxy_rps == proxy_rps:  # 非 NaN
                rps_mode = "real"
            else:
                proxy_rps = min(99.0, max(self.broad_rps_hard_min, 50.0 + pct * 2))
                rps_mode = "approximate"
            cands.append(StockCandidate(
                code=code, name=name, board=self._board_for_code(code, "") or self._infer_board_from_name(name) or "观察池",
                close=close, pct_change=pct, turnover_rate=turnover,
                circ_mv_yi=mv_yi, rps=float(proxy_rps), rps_mode=rps_mode,
                reasons=[f"观察池：快照涨幅 {pct:.2f}%，流通市值 {mv_yi:.1f}亿"],
                fund_inflow_days=0, fund_flow_status="unavailable",
                score=0.0, is_watchlist=True,
            ))
            if len(cands) >= need:
                break
        logger.info("宽松观察池补充：%d 只", len(cands))
        return cands

    def _enrich_spot_with_tencent(self, df: pd.DataFrame, code_col: str) -> pd.DataFrame:
        """用腾讯批量行情补齐同花顺源缺失的市值/PE/PB字段。

        同花顺全市场行情稳定但缺流通市值，直接用 0 会让市值过滤失效，
        前端也会误显示为 0 亿。这里只对待深度扫描的 Top N 做一次腾讯批量
        请求，成本低，失败则保留原数据走降级逻辑。
        """
        if df is None or len(df) == 0 or code_col not in df.columns:
            return df

        if "流通市值" not in df.columns:
            need_mv = True
        else:
            mv_values = pd.to_numeric(df["流通市值"], errors="coerce").fillna(0)
            need_mv = (mv_values <= 0).any()
        if not need_mv:
            return df

        codes = [
            str(c).strip().zfill(6)
            for c in df[code_col].tolist()
            if str(c).strip().isdigit()
        ]
        if not codes:
            return df

        try:
            from .tdx_source import tencent_quote
            quotes = tencent_quote(codes)
        except Exception as e:  # noqa: BLE001
            logger.debug("腾讯行情增强失败: %s", e)
            return df
        if not quotes:
            return df

        out = df.copy()
        for col in ("流通市值", "总市值", "市盈率-动态", "市净率"):
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)

        for idx, row in out.iterrows():
            code = str(row.get(code_col, "")).strip().zfill(6)
            q = quotes.get(code)
            if not q:
                continue
            if q.get("float_mcap_yi", 0) > 0:
                out.at[idx, "流通市值"] = q["float_mcap_yi"] * 1e8
            if q.get("mcap_yi", 0) > 0:
                out.at[idx, "总市值"] = q["mcap_yi"] * 1e8
            if q.get("pe_ttm", 0) != 0:
                out.at[idx, "市盈率-动态"] = q["pe_ttm"]
            if q.get("pb", 0) != 0:
                out.at[idx, "市净率"] = q["pb"]
        return out

    @staticmethod
    def _infer_board_from_name(name: Any) -> str:
        """Conservative local industry fallback from stock name keywords.

        This is intentionally labeled as ``行业:`` so downstream scoring treats it as
        weak evidence, not as a confirmed hot concept.
        """
        text = str(name or "")
        rules = [
            ("银行", ("银行",)),
            ("证券", ("证券", "券商")),
            ("保险", ("保险",)),
            ("房地产", ("地产", "城开", "置业", "发展")),
            ("医药", ("药", "医", "生物", "医疗", "健康")),
            ("化工", ("化", "材料", "新材", "化学", "氟材", "氟")),
            ("有色金属", ("金属", "锂", "铜", "铝", "锆", "钨", "钼", "稀土")),
            ("半导体", ("芯", "微", "半导", "电子")),
            ("智能制造", ("机器人", "智能", "自动化", "步科", "集智", "联众")),
            ("汽车", ("汽车", "汽配", "车", "轮胎", "恒帅", "正强", "科达利")),
            ("电力设备", ("电气", "电力", "电网", "绿能", "冠电")),
            ("光伏储能", ("光伏", "储能", "电池", "能源")),
            ("军工", ("航天", "航空", "兵装", "军工", "雷达")),
            ("机械设备", ("数控", "机床", "精机", "装备")),
            ("消费", ("食品", "酒", "饮料", "百货", "家居", "中炬")),
            ("农业食品", ("农业", "种业", "菌业", "牧业")),
            ("传媒游戏", ("传媒", "游戏", "影视", "文化", "广告", "营销", "因赛", "安妮")),
            ("软件AI", ("软件", "数据", "信息", "科技", "数码", "网络")),
            ("环保", ("环保", "环境", "节能", "科净源", "海洋")),
            ("商业物业", ("商管", "物业")),
            ("黄金", ("黄金", "山金")),
            ("纺织服饰", ("纺", "服饰", "百隆")),
        ]
        for label, keys in rules:
            if any(k in text for k in keys):
                return f"行业:{label}"
        return ""


# ---------------------------------------------------------------------- #
# 指标计算（纯 pandas）
# ---------------------------------------------------------------------- #
def _ma(k: pd.DataFrame, n: int) -> Optional[float]:
    col = _find_col(k, ["收盘", "close", "收盘价"])
    if col is None:
        return None
    s = pd.to_numeric(k[col], errors="coerce").dropna()
    if len(s) < n:
        return None
    return float(s.iloc[-n:].mean())


class RPSCalculator:
    """轻量 RPS 查询器，供策略池等独立模块使用。"""

    def __init__(self, dl: DataLoader, cfg: dict[str, Any] | None = None) -> None:
        self.dl = dl
        self.cfg = cfg or {}
        self.rps_map: dict[str, float] = {}
        self.rps_date: Optional[str] = None

    def set_rps_map(self, rps_map: dict[str, float], date: Optional[str] = None) -> None:
        self.rps_map = rps_map or {}
        self.rps_date = date

    def rps_for_codes(self, codes: list[str], date: Optional[str] = None) -> dict[str, dict[str, Any]]:
        """返回 {code: {"rps": float, "mode": str}}。"""
        date = date or pd.Timestamp.now().strftime("%Y%m%d")
        out: dict[str, dict[str, Any]] = {}
        try:
            spot = self.dl.all_spot()
        except Exception:  # noqa: BLE001
            spot = pd.DataFrame()
        for code in codes:
            try:
                k = self.dl.daily_kline(code, n=30, date=date)
                closes = pd.to_numeric(k["close"], errors="coerce").dropna() if not k.empty else pd.Series(dtype=float)
            except Exception:  # noqa: BLE001
                closes = pd.Series(dtype=float)
            rps = _rps(code, closes, spot, self.rps_map)
            mode = "real" if self.rps_map and code in self.rps_map else "approximate" if len(closes) >= 21 else "unavailable"
            out[code] = {"rps": rps, "mode": mode}
        return out


def _rps(code: str, hist_closes: pd.Series, spot: pd.DataFrame, rps_map: dict | None = None) -> float:
    """个股 20 日 RPS：近 20 日累计涨幅在全市场的百分位排名。

    优先用预计算的真实 RPS 表（rps_map，由 rps.compute_all_rps 离线生成，
    与真实 RPS 相关≈1.0）。表为空时回退到旧的近似版（相关仅 0.19，带偏差，
    仅作兜底，建议盘后跑 `python -m engine.cli rps-build` 预计算）。
    """
    # 1. 优先查真实 RPS 表
    if rps_map:
        v = rps_map.get(code)
        if v is not None and v == v:  # 非 NaN
            return float(v)
    # 2. 回退：近似版（尺度不匹配，已知失真，仅兜底）
    if len(hist_closes) < 21:
        return 50.0
    ret_20d = float(hist_closes.iloc[-1] / hist_closes.iloc[-21] - 1)
    pct_col = _find_col(spot, ["涨跌幅"])
    if pct_col is None:
        return 50.0
    all_pct = pd.to_numeric(spot[pct_col], errors="coerce").dropna()
    rank = (all_pct < ret_20d * 1).mean()  # 近似（失真）
    return float(rank * 100)


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / std


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """匹配列名：精确 > 前缀 > 子串（子串取最短避免误匹配 '涨跌幅.1'）。"""
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
        for real in cols:
            if str(real).startswith(c):
                return real
        matches = [real for real in cols if c in str(real)]
        if matches:
            return min(matches, key=lambda x: len(str(x)))
    return None
