"""历史回放回测：对正式推荐做因果 1-3 日回放，度量真实成功率。

流程：
1. 对回放区间内每个交易日用 ``Pipeline(replay=True)`` 做盘后选股
   （PIT-safe：全部数据来自本地全市场日线档案 + 档案 RPS + 本地公告档案）；
2. 对每只 ``final_recommendations``，按其结构化买卖计划用
   ``ShortTermReplayEngine`` 模拟「次日入场 → 1-3 日内退出」的真实成交
   （含佣金/印花税/滑点，同日双触发止损优先）；
3. 汇总：净费用后成功率（win = net_pnl > 0）、成交率、平均净收益、
   盈亏比、逐日/逐策略分解，以及 1/2/3 日前瞻收益参考。

诚实性约束（与 short_term_replay 一致）：
- 信号日收盘后才能产生信号，次日才允许买入；
- 未触发条件买点记为 no_entry（不算赢也不算输，单列成交率）；
- 样本不足时明确标注 insufficient_sample，不外推结论。
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import load_config
from .pipeline import Pipeline
from .replay_loader import ReplayDataLoader
from .short_term_replay import ReplayOutcome, ShortTermReplayEngine

logger = logging.getLogger("pangu.replay_backtest")


@dataclass
class ReplayBacktestConfig:
    """回放回测参数。"""

    start_date: str
    end_date: str
    settings_path: str = "config/settings.yaml"
    settings_override: dict[str, Any] = field(default_factory=dict)
    db_path: str = "data/pangu.db"
    archive_db: str = "data/market_breadth/raw.sqlite3"
    announcement_dir: str = "data/announcement_archive"
    output_dir: str = "data/research"
    entry_window_days: int = 1
    max_holding_days: int = 3     # 与 exit_plan 一致的最长持有天数（评估窗口）
    min_trades_for_claim: int = 30
    enable_progress: bool = True
    tag: str = ""

    @classmethod
    def from_args(
        cls,
        start_date: str,
        end_date: str,
        settings_path: str = "config/settings.yaml",
        override: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "ReplayBacktestConfig":
        return cls(
            start_date=start_date,
            end_date=end_date,
            settings_path=settings_path,
            settings_override=override or {},
            **kwargs,
        )


@dataclass
class HorizonStat:
    """某前瞻期的上涨率统计。"""

    horizon: int
    samples: int = 0
    ups: int = 0
    sum_ret: float = 0.0

    def add(self, ret: float) -> None:
        self.samples += 1
        self.sum_ret += ret
        if ret > 0:
            self.ups += 1

    @property
    def up_rate(self) -> float:
        return self.ups / self.samples if self.samples else 0.0

    @property
    def avg_ret(self) -> float:
        return self.sum_ret / self.samples if self.samples else 0.0


class ReplayBacktester:
    """PIT-safe 推荐回测器。"""

    def __init__(
        self,
        cfg: ReplayBacktestConfig,
        loader: Optional[ReplayDataLoader] = None,
    ) -> None:
        self.cfg = cfg
        # 档案日期为紧凑 YYYYMMDD；入口统一归一，接受 ISO 输入
        cfg.start_date = str(cfg.start_date).replace("-", "")
        cfg.end_date = str(cfg.end_date).replace("-", "")
        self.full_cfg = load_config(cfg.settings_path)
        if cfg.settings_override:
            self._deep_merge(self.full_cfg, cfg.settings_override)
        self.pipeline = Pipeline(full_cfg=self.full_cfg, replay=True)
        # 优化循环中可共享同一个已装载档案的 loader，避免重复预热
        if loader is not None:
            self.pipeline._replay_loader = loader
        self.loader: ReplayDataLoader = self.pipeline._activate_replay(cfg.start_date)
        self.replay_engine = ShortTermReplayEngine(cfg=self.full_cfg)

    @staticmethod
    def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                ReplayBacktester._deep_merge(dst[k], v)
            else:
                dst[k] = v

    # ------------------------------------------------------------------ #
    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        all_days = self.loader.trading_days(cfg.start_date, cfg.end_date)
        # 信号日之后需要 entry_window + max_holding 个交易日来完成评估
        horizon = cfg.entry_window_days + cfg.max_holding_days
        evaluable = [d for d in all_days if self._future_days(d) >= horizon]
        if len(evaluable) < len(all_days):
            logger.info(
                "%d 个交易日中 %d 个可完整评估（尾部 %d 天窗口不足）",
                len(all_days), len(evaluable), len(all_days) - len(evaluable),
            )

        signals: list[dict[str, Any]] = []
        daily_summary: list[dict[str, Any]] = []
        t_start = time.time()
        for i, date in enumerate(evaluable):
            t0 = time.time()
            result = self.pipeline.run(date, replay=True)
            finals = result.final_recommendations or []
            for item in finals:
                signals.append({
                    "signal_date": date,
                    "code": str(item.get("code") or "").zfill(6),
                    "name": item.get("name"),
                    "strategy": item.get("strategy") or item.get("strategy_name"),
                    "role": item.get("role"),
                    "score": item.get("score"),
                    "rps": item.get("rps"),
                    "item": item,
                })
            daily_summary.append({
                "date": date,
                "final_count": len(finals),
                "watch_count": len(result.watchlist or []),
                "signal_count": sum(len(v) for v in (result.strategy_signals or {}).values()),
                "temperature": (result.sentiment or {}).get("temperature"),
                "data_quality": result.data_quality,
                "elapsed_s": round(time.time() - t0, 1),
            })
            if cfg.enable_progress:
                logger.info(
                    "[回放 %d/%d] %s final=%d watch=%d (%.1fs)",
                    i + 1, len(evaluable), date, len(finals),
                    len(result.watchlist or []), time.time() - t0,
                )
        elapsed = time.time() - t_start

        outcomes = self._evaluate(signals)
        report = self._build_report(signals, outcomes, daily_summary, elapsed)
        self._save(report)
        return report

    def _future_days(self, date: str) -> int:
        days = self.loader._trade_dates
        idx = days.index(date)
        return len(days) - idx - 1

    # ------------------------------------------------------------------ #
    def _evaluate(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        horizons = {h: HorizonStat(horizon=h) for h in (1, 2, 3)}
        open_entry = HorizonStat(horizon=0)  # 次日开盘买入持有到信号后第3日收盘
        for sig in signals:
            code = sig["code"]
            date = sig["signal_date"]
            item = sig["item"]
            kline = self.loader.daily_kline(code, days=260, date="99999999")
            outcome: ReplayOutcome = self.replay_engine.replay(
                item, kline, signal_date=date,
            )
            row = outcome.to_dict()
            row["strategy"] = sig.get("strategy")
            row["role"] = sig.get("role")
            row["score"] = sig.get("score")
            outcomes.append(row)

            # 前瞻参考收益（无费用，收盘对收盘 + 次日开盘入场版）
            frame = self.replay_engine._normalize_kline(kline)
            if frame is not None:
                future = frame[frame["_date"] > str(date)].reset_index(drop=True)
                if not future.empty:
                    sig_close_rows = frame[frame["_date"] <= str(date)]
                    if not sig_close_rows.empty:
                        base_close = float(sig_close_rows.iloc[-1]["_close"])
                        next_open = float(future.iloc[0]["_open"])
                        for h, stat in horizons.items():
                            if len(future) >= h:
                                ret = float(future.iloc[h - 1]["_close"]) / base_close - 1
                                stat.add(ret)
                        if len(future) >= 3:
                            open_entry.add(float(future.iloc[2]["_close"]) / next_open - 1)
        self._horizons = {h: s for h, s in horizons.items()}
        self._open_entry = open_entry
        return outcomes

    # ------------------------------------------------------------------ #
    def _build_report(
        self,
        signals: list[dict[str, Any]],
        outcomes: list[dict[str, Any]],
        daily_summary: list[dict[str, Any]],
        elapsed: float,
    ) -> dict[str, Any]:
        cfg = self.cfg
        closed = [o for o in outcomes if o.get("status") == "closed"]
        no_entry = [o for o in outcomes if o.get("status") == "no_entry"]
        pending = [o for o in outcomes if o.get("status") == "pending"]
        invalid = [o for o in outcomes if o.get("status") == "invalid"]

        wins = [o for o in closed if (o.get("win") is True)]
        losses = [o for o in closed if (o.get("win") is False)]
        n_closed = len(closed)
        success_rate = len(wins) / n_closed if n_closed else None

        rets = [float(o.get("net_return") or 0.0) for o in closed]
        gross_win = sum(r for r in rets if r > 0)
        gross_loss = abs(sum(r for r in rets if r < 0))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_win > 0 else 0.0
        )
        avg_ret = sum(rets) / n_closed if n_closed else 0.0

        # 逐笔等权净值与最大回撤
        equity, peak, max_dd = 1.0, 1.0, 0.0
        for r in rets:
            equity *= (1 + r)
            peak = max(peak, equity)
            max_dd = max(max_dd, 1 - equity / peak)

        # Wilson 下界
        wilson_lo = None
        if n_closed:
            z = 1.96
            p = len(wins) / n_closed
            denom = 1 + z * z / n_closed
            center = p + z * z / (2 * n_closed)
            margin = z * math.sqrt(p * (1 - p) / n_closed + z * z / (4 * n_closed * n_closed))
            wilson_lo = (center - margin) / denom

        # 分策略
        by_strategy: dict[str, dict[str, Any]] = {}
        for o in closed:
            s = str(o.get("strategy") or "未知")
            st = by_strategy.setdefault(
                s, {"closed": 0, "wins": 0, "net_return_sum": 0.0},
            )
            st["closed"] += 1
            st["wins"] += 1 if o.get("win") else 0
            st["net_return_sum"] += float(o.get("net_return") or 0.0)
        for s, st in by_strategy.items():
            st["success_rate"] = round(st["wins"] / st["closed"], 4) if st["closed"] else None
            st["avg_net_return"] = round(st["net_return_sum"] / st["closed"], 5) if st["closed"] else None
            del st["net_return_sum"]

        n_signals = len(signals)
        execution_rate = n_closed / n_signals if n_signals else 0.0
        insufficient = n_closed < cfg.min_trades_for_claim

        horizon_stats = {
            f"{h}d": {
                "samples": s.samples,
                "up_rate": round(s.up_rate, 4),
                "avg_return": round(s.avg_ret, 5),
            }
            for h, s in sorted(self._horizons.items())
        }

        return {
            "research_only": True,
            "pit_safe": True,
            "engine": "replay_backtest",
            "start": cfg.start_date,
            "end": cfg.end_date,
            "entry_window_days": cfg.entry_window_days,
            "max_holding_days": cfg.max_holding_days,
            "generated_at": pd.Timestamp.now().isoformat(),
            "elapsed_s": round(elapsed, 1),
            "settings_override": cfg.settings_override,
            "signals": {
                "total": n_signals,
                "trading_days": len(daily_summary),
                "execution_rate": round(execution_rate, 4),
                "no_entry": len(no_entry),
                "invalid": len(invalid),
                "pending": len(pending),
                "closed": n_closed,
            },
            "success_metrics": {
                "definition": "净费用后 net_pnl > 0（次日入场，1-3 日结构化退出）",
                "success_rate": round(success_rate, 4) if success_rate is not None else None,
                "wilson_95_lower": round(wilson_lo, 4) if wilson_lo is not None else None,
                "wins": len(wins),
                "losses": len(losses),
                "avg_net_return": round(avg_ret, 5) if n_closed else None,
                "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else None,
                "max_drawdown": round(max_dd, 4),
                "final_equity": round(equity, 4),
                "insufficient_sample": insufficient,
                "min_trades_for_claim": cfg.min_trades_for_claim,
            },
            "horizon_reference": {
                **horizon_stats,
                "next_open_to_3d_close": {
                    "samples": self._open_entry.samples,
                    "up_rate": round(self._open_entry.up_rate, 4),
                    "avg_return": round(self._open_entry.avg_ret, 5),
                },
            },
            "by_strategy": by_strategy,
            "daily": daily_summary,
            "outcomes": outcomes,
        }

    # ------------------------------------------------------------------ #
    def _save(self, report: dict[str, Any]) -> None:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{self.cfg.tag}" if self.cfg.tag else ""
        path = out_dir / (
            f"replay_backtest_{self.cfg.start_date}_{self.cfg.end_date}{suffix}.json"
        )
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        logger.info("回放回测报告已写入 %s", path)

    # ------------------------------------------------------------------ #
    @staticmethod
    def summary_text(report: dict[str, Any]) -> str:
        """人类可读摘要。"""
        sm = report.get("success_metrics", {})
        sg = report.get("signals", {})
        hz = report.get("horizon_reference", {})
        lines = [
            f"回放区间 {report['start']} ~ {report['end']}"
            f"（{sg.get('trading_days')} 个交易日，信号 {sg.get('total')} 笔）",
            f"成交 {sg.get('closed')} 笔（成交率 {sg.get('execution_rate')*100:.0f}%，"
            f"未触发 {sg.get('no_entry')}，无效 {sg.get('invalid')}）",
            f"成功率: {sm.get('success_rate')*100:.2f}%" if sm.get("success_rate") is not None else "成功率: 无成交样本",
            f"Wilson95 下界: {sm.get('wilson_95_lower')}",
            f"平均净收益/笔: {sm.get('avg_net_return')}",
            f"盈亏比 PF: {sm.get('profit_factor')} | 最大回撤: {sm.get('max_drawdown')}",
            f"前瞻参考: 1日上涨率 {hz.get('1d', {}).get('up_rate')}, "
            f"2日 {hz.get('2d', {}).get('up_rate')}, 3日 {hz.get('3d', {}).get('up_rate')} | "
            f"次日开盘→3日收盘 {hz.get('next_open_to_3d_close', {}).get('up_rate')}",
        ]
        if sm.get("insufficient_sample"):
            lines.append(
                f"⚠ 成交样本 {sm.get('wins', 0) + sm.get('losses', 0)} < "
                f"{sm.get('min_trades_for_claim')}，以下数字不构成统计结论（insufficient_sample）"
            )
        by = report.get("by_strategy") or {}
        if by:
            lines.append("分策略：")
            for s, st in sorted(by.items()):
                lines.append(
                    f"  {s}: {st.get('closed')} 笔, 成功率 {st.get('success_rate')}, "
                    f"均净收益 {st.get('avg_net_return')}"
                )
        return "\n".join(lines)


def run_replay_backtest(
    start_date: str,
    end_date: str,
    settings_path: str = "config/settings.yaml",
    override: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """便捷入口：构建配置并跑完整回放回测。"""
    cfg = ReplayBacktestConfig.from_args(
        start_date, end_date, settings_path=settings_path, override=override, **kwargs,
    )
    bt = ReplayBacktester(cfg)
    return bt.run()
