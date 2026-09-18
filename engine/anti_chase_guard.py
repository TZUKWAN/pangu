"""反追涨闸门：把明显高位加速、远离均线的候选降级到观察池或 blocked。

设计原则：
- 默认降级为 watch，保留市场结构信息；
- 只有极端情况（连续缩量加速、非核心票短期暴涨）才 blocked；
- 根据 strategy_name / role / theme / entry_style 做差异化阈值。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader, safe_float

logger = logging.getLogger("pangu.anti_chase")


class AntiChaseGuard:
    """反追涨闸门。"""

    # 默认阈值（比例：0.07 = 7%）。candidate 的 pct_change 为百分点（10.0 = 10%）
    # 在 _judge_one 中统一除以 100 换算成比例后比较。
    DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
        "default": {
            "ret_3d": 0.18, "ret_5d": 0.30,
            "dist_ma5": 0.10, "dist_ma20": 0.20, "daily_pct": 0.08,
        },
        "leader": {
            "ret_3d": 0.25, "ret_5d": 0.40,
            "dist_ma5": 0.15, "dist_ma20": 0.25, "daily_pct": 0.105,
        },
        "limit_up_core": {
            "ret_3d": 0.30, "ret_5d": 0.50,
            "dist_ma5": 0.20, "dist_ma20": 0.30, "daily_pct": 0.105,
        },
        "limit_up_back": {
            "ret_3d": 0.15, "ret_5d": 0.25,
            "dist_ma5": 0.10, "dist_ma20": 0.18, "daily_pct": 0.075,
        },
        "trend_pullback": {
            "ret_3d": 0.15, "ret_5d": 0.25,
            "dist_ma5": 0.08, "dist_ma20": 0.18, "daily_pct": 0.06,
        },
        "oversold_rebound": {
            "ret_3d": 0.12, "ret_5d": 0.20,
            "dist_ma5": 0.08, "dist_ma20": 0.15, "daily_pct": 0.06,
        },
        "large_cap_low_vol": {
            "ret_3d": 0.20, "ret_5d": 0.35,
            "dist_ma5": 0.12, "dist_ma20": 0.25, "daily_pct": 0.08,
        },
    }

    def __init__(self, dl: DataLoader, cfg: dict[str, Any] | None = None) -> None:
        self.dl = dl
        full_cfg = cfg or {}
        self.cfg = full_cfg.get("anti_chase", {})
        # 允许经 settings.yaml 覆盖任一策略的阈值
        self._threshold_overrides = self.cfg.get("thresholds") or {}

    def _thresholds(self, strategy: str, role: str) -> dict[str, float]:
        if strategy == "limit_up":
            base = "limit_up_core" if role in ("龙头", "中军", "leader", "core") else "limit_up_back"
        else:
            base = strategy if strategy in self.DEFAULT_THRESHOLDS else "default"
        th = dict(self.DEFAULT_THRESHOLDS.get(base, self.DEFAULT_THRESHOLDS["default"]))
        override = self._threshold_overrides.get(base)
        if isinstance(override, dict):
            for k, v in override.items():
                try:
                    th[k] = float(v)
                except (TypeError, ValueError):
                    continue
        return th

    def guard(
        self,
        candidates: list[dict[str, Any]],
        date: str | None = None,
    ) -> list[dict[str, Any]]:
        """对候选列表统一打 anti_chase 标记，返回原列表（字段被修改）。"""
        for c in candidates:
            c["anti_chase"] = self._judge_one(c, date)
        return candidates

    def _judge_one(self, item: dict[str, Any], date: str | None) -> dict[str, Any]:
        code = str(item.get("code", ""))
        close = safe_float(item.get("close"), 0.0)
        # pct_change 以百分点传入（10.0 = 10%），统一换算成比例（0.10）
        pct = safe_float(item.get("pct_change"), 0.0) / 100.0

        # 优先用 technical 里的均线，避免重复取 K 线
        ma_map = (item.get("technical") or {}).get("ma") or {}
        ma5 = safe_float(ma_map.get("ma5"), 0.0)
        ma20 = safe_float(ma_map.get("ma20"), 0.0)

        # 计算短期涨幅
        technical_kline = (item.get("technical") or {}).get("kline")
        ret_3d, ret_5d = self._short_term_returns(code, date, technical_kline)

        metrics = {
            "return_3d": round(ret_3d, 4) if ret_3d is not None else None,
            "return_5d": round(ret_5d, 4) if ret_5d is not None else None,
            "dist_ma5": round((close - ma5) / ma5, 4) if ma5 and ma5 > 0 else None,
            "dist_ma20": round((close - ma20) / ma20, 4) if ma20 and ma20 > 0 else None,
            "daily_pct": round(pct, 4),
        }

        strategy = self._strategy_type(item)
        role = (item.get("role") or "").lower()
        th = self._thresholds(strategy, role)

        watch_reasons: list[str] = []
        blocked_reasons: list[str] = []

        def _hit(name: str, val: float | None, threshold: float) -> bool:
            return val is not None and val > threshold

        if _hit("3日涨幅", metrics["return_3d"], th["ret_3d"]):
            watch_reasons.append(f"近3日涨幅 {metrics['return_3d']*100:.1f}% 超出阈值 {th['ret_3d']*100:.0f}%")
        if _hit("5日涨幅", metrics["return_5d"], th["ret_5d"]):
            if strategy == "limit_up" and role not in ("龙头", "中军", "leader", "core", "中军"):
                # 连板梯队后排补涨：5 日涨幅过高直接 blocked，禁入 final
                blocked_reasons.append(f"连板梯队后排补涨，近5日涨幅 {metrics['return_5d']*100:.1f}% 过高")
            elif strategy in ("leader", "limit_up") or role in ("龙头", "中军"):
                watch_reasons.append(f"近5日涨幅 {metrics['return_5d']*100:.1f}% 过高")
            else:
                blocked_reasons.append(f"近5日涨幅 {metrics['return_5d']*100:.1f}% 过高，非核心票")
        if _hit("距MA5", metrics["dist_ma5"], th["dist_ma5"]):
            watch_reasons.append(f"股价距MA5 {metrics['dist_ma5']*100:.1f}% 超出阈值")
        if _hit("距MA20", metrics["dist_ma20"], th["dist_ma20"]):
            watch_reasons.append(f"股价距MA20 {metrics['dist_ma20']*100:.1f}% 超出阈值")
        if _hit("当日涨幅", metrics["daily_pct"], th["daily_pct"]):
            if strategy == "breakout_confirm" and role in ("龙头", "中军"):
                watch_reasons.append("当日涨幅较大，需分歧/回封确认")
            else:
                watch_reasons.append(f"当日涨幅 {metrics['daily_pct']*100:.1f}% 超出阈值")

        # 缩量加速：近3日连续上涨且成交量递减（需要 K 线）
        if self._is_shrinking_acceleration(code, date, technical_kline):
            blocked_reasons.append("连续缩量加速，风险过高")

        # 结果判定
        if blocked_reasons:
            return {
                "status": "blocked",
                "reason": "；".join(blocked_reasons + watch_reasons),
                "metrics": metrics,
            }
        if watch_reasons:
            return {
                "status": "watch",
                "reason": "；".join(watch_reasons),
                "metrics": metrics,
            }
        return {
            "status": "ok",
            "reason": "",
            "metrics": metrics,
        }

    def _strategy_type(self, item: dict[str, Any]) -> str:
        """从候选字段推断策略类型。"""
        strategy = str(item.get("strategy_name") or item.get("strategy") or "").lower()
        if "leader" in strategy or "龙头" in strategy:
            return "leader"
        if "limit" in strategy or "连板" in strategy:
            return "limit_up"
        if "pullback" in strategy or "回踩" in strategy or "趋势回踩" in strategy:
            return "trend_pullback"
        if "oversold" in strategy or "超跌" in strategy:
            return "oversold_rebound"
        if "dividend" in strategy or "large" in strategy or "低波" in strategy:
            return "large_cap_low_vol"
        entry_style = str(item.get("entry_style") or "").lower()
        if "breakout" in entry_style:
            return "breakout_confirm"
        return "default"

    def _short_term_returns(
        self,
        code: str,
        date: str | None,
        technical_kline: Any = None,
    ) -> tuple[Optional[float], Optional[float]]:
        """返回 (3日涨幅, 5日涨幅)。"""
        if not code:
            return None, None
        try:
            k = pd.DataFrame(technical_kline) if technical_kline else self.dl.daily_kline(code, days=10, date=date)
            if k is None or len(k) < 6:
                return None, None
            close_col = None
            for col in ("收盘", "close", "收盘价"):
                if col in k.columns:
                    close_col = col
                    break
            if close_col is None:
                return None, None
            closes = pd.to_numeric(k[close_col], errors="coerce").dropna()
            if len(closes) < 6:
                return None, None
            latest = float(closes.iloc[-1])
            ret_3d = (latest / float(closes.iloc[-4]) - 1) if len(closes) >= 4 else None
            ret_5d = (latest / float(closes.iloc[-6]) - 1) if len(closes) >= 6 else None
            return ret_3d, ret_5d
        except Exception as e:  # noqa: BLE001
            logger.debug("%s 短期涨幅计算失败: %s", code, e)
            return None, None

    def _is_shrinking_acceleration(
        self,
        code: str,
        date: str | None,
        technical_kline: Any = None,
    ) -> bool:
        """判断近3日是否连续上涨且成交量递减。"""
        if not code:
            return False
        try:
            k = pd.DataFrame(technical_kline) if technical_kline else self.dl.daily_kline(code, days=10, date=date)
            if k is None or len(k) < 4:
                return False
            close_col = None
            vol_col = None
            for col in ("收盘", "close", "收盘价"):
                if col in k.columns:
                    close_col = col
                    break
            for col in ("成交量", "volume", "vol"):
                if col in k.columns:
                    vol_col = col
                    break
            if close_col is None or vol_col is None:
                return False
            closes = pd.to_numeric(k[close_col], errors="coerce").dropna()
            vols = pd.to_numeric(k[vol_col], errors="coerce").dropna()
            if len(closes) < 4 or len(vols) < 4:
                return False
            # 近3根 K 线连续收阳且成交量递减
            if not all(closes.iloc[-1] > closes.iloc[-2] > closes.iloc[-3] > closes.iloc[-4]):
                return False
            if not (vols.iloc[-1] < vols.iloc[-2] < vols.iloc[-3]):
                return False
            # 近3日累计涨幅 > 15%
            return (closes.iloc[-1] / closes.iloc[-4] - 1) > 0.15
        except Exception as e:  # noqa: BLE001
            logger.debug("%s 缩量加速判断失败: %s", code, e)
            return False
