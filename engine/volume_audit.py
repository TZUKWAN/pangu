"""Volume audit for candidate recommendation gating."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .data_loader import DataLoader, safe_float


class VolumeAudit:
    def __init__(self, dl: DataLoader | None = None, cfg: dict[str, Any] | None = None) -> None:
        self.dl = dl
        self.cfg = (cfg or {}).get("volume_audit", {})

    def audit_candidates(self, candidates: list[dict[str, Any]], date: str | None = None) -> list[dict[str, Any]]:
        for item in candidates:
            item["volume_audit"] = self.audit(item, date=date)
        return candidates

    def audit(self, item: dict[str, Any], date: str | None = None) -> dict[str, Any]:
        rows = self._kline_rows(item, date)
        turnover_rate, turnover_status = self._turnover(item)
        if rows is None or len(rows) < 10:
            return {
                "status": "missing",
                "volume_ratio_5": None,
                "volume_ratio_10": None,
                "amount_ratio_5": None,
                "turnover_rate": turnover_rate,
                "turnover_status": turnover_status,
                "obv_trend": "missing",
                "price_volume_pattern": "missing",
                "volume_anomaly": None,
                "signals": [],
                "risks": ["量能数据不足"],
                "reason": "量能缺失，不能进入正式推荐",
            }

        close = pd.to_numeric(self._col(rows, ["收盘", "close"]), errors="coerce")
        volume = pd.to_numeric(self._col(rows, ["成交量", "volume", "vol"]), errors="coerce")
        amount = pd.to_numeric(self._col(rows, ["成交额", "amount"]), errors="coerce")
        valid = pd.DataFrame({"close": close, "volume": volume, "amount": amount}).dropna(subset=["close", "volume"])
        if len(valid) < 10 or valid["volume"].tail(10).sum() <= 0:
            return {
                "status": "missing",
                "volume_ratio_5": None,
                "volume_ratio_10": None,
                "amount_ratio_5": None,
                "turnover_rate": turnover_rate,
                "turnover_status": turnover_status,
                "obv_trend": "missing",
                "price_volume_pattern": "missing",
                "volume_anomaly": None,
                "signals": [],
                "risks": ["成交量字段缺失或无效"],
                "reason": "成交量无效，不能进入正式推荐",
            }

        latest_vol = float(valid["volume"].iloc[-1])
        avg5 = float(valid["volume"].iloc[-6:-1].mean()) if len(valid) >= 6 else 0.0
        avg10 = float(valid["volume"].iloc[-11:-1].mean()) if len(valid) >= 11 else 0.0
        volume_ratio_5 = latest_vol / avg5 if avg5 > 0 else None
        volume_ratio_10 = latest_vol / avg10 if avg10 > 0 else None

        amount_ratio_5 = None
        if valid["amount"].notna().sum() >= 6:
            latest_amount = float(valid["amount"].iloc[-1])
            avg_amount5 = float(valid["amount"].iloc[-6:-1].mean())
            amount_ratio_5 = latest_amount / avg_amount5 if avg_amount5 > 0 else None

        latest_close = float(valid["close"].iloc[-1])
        prev_close = float(valid["close"].iloc[-2]) if len(valid) >= 2 else latest_close
        recent_high = float(valid["close"].iloc[-21:-1].max()) if len(valid) >= 21 else float(valid["close"].iloc[:-1].max())
        recent_low = float(valid["close"].iloc[-21:-1].min()) if len(valid) >= 21 else float(valid["close"].iloc[:-1].min())
        ret_5 = latest_close / float(valid["close"].iloc[-6]) - 1 if len(valid) >= 6 and valid["close"].iloc[-6] else 0.0
        ret_10 = latest_close / float(valid["close"].iloc[-11]) - 1 if len(valid) >= 11 and valid["close"].iloc[-11] else 0.0

        obv_trend = self._obv_trend(valid["close"], valid["volume"])
        pattern, status, signals, risks = self._pattern(
            latest_close=latest_close,
            prev_close=prev_close,
            recent_high=recent_high,
            recent_low=recent_low,
            ret_5=ret_5,
            ret_10=ret_10,
            volume_ratio_5=volume_ratio_5,
            turnover_status=turnover_status,
        )
        reason = self._reason(pattern, status, turnover_status, signals, risks)
        return {
            "status": status,
            "volume_ratio_5": round(volume_ratio_5, 3) if volume_ratio_5 is not None else None,
            "volume_ratio_10": round(volume_ratio_10, 3) if volume_ratio_10 is not None else None,
            "amount_ratio_5": round(amount_ratio_5, 3) if amount_ratio_5 is not None else None,
            "turnover_rate": turnover_rate,
            "turnover_status": turnover_status,
            "obv_trend": obv_trend,
            "price_volume_pattern": pattern,
            "volume_anomaly": self._anomaly(volume_ratio_5),
            "signals": signals,
            "risks": risks,
            "reason": reason,
        }

    def _pattern(
        self,
        *,
        latest_close: float,
        prev_close: float,
        recent_high: float,
        recent_low: float,
        ret_5: float,
        ret_10: float,
        volume_ratio_5: float | None,
        turnover_status: str,
    ) -> tuple[str, str, list[str], list[str]]:
        signals: list[str] = []
        risks: list[str] = []
        if turnover_status == "missing":
            risks.append("换手率缺失")
        if volume_ratio_5 is None:
            return "missing", "missing", signals, risks + ["量比缺失"]

        breakout = latest_close > recent_high * 1.005 if recent_high > 0 else False
        near_high = latest_close >= recent_high * 0.97 if recent_high > 0 else False
        rebound = latest_close > prev_close and latest_close > recent_low * 1.05 if recent_low > 0 else False
        pullback = ret_5 < 0

        if breakout and volume_ratio_5 >= 1.35:
            signals.append("放量突破")
            return "breakout_with_volume", "ok" if turnover_status != "missing" else "watch", signals, risks
        if breakout and volume_ratio_5 < 1.1:
            risks.append("无量突破")
            return "breakout_without_volume", "watch", signals, risks
        if pullback and volume_ratio_5 <= 0.85:
            signals.append("缩量回踩")
            return "pullback_shrink", "ok" if turnover_status != "missing" else "watch", signals, risks
        if near_high and abs(latest_close / prev_close - 1) < 0.015 and volume_ratio_5 >= 1.8:
            risks.append("高位放量滞涨")
            return "distribution_volume", "watch", signals, risks
        if ret_5 <= -0.08 and volume_ratio_5 >= 2.0:
            risks.append("恐慌放量")
            return "panic_volume", "watch", signals, risks
        if rebound and volume_ratio_5 < 0.9:
            risks.append("无量反弹")
            return "weak_rebound_no_volume", "watch", signals, risks
        return "normal", "watch" if turnover_status == "missing" else "ok", signals, risks

    def _turnover(self, item: dict[str, Any]) -> tuple[float | None, str]:
        status = str(item.get("turnover_status") or "").lower()
        missing = item.get("turnover_missing")
        value = item.get("turnover_rate")
        if value is None and isinstance(item.get("liquidity"), dict):
            value = item["liquidity"].get("turnover_rate")
        rate = safe_float(value, None)
        if status in {"missing", "estimated", "stale", "invalid", "ok"}:
            return rate, status
        if missing is True or rate is None:
            return rate, "missing"
        return rate, "ok"

    def _kline_rows(self, item: dict[str, Any], date: str | None) -> pd.DataFrame | None:
        technical = item.get("technical") or {}
        rows = technical.get("kline") or []
        if rows:
            return pd.DataFrame(rows)
        if self.dl is None:
            return None
        code = str(item.get("code") or "")
        if not code:
            return None
        try:
            return self.dl.daily_kline(code, days=60, date=date)
        except Exception:
            return None

    def _col(self, df: pd.DataFrame, names: list[str]) -> pd.Series:
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series([pd.NA] * len(df))

    def _obv_trend(self, close: pd.Series, volume: pd.Series) -> str:
        if len(close) < 8:
            return "missing"
        direction = close.diff().fillna(0).map(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * volume.fillna(0)).cumsum()
        recent = float(obv.iloc[-1] - obv.iloc[-6]) if len(obv) >= 6 else 0.0
        if recent > 0:
            return "rising"
        if recent < 0:
            return "falling"
        return "flat"

    def _anomaly(self, volume_ratio_5: float | None) -> str | None:
        if volume_ratio_5 is None:
            return None
        if volume_ratio_5 >= 3:
            return "extreme_volume"
        if volume_ratio_5 <= 0.4:
            return "extreme_shrink"
        return None

    def _reason(self, pattern: str, status: str, turnover_status: str, signals: list[str], risks: list[str]) -> str:
        if status == "missing":
            return "量能数据缺失，不能进入正式推荐"
        if turnover_status == "missing":
            return "换手率缺失，量能证据不足"
        if risks:
            return "；".join(risks)
        if signals:
            return "；".join(signals)
        return f"量能形态 {pattern}"
