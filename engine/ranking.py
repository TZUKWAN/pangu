"""A股全市场排行榜：涨幅榜/跌幅榜/成交额榜/换手率榜/主力净流入榜。

基于 DataLoader.all_spot() 的全市场实时行情（同花顺源，不封IP，5194只）。
所有排名在内存中完成，毫秒级响应。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("pangu.ranking")

# 预定义排行榜类型
RANK_TYPES = {
    "gainers":      {"col": "涨跌幅",       "ascending": False, "label": "涨幅榜"},
    "losers":       {"col": "涨跌幅",       "ascending": True,  "label": "跌幅榜"},
    "volume":       {"col": "成交额",       "ascending": False, "label": "成交额榜"},
    "turnover":     {"col": "换手率",       "ascending": False, "label": "换手率榜"},
    "net_inflow":   {"col": "主力净流入-净额", "ascending": False, "label": "主力净流入榜"},
    "net_outflow":  {"col": "主力净流入-净额", "ascending": True,  "label": "主力净流出榜"},
    "cap":          {"col": "流通市值",     "ascending": False, "label": "流通市值榜"},
}


def _safe_float_series(series: pd.Series) -> pd.Series:
    """安全转换列到 float，非数值填 0。"""

    def _convert(val: Any) -> float:
        if val is None or (isinstance(val, str) and val.strip() in ("", "-")):
            return 0.0
        if isinstance(val, str):
            # 处理 "175.43亿" 这类中文金额
            s = val.strip().replace(",", "").replace("%", "")
            unit = 1.0
            if s.endswith("亿"):
                unit = 1e8
                s = s[:-1]
            elif s.endswith("万"):
                unit = 1e4
                s = s[:-1]
            try:
                return float(s) * unit
            except ValueError:
                return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    return series.apply(_convert)


@dataclass
class RankResult:
    """单榜结果。"""
    name: str
    label: str
    updated: str
    total_stocks: int
    top_n: int
    rows: list[dict[str, Any]] = field(default_factory=list)


class MarketRanking:
    """全市场排行榜。"""

    def __init__(self, dl=None) -> None:
        self._dl = dl

    def _get_dl(self):
        if self._dl is None:
            from .data_loader import DataLoader
            self._dl = DataLoader()
        return self._dl

    def get_rank(
        self,
        rank_type: str = "gainers",
        top_n: int = 30,
        min_price: float = 0.0,
        exclude_st: bool = True,
    ) -> RankResult:
        """获取单个排行榜。

        Args:
            rank_type: 排行榜类型 (gainers/losers/volume/turnover/net_inflow/net_outflow/cap)
            top_n: 返回前 N 只
            min_price: 最低股价过滤（默认不过滤）
            exclude_st: 是否排除 ST
        """
        cfg = RANK_TYPES.get(rank_type)
        if cfg is None:
            available = ", ".join(RANK_TYPES)
            return RankResult(
                name=rank_type, label=f"未知类型（可用: {available}）",
                updated="", total_stocks=0, top_n=top_n,
            )

        try:
            dl = self._get_dl()
            df = dl.all_spot()
        except Exception as e:  # noqa: BLE001
            logger.warning("all_spot 取数失败: %s", e)
            return RankResult(
                name=rank_type, label=cfg["label"],
                updated=datetime.now().strftime("%H:%M:%S"),
                total_stocks=0, top_n=top_n,
            )

        if df is None or len(df) == 0:
            return RankResult(
                name=rank_type, label=cfg["label"],
                updated=datetime.now().strftime("%H:%M:%S"),
                total_stocks=0, top_n=top_n,
            )

        # 过滤
        if exclude_st:
            if "名称" in df.columns:
                df = df[~df["名称"].str.contains("ST|退", na=False)]

        if min_price > 0 and "最新价" in df.columns:
            df = df[_safe_float_series(df["最新价"]) >= min_price]

        sort_col = cfg["col"]
        if sort_col not in df.columns:
            return RankResult(
                name=rank_type, label=cfg["label"],
                updated=datetime.now().strftime("%H:%M:%S"),
                total_stocks=len(df), top_n=top_n,
                rows=[],
            )

        # 转换排序列为数值
        df = df.copy()
        df["_sort_val"] = _safe_float_series(df[sort_col])

        # 排序
        df = df.sort_values("_sort_val", ascending=cfg["ascending"])

        # 取前 N
        top = df.head(top_n)

        # 构造输出
        display_cols = ["代码", "名称", "最新价", "涨跌幅", "成交额", "换手率", "主力净流入-净额"]
        available = [c for c in display_cols if c in top.columns]

        rows = []
        for rank_idx, (_, row) in enumerate(top.iterrows(), 1):
            item = {"rank": rank_idx}
            for c in available:
                val = row[c]
                if isinstance(val, (int, float)):
                    item[c] = round(val, 2)
                else:
                    item[c] = str(val) if val is not None else ""
            # 附加值（用于展示排序依据）
            item["_sort_value"] = round(row["_sort_val"], 2)
            rows.append(item)

        return RankResult(
            name=rank_type,
            label=cfg["label"],
            updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_stocks=len(df),
            top_n=top_n,
            rows=rows,
        )

    def get_all_ranks(self, top_n: int = 20, **kwargs: Any) -> dict[str, RankResult]:
        """获取所有排行榜。"""
        results = {}
        for rank_type in RANK_TYPES:
            results[rank_type] = self.get_rank(rank_type, top_n=top_n, **kwargs)
        return results

    def get_market_breadth(self) -> dict[str, Any]:
        """市场宽度统计（涨跌家数/涨跌停家数等）。"""
        try:
            dl = self._get_dl()
            df = dl.all_spot()
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

        if df is None or len(df) == 0:
            return {"error": "全市场行情不可用"}

        pct = _safe_float_series(df["涨跌幅"]) if "涨跌幅" in df.columns else pd.Series()

        up = int((pct > 0).sum())
        down = int((pct < 0).sum())
        flat = int((pct == 0).sum())
        limit_up = int((pct >= 9.8).sum())
        limit_down = int((pct <= -9.8).sum())

        # 金额统计
        if "成交额" in df.columns:
            total_amt = _safe_float_series(df["成交额"]).sum()
            avg_amt = total_amt / max(1, len(df))
        else:
            total_amt = avg_amt = 0.0

        return {
            "total": len(df),
            "up": up, "down": down, "flat": flat,
            "up_pct": round(up / max(1, len(df)) * 100, 1),
            "down_pct": round(down / max(1, len(df)) * 100, 1),
            "limit_up": limit_up,
            "limit_down": limit_down,
            "total_amount_yi": round(total_amt / 1e8, 1),
            "avg_amount_wan": round(avg_amt / 1e4, 1),
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
