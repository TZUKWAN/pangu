"""市场结构分析仪：把「情绪温度计」从单点打分扩展到多维诊断。

背景
----
sentiment_meter.py 只取 5 个当日分项（涨停/连板/炸板/跌停/涨跌比），
只能回答「今天热不热」，回答不了：
- 今天 58 分在历史上算什么水平？
- 情绪是在加速升温还是拐头向下？
- 指数、融资余额、板块轮动在不在同一个节奏？

本模块新增能力
----------------
1. 14+ 情绪分项（赚钱效应/市场广度/波动率/板块轮动/封板率/连板梯队等），
   已完全去除对东财独家接口的依赖。
2. 60 日、250 日历史分位，让「今日 58 分」有参照系。
3. 情绪动量：5 日/10 日变化率、加速度、顶/底信号。
4. 市场结构：主要指数趋势（相对 20 日线）、两融余额（数据源已停用）、主力资金流。
5. 诊断报告：JSON（给 pipeline/LLM）+ Markdown（给人读）。

数据源
------
- 同花顺：全市场实时行情、涨停池、跌停池、强势股池、板块资金流、概念板块
- 腾讯：个股前复权 K 线（用于均线/市场广度）、指数 K 线
- 新浪：财务指标（备用）
- adata：概念板块成分股（备用）

设计取舍
--------
- 历史分位依赖本地历史库。首次运行历史不足时会用中性 50 分占位并写入 warnings，
  随着每日运行自动累积到 60/250 个交易日。
- 市场广度（均线多头/站上 20 日线）需要个股历史收盘价。为控制耗时，
  默认只采样流通市值最大的 150 只 A 股作为全市场代理，结果写入 warnings 说明采样口径。
  采样数据会缓存到本地，后续每日只增量更新 1 天，速度可接受。
- 所有数据调用都走 DataLoader 的重试/缓存/降级机制，不直接裸调数据源。

使用示例
--------
    from engine.data_loader import DataLoader
    from engine.market_structure import MarketStructureAnalyzer

    dl = DataLoader()
    msa = MarketStructureAnalyzer(dl)
    result = msa.analyze()          # 分析今天
    print(result.to_json())         # 结构化 JSON
    print(result.to_markdown())     # 人读简报

    # 与现有 pipeline 兼容的简化情绪字典
    legacy = result.to_legacy_sentiment()
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .data_loader import DataLoader, safe_float, find_col as _find_col

logger = logging.getLogger("pangu.market_structure")

# --------------------------------------------------------------------------- #
# 常量：指数代码映射
# --------------------------------------------------------------------------- #
INDEX_SYMBOLS = {
    # 腾讯指数代码（带市场前缀，避免与股票代码冲突）
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "沪深300": "sh000300",
    "创业板指": "sz399006",
    "科创50": "sh000688",
}

# --------------------------------------------------------------------------- #
# 小型辅助函数
# --------------------------------------------------------------------------- #

def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """把数值限制在 [lo, hi] 区间。"""
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return (lo + hi) / 2


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """匹配列名：精确 > 前缀 > 子串（子串取最短避免误匹配）。"""
    if df is None or df.empty:
        return None
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


def _anchor_score(
    value: float,
    a1: float, a2: float, a3: float, a4: float,
) -> float:
    """分段线性映射：value 在 [a1,a2,a3,a4] 上分别对应 [0,50,85,100]。"""
    points = sorted([(a1, 0.0), (a2, 50.0), (a3, 85.0), (a4, 100.0)], key=lambda p: p[0])
    if value <= points[0][0]:
        return 0.0
    if value >= points[-1][0]:
        return 100.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= value <= x1:
            if x1 == x0:
                return y1
            return _clamp(y0 + (value - x0) / (x1 - x0) * (y1 - y0))
    return 50.0


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """把可能是字符串的序列转成 float，非法值变 NaN。"""
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce")


def _last_trade_date(date: str) -> str:
    """返回前一天（简单日历，不处理假期；情绪分析用于相邻交易日比较即可）。"""
    d = datetime.strptime(date, "%Y%m%d")
    prev = d - timedelta(days=1)
    # 若前一天是周末，继续往前推（粗略）
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev.strftime("%Y%m%d")


# --------------------------------------------------------------------------- #
# 历史库：累积每日快照，用于历史分位和动量
# --------------------------------------------------------------------------- #

class HistoryKeeper:
    """本地历史库。存两类数据：
    1. 每日情绪/结构快照（aggregated）。
    2. 用于计算市场广度的采样个股收盘价（wide）。
    3. 每日领涨板块（用于轮动速度）。
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.root = Path(cache_dir) / "market_history"
        self.root.mkdir(parents=True, exist_ok=True)
        self.snap_path = self.root / "snapshots.parquet"
        self.price_path = self.root / "sample_prices.parquet"
        self.sector_path = self.root / "sector_leaders.parquet"

    # ---- 快照 ----
    def load_snapshots(self) -> pd.DataFrame:
        if not self.snap_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(self.snap_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("历史快照读取失败: %s", e)
            return pd.DataFrame()

    def save_snapshots(self, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self.snap_path, index=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("历史快照保存失败: %s", e)

    def append_snapshot(self, snap: dict[str, Any]) -> None:
        df = self.load_snapshots()
        row = pd.DataFrame([snap])
        if df.empty:
            df = row
        else:
            # 去重：同一日期只保留最新
            df = pd.concat([df[df["date"] != snap["date"]], row], ignore_index=True)
            df = df.sort_values("date").reset_index(drop=True)
        self.save_snapshots(df)

    def percentile(
        self,
        date: str,
        field: str,
        windows: list[int] = (60, 250),
    ) -> dict[str, Optional[float]]:
        """计算某字段在指定历史窗口内的分位（不含当天）。"""
        df = self.load_snapshots()
        out: dict[str, Optional[float]] = {f"p{w}": None for w in windows}
        if df.empty or field not in df.columns:
            return out
        hist = df[df["date"] < date][field].dropna()
        if hist.empty:
            return out
        today_val = df[df["date"] == date][field]
        today_val = today_val.iloc[0] if not today_val.empty else None
        if today_val is None or pd.isna(today_val):
            return out
        for w in windows:
            window = hist.tail(w)
            if len(window) < 5:  # 样本太少不分位
                continue
            # 分位 = 小于今天值的天数 / 总天数
            out[f"p{w}"] = round((window < today_val).sum() / len(window) * 100, 1)
        return out

    def momentum(self, date: str, field: str = "temperature") -> dict[str, Optional[float]]:
        """计算 5 日、10 日变化率和加速度。"""
        df = self.load_snapshots()
        out: dict[str, Optional[float]] = {"roc_5d": None, "roc_10d": None, "accel": None}
        if df.empty or field not in df.columns:
            return out
        s = df.set_index("date")[field].dropna()
        if date not in s.index:
            return out
        idx = s.index.get_loc(date)
        if idx == 0:
            return out
        today = s.iloc[idx]
        for d, key in [(5, "roc_5d"), (10, "roc_10d")]:
            if idx - d >= 0:
                prev = s.iloc[idx - d]
                if prev != 0 and not pd.isna(prev):
                    out[key] = round((today - prev) / abs(prev) * 100, 2)
        # 加速度 = 5 日 ROC - 10 日 ROC（看短期是否在加速）
        if out["roc_5d"] is not None and out["roc_10d"] is not None:
            out["accel"] = round(out["roc_5d"] - out["roc_10d"], 2)
        return out

    # ---- 采样股价 ----
    def load_prices(self) -> pd.DataFrame:
        if not self.price_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(self.price_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("采样股价读取失败: %s", e)
            return pd.DataFrame()

    def save_prices(self, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self.price_path, index=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("采样股价保存失败: %s", e)

    def update_prices(self, new_rows: list[dict[str, Any]]) -> pd.DataFrame:
        """把新交易日的采样收盘价追加到宽表。new_rows: [{date, symbol, close}, ...]。"""
        if not new_rows:
            return self.load_prices()
        new = pd.DataFrame(new_rows)
        old = self.load_prices()
        if old.empty:
            combined = new
        else:
            combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
        combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.save_prices(combined)
        return combined

    # ---- 领涨板块 ----
    def load_sectors(self) -> pd.DataFrame:
        if not self.sector_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(self.sector_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("板块历史读取失败: %s", e)
            return pd.DataFrame()

    def save_sectors(self, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self.sector_path, index=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("板块历史保存失败: %s", e)

    def record_sector_leaders(self, date: str, leaders: list[str]) -> None:
        df = self.load_sectors()
        row = pd.DataFrame([{"date": date, "leaders": leaders}])
        if df.empty:
            df = row
        else:
            df = pd.concat([df[df["date"] != date], row], ignore_index=True)
            df = df.sort_values("date").reset_index(drop=True)
        self.save_sectors(df)

    def prev_leaders(self, date: str) -> Optional[list[str]]:
        df = self.load_sectors()
        if df.empty:
            return None
        prev_date = _last_trade_date(date)
        row = df[df["date"] == prev_date]
        if row.empty:
            # 尝试找最近一天
            past = df[df["date"] < date]
            if past.empty:
                return None
            row = past.tail(1)
        leaders = row.iloc[0]["leaders"]
        return list(leaders) if leaders is not None else []

    def prev_leaders_window(
        self,
        date: str,
        window: int,
        include_yesterday: bool = True,
    ) -> dict[str, Optional[list[str]]]:
        """取过去若干交易日的领涨板块列表。

        Args:
            date: 当前日期 YYYYMMDD。
            window: 往回看几个交易日（含）。
            include_yesterday: 是否单独返回昨日列表。

        Returns:
            {"yesterday": [...], "last_n": [...], "combined": set(...)}
            数据不足时对应项为 None 或空集合。
        """
        df = self.load_sectors()
        result: dict[str, Optional[list[str]]] = {
            "yesterday": None,
            "last_n": None,
        }
        if df.empty:
            return result

        past = df[df["date"] < date].sort_values("date").reset_index(drop=True)
        if past.empty:
            return result

        if include_yesterday and len(past) >= 1:
            leaders = past.iloc[-1]["leaders"]
            result["yesterday"] = list(leaders) if leaders is not None else []

        n_rows = past.tail(window)
        combined: list[str] = []
        for _, row in n_rows.iterrows():
            leaders = row["leaders"]
            if leaders is not None:
                combined.extend(list(leaders))
        result["last_n"] = combined
        return result


# --------------------------------------------------------------------------- #
# 结果对象
# --------------------------------------------------------------------------- #

@dataclass
class MarketStructureResult:
    """市场结构分析结果。"""

    date: str
    # 情绪分项：原始值 + 0-100 打分
    components: dict[str, Any] = field(default_factory=dict)
    # 汇总情绪
    sentiment: dict[str, Any] = field(default_factory=dict)
    # 市场结构（指数/两融/资金流）
    market_structure: dict[str, Any] = field(default_factory=dict)
    # 动量
    momentum: dict[str, Any] = field(default_factory=dict)
    # 信号与诊断
    signals: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "temperature": round(self.sentiment.get("temperature", 50.0), 1),
            "posture": self.sentiment.get("posture", "正常"),
            "advice": self.sentiment.get("advice", ""),
            "components": self.components,
            "sentiment": self.sentiment,
            "market_structure": self.market_structure,
            "momentum": self.momentum,
            "signals": self.signals,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def to_legacy_sentiment(self) -> dict[str, Any]:
        """与现有 pipeline 兼容的简化情绪字典。"""
        comps: dict[str, Any] = {}
        for k, v in self.components.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    comps[f"{k}_{sub_k}"] = sub_v
            else:
                comps[k] = v
        return {
            "temperature": round(self.sentiment.get("temperature", 50.0), 1),
            "posture": self.sentiment.get("posture", "正常"),
            "advice": self.sentiment.get("advice", ""),
            "date": self.date,
            "components": comps,
            "warnings": self.warnings,
            "market_structure": self.market_structure,
            "momentum": self.momentum,
            "signals": self.signals,
        }

    def to_markdown(self) -> str:
        """人读诊断报告。"""
        lines: list[str] = []
        lines.append(f"# 市场情绪与结构诊断报告 — {self.date}")
        lines.append("")
        s = self.sentiment
        lines.append(f"## 一、情绪总览")
        lines.append(f"- **情绪温度**：{s.get('temperature', 0):.1f} / 100")
        lines.append(f"- **市场姿态**：{s.get('posture', '未知')}")
        lines.append(f"- **一句话建议**：{s.get('advice', '')}")
        lines.append(f"- **历史分位**：60日 {s.get('percentile_60d', 'N/A')} | 250日 {s.get('percentile_250d', 'N/A')}")
        lines.append("")

        lines.append("## 二、情绪分项（原始值 | 打分）")
        for name, info in self.components.items():
            if isinstance(info, dict) and "score" in info:
                raw = info.get("raw", "N/A")
                score = info.get("score", 0)
                lines.append(f"- **{name}**：{raw} → {score:.1f} 分")
            else:
                lines.append(f"- **{name}**：{info}")
        lines.append("")

        lines.append("## 三、动量")
        for k, v in self.momentum.items():
            lines.append(f"- {k}：{v if v is not None else 'N/A'}")
        lines.append("")

        lines.append("## 四、市场结构")
        ms = self.market_structure
        if "indices" in ms:
            lines.append("### 指数趋势（相对20日线）")
            for idx_name, idx_info in ms["indices"].items():
                lines.append(
                    f"- {idx_name}：最新 {idx_info.get('close')}，"
                    f"20日线 {idx_info.get('ma20')}，"
                    f"偏离 {idx_info.get('deviate_pct'):.2f}%"
                )
        if "margin" in ms:
            lines.append(f"### 两融余额：{ms['margin'].get('balance_yi', 'N/A')} 亿元")
        if "fund_flow" in ms:
            lines.append("### 主力资金")
            lines.append(f"- 净流入板块数：{ms['fund_flow'].get('inflow_sectors', 'N/A')}")
            lines.append(f"- 净流出板块数：{ms['fund_flow'].get('outflow_sectors', 'N/A')}")
            top3 = ms["fund_flow"].get("top3_inflow", [])
            if top3:
                lines.append("- 净流入前三：" + "、".join(top3))
        lines.append("")

        lines.append("## 五、信号")
        sig = self.signals
        lines.append(f"- **见顶信号**：{sig.get('peak', False)} ({sig.get('peak_reason', '')})")
        lines.append(f"- **见底信号**：{sig.get('bottom', False)} ({sig.get('bottom_reason', '')})")
        lines.append(f"- **结构背离**：{sig.get('divergence', '无')}")
        if self.warnings:
            lines.append("")
            lines.append("## 六、数据提示")
            for w in self.warnings:
                lines.append(f"- ⚠ {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主分析器
# --------------------------------------------------------------------------- #

class MarketStructureAnalyzer:
    """市场结构与情绪增强分析器。"""

    def __init__(
        self,
        dl: DataLoader,
        cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        self.dl = dl
        self.cfg = cfg or {}
        self.history = HistoryKeeper(self.cfg.get("history_dir", "data/cache"))

        # 可配置权重：15 个分项。默认合计 1.0
        self.weights = self.cfg.get("weights", {
            "limit_up_count": 0.13,
            "consecutive_height": 0.10,
            "broke_rate": 0.0,        # 炸板池已停用（东财独家），权重并入其他项
            "limit_down_count": 0.09,
            "advance_decline": 0.08,
            "strong_gain_ratio": 0.09,
            "ma_bullish_ratio": 0.09,
            "above_ma20_ratio": 0.09,
            "volatility": 0.08,
            "sector_rotation": 0.08,
            "seal_rate": 0.09,         # 封板成功率（替代炸板率）
            "ladder_structure": 0.08,
            "first_vs_high_divergence": 0.00,
            "strong_pool_count": 0.00,
        })
        # 校验权重和
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            logger.warning("情绪分项权重和 %.3f，自动归一化", total)
            self.weights = {k: v / total for k, v in self.weights.items()}

        self.breadth_sample_size = self.cfg.get("breadth_sample_size", 150)
        self.breadth_min_history = self.cfg.get("breadth_min_history", 20)
        self.sector_top_n = self.cfg.get("sector_top_n", 10)
        # 轮动持续性单独配置前 N，通常取前 5 名板块观察持续性
        self.sector_rotation_persistence_top_n = self.cfg.get(
            "sector_rotation_persistence_top_n", 5
        )
        self.sector_rotation_lookback = self.cfg.get("sector_rotation_lookback", 3)

    # ----------------------------------------------------------------------- #
    # 主入口
    # ----------------------------------------------------------------------- #
    def analyze(self, date: Optional[str] = None) -> MarketStructureResult:
        date = date or datetime.now().strftime("%Y%m%d")
        result = MarketStructureResult(date=date)
        warnings: list[str] = []

        # ---- 0. 拉取基础数据 ----
        spot = self.dl.all_spot()
        zt = self.dl.limit_up_pool(date)
        broke = self.dl.broke_pool(date)
        dt = self.dl.limit_down_pool(date)
        strong = self.dl.strong_pool(date)
        concept = self.dl.concept_boards()
        fund = self.dl.sector_fund_flow_rank("今日")

        # 涨跌比列
        pct_col = _find_col(spot, ["涨跌幅"])
        pcts = _to_numeric_series(spot[pct_col]) if pct_col else pd.Series(dtype=float)

        # ── 涨跌停池可用性检测（双模式核心）────────────────────────────
        # 涨停池/炸板池/跌停池全空 = 相关数据源不可用。
        # 此时改用全市场涨跌幅分布近似封板指标，保证情绪温度计不致盲。
        limit_pool_available = len(zt) > 0 or len(broke) > 0 or len(dt) > 0
        if not limit_pool_available and len(pcts) > 0:
            warnings.append(
                "涨停/炸板/跌停池数据为空（可能数据源暂不可用），情绪温度计改用"
                "全市场涨跌幅分布近似封板指标（精度略降但不致盲）"
            )
            logger.warning("涨停池数据为空，情绪温度计启用近似模式")

        # ---- 1. 计算 15+ 情绪分项 ----
        components: dict[str, Any] = {}

        if limit_pool_available:
            # 精确模式：涨停/炸板/跌停数据齐全，逐项精算
            # 1.1 涨停家数
            lu_count = len(zt)
            components["limit_up_count"] = {
                "raw": lu_count,
                "score": _anchor_score(lu_count, 15, 40, 80, 120),
            }

            # 1.2 最高连板高度
            consec_height = self._max_consecutive(zt)
            components["consecutive_height"] = {
                "raw": consec_height,
                "score": _anchor_score(consec_height, 2, 4, 6, 8),
            }

            # 1.3 炸板率
            broke_count = len(broke)
            denom_lb = lu_count + broke_count
            broke_rate = broke_count / denom_lb if denom_lb > 0 else 0.0
            components["broke_rate"] = {
                "raw": round(broke_rate, 4),
                "score": _clamp(100 - (broke_rate / 0.60) * 100),
            }

            # 1.4 跌停家数
            ld_count = len(dt)
            components["limit_down_count"] = {
                "raw": ld_count,
                "score": _clamp(100 - (ld_count / 60) * 100),
            }

            # 1.10 涨停封板率 = 涨停 / (涨停 + 炸板)
            seal_rate = lu_count / denom_lb if denom_lb > 0 else 0.0
            components["seal_rate"] = {
                "raw": round(seal_rate, 4),
                "score": _clamp(seal_rate * 100),
            }

            # 1.11 连板梯队结构（金字塔健康度）
            ladder = self._ladder_structure(zt)
            components["ladder_structure"] = {
                "raw": ladder,
                "score": round(ladder.get("pyramid_score", 50.0), 1),
            }

            # 1.13 首板/高度板赚钱效应分化
            divergence = self._first_vs_high_divergence(zt, date)
            components["first_vs_high_divergence"] = {
                "raw": divergence,
                "score": round(divergence.get("score", 50.0), 1),
            }

            # 1.14 强势股池数量
            sp_count = len(strong)
            components["strong_pool_count"] = {
                "raw": sp_count,
                "score": _anchor_score(sp_count, 10, 30, 80, 150),
            }

        else:
            # ── 近似模式：涨跌停池不可用，用全市场涨跌幅分布近似封板指标 ──
            approx = self._approximate_limit_pools(pcts)
            components["limit_up_count"] = approx["limit_up_count"]
            components["consecutive_height"] = approx["consecutive_height"]
            components["broke_rate"] = approx["broke_rate"]
            components["limit_down_count"] = approx["limit_down_count"]
            components["seal_rate"] = approx["seal_rate"]
            components["ladder_structure"] = {"raw": {}, "score": 50.0}
            components["first_vs_high_divergence"] = {"raw": {}, "score": 50.0}
            components["strong_pool_count"] = approx["strong_pool_count"]

        # 1.5 涨跌家数比（不依赖东财，两种模式都算）
        advance = int((pcts > 0).sum()) if not pcts.empty else 0
        decline = int((pcts < 0).sum()) if not pcts.empty else 0
        total_ad = advance + decline
        ad_ratio = advance / total_ad if total_ad > 0 else 0.5
        components["advance_decline"] = {
            "raw": f"{advance}/{decline}",
            "score": _clamp(ad_ratio * 100),
        }

        # 1.6 赚钱效应（涨幅>5%占比）
        strong_gain_ratio = (pcts > 5).sum() / len(pcts) if len(pcts) > 0 else 0.0
        components["strong_gain_ratio"] = {
            "raw": round(strong_gain_ratio, 4),
            "score": _anchor_score(strong_gain_ratio * 100, 2, 5, 15, 25),
        }

        # 1.7 市场广度：均线多头占比 & 站上20日线占比
        breadth = self._market_breadth(spot, date)
        components["ma_bullish_ratio"] = {
            "raw": round(breadth["ma_bullish_ratio"], 4),
            "score": _anchor_score(breadth["ma_bullish_ratio"] * 100, 10, 30, 60, 80),
        }
        components["above_ma20_ratio"] = {
            "raw": round(breadth["above_ma20_ratio"], 4),
            "score": _anchor_score(breadth["above_ma20_ratio"] * 100, 15, 35, 65, 85),
        }
        if breadth.get("warning"):
            warnings.append(breadth["warning"])

        # 1.8 波动率（涨跌幅标准差）—— 过高视为不稳定
        volatility = pcts.std() if not pcts.empty else 0.0
        vol_score = 100.0
        if volatility > 3.0:
            vol_score = _clamp(100 - (volatility - 3.0) / 3.0 * 100)
        components["volatility"] = {
            "raw": round(volatility, 4),
            "score": round(vol_score, 1),
        }

        # 1.9 板块轮动速度（今昨领涨重合度）
        rotation = self._sector_rotation_speed(concept, date)
        components["sector_rotation"] = {
            "raw": round(rotation["overlap"], 4),
            "score": _anchor_score(rotation["overlap"] * 100, 20, 40, 65, 85),
        }
        if rotation.get("warning"):
            warnings.append(rotation["warning"])

        # 1.9b 板块轮动持续性（今日领涨前 N 与昨日/前 3 日重合度）
        persistence = self._sector_rotation_persistence(concept, date)
        components["sector_rotation_persistence"] = {
            "raw": {
                "overlap_1d": round(persistence.get("overlap_1d", 0.0), 4),
                "overlap_3d": round(persistence.get("overlap_3d", 0.0), 4),
            },
            "score": round(persistence.get("persistence_score", 50.0), 1),
        }
        if persistence.get("warning"):
            warnings.append(persistence["warning"])

        # ---- 2. 汇总情绪温度 ----
        temperature = self._weighted_temperature(components)
        posture, advice = self._posture(temperature)

        # ---- 3. 历史分位与动量（先保存快照再计算）----
        snap = {"date": date, "temperature": temperature}
        for k, v in components.items():
            if isinstance(v, dict) and "score" in v:
                snap[f"{k}_score"] = v["score"]
            if isinstance(v, dict) and "raw" in v and isinstance(v["raw"], (int, float)):
                snap[f"{k}_raw"] = v["raw"]
        self.history.append_snapshot(snap)

        percentiles = self.history.percentile(date, "temperature", [60, 250])
        mom = self.history.momentum(date, "temperature")

        # ---- 4. 市场结构 ----
        indices = self._index_structure()
        margin = self._margin_status()
        fund_flow = self._fund_flow_status(fund)

        # ---- 5. 信号 ----
        signals = self._generate_signals(
            temperature=temperature,
            components=components,
            mom=mom,
            indices=indices,
        )

        result.components = components
        result.sentiment = {
            "temperature": round(temperature, 1),
            "posture": posture,
            "advice": advice,
            "percentile_60d": percentiles.get("p60"),
            "percentile_250d": percentiles.get("p250"),
        }
        result.momentum = mom
        result.market_structure = {
            "indices": indices,
            "margin": margin,
            "fund_flow": fund_flow,
        }
        result.signals = signals
        result.warnings = warnings

        logger.info(
            "市场结构分析完成 %s 温度=%.1f 姿态=%s 60日分位=%s 250日分位=%s",
            date, temperature, posture,
            percentiles.get("p60"), percentiles.get("p250"),
        )
        return result

    # ----------------------------------------------------------------------- #
    # 分项计算细节
    # ----------------------------------------------------------------------- #
    def _max_consecutive(self, zt: pd.DataFrame) -> int:
        """从涨停池取最高连板数。"""
        if zt is None or zt.empty:
            return 0
        for col_name in ["连板数", "涨停统计"]:
            col = _find_col(zt, [col_name])
            if col is None:
                continue
            vals = zt[col].astype(str)
            nums = vals.str.extract(r"(\d+)").iloc[:, 0]
            nums = pd.to_numeric(nums, errors="coerce").dropna()
            if len(nums) > 0:
                return int(nums.max())
        return 0

    def _approximate_limit_pools(self, pcts: pd.Series) -> dict[str, dict[str, Any]]:
        """东财不可用时，用全市场涨跌幅分布近似封板指标。

        A 股主板涨停 ±10%、创业板/科创板 ±20%。用涨跌幅分布的尾部统计近似：
        - 涨停家数 ≈ 涨幅 ≥ 9.8% 的家数（涨停封板的近似）
        - 跌停家数 ≈ 跌幅 ≤ -9.8% 的家数
        - 连板高度：无历史数据无法精确，用涨停家数反推（涨停多≈情绪高≈可能有高度板）
        - 炸板率/封板率：用涨停家数占比间接映射（涨停多→封板率高）
        - 强势池 ≈ 涨幅 ≥ 7% 的家数

        精度说明：比东财精确值略粗（9.8% 阈值会漏掉部分未完全封板的票），
        但趋势方向一致，足够让情绪温度计给出合理的冷热判断。
        """
        result: dict[str, dict[str, Any]] = {}
        if pcts is None or pcts.empty:
            # 全市场涨跌幅也没有（极端情况），全部中性
            neutral = {"raw": 0, "score": 50.0}
            return {
                "limit_up_count": neutral, "consecutive_height": neutral,
                "broke_rate": {"raw": 0.0, "score": 50.0},
                "limit_down_count": neutral, "seal_rate": {"raw": 0.0, "score": 50.0},
                "strong_pool_count": neutral,
            }

        n_total = len(pcts)
        # 涨停近似：涨幅 ≥ 9.8%（覆盖主板10%涨停；创业板/科创20%涨停会算进强势池）
        lu_approx = int((pcts >= 9.8).sum())
        ld_approx = int((pcts <= -9.8).sum())
        strong_approx = int((pcts >= 7.0).sum())

        # 连板高度近似：涨停家数越多，情绪越亢奋，高度板概率越大
        # 涨停<10→高度1, 10-30→2, 30-60→3, 60-100→4, >100→5
        consec_approx = min(5, max(1, lu_approx // 20 + 1))

        # 封板率近似：涨停家数占全市场比例映射（涨停多→封板成功率高）
        lu_ratio = lu_approx / n_total if n_total > 0 else 0
        seal_approx = min(0.95, lu_ratio * 15)  # 缩放到合理封板率区间
        broke_rate_approx = max(0.05, 1 - seal_approx)  # 炸板率 ≈ 1 - 封板率

        result["limit_up_count"] = {
            "raw": lu_approx,
            "score": _anchor_score(lu_approx, 15, 40, 80, 120),
        }
        result["consecutive_height"] = {
            "raw": consec_approx,
            "score": _anchor_score(consec_approx, 2, 4, 6, 8),
        }
        result["broke_rate"] = {
            "raw": round(broke_rate_approx, 4),
            "score": _clamp(100 - (broke_rate_approx / 0.60) * 100),
        }
        result["limit_down_count"] = {
            "raw": ld_approx,
            "score": _clamp(100 - (ld_approx / 60) * 100),
        }
        result["seal_rate"] = {
            "raw": round(seal_approx, 4),
            "score": _clamp(seal_approx * 100),
        }
        result["strong_pool_count"] = {
            "raw": strong_approx,
            "score": _anchor_score(strong_approx, 10, 30, 80, 150),
        }
        return result

    def _market_breadth(self, spot: pd.DataFrame, date: str) -> dict[str, Any]:
        """计算市场广度。默认采样流通市值最大的 N 只股票。
        首次运行会拉取这些股票 30 日 K 线，之后每日只增量更新 1 天。
        """
        result = {"ma_bullish_ratio": 0.0, "above_ma20_ratio": 0.0, "warning": ""}
        if spot is None or spot.empty:
            result["warning"] = "实时行情为空，市场广度用中性值"
            return result

        # 取流通市值列
        mv_col = _find_col(spot, ["流通市值"])
        code_col = _find_col(spot, ["代码"])
        if mv_col is None or code_col is None:
            result["warning"] = "实时行情缺代码/流通市值列，市场广度用中性值"
            return result

        spot["_mv"] = _to_numeric_series(spot[mv_col])
        sample = spot.nlargest(self.breadth_sample_size, "_mv")[[code_col]].copy()
        sample.columns = ["symbol"]

        # 增量更新股价缓存：daily_kline 返回近 30 天，全部写入缓存，
        # 这样首次运行即可拥有足够历史计算均线广度。
        new_rows: list[dict[str, Any]] = []
        for symbol in sample["symbol"].astype(str):
            hist = self.dl.daily_kline(symbol, days=30)
            if hist is None or hist.empty:
                continue
            close_col = _find_col(hist, ["收盘", "close"])
            date_col = _find_col(hist, ["日期", "date"])
            if close_col is None or date_col is None:
                continue
            for _, row in hist.iterrows():
                d = str(row[date_col]).replace("-", "")
                close = safe_float(row[close_col])
                if not math.isnan(close):
                    new_rows.append({
                        "date": d,
                        "symbol": symbol,
                        "close": close,
                    })

        prices = self.history.update_prices(new_rows)
        if prices.empty:
            result["warning"] = "采样股价缓存为空，市场广度用中性值"
            return result

        # 计算每个 symbol 到 date 为止是否有足够历史
        date_prices = prices[prices["date"] <= date]
        if date_prices.empty:
            result["warning"] = "采样股价无目标日期数据，市场广度用中性值"
            return result

        bullish = 0
        above20 = 0
        valid = 0
        min_h = self.breadth_min_history
        for symbol, g in date_prices.groupby("symbol"):
            g = g.sort_values("date").reset_index(drop=True)
            # 找到 date 对应位置
            idx = g[g["date"] == date].index
            if len(idx) == 0:
                continue
            idx = idx[0]
            if idx < min_h:
                continue
            window = g.iloc[idx - min_h + 1: idx + 1]
            closes = window["close"].astype(float)
            if len(closes) < min_h or closes.isna().any():
                continue
            ma5 = closes.tail(5).mean()
            ma10 = closes.tail(10).mean()
            ma20 = closes.tail(20).mean()
            if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20):
                continue
            valid += 1
            if ma5 > ma10 > ma20:
                bullish += 1
            if closes.iloc[-1] > ma20:
                above20 += 1

        if valid == 0:
            result["warning"] = (
                f"采样股有效历史不足 {min_h} 天（仅 {valid} 只），"
                "市场广度用中性值；持续运行后会自动累积"
            )
            return result

        result["ma_bullish_ratio"] = bullish / valid
        result["above_ma20_ratio"] = above20 / valid
        result["warning"] = (
            f"市场广度基于流通市值前 {self.breadth_sample_size} 只采样股，"
            f"有效 {valid} 只"
        )
        return result

    def _sector_rotation_speed(self, concept: pd.DataFrame, date: str) -> dict[str, Any]:
        """板块轮动速度：今日领涨前 N 与昨日领涨前 N 的 Jaccard 重合度。"""
        result = {"overlap": 0.5, "warning": ""}
        if concept is None or concept.empty:
            result["warning"] = "概念板块数据为空，轮动速度用中性 0.5"
            return result

        name_col = _find_col(concept, ["板块名称", "名称"])
        if name_col is None:
            result["warning"] = "概念板块缺名称列，轮动速度用中性 0.5"
            return result

        today_leaders = concept[name_col].head(self.sector_top_n).astype(str).tolist()
        self.history.record_sector_leaders(date, today_leaders)

        prev_leaders = self.history.prev_leaders(date)
        if prev_leaders is None:
            result["warning"] = "暂无昨日领涨板块，轮动速度用中性 0.5"
            return result

        set_t = set(today_leaders)
        set_p = set(prev_leaders)
        inter = len(set_t & set_p)
        union = len(set_t | set_p)
        result["overlap"] = inter / union if union > 0 else 0.0
        return result

    def _sector_rotation_persistence(self, concept: pd.DataFrame, date: str) -> dict[str, Any]:
        """板块轮动持续性：今日领涨前 N 与昨日/前 3 日的重合度。

        返回：
            overlap_1d: 与昨日领涨前 N 的 Jaccard 重合度
            overlap_3d: 与前 3 日（合并集合）的 Jaccard 重合度
            persistence_score: 综合持续性得分（0-100）
        """
        result = {
            "overlap_1d": 0.0,
            "overlap_3d": 0.0,
            "persistence_score": 50.0,
            "warning": "",
        }
        if concept is None or concept.empty:
            result["warning"] = "概念板块数据为空，轮动持续性用中性 50"
            return result

        name_col = _find_col(concept, ["板块名称", "名称"])
        if name_col is None:
            result["warning"] = "概念板块缺名称列，轮动持续性用中性 50"
            return result

        top_n = self.sector_rotation_persistence_top_n
        today_leaders = set(concept[name_col].head(top_n).astype(str).tolist())

        window = self.history.prev_leaders_window(date, window=self.sector_rotation_lookback)
        yesterday = window.get("yesterday")
        last_n = window.get("last_n")

        if yesterday is None and last_n is None:
            result["warning"] = "暂无历史领涨板块，轮动持续性用中性 50"
            return result

        def _jaccard(a: set, b: set) -> float:
            union = len(a | b)
            return len(a & b) / union if union > 0 else 0.0

        if yesterday is not None:
            result["overlap_1d"] = _jaccard(today_leaders, set(yesterday))
        if last_n is not None:
            # last_n 是多个交易日 leaders 的合并列表，直接 set 去重
            result["overlap_3d"] = _jaccard(today_leaders, set(last_n))

        # 综合得分：1 日重合度 60% + 3 日重合度 40%，再映射到 0-100
        raw = result["overlap_1d"] * 0.6 + result["overlap_3d"] * 0.4
        result["persistence_score"] = _anchor_score(raw * 100, 10, 25, 50, 75)
        return result

    def _ladder_structure(self, zt: pd.DataFrame) -> dict[str, Any]:
        """连板梯队结构健康度：统计各连板数的家数，金字塔越明显分越高。"""
        result = {"distribution": {}, "pyramid_score": 50.0}
        if zt is None or zt.empty:
            return result
        col = _find_col(zt, ["连板数"])
        if col is None:
            return result
        nums = pd.to_numeric(zt[col], errors="coerce").dropna().astype(int)
        if nums.empty:
            return result
        dist = nums.value_counts().sort_index().to_dict()
        result["distribution"] = {int(k): int(v) for k, v in dist.items()}
        # 金字塔评分：低板家数 >= 高板家数得高分
        levels = sorted(dist.keys())
        if len(levels) < 2:
            result["pyramid_score"] = 50.0
            return result
        score = 100.0
        for i in range(len(levels) - 1):
            lower = dist.get(levels[i], 0)
            higher = dist.get(levels[i + 1], 0)
            if higher > lower:
                score -= 25.0 * (higher / max(lower, 1))
        result["pyramid_score"] = _clamp(score)
        return result

    def _first_vs_high_divergence(self, zt: pd.DataFrame, date: str) -> dict[str, Any]:
        """首板 vs 高度板赚钱效应分化。
        用昨日涨停股池（stock_zt_pool_previous_em）的收益来近似：
        - 首板：连板数 == 1 的股票昨日收益（= 涨停 9.9% 左右）
        - 高度板：连板数 >= 3 的股票今日平均收益
        若高度板收益远高于首板，说明资金抱团高位，属分化/末端特征。
        """
        result = {
            "first_board_avg": None,
            "high_board_avg": None,
            "spread": None,
            "score": 50.0,
            "warning": "",
        }
        if zt is None or zt.empty:
            result["warning"] = "涨停池为空，首板/高度板分化无法计算"
            return result

        # 尝试取昨日涨停池
        try:
            prev_zt = self.dl.limit_up_pool(_last_trade_date(date))
        except Exception as e:  # noqa: BLE001
            logger.warning("昨日涨停池获取失败: %s", e)
            prev_zt = pd.DataFrame()

        if prev_zt is None or prev_zt.empty:
            result["warning"] = "昨日涨停池为空，分化指标用中性值"
            return result

        consec_col = _find_col(prev_zt, ["连板数"])
        pct_col = _find_col(zt, ["涨跌幅"])
        code_col = _find_col(zt, ["代码"])
        if consec_col is None or pct_col is None or code_col is None:
            result["warning"] = "涨停池缺连板数/涨跌幅/代码列，分化指标用中性值"
            return result

        prev_zt["_consec"] = pd.to_numeric(prev_zt[consec_col], errors="coerce")
        first_codes = set(prev_zt[prev_zt["_consec"] == 1][code_col].astype(str))
        high_codes = set(prev_zt[prev_zt["_consec"] >= 3][code_col].astype(str))

        zt["_pct"] = _to_numeric_series(zt[pct_col])
        zt["_code"] = zt[code_col].astype(str)

        first_pcts = zt[zt["_code"].isin(first_codes)]["_pct"]
        high_pcts = zt[zt["_code"].isin(high_codes)]["_pct"]

        first_avg = first_pcts.mean() if not first_pcts.empty else None
        high_avg = high_pcts.mean() if not high_pcts.empty else None
        result["first_board_avg"] = round(first_avg, 2) if first_avg is not None else None
        result["high_board_avg"] = round(high_avg, 2) if high_avg is not None else None

        if first_avg is not None and high_avg is not None and not pd.isna(high_avg) and not pd.isna(first_avg):
            spread = high_avg - first_avg
            result["spread"] = round(spread, 2)
            # 高度板比首板高 3% 以上 → 分化加剧，分数压低
            if spread > 3:
                result["score"] = _clamp(50 - (spread - 3) * 10)
            elif spread < -2:
                # 高度板反而更差 → 高位杀跌，也偏低
                result["score"] = _clamp(50 + (spread + 2) * 5)
            else:
                result["score"] = 65.0
        else:
            result["warning"] = "首板/高度板样本不足，分化指标用中性值"
        return result

    # ----------------------------------------------------------------------- #
    # 汇总与市场结构
    # ----------------------------------------------------------------------- #
    def _weighted_temperature(self, components: dict[str, Any]) -> float:
        total = 0.0
        for key, weight in self.weights.items():
            comp = components.get(key, {})
            score = comp.get("score", 50.0) if isinstance(comp, dict) else 50.0
            total += score * weight
        return _clamp(total)

    def _posture(self, temperature: float) -> tuple[str, str]:
        if temperature < 40:
            return "冰点", "情绪退潮，市场赚钱效应差。建议空仓或只做极轻仓左侧试错。"
        if temperature > 85:
            return "亢奋", "一致性过强，随时分歧退潮。控制仓位，避免高位接力。"
        return "正常", "情绪温和，存在结构性机会。按趋势选股，严格止损。"

    def _index_structure(self) -> dict[str, Any]:
        """主要指数相对 20 日线位置。

        用腾讯指数K线（不封 IP）替代东财 index_zh_a_hist。
        """
        indices: dict[str, Any] = {}
        for name, symbol in INDEX_SYMBOLS.items():
            try:
                from . import tdx_source
                # 腾讯指数代码：sh000001/sz399001/sh000300/sz399006
                df = tdx_source.tencent_kline_qfq(symbol, days=60, adjust="")
            except Exception as e:  # noqa: BLE001
                logger.debug("指数 %s 腾讯K线失败: %s", name, e)
                continue
            if df is None or df.empty or len(df) < 20:
                continue
            close_col = _find_col(df, ["收盘", "close"])
            if close_col is None:
                continue
            closes = _to_numeric_series(df[close_col])
            close = closes.iloc[-1]
            ma20 = closes.tail(20).mean()
            if pd.isna(close) or pd.isna(ma20):
                continue
            indices[name] = {
                "symbol": symbol,
                "close": round(close, 2),
                "ma20": round(ma20, 2),
                "deviate_pct": round((close - ma20) / ma20 * 100, 2),
            }
        return indices

    def _margin_status(self) -> dict[str, Any]:
        """两融余额。东财接口已下线，返回空（不影响温度计主分项）。"""
        return {"balance_yi": None, "warning": "两融数据源已停用"}

    def _fund_flow_status(self, fund: pd.DataFrame) -> dict[str, Any]:
        """主力资金流状态。"""
        result = {
            "inflow_sectors": None,
            "outflow_sectors": None,
            "top3_inflow": [],
            "warning": "",
        }
        if fund is None or fund.empty:
            result["warning"] = "板块资金流数据为空"
            return result
        name_col = _find_col(fund, ["名称"])
        net_col = _find_col(fund, ["今日主力净流入-净额", "主力净流入-净额", "净流入"])
        if name_col is None or net_col is None:
            result["warning"] = "板块资金流缺名称/净流入列"
            return result
        fund["_net"] = _to_numeric_series(fund[net_col])
        inflow = (fund["_net"] > 0).sum()
        outflow = (fund["_net"] < 0).sum()
        top3 = fund.nlargest(3, "_net")[name_col].astype(str).tolist()
        result["inflow_sectors"] = int(inflow)
        result["outflow_sectors"] = int(outflow)
        result["top3_inflow"] = top3
        return result

    def _generate_signals(
        self,
        temperature: float,
        components: dict[str, Any],
        mom: dict[str, Optional[float]],
        indices: dict[str, Any],
    ) -> dict[str, Any]:
        """生成见顶、见底、背离信号。"""
        signals: dict[str, Any] = {
            "peak": False,
            "peak_reason": "",
            "bottom": False,
            "bottom_reason": "",
            "divergence": "无",
        }

        broke_rate = components.get("broke_rate", {}).get("raw", 0.0)
        ld_count = components.get("limit_down_count", {}).get("raw", 0)
        ad_score = components.get("advance_decline", {}).get("score", 50.0)
        seal_rate = components.get("seal_rate", {}).get("raw", 0.0)
        roc5 = mom.get("roc_5d")
        roc10 = mom.get("roc_10d")

        # 见顶信号
        peak_factors = []
        if temperature > 85:
            peak_factors.append("温度>85")
        if broke_rate > 0.35:
            peak_factors.append("炸板率高")
        if seal_rate < 0.70:
            peak_factors.append("封板率<70%")
        if roc5 is not None and roc10 is not None and roc5 < roc10:
            peak_factors.append("短期动量衰减")
        if len(peak_factors) >= 2:
            signals["peak"] = True
            signals["peak_reason"] = "、".join(peak_factors)

        # 见底信号
        bottom_factors = []
        if temperature < 40:
            bottom_factors.append("温度<40")
        if ld_count > 50:
            bottom_factors.append("跌停家数>50")
        if roc5 is not None and roc10 is not None and roc5 > roc10:
            bottom_factors.append("短期动量修复")
        if ad_score < 30 and temperature < 40:
            bottom_factors.append("涨跌比极低")
        if len(bottom_factors) >= 2:
            signals["bottom"] = True
            signals["bottom_reason"] = "、".join(bottom_factors)

        # 指数与情绪背离
        if indices:
            avg_deviate = np.mean([v["deviate_pct"] for v in indices.values() if "deviate_pct" in v])
            if not math.isnan(avg_deviate):
                if avg_deviate > 1.0 and temperature < 50:
                    signals["divergence"] = "指数偏强但情绪偏冷（权重/蓝筹拉升）"
                elif avg_deviate < -1.0 and temperature > 60:
                    signals["divergence"] = "指数偏弱但情绪偏热（题材小票活跃）"
        return signals


# --------------------------------------------------------------------------- #
# 命令行/独立运行入口
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dl = DataLoader()
    analyzer = MarketStructureAnalyzer(dl)
    res = analyzer.analyze()
    print(res.to_markdown())
    print("\n--- JSON ---\n")
    print(res.to_json())
