"""量化护栏：对候选池做「排雷」过滤，宁可错过不可买错。

这是趋势选股之后的「安全网」，剔除有重大风险的票，避免踩雷。
属于「量化辅助」范畴 —— 不主导选股，只做减法。

护栏规则（可经 settings.yaml 调）：
- ST / *ST / 退市风险：直接剔除
- 上市不足 N 日次新：剔除（波动异常、无历史参照）
- 一字板：可选剔除（买不到，短线打板可关）
- 估值：PE/PB 超上限剔除（泡沫）；PE 为负（亏损）剔除
- 财务风险：最近报告期亏损 / 资产负债率过高 → 剔除

注意：财务数据取数较慢（逐只取），护栏按「快规则在前」排序，
先剔 ST/次新/估值（用已有 spot 数据，秒级），最后才取财务（慢）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader, safe_float, find_col as _find_col
from .trend_scanner import StockCandidate

logger = logging.getLogger("pangu.guard")


@dataclass
class GuardResult:
    """护栏结果：kept 是真正通过硬护栏的候选；watch 是轻微风险只能观察；
    rejected 是明确排除的候选。"""

    kept: list[StockCandidate]            # 真正通过硬护栏，可进入正式候选
    watch: list[StockCandidate]           # 轻微风险，只能观察，不能最终推荐
    rejected: list[dict[str, Any]]        # 明确排除（含原因）
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": [c.to_dict() for c in self.kept],
            "watch": [c.to_dict() for c in self.watch],
            "rejected": self.rejected,
            "warnings": self.warnings,
        }


class QuantGuard:
    """量化护栏。"""

    def __init__(self, dl: DataLoader, cfg: dict[str, Any]) -> None:
        self.dl = dl
        self.cfg = cfg or {}
        self.exclude_st = self.cfg.get("exclude_st", True)
        self.exclude_new_days = self.cfg.get("exclude_new_days", 5)
        self.exclude_one_word = self.cfg.get("exclude_one_word_limit", False)

        vcfg = self.cfg.get("valuation", {})
        self.pe_max = vcfg.get("pe_max", 200)
        self.pe_min = vcfg.get("pe_min", 0)
        self.pb_max = vcfg.get("pb_max", 15)

        fcfg = self.cfg.get("financial_risk", {})
        self.exclude_loss = fcfg.get("exclude_loss", True)
        self.debt_max = fcfg.get("debt_ratio_max", 0.80)

    # ------------------------------------------------------------------ #
    def filter(self, candidates: list[StockCandidate], date: Optional[str] = None) -> GuardResult:
        res = GuardResult(kept=[], watch=[], rejected=[])
        # 注意：all_spot 只有实时快照，没有历史参数；历史回看时市值/PE 等字段为当天数据，
        # 收盘价/次新判定等已改用 daily_kline(date) 取历史。
        spot = self.dl.all_spot()
        spot_map: dict[str, pd.Series] = {}
        if len(spot) > 0:
            code_col = _find_col(spot, ["代码"]) or spot.columns[1]
            spot_map = {str(r[code_col]).strip(): r for _, r in spot.iterrows()}

        for c in candidates:
            check = self._check(c, spot_map.get(c.code), date=date)
            reason = None
            watch_reason = None
            if isinstance(check, tuple):
                reason, watch_reason = check
            else:
                reason = check

            if reason:
                # 硬剔除：明确风险
                c.risk_flags.append(reason)
                res.rejected.append({"code": c.code, "name": c.name, "reason": reason})
                continue
            if watch_reason:
                # 轻微风险：进入观察池，不能最终推荐
                c.risk_flags.append(watch_reason)
                res.watch.append(c)
                continue
            res.kept.append(c)

        logger.info("护栏：候选 %d → 通过 %d，观察 %d，硬剔除 %d",
                    len(candidates), len(res.kept), len(res.watch), len(res.rejected))
        return res

    # ------------------------------------------------------------------ #
    def _check(self, c: StockCandidate, spot_row: Optional[pd.Series], date: Optional[str] = None) -> Optional[str] | tuple[Optional[str], Optional[str]]:
        """返回剔除原因或 (剔除原因, 观察原因)。

        - 返回字符串：硬剔除原因。
        - 返回 (None, watch_reason)：轻微风险，进入 watch 池。
        - 返回 None：通过。
        """
        watch_reasons: list[str] = []

        # 1. ST / *ST / 退市：硬剔除
        if self.exclude_st:
            name = c.name.upper()
            if "ST" in name or "退" in c.name:
                return "ST/*ST/退市风险"

        # 2. 一字板（买不到）
        if self.exclude_one_word and spot_row is not None:
            open_v = safe_float(spot_row.get("今开"))
            close_v = safe_float(spot_row.get("最新价"))
            low_v = safe_float(spot_row.get("最低"))
            if close_v and close_v > 0 and open_v and low_v:
                if abs(open_v - close_v) / close_v < 0.001 and abs(close_v - low_v) / close_v < 0.001:
                    return "一字涨停（无法买入）"

        # 3. 估值过滤（PE/PB）
        if spot_row is not None:
            pe = safe_float(spot_row.get("市盈率-动态"))
            pb = safe_float(spot_row.get("市净率"))

            pe_missing = pd.isna(pe)
            pb_missing = pd.isna(pb)
            if pe_missing and pb_missing:
                watch_reasons.append("估值数据缺失")
            else:
                if pe_missing:
                    watch_reasons.append("PE数据缺失")
                elif not pd.isna(pe):
                    if pe < self.pe_min:
                        if self.exclude_loss:
                            return f"亏损或PE过低（PE={pe:.1f}）"
                    elif pe > self.pe_max:
                        return f"估值过高（PE={pe:.1f}）"

                if pb_missing:
                    watch_reasons.append("PB数据缺失")
                elif not pd.isna(pb):
                    if pb > self.pb_max:
                        return f"PB 过高（PB={pb:.1f}）"

        # 4. 次新（上市天数不足）
        if self.exclude_new_days > 0:
            k = self.dl.daily_kline(c.code, days=self.exclude_new_days + 5, date=date)
            if len(k) == 0:
                return "无足够历史数据"
            if len(k) < self.exclude_new_days:
                return f"次新股（上市不足 {self.exclude_new_days} 日）"

        # 5. 财务风险（慢，放最后；逐只取）
        if self.exclude_loss or self.debt_max < 1.0:
            fin_reason = self._check_financial(c.code)
            if fin_reason:
                return fin_reason

        if watch_reasons:
            return None, ";".join(watch_reasons)
        return None

    def _check_financial(self, code: str) -> Optional[str]:
        """财务排雷：最近报告期亏损 / 高负债。"""
        fin = self.dl.financial_indicator(code)
        if len(fin) == 0:
            return None  # 取不到不强拦（避免误杀）
        # akshare 列名：选项/日期/加权净资产收益率(%)/资产负债率(%) 等
        latest = fin.iloc[-1]
        debt_col = _find_col(fin, ["资产负债率"])
        if debt_col and self.debt_max < 1.0:
            debt = safe_float(latest.get(debt_col))
            if debt and debt / 100 > self.debt_max:
                return f"资产负债率过高（{debt:.0f}%）"

        if self.exclude_loss:
            # ROE 显著为负通常意味着亏损；更直接看净利润列
            profit_col = _find_col(fin, ["净利润", "归属母公司股东的净利润"])
            if profit_col:
                profit = safe_float(latest.get(profit_col))
                if profit and profit < 0:
                    return "最近报告期亏损"
        return None
