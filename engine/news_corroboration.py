"""Cross-check exact-time third-party news against official CNINFO disclosures."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from .announcement_archive import classify_announcement_title


_COMPATIBLE_OFFICIAL_EVENTS = {
    "earnings_positive": {"performance_growth"},
    "major_contract": {"major_contract", "project_win"},
    "product_breakthrough": {"product_breakthrough"},
    "restructuring": {"restructuring"},
}
_AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(亿美元|亿元|万美元|万元|元)")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_MULTIPLE_RE = re.compile(r"(?:增长|预增|增加|上升)?\s*(\d+(?:\.\d+)?)\s*倍")
_AMOUNT_MULTIPLIERS = {
    "元": 1.0,
    "万元": 10_000.0,
    "亿元": 100_000_000.0,
    "万美元": 70_000.0,
    "亿美元": 700_000_000.0,
}


@dataclass(frozen=True)
class OfficialCorroboration:
    confirmed: bool
    status: str
    reason: str
    signal_event_type: str
    supported_event_family: bool
    role_confirmed: bool = False
    causal_order_known: bool = False
    time_precision: str = "unknown"
    announcement_id: str = ""
    official_event_type: str = ""
    title: str = ""
    published_at: str = ""
    adjunct_url: str = ""
    source: str = "cninfo"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventIdentityAudit:
    matched: bool
    status: str
    reason: str
    news_amounts_yuan: tuple[float, ...] = ()
    official_amounts_yuan: tuple[float, ...] = ()
    news_percentages: tuple[float, ...] = ()
    official_growth_lower_pct: float | None = None
    official_growth_upper_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OfficialNewsCorroborator:
    """Require the same stock and a compatible official event family.

    CNINFO's historical query commonly exposes only calendar-date precision
    (timestamps at 00:00:00).  A same-day match therefore corroborates the
    claim and beneficiary role, but it must not be represented as proof of
    exact intraday ordering.  A prior-date match has known causal ordering.
    """

    def __init__(
        self,
        root: str | Path = "data/announcement_archive",
        *,
        lookback_calendar_days: int = 1,
    ) -> None:
        self.root = Path(root)
        self.lookback_calendar_days = max(0, int(lookback_calendar_days))
        self._cache: dict[str, dict[str, Any]] = {}

    def corroborate(
        self,
        *,
        code: str,
        signal_date: str,
        signal_event_type: str,
        signal_published_at: str = "",
    ) -> OfficialCorroboration:
        normalized_code = str(code).zfill(6)
        compatible = _COMPATIBLE_OFFICIAL_EVENTS.get(str(signal_event_type))
        if not compatible:
            return OfficialCorroboration(
                confirmed=False,
                status="unsupported_event_family",
                reason=(
                    "当前官方公告语料和正文分析器尚未覆盖该事件族，"
                    "第三方快讯不能单独作为正式推荐的一手证据"
                ),
                signal_event_type=str(signal_event_type),
                supported_event_family=False,
            )

        try:
            signal_day = datetime.strptime(str(signal_date), "%Y%m%d").date()
        except ValueError:
            return OfficialCorroboration(
                confirmed=False,
                status="invalid_signal_date",
                reason="信号日期不是 YYYYMMDD",
                signal_event_type=str(signal_event_type),
                supported_event_family=True,
            )

        signal_payload = self._payload(signal_day.strftime("%Y%m%d"))
        if not signal_payload or not signal_payload.get("complete"):
            return OfficialCorroboration(
                confirmed=False,
                status="official_archive_missing_or_incomplete",
                reason="信号日巨潮公告归档缺失或不完整",
                signal_event_type=str(signal_event_type),
                supported_event_family=True,
            )

        candidates: list[tuple[Any, Mapping[str, Any], str]] = []
        for offset in range(self.lookback_calendar_days + 1):
            date = signal_day - timedelta(days=offset)
            payload = self._payload(date.strftime("%Y%m%d"))
            if not payload or not payload.get("complete"):
                continue
            for event in payload.get("events") or []:
                if not isinstance(event, Mapping) or str(event.get("code") or "").zfill(6) != normalized_code:
                    continue
                polarity, official_event_type = classify_announcement_title(
                    str(event.get("title") or "")
                )
                if polarity != "positive" or official_event_type not in compatible:
                    continue
                official_date = self._official_date(event, date)
                if official_date > signal_day:
                    continue
                candidates.append((official_date, event, official_event_type))

        if not candidates:
            return OfficialCorroboration(
                confirmed=False,
                status="no_compatible_official_announcement",
                reason="信号日或前一自然日没有同代码、同事件族的巨潮正式公告",
                signal_event_type=str(signal_event_type),
                supported_event_family=True,
            )

        official_date, event, official_event_type = max(
            candidates,
            key=lambda item: (
                item[0],
                str(item[1].get("announcement_id") or ""),
            ),
        )
        published_at = str(event.get("published_at") or "")
        exact_time = self._has_intraday_precision(published_at)
        same_day = official_date == signal_day
        causal_order_known = bool(not same_day or exact_time)
        status = (
            "confirmed_previous_date"
            if not same_day
            else (
                "confirmed_same_day_exact_time"
                if exact_time
                else "confirmed_same_day_date_only"
            )
        )
        reason = (
            "巨潮正式公告早于快讯信号日，因果顺序已知"
            if not same_day
            else (
                "巨潮正式公告具有同日精确时间，可核验先后"
                if exact_time
                else (
                    "巨潮正式公告确认事实与受益主体；历史接口仅有日期精度，"
                    "不能声称知道同日精确先后"
                )
            )
        )
        return OfficialCorroboration(
            confirmed=True,
            status=status,
            reason=reason,
            signal_event_type=str(signal_event_type),
            supported_event_family=True,
            role_confirmed=True,
            causal_order_known=causal_order_known,
            time_precision="intraday" if exact_time else "date_only",
            announcement_id=str(event.get("announcement_id") or ""),
            official_event_type=official_event_type,
            title=str(event.get("title") or ""),
            published_at=published_at,
            adjunct_url=str(event.get("adjunct_url") or ""),
        )

    @staticmethod
    def audit_event_identity(
        *,
        news_evidence: str,
        signal_event_type: str,
        announcement_detail: Mapping[str, Any],
    ) -> EventIdentityAudit:
        evidence = str(news_evidence or "")
        if signal_event_type == "major_contract":
            news_amounts = OfficialNewsCorroborator._amounts_yuan(evidence)
            official_amounts = tuple(
                amount
                for value in (announcement_detail.get("amounts") or [])
                for amount in OfficialNewsCorroborator._amounts_yuan(str(value))
            )
            if not news_amounts:
                return EventIdentityAudit(
                    False,
                    "news_amount_missing",
                    "合同/中标快讯缺少可核对金额，无法证明与官方 PDF 是同一事件",
                    official_amounts_yuan=official_amounts,
                )
            if not official_amounts:
                return EventIdentityAudit(
                    False,
                    "official_amount_missing",
                    "官方 PDF 正文没有提取到可核对金额",
                    news_amounts_yuan=news_amounts,
                )
            matched = any(
                abs(news - official) / max(abs(news), abs(official), 1.0) <= 0.08
                for news in news_amounts
                for official in official_amounts
            )
            return EventIdentityAudit(
                matched,
                "amount_match" if matched else "amount_mismatch",
                (
                    "新闻金额与官方 PDF 金额在 8% 容差内一致"
                    if matched
                    else "新闻金额与官方 PDF 金额不一致，不是同一合同/中标事件"
                ),
                news_amounts_yuan=news_amounts,
                official_amounts_yuan=official_amounts,
            )

        if signal_event_type == "earnings_positive":
            news_percentages = tuple(
                [float(value) for value in _PERCENT_RE.findall(evidence)]
                + [float(value) * 100.0 for value in _MULTIPLE_RE.findall(evidence)]
                + ([100.0] if "翻一倍" in evidence else [])
            )
            lower = announcement_detail.get("growth_lower_pct")
            upper = announcement_detail.get("growth_upper_pct")
            lower_value = float(lower) if lower is not None else None
            upper_value = float(upper) if upper is not None else lower_value
            if not news_percentages:
                return EventIdentityAudit(
                    False,
                    "news_growth_missing",
                    "业绩快讯缺少可核对的同比百分比",
                    official_growth_lower_pct=lower_value,
                    official_growth_upper_pct=upper_value,
                )
            if lower_value is None or upper_value is None:
                return EventIdentityAudit(
                    False,
                    "official_growth_missing",
                    "官方 PDF 未提取到归母净利润同比区间",
                    news_percentages=news_percentages,
                )
            matched = any(
                lower_value - 5.0 <= value <= upper_value + 5.0
                for value in news_percentages
            )
            return EventIdentityAudit(
                matched,
                "growth_range_match" if matched else "growth_range_mismatch",
                (
                    "新闻增长百分比与官方归母净利润区间一致"
                    if matched
                    else "新闻增长百分比与官方归母净利润区间不一致"
                ),
                news_percentages=news_percentages,
                official_growth_lower_pct=lower_value,
                official_growth_upper_pct=upper_value,
            )

        if signal_event_type == "restructuring":
            news_categories = OfficialNewsCorroborator._restructuring_categories(evidence)
            official_text = " ".join([
                str(announcement_detail.get("title") or ""),
                " ".join(str(value) for value in (announcement_detail.get("evidence_snippets") or [])),
            ])
            official_categories = OfficialNewsCorroborator._restructuring_categories(
                official_text
            )
            matched = bool(news_categories & official_categories)
            return EventIdentityAudit(
                matched,
                "restructuring_category_match" if matched else "restructuring_category_mismatch",
                (
                    "新闻与官方 PDF 的重组子类型一致"
                    if matched else "新闻与官方 PDF 的重组子类型不一致"
                ),
            )

        if signal_event_type == "product_breakthrough":
            news_categories = OfficialNewsCorroborator._product_categories(evidence)
            official_title = str(announcement_detail.get("title") or "")
            official_text = " ".join([
                official_title,
                " ".join(str(value) for value in (announcement_detail.get("evidence_snippets") or [])),
            ])
            official_categories = OfficialNewsCorroborator._product_categories(official_text)
            category_match = bool(news_categories & official_categories)
            anchor_match = OfficialNewsCorroborator._distinctive_bigram_overlap(
                evidence, official_title
            ) >= 2
            matched = bool(category_match and anchor_match)
            return EventIdentityAudit(
                matched,
                (
                    "product_category_and_anchor_match"
                    if matched else (
                        "product_category_mismatch"
                        if not category_match else "product_anchor_mismatch"
                    )
                ),
                (
                    "产品事件子类型及产品名称锚点均一致"
                    if matched else (
                        "新闻与官方 PDF 的产品事件子类型不一致"
                        if not category_match else "产品事件类型一致，但产品名称/型号锚点不足"
                    )
                ),
            )

        return EventIdentityAudit(
            False,
            "unsupported_identity_rule",
            "该事件族尚无可审计的事件身份规则",
        )

    def _payload(self, date: str) -> dict[str, Any]:
        if date not in self._cache:
            try:
                payload = json.loads((self.root / f"{date}.json").read_text(encoding="utf-8"))
                self._cache[date] = payload if isinstance(payload, dict) else {}
            except (OSError, ValueError, TypeError):
                self._cache[date] = {}
        return self._cache[date]

    @staticmethod
    def _official_date(event: Mapping[str, Any], fallback: Any) -> Any:
        value = str(event.get("published_at") or "")[:10]
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return fallback

    @staticmethod
    def _has_intraday_precision(value: str) -> bool:
        text = str(value or "")
        if "T" not in text:
            return False
        time_text = text.split("T", 1)[1][:8]
        return bool(time_text and time_text != "00:00:00")

    @staticmethod
    def _amounts_yuan(text: str) -> tuple[float, ...]:
        output: list[float] = []
        for number, unit in _AMOUNT_RE.findall(str(text or "")):
            try:
                output.append(
                    float(number.replace(",", "")) * _AMOUNT_MULTIPLIERS[unit]
                )
            except (KeyError, ValueError):
                continue
        return tuple(output)

    @staticmethod
    def _restructuring_categories(text: str) -> set[str]:
        value = str(text or "")
        categories: set[str] = set()
        if "发行股份购买资产" in value:
            categories.add("share_issue_asset_purchase")
        if "资产注入" in value:
            categories.add("asset_injection")
        if "控制权变更" in value:
            categories.add("control_change")
        if "重大资产重组" in value:
            categories.add("major_restructuring")
        if "收购" in value:
            categories.add("acquisition")
        return categories

    @staticmethod
    def _product_categories(text: str) -> set[str]:
        value = str(text or "")
        categories: set[str] = set()
        if any(marker in value for marker in (
            "批准上市", "获批上市", "注册证", "注册批件",
        )):
            categories.add("regulatory_approval")
        if "量产" in value:
            categories.add("mass_production")
        if "技术突破" in value:
            categories.add("technical_breakthrough")
        return categories

    @staticmethod
    def _distinctive_bigram_overlap(left: str, right: str) -> int:
        generic = (
            "关于", "公告", "公司", "子公司", "全资", "控股", "获得", "取得",
            "批准上市", "获批上市", "注册证", "注册批件", "实现量产", "正式量产",
            "产品量产", "技术突破", "创新药", "药品", "产品", "医疗器械",
        )

        def bigrams(value: str) -> set[str]:
            normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or ""))
            for marker in generic:
                normalized = normalized.replace(marker, "")
            return {
                normalized[index:index + 2]
                for index in range(max(0, len(normalized) - 1))
            }

        return len(bigrams(left) & bigrams(right))
