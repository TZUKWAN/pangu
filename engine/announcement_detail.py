"""Extract and score the economic substance of official announcement PDFs."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import requests
from pypdf import PdfReader


_YOY_RANGE_RE = re.compile(
    r"(?:同比|比上年同期)[^。；%]{0,30}?(?:增加|增长|上升)?[^。；%]{0,12}?"
    r"(\d+(?:\.\d+)?)\s*%\s*(?:到|至|[-~—])\s*(\d+(?:\.\d+)?)\s*%"
)
_YOY_SINGLE_RE = re.compile(
    r"(?:同比|比上年同期)[^。；%]{0,30}?(?:增加|增长|上升)[^。；%]{0,12}?(\d+(?:\.\d+)?)\s*%"
)
_AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(亿美元|亿元|万元|万美元|元)")
_ATTRIBUTABLE_PROFIT_ANCHORS = (
    "归属于上市公司股东的净利润",
    "归属于母公司所有者的净利润",
    "归属于母公司股东的净利润",
    "归母净利润",
)


@dataclass
class AnnouncementDetail:
    announcement_id: str
    code: str
    event_type: str
    title: str
    text_chars: int
    catalyst_strength: float
    tradable_catalyst: bool
    contract_direction: str = "unknown"
    event_stage: str = "unknown"
    capital_purpose: str = "unknown"
    growth_lower_pct: float | None = None
    growth_upper_pct: float | None = None
    amounts: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    extraction_status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "announcement_id": self.announcement_id,
            "code": self.code,
            "event_type": self.event_type,
            "title": self.title,
            "text_chars": self.text_chars,
            "catalyst_strength": round(self.catalyst_strength, 2),
            "tradable_catalyst": self.tradable_catalyst,
            "contract_direction": self.contract_direction,
            "event_stage": self.event_stage,
            "capital_purpose": self.capital_purpose,
            "growth_lower_pct": self.growth_lower_pct,
            "growth_upper_pct": self.growth_upper_pct,
            "amounts": self.amounts,
            "evidence_snippets": self.evidence_snippets,
            "risk_flags": self.risk_flags,
            "reasons": self.reasons,
            "extraction_status": self.extraction_status,
        }


class AnnouncementDetailAnalyzer:
    ALLOWED_HOSTS = {"static.cninfo.com.cn", "www.cninfo.com.cn"}

    def __init__(
        self,
        cache_dir: str | Path = "data/announcement_pdf",
        *,
        session: requests.Session | None = None,
        timeout: float = 25.0,
        max_bytes: int = 15 * 1024 * 1024,
        max_pages: int = 20,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_pages = max_pages

    def analyze(self, event: Mapping[str, Any]) -> AnnouncementDetail:
        announcement_id = str(event.get("announcement_id") or "")
        code = str(event.get("code") or "").zfill(6)
        event_type = str(event.get("event_type") or "other")
        title = str(event.get("title") or "")
        url = str(event.get("adjunct_url") or "")
        try:
            pdf_bytes = self._download(announcement_id, url)
            text = self._extract_text(pdf_bytes)
            if len(text.strip()) < 80:
                return AnnouncementDetail(
                    announcement_id, code, event_type, title, len(text), 0.0, False,
                    extraction_status="text_too_short",
                    risk_flags=["PDF正文无法可靠提取，禁止作为强催化"],
                )
            return self.analyze_text(event, text)
        except Exception as exc:  # noqa: BLE001
            return AnnouncementDetail(
                announcement_id, code, event_type, title, 0, 0.0, False,
                extraction_status="failed",
                risk_flags=[f"PDF提取失败: {exc}"],
            )

    def analyze_text(self, event: Mapping[str, Any], text: str) -> AnnouncementDetail:
        normalized = re.sub(r"\s+", "", str(text))
        announcement_id = str(event.get("announcement_id") or "")
        code = str(event.get("code") or "").zfill(6)
        event_type = str(event.get("event_type") or "other")
        title = str(event.get("title") or "")
        name = str(event.get("name") or "")
        risk_flags = [
            label for keyword, label in (
                ("重大不确定性", "存在重大不确定性"),
                ("尚需提交", "尚需审批"),
                ("可能因", "存在条件性风险"),
                ("不能保证", "无法保证履行"),
                ("注意投资风险", "公告提示投资风险"),
            )
            if keyword in normalized
        ]
        amounts = list(dict.fromkeys(match.group(0) for match in _AMOUNT_RE.finditer(normalized)))[:8]
        strength = 35.0
        tradable = False
        reasons: list[str] = []
        direction = "unknown"
        event_stage = "unknown"
        capital_purpose = "unknown"
        growth_lower: float | None = None
        growth_upper: float | None = None
        evidence_snippets: list[str] = []

        if event_type == "performance_growth":
            values, snippet = self._extract_attributable_growth(normalized)
            if snippet:
                evidence_snippets.append(snippet)
            if values:
                growth_lower, growth_upper = min(values), max(values)
                strength = min(95.0, 45.0 + min(growth_lower, 150.0) * 0.30)
                reasons.append(f"公告正文可提取业绩增长下限 {growth_lower:.1f}%")
                tradable = growth_lower >= 50.0
            else:
                strength = 30.0
                reasons.append("未从正文提取到可验证的业绩增长下限")
            if "扭亏为盈" in normalized:
                strength = max(strength, 75.0)
                tradable = bool(values)
                reasons.append("正文明确扭亏为盈；仍要求归母净利润同比幅度可核验")

        elif event_type in {"major_contract", "project_win"}:
            purchase_markers = ("采购合同", "向该供应商采购", "承诺采购", "采购产品")
            sales_markers = ("销售合同", "向客户销售", "向客户提供", "中标通知书", "收到中标")
            if any(marker in normalized for marker in purchase_markers):
                direction = "purchase"
                strength = 20.0
                tradable = False
                reasons.append("正文显示为采购/成本锁定合同，不是新增销售收入")
                evidence_snippets.append(self._snippet(normalized, "采购", 220))
            elif any(marker in normalized for marker in sales_markers):
                direction = "sales_or_project_revenue"
                strength = 75.0 if amounts else 65.0
                tradable = True
                reasons.append("正文显示为销售或项目收入类合同")
                marker = next(marker for marker in sales_markers if marker in normalized)
                evidence_snippets.append(self._snippet(normalized, marker, 220))
            else:
                strength = 45.0
                reasons.append("合同方向无法从正文可靠确认")
            if "占公司最近一个会计年度经审计营业收入" in normalized:
                strength += 5.0
                reasons.append("正文披露合同与年度营业收入的相对关系")

        elif event_type == "product_breakthrough":
            renewal_or_change = any(marker in title for marker in (
                "变更", "延续", "续展", "换发", "重新注册",
            ))
            approval_markers = (
                "获得批准上市", "获批上市", "取得注册证", "获得注册证",
                "取得注册批件", "获得注册批件",
            )
            mass_production_markers = ("实现量产", "正式量产", "产品量产")
            if renewal_or_change:
                event_stage = "renewal_or_change"
                strength = 25.0
                reasons.append("注册证变更/延续/换发不是新增产品催化")
            elif any(marker in title or marker in normalized for marker in approval_markers):
                event_stage = "regulatory_approval"
                strength = 72.0
                tradable = True
                reasons.append("官方正文确认新增药品/器械获得监管批准或注册证")
                marker = next(
                    marker for marker in approval_markers
                    if marker in title or marker in normalized
                )
                evidence_snippets.append(self._snippet(normalized, marker, 260))
                if any(marker in normalized for marker in (
                    "尚未实现商业化", "短期内不会对经营业绩产生重大影响",
                    "近期经营业绩不会产生重大影响",
                )):
                    strength = 55.0
                    tradable = False
                    risk_flags.append("公告明确短期商业化/业绩贡献有限")
            elif any(marker in title or marker in normalized for marker in mass_production_markers):
                event_stage = "mass_production"
                commercial = bool(amounts or any(marker in normalized for marker in (
                    "销售订单", "客户订单", "营业收入", "批量交付",
                )))
                strength = 68.0 if commercial else 50.0
                tradable = commercial
                reasons.append(
                    "正式量产且正文存在商业化证据"
                    if commercial else "仅确认量产，未见订单/收入等商业化证据"
                )
                marker = next(
                    marker for marker in mass_production_markers
                    if marker in title or marker in normalized
                )
                evidence_snippets.append(self._snippet(normalized, marker, 260))
            else:
                event_stage = "unverified_claim"
                strength = 35.0
                reasons.append("技术突破缺少监管批准、量产或商业化正文证据")

        elif event_type == "restructuring":
            terminated = any(marker in title for marker in ("终止", "取消"))
            completed = any(marker in title for marker in ("完成", "实施完毕", "过户完成"))
            progress = "进展" in title
            planning = any(marker in title for marker in ("筹划", "提示性公告"))
            formal = any(marker in title for marker in (
                "预案", "草案", "报告书", "发行股份购买资产", "重大资产重组方案",
            ))
            if terminated:
                event_stage = "terminated"
                strength = 0.0
                reasons.append("重组已经终止或取消")
            elif completed:
                event_stage = "completed"
                strength = 25.0
                reasons.append("重组已经完成，不是新增短期催化")
            elif progress and not formal:
                event_stage = "progress"
                strength = 30.0
                reasons.append("仅为既有重组进展公告")
            elif planning and not formal:
                event_stage = "planning"
                strength = 45.0
                reasons.append("仅处于筹划阶段，方案与审批存在较大不确定性")
            elif formal:
                event_stage = "formal_plan"
                strength = 72.0
                tradable = True
                reasons.append("官方披露正式重组预案/草案/报告书")
                marker = next((marker for marker in (
                    "发行股份购买资产", "重大资产重组", "本次交易",
                ) if marker in normalized), "")
                if marker:
                    evidence_snippets.append(self._snippet(normalized, marker, 280))
            else:
                event_stage = "unknown"
                strength = 35.0
                reasons.append("无法确认是新增正式重组方案")

        elif event_type == "shareholder_increase":
            proposed = any(marker in normalized for marker in ("增持计划", "拟增持", "计划增持"))
            completed = any(marker in normalized for marker in ("增持完成", "实施完毕"))
            if proposed and not completed:
                strength = 70.0 if amounts else 60.0
                tradable = True
                reasons.append("股东增持计划尚处执行期")
            else:
                strength = 35.0
                reasons.append("增持已完成或计划阶段无法确认")

        elif event_type == "share_buyback":
            title_completed = "回购" in title and any(marker in title for marker in (
                "完成", "实施完毕", "期限届满", "结果暨股份变动",
            ))
            title_progress = "回购" in title and any(marker in title for marker in (
                "进展", "首次回购", "比例达到", "回购股份达到",
            ))
            title_formal = "回购" in title and any(marker in title for marker in (
                "方案", "报告书",
            ))
            body_completed = any(marker in normalized for marker in (
                "回购实施完成", "回购完成", "实施完毕", "回购期限届满",
            ))
            body_progress = any(marker in normalized[:500] for marker in (
                "首次回购公司股份", "回购进展情况",
            ))
            body_formal = any(marker in normalized[:800] for marker in (
                "回购股份方案", "回购股份报告书",
            ))
            proposal_only = "提议回购" in title or "提议公司回购" in normalized[:800]
            cancellation = any(marker in normalized for marker in (
                "用于注销", "减少注册资本", "依法注销",
            ))
            if title_completed:
                event_stage = "completed"
                strength = 20.0
                reasons.append("回购已经完成或期限届满，不是新增催化")
            elif title_progress:
                event_stage = "execution_progress"
                strength = 30.0
                reasons.append("首次回购/比例进展属于既有方案执行，不重复作为新增催化")
            elif proposal_only and not title_formal:
                event_stage = "informal_proposal"
                strength = 45.0
                reasons.append("仅有股东或管理层提议，尚非正式回购方案")
            elif title_formal:
                event_stage = "formal_plan"
                capital_purpose = "cancellation" if cancellation else "other"
                strength = 72.0 if amounts else 55.0
                if cancellation:
                    strength += 8.0
                    reasons.append("正式回购方案且股份拟注销，减少流通/注册资本")
                else:
                    reasons.append("正式回购方案或回购报告书")
                tradable = bool(amounts)
                evidence_snippets.append(self._snippet(normalized, "回购", 260))
            elif body_completed:
                event_stage = "completed"
                strength = 20.0
                reasons.append("正文显示回购已经完成或期限届满，不是新增催化")
            elif body_progress:
                event_stage = "execution_progress"
                strength = 30.0
                reasons.append("正文显示为既有回购方案执行进展")
            elif body_formal:
                event_stage = "formal_plan"
                capital_purpose = "cancellation" if cancellation else "other"
                strength = 72.0 if amounts else 55.0
                if cancellation:
                    strength += 8.0
                reasons.append("正文可确认正式回购方案")
                tradable = bool(amounts)
                evidence_snippets.append(self._snippet(normalized, "回购", 260))
            else:
                strength = 30.0
                reasons.append("无法确认是新增正式回购方案")

        if name.upper().startswith("*ST") or name.upper().startswith("ST") or "退市风险" in normalized:
            tradable = False
            strength = min(strength, 20.0)
            risk_flags.append("ST/退市风险股票禁止进入正式推荐")
        if "不构成重大资产重组" in normalized:
            risk_flags.append("不构成重大资产重组，不得按重组催化理解")
        if len(risk_flags) >= 3:
            strength = max(0.0, strength - 15.0)
        tradable = bool(tradable and strength >= 60.0)
        return AnnouncementDetail(
            announcement_id=announcement_id,
            code=code,
            event_type=event_type,
            title=title,
            text_chars=len(normalized),
            catalyst_strength=strength,
            tradable_catalyst=tradable,
            contract_direction=direction,
            event_stage=event_stage,
            capital_purpose=capital_purpose,
            growth_lower_pct=growth_lower,
            growth_upper_pct=growth_upper,
            amounts=amounts,
            evidence_snippets=evidence_snippets,
            risk_flags=list(dict.fromkeys(risk_flags)),
            reasons=reasons,
        )

    @staticmethod
    def _extract_attributable_growth(text: str) -> tuple[list[float], str]:
        """Return YoY growth tied specifically to attributable net profit.

        Exchange templates often contain a generic "up more than 50%" listing
        rule before the company's actual estimate.  Searching the whole PDF
        therefore creates a false 50% lower bound.  This parser only considers
        the primary attributable-net-profit row and stops before the deducted
        non-recurring-profit row.
        """
        matches: list[tuple[int, str]] = []
        for anchor in _ATTRIBUTABLE_PROFIT_ANCHORS:
            matches.extend((match.start(), anchor) for match in re.finditer(anchor, text))
        for start, anchor in sorted(matches):
            segment = text[start:start + 520]
            deduct_at = segment.find("扣除非经常")
            if 0 <= deduct_at < 45:
                continue
            if deduct_at >= 45:
                segment = segment[:deduct_at]
            range_matches = list(_YOY_RANGE_RE.finditer(segment))
            if range_matches:
                values = [
                    value
                    for match in range_matches
                    for value in (float(match.group(1)), float(match.group(2)))
                ]
                return values, segment[:300]
            singles = [float(match.group(1)) for match in _YOY_SINGLE_RE.finditer(segment)]
            if singles:
                return singles, segment[:300]
        return [], ""

    @staticmethod
    def _snippet(text: str, marker: str, width: int = 220) -> str:
        index = text.find(marker)
        if index < 0:
            return ""
        start = max(0, index - width // 3)
        return text[start:start + width]

    def _download(self, announcement_id: str, url: str) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in self.ALLOWED_HOSTS:
            raise ValueError("公告PDF URL不在巨潮官方允许域名内")
        safe_id = announcement_id if announcement_id.isdigit() else hashlib.sha256(url.encode()).hexdigest()[:20]
        path = self.cache_dir / f"{safe_id}.pdf"
        if path.exists():
            data = path.read_bytes()
            if data.startswith(b"%PDF"):
                return data
        response = self.session.get(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = bytes(response.content)
        if len(data) > self.max_bytes:
            raise ValueError(f"公告PDF超过大小上限 {self.max_bytes}")
        if not data.startswith(b"%PDF"):
            raise ValueError("巨潮附件不是有效PDF")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_bytes(data)
        temp.replace(path)
        return data

    def _extract_text(self, pdf_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts: list[str] = []
        for page in reader.pages[: self.max_pages]:
            try:
                contents = page.get_contents()
                if contents is not None and len(contents.get_data()) > 25 * 1024 * 1024:
                    raise ValueError("单页PDF内容流过大，停止提取")
                parts.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001
                parts.append(f"[page_extract_error:{exc}]")
        return "\n".join(parts)
