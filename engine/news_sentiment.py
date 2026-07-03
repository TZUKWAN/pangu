"""新闻情绪打分：解析 capitalise-finnews 简报，输出个股/题材情绪得分。

设计：
- 读取 data/reports/ 下最新的 FinNews Markdown 简报。
  文件名匹配规则：包含 "finnews" / "FinNews" / 日期格式 YYYY-MM-DD.md。
- 分块解析「今日看点 / A股异动 / 领涨板块 / 概念科普 / 资金动向」等章节。
- 对每个题材/概念做关键词情感打分：
    positive：上涨、领涨、突破、利好、强劲、超预期、净流入...
    negative：下跌、领跌、调整、利空、疲软、不及预期、净流出...
- 从正文提取 6 位 A 股代码，建立 code -> 题材/情感 的映射。
- 输出：{code: {"sentiment_score": 0-100, "themes": [...], "risks": [...]}}
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("pangu.news_sentiment")

# 默认情感词表（可在 settings.yaml 覆盖）
_POSITIVE_WORDS = [
    "上涨", "领涨", "涨幅", "走强", "强劲", "强势", "反弹", "复苏",
    "突破", "创新高", "新高", "利好", "超预期", "增长", "增持",
    "净流入", "流入", "涨停", "封板", "放量", "攀升", "大涨",
    "上行", "多头", "景气", "繁荣", "乐观", "机会", "看好",
]

_NEGATIVE_WORDS = [
    "下跌", "领跌", "跌幅", "走弱", "疲软", "弱势", "调整", "回调",
    "破位", "创新低", "新低", "利空", "不及预期", "下降", "减持",
    "净流出", "流出", "跌停", "开板", "炸板", "缩量", "暴跌",
    "下行", "空头", "衰退", "悲观", "风险", "回避", "谨慎",
]

_RISK_WORDS = [
    "风险", "亏损", "暴雷", "退市", "警示", "监管", "立案调查",
    "业绩变脸", "不及预期", "减持", "净流出", "回避", "谨慎",
]

# 行业/概念关键词，用于从个股新闻中提取题材（避免把情感词当题材）
_INDUSTRY_KEYWORDS = {
    "AI", "人工智能", "算力", "芯片", "半导体", "光模块", "CPO", "服务器",
    "机器人", "人形机器人", "工业母机", "华为", "消费电子", "MR", "VR",
    "数据要素", "信创", "国产替代", "量子", "脑机接口",
    "新能源", "光伏", "储能", "锂电", "固态电池", "充电桩", "风电", "氢能",
    "核电", "稀土", "特高压", "医药", "创新药", "中药", "医疗器械", "CRO",
    "疫苗", "合成生物", "券商", "银行", "保险", "地产", "建材", "钢铁",
    "有色", "煤炭", "化工", "石油", "黄金", "铜", "白酒", "食品", "零售",
    "旅游", "传媒", "游戏", "教育", "纺织", "军工", "航天", "航空",
    "低空经济", "商业航天", "卫星", "国企改革", "重组", "并购", "回购",
    "增持", "一带一路", "碳中和", "猴痘", "重组蛋白",
}

# 6 位 A 股代码正则（仅匹配 0/3/6 开头；排除小数/大数字中的片段）
_CODE_RE = re.compile(r"(?<![\d.])([036]\d{5})(?![\d.])")
# Markdown 标题
_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_SUBSECTION_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)


@dataclass
class ThemeSentiment:
    """单个题材/概念的情绪解析结果。"""

    name: str
    label: str          # positive / neutral / negative
    raw_score: float    # [-1, 1]，正数偏正面
    score: float        # [0, 100]，方便与推荐分同尺度加权
    snippets: list[str] = field(default_factory=list)


class NewsSentimentScorer:
    """FinNews 简报情绪打分器。"""

    def __init__(
        self,
        cfg: Optional[dict[str, Any]] = None,
        report_dir: str = "data/reports",
    ) -> None:
        """初始化。

        Args:
            cfg: 配置字典，可覆盖 positive_words / negative_words / risk_words / sentiment_window。
            report_dir: 简报存放目录。
        """
        self.cfg = cfg or {}
        self.report_dir = Path(report_dir)
        self.positive = set(self.cfg.get("positive_words", _POSITIVE_WORDS))
        self.negative = set(self.cfg.get("negative_words", _NEGATIVE_WORDS))
        self.risk = set(self.cfg.get("risk_words", _RISK_WORDS))
        # 代码上下文窗口（字符数），用于判断代码归属哪个题材
        self.window = int(self.cfg.get("sentiment_window", 300))

    # ------------------------------------------------------------------ #
    def score(self) -> dict[str, dict[str, Any]]:
        """解析最新简报并返回个股情绪得分映射。

        Returns:
            {code: {"sentiment_score": float, "themes": [str], "risks": [str]}}
        """
        report_path = self._find_latest_report()
        if not report_path:
            logger.warning("未找到 FinNews 简报，跳过新闻情绪打分")
            return {}
        return self.parse_report(report_path)

    def parse_report(self, path: str | Path) -> dict[str, dict[str, Any]]:
        """解析指定 Markdown 简报，返回个股情绪映射。"""
        text = Path(path).read_text(encoding="utf-8")
        return self.score_from_text(text, source=str(path))

    def score_from_text(
        self,
        text: str,
        source: str = "realtime",
        stock_news: Optional[dict[str, list[dict[str, Any]]]] = None,
    ) -> dict[str, dict[str, Any]]:
        """直接从文本解析情绪，不依赖文件（供 news_fetcher 实时新闻用）。

        复用 parse_report 的打分逻辑，但接收实时拼接的新闻文本而非读盘。
        text 应是 Markdown 格式（## 标题分题材），对齐 NewsFetcher.to_markdown() 输出。

        Args:
            text: 新闻 Markdown 文本（快讯 + 题材热度）。
            source: 来源标识，仅用于日志。
            stock_news: {code: [{title, content, source, time}, ...]} 个股关联新闻。
                若提供，会按 code 对个股新闻标题/摘要做独立情感打分，并与快讯情绪融合。
        """
        themes = self._extract_themes(text)
        code_themes, code_snippets = self._map_codes_to_themes(text, themes)
        risks = self._extract_risks(text)

        result: dict[str, dict[str, Any]] = {}
        for code, theme_names in code_themes.items():
            theme_objs = [t for t in themes if t.name in theme_names]
            if not theme_objs:
                continue
            avg_score = sum(t.score for t in theme_objs) / len(theme_objs)
            avg_raw = sum(t.raw_score for t in theme_objs) / len(theme_objs)
            label = self._label_from_raw(avg_raw)
            snippet_text = " ".join(code_snippets.get(code, []))
            pos, neg = self._count_sentiment(snippet_text)
            result[code] = {
                "sentiment_score": round(avg_score, 1),
                "sentiment_label": label,
                "themes": sorted({t.name for t in theme_objs}),
                "risks": [
                    r for r in risks
                    if code in r or any(t.name in r for t in theme_objs)
                ][:5],
                "snippets": code_snippets.get(code, [])[:3],
                "sources": ["flash"],
                "explain": self._explain(label, avg_score, pos, neg, ["flash"], []),
            }

        # 个股新闻 per-code 情感打分，并与快讯结果融合
        if stock_news:
            for code, items in stock_news.items():
                stock = self._score_stock_news(code, items)
                if stock is None:
                    continue
                existing = result.get(code)
                if existing:
                    # 快讯与个股新闻融合：等权平均
                    merged_score = (existing["sentiment_score"] + stock["sentiment_score"]) / 2
                    merged_raw = ((existing["sentiment_score"] / 100.0 * 2) - 1 +
                                  (stock["sentiment_score"] / 100.0 * 2) - 1) / 2
                    merged_label = self._label_from_raw(merged_raw)
                    existing["sentiment_score"] = round(merged_score, 1)
                    existing["sentiment_label"] = merged_label
                    existing["themes"] = sorted(set(existing.get("themes", [])) | set(stock.get("themes", [])))
                    existing["risks"] = sorted(set(existing.get("risks", [])) | set(stock.get("risks", [])))[:5]
                    existing["snippets"] = (existing.get("snippets", []) + stock.get("snippets", []))[:3]
                    existing["sources"] = sorted(set(existing.get("sources", [])) | {"stock_news"})
                    # 合并 explain
                    existing["explain"] = self._merge_explain(
                        existing.get("explain", ""),
                        stock.get("explain", ""),
                        merged_label, merged_score,
                    )
                else:
                    stock["sources"] = ["stock_news"]
                    result[code] = stock

        logger.info("新闻情绪解析完成：%s 共 %d 只相关个股", source, len(result))
        return result

    # ------------------------------------------------------------------ #
    def _find_latest_report(self) -> Optional[Path]:
        """在 report_dir 中寻找最新的 FinNews 风格 Markdown 简报。

        排序优先级：文件名中的日期 > 文件修改时间，确保测试结果稳定且符合直觉。
        """
        if not self.report_dir.exists():
            return None
        candidates = []
        for p in self.report_dir.glob("*.md"):
            name = p.name.lower()
            # 匹配 FinNews / YYYY-MM-DD / YYYYMMDD / pangu 简报
            if (
                "finnews" in name
                or re.search(r"\d{4}-\d{2}-\d{2}", name)
                or re.search(r"\d{8}", name)
            ):
                m = re.search(r"(\d{4})-(\d{2})-(\d{2})", p.name)
                if not m:
                    m = re.search(r"(\d{4})(\d{2})(\d{2})", p.name)
                date_key = m.group(0) if m else ""
                candidates.append((date_key, p.stat().st_mtime, p))
        if not candidates:
            return None
        # 日期降序，同日期按修改时间降序
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    def _extract_themes(self, text: str) -> list[ThemeSentiment]:
        """把 Markdown 按 ## / ### 标题切分成题材，并打情感标签。"""
        themes: list[ThemeSentiment] = []
        # 先按一级标题拆分
        sections = _SECTION_RE.split(text)
        # sections[0] 是文前内容，后面成对出现 (标题, 内容)
        for i in range(1, len(sections), 2):
            sec_title = sections[i].strip()
            sec_body = sections[i + 1] if i + 1 < len(sections) else ""
            # 如果该节还有子标题，再细分；子标题内若含板块/题材表格，也逐行打分为子题材
            subs = _SUBSECTION_RE.split(sec_body)
            if len(subs) >= 3:
                for j in range(1, len(subs), 2):
                    sub_title = subs[j].strip()
                    sub_body = subs[j + 1] if j + 1 < len(subs) else ""
                    rows = self._extract_table_rows(sub_body)
                    if rows and ("板块" in sub_title or "题材" in sub_title):
                        for row in rows[:10]:
                            themes.append(self._score_theme(f"{sec_title}>{sub_title}>{row}", row))
                    else:
                        themes.append(self._score_theme(f"{sec_title}>{sub_title}", sub_body))
            else:
                # 无子标题时，把表格行也视作子题材（如领涨板块每行）
                rows = self._extract_table_rows(sec_body)
                if rows and ("板块" in sec_title or "题材" in sec_title):
                    for row in rows[:10]:
                        themes.append(self._score_theme(f"{sec_title}>{row}", row))
                else:
                    themes.append(self._score_theme(sec_title, sec_body))
        # 去重并过滤空标题
        seen = set()
        unique = []
        for t in themes:
            if t.name and t.name not in seen:
                seen.add(t.name)
                unique.append(t)
        return unique

    def _extract_table_rows(self, body: str) -> list[str]:
        """提取 Markdown 表格的每一行文本（用于板块/题材列表）。"""
        rows = []
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("|") and "---" in line:
                continue
            if line.startswith("|"):
                cells = [c.strip() for c in line.split("|") if c.strip()]
                rows.append(" ".join(cells))
        return rows

    def _score_theme(self, name: str, body: str) -> ThemeSentiment:
        """对单个题材文本打分。"""
        text = f"{name} {body}"
        pos = sum(1 for w in self.positive if w in text)
        neg = sum(1 for w in self.negative if w in text)
        total = pos + neg
        if total == 0:
            raw = 0.0
        else:
            raw = (pos - neg) / total
        label = self._label_from_raw(raw)
        score = (raw + 1.0) / 2.0 * 100.0
        # 摘要：取前 3 句非空文本
        snippets = [s.strip() for s in re.split(r"[。\n]", body) if s.strip()][:3]
        return ThemeSentiment(
            name=name, label=label, raw_score=raw,
            score=score, snippets=snippets,
        )

    def _label_from_raw(self, raw: float) -> str:
        """把原始情感分映射为标签。"""
        if raw > 0.15:
            return "positive"
        if raw < -0.15:
            return "negative"
        return "neutral"

    def _count_sentiment(self, text: str) -> tuple[int, int]:
        """统计文本中正向/负向情感词命中次数。"""
        pos = sum(1 for w in self.positive if w in text)
        neg = sum(1 for w in self.negative if w in text)
        return pos, neg

    def _explain(
        self,
        label: str,
        score: float,
        pos: int,
        neg: int,
        sources: list[str],
        themes: list[str],
    ) -> str:
        """生成人类可读的情绪说明。"""
        label_cn = {"positive": "正面", "negative": "负面", "neutral": "中性"}.get(label, "中性")
        src = "/".join(sources) if sources else "未知来源"
        total = pos + neg
        if total == 0:
            reason = "未命中任何情感词，按中性处理"
        elif label == "positive":
            reason = f"正向词 {pos} 个 > 负向词 {neg} 个"
        elif label == "negative":
            reason = f"负向词 {neg} 个 > 正向词 {pos} 个"
        else:
            reason = f"正/负向词接近（{pos}/{neg}），多空交织"
        theme_part = f"；题材：{', '.join(themes[:5])}" if themes else ""
        return f"来源：{src}；{reason}；情绪：{label_cn}，得分 {round(score, 1)}{theme_part}"

    def _merge_explain(
        self,
        flash_explain: str,
        stock_explain: str,
        label: str,
        score: float,
    ) -> str:
        """合并快讯与个股新闻的 explain。"""
        label_cn = {"positive": "正面", "negative": "负面", "neutral": "中性"}.get(label, "中性")
        parts = [f"融合后情绪：{label_cn}，得分 {round(score, 1)}"]
        if flash_explain:
            parts.append(f"[快讯] {flash_explain}")
        if stock_explain:
            parts.append(f"[个股新闻] {stock_explain}")
        return "；".join(parts)

    @staticmethod
    def _extract_codes(text: str) -> list[str]:
        """从文本中提取 6 位 A 股代码列表。"""
        return _CODE_RE.findall(text)

    def _map_codes_to_themes(
        self,
        text: str,
        themes: list[ThemeSentiment],
    ) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
        """根据代码出现位置，把它归属到最近的题材。"""
        code_themes: dict[str, set[str]] = {}
        code_snippets: dict[str, list[str]] = {}

        # 简单策略：对每个题材，在该题材文本区域（标题到下一个同级/上级标题）内搜索代码
        theme_regions = self._theme_regions(text, themes)
        for region_name, region_text in theme_regions.items():
            for code in set(_CODE_RE.findall(region_text)):
                code_themes.setdefault(code, set()).add(region_name)
                # 提取代码上下文片段
                for m in _CODE_RE.finditer(region_text):
                    if m.group() != code:
                        continue
                    start = max(m.start() - self.window // 2, 0)
                    end = min(m.end() + self.window // 2, len(region_text))
                    snippet = region_text[start:end].replace("\n", " ").strip()
                    code_snippets.setdefault(code, []).append(snippet)
        return code_themes, code_snippets

    def _theme_regions(self, text: str, themes: list[ThemeSentiment]) -> dict[str, str]:
        """返回每个题材名对应的文本区域（从该题材标题到下一标题）。"""
        regions: dict[str, str] = {}
        # 按标题在原文中出现的顺序切分
        positions = []
        for t in themes:
            # 题材名可能包含 >（嵌套章节/表格行），逐段尝试匹配标题
            candidates = {t.name}
            candidates.update(t.name.split(">"))
            for title in (c.strip() for c in candidates if c.strip()):
                idx = text.find(f"## {title}")
                if idx == -1:
                    idx = text.find(f"### {title}")
                if idx != -1:
                    positions.append((idx, t.name))
                    break
        positions.sort(key=lambda x: x[0])
        for i, (pos, name) in enumerate(positions):
            nxt = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            regions[name] = text[pos:nxt]
        # 兜底：如果按标题没找到区域，用全文
        if not regions and themes:
            for t in themes:
                regions[t.name] = text
        return regions

    def _score_stock_news(
        self,
        code: str,
        items: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """对单只个股的关联新闻标题/摘要做情感打分。

        返回与 score_from_text 兼容的字典，供后续融合。
        """
        if not items:
            return None
        texts: list[str] = []
        snippets: list[str] = []
        sources: set[str] = set()
        for item in items:
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            src = str(item.get("source", "")).strip() or "个股新闻"
            if title:
                texts.append(title)
                snippets.append(title[:80])
                sources.add(src)
            if content:
                texts.append(content[:300])
                sources.add(src)
        if not texts:
            return None
        full_text = "\n".join(texts)

        pos, neg = self._count_sentiment(full_text)
        total = pos + neg
        if total == 0:
            raw = 0.0
        else:
            raw = (pos - neg) / total
        score = (raw + 1.0) / 2.0 * 100.0
        label = self._label_from_raw(raw)

        # 题材：仅提取行业/概念关键词，避免把情感词本身当题材
        themes = set()
        for keyword in _INDUSTRY_KEYWORDS:
            if keyword in full_text:
                themes.add(keyword)

        # 风险：提取含风险词的句子
        risks = []
        for sentence in re.split(r"[。\n]", full_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            if any(w in sentence for w in self.risk):
                risks.append(sentence[:120])

        return {
            "sentiment_score": round(score, 1),
            "sentiment_label": label,
            "themes": sorted(themes)[:10],
            "risks": risks[:5],
            "snippets": snippets[:3],
            "sources": sorted(sources),
            "explain": self._explain(label, score, pos, neg, sorted(sources), sorted(themes)),
        }

    def _extract_risks(self, text: str) -> list[str]:
        """提取含风险关键词的句子/片段。"""
        risks = []
        for sentence in re.split(r"[。\n]", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            if any(w in sentence for w in self.risk):
                risks.append(sentence)
        # 去重并限制数量
        seen = set()
        out = []
        for r in risks:
            if r not in seen and len(out) < 10:
                seen.add(r)
                out.append(r)
        return out
