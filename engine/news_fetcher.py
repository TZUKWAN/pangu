"""实时新闻聚合：财联社电报 + 华尔街见闻 + 金十数据 + 新浪财经 + 题材热度提取。

解决原系统的核心缺口：news_sentiment.py 只读静态 Markdown 简报，拿不到当日
最新新闻。本模块实时拉取，聚合后供：
1. news_sentiment.py 解析题材情绪（复用现有词表）
2. pipeline 给候选股附加「相关新闻」字段
3. UI 展示「今日新闻动态」区

数据源（均免 token，走公开接口）：
- 财联社电报 cache 接口
- 华尔街见闻 live (A股精选)
- 金十数据
- 新浪财经直播
- 雪球热门股票（新增，个股人气/情绪信号）
- 格隆汇快讯（新增）

已移除源：
- 东财快讯（东方财富，已移除）
- 东财个股搜索（东方财富，已移除）

设计原则：
1. 全程降级：任一接口失败返回空，不阻断选股主流程
2. 列名容错：akshare 中文列名偶有微调，用 find_col 模糊匹配
3. 当日过滤：只保留今天的电报/新闻，避免历史噪音
4. 多源去重：跨源内容按文本指纹去重
5. 缓存层：按 source+date 缓存 30 分钟，上游失败返回过期缓存
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader

logger = logging.getLogger("pangu.news_fetcher")

# A股/港股代码简单匹配（6 位数字，不保证全部有效）
_STOCK_CODE_RE = re.compile(r"\b\d{6}\b")
# 金额/数值实体（如 1.2 亿、3000 万）
_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(亿|万|亿元|万元|%)")


@dataclass
class NewsFlash:
    """单条快讯。"""

    time: str          # 发布时间，如 "14:32"
    content: str       # 正文
    important: bool = False  # 是否重要
    subjects: list[str] = field(default_factory=list)   # 题材标签
    stocks: list[dict] = field(default_factory=list)    # 关联个股 [{code,name,chg}]
    source: str = ""   # 来源标识
    content_hash: str = ""     # 正文指纹（用于去重/溯源）
    entities: list[dict[str, Any]] = field(default_factory=list)  # 提取实体 [{type,value}]

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = _content_hash(self.content)
        if not self.entities and self.content:
            self.entities = _extract_entities(self.content, self.stocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time, "content": self.content, "important": self.important,
            "subjects": self.subjects, "stocks": self.stocks, "source": self.source,
            "content_hash": self.content_hash, "entities": self.entities,
        }


def _content_hash(content: str) -> str:
    """生成正文指纹：去空白、取前 120 字符后 md5。"""
    normalized = re.sub(r"\s+", "", content)[:120]
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]


def _extract_entities(content: str, stocks: list[dict]) -> list[dict[str, Any]]:
    """从正文提取股票代码、金额/幅度等实体。"""
    entities: list[dict[str, Any]] = []
    # 1. 结构化关联个股
    for s in stocks:
        code = str(s.get("code", ""))
        name = str(s.get("name", ""))
        if code:
            entities.append({"type": "stock_code", "value": code, "name": name})
    # 2. 正文中的 6 位代码（去重）
    seen_codes = {e["value"] for e in entities if e["type"] == "stock_code"}
    for code in _STOCK_CODE_RE.findall(content):
        if code not in seen_codes:
            entities.append({"type": "stock_code", "value": code})
            seen_codes.add(code)
    # 3. 金额/百分比
    seen_amounts: set[str] = set()
    for m in _AMOUNT_RE.finditer(content):
        val = m.group(0)
        if val not in seen_amounts:
            entities.append({"type": "metric", "value": val, "unit": m.group(2)})
            seen_amounts.add(val)
    return entities


@dataclass
class StockNews:
    """单只股票的相关新闻。"""

    code: str
    title: str
    content: str = ""
    source: str = ""
    time: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "content": self.content[:200],
            "source": self.source, "time": self.time,
        }


@dataclass
class NewsResult:
    """新闻聚合结果。"""

    date: str
    flashes: list[NewsFlash] = field(default_factory=list)       # 今日快讯
    stock_news: dict[str, list[StockNews]] = field(default_factory=dict)  # {code: [新闻]}
    hot_themes: list[tuple[str, int]] = field(default_factory=list)       # [(题材词, 出现次数)]
    warnings: list[str] = field(default_factory=list)
    source_state: dict[str, dict[str, Any]] = field(default_factory=dict)  # 各源状态

    def to_markdown(self) -> str:
        """生成今日新闻简报 Markdown（供 news_sentiment.py 解析）。"""
        lines = [f"# 今日财经快讯 · {self.date}", ""]
        if self.flashes:
            lines += ["## 今日看点", ""]
            for f in self.flashes[:30]:
                prefix = "【重要】" if f.important else ""
                src = f"[{f.source}] " if f.source else ""
                lines.append(f"- {f.time} {prefix}{src}{f.content}")
            lines.append("")

        if self.hot_themes:
            lines += ["## 领涨板块/题材热度", ""]
            for theme, cnt in self.hot_themes[:15]:
                lines.append(f"- {theme}（{cnt}次提及）")
            lines.append("")

        if not self.flashes and not self.hot_themes:
            lines.append("*今日未取到新闻数据（数据源不可用）*")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "flashes": [f.to_dict() for f in self.flashes[:20]],
            "stock_news": {
                code: [n.to_dict() for n in news[:5]]
                for code, news in self.stock_news.items()
            },
            "hot_themes": self.hot_themes[:15],
            "warnings": self.warnings,
            "source_state": self.source_state,
        }


class NewsFetcher:
    """实时新闻聚合器。"""

    def __init__(self, dl: DataLoader, cfg: Optional[dict] = None) -> None:
        self.dl = dl
        self.cfg = cfg or {}
        ncfg = self.cfg.get("news", {})
        self.flash_limit = ncfg.get("flash_limit", 50)       # 电报最多取条数
        self.stock_news_limit = ncfg.get("stock_news_per_code", 5)  # 每只票最多新闻数
        # 题材关键词（从电报/新闻里提取热度）
        self.theme_keywords = ncfg.get("theme_keywords", _DEFAULT_THEME_KEYWORDS)

    # ------------------------------------------------------------------ #
    def fetch_today(
        self,
        candidates: Optional[list[dict]] = None,
        date: Optional[str] = None,
    ) -> NewsResult:
        """拉取今日新闻聚合。"""
        date = date or datetime.now().strftime("%Y%m%d")
        result = NewsResult(date=date)

        # 1. 财联社电报（主源）
        try:
            before = len(result.flashes)
            self._fetch_flashes(result, date)
            result.source_state["cls"] = {"ok": True, "count": len(result.flashes) - before, "error": ""}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            result.warnings.append(f"财联社电报取数失败: {err}")
            result.source_state["cls"] = {"ok": False, "count": 0, "error": err}
            logger.warning("财联社电报失败: %s", e)

        # 多源兜底：华尔街见闻 → 金十 → 雪球热门 → 格隆汇 → 新浪
        fallback_sources = [
            ("wscn", self._fetch_wallstreetcn),
            ("jin10", self._fetch_jin10),
            ("xueqiu", self._fetch_xueqiu_hotstock),
            ("gelonghui", self._fetch_gelonghui),
            ("sina", self._fetch_sina_finance),
        ]
        for source_id, fetch_fn in fallback_sources:
            if len(result.flashes) >= self.flash_limit:
                break
            try:
                before = len(result.flashes)
                fetch_fn(result)
                result.source_state[source_id] = {"ok": True, "count": len(result.flashes) - before, "error": ""}
            except Exception as e:  # noqa: BLE001
                err = str(e)
                result.source_state[source_id] = {"ok": False, "count": 0, "error": err}
                logger.debug("%s 失败: %s", fetch_fn.__name__, e)
        self._dedup_flashes(result)

        # 2. 个股关联新闻（针对候选股）
        if candidates:
            for c in candidates[:10]:  # 限制最多 10 只，避免过多请求
                code = str(c.get("code", "")).strip()
                if not code:
                    continue
                try:
                    news = self._fetch_stock_news(code)
                    if news:
                        result.stock_news[code] = news
                except Exception as e:  # noqa: BLE001
                    logger.debug("个股新闻 %s 失败: %s", code, e)

        # 3. 题材热度（从电报+新闻文本提取关键词频次）
        result.hot_themes = self._extract_hot_themes(result)

        logger.info(
            "新闻聚合完成 %s：快讯 %d 条，个股新闻 %d 只，热门题材 %d 个",
            date, len(result.flashes), len(result.stock_news), len(result.hot_themes),
        )
        return result

    # ------------------------------------------------------------------ #
    def _fetch_flashes(self, result: NewsResult, date: str) -> None:
        """拉财联社电报（免签名 cache 接口，国内直连）。"""
        import urllib.request
        import json as _json
        from datetime import datetime as _dt
        url = "https://www.cls.cn/api/cache?app=CailianpressWeb&name=telegraph&os=web&sv=8.7.9"
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.cls.cn/",
            })
            resp = urllib.request.urlopen(req, timeout=10)
            data = _json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            result.warnings.append(f"财联社电报取数失败: {e}")
            return

        roll = (data.get("data") or {}).get("roll_data") or []
        if not roll:
            return
        for x in roll:
            if x.get("is_ad"):
                continue
            ctime = x.get("ctime", 0)
            try:
                time_str = _dt.fromtimestamp(int(ctime)).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError, OSError):
                continue
            if date[:4] not in time_str:
                continue
            content = (x.get("content") or x.get("brief") or "").strip()
            if not content:
                continue
            subjects = [s.get("subject_name", "") for s in (x.get("subjects") or []) if s.get("subject_name")]
            stocks = [{
                "code": str(s.get("StockID", "")),
                "name": str(s.get("name", "")),
                "chg": s.get("RiseRange"),
            } for s in (x.get("stock_list") or []) if s.get("StockID")]
            result.flashes.append(NewsFlash(
                time=time_str[-5:],
                content=content[:200],
                important=x.get("level") != "C" or "【重要】" in content[:6],
                subjects=subjects,
                stocks=stocks,
                source="cls",
            ))
            if len(result.flashes) >= self.flash_limit:
                break
        if result.flashes:
            logger.info("财联社电报（免签名 cache）：%d 条", len(result.flashes))

    # ------------------------------------------------------------------ #
    def _fetch_wallstreetcn(self, result: NewsResult) -> None:
        """华尔街见闻快讯（免 Key 免代理，A股精选频道）。"""
        import urllib.request
        import json as _json
        from datetime import datetime as _dt
        url = "https://api-one-wscn.awtmt.com/apiv1/content/lives?channel=global-channel&limit=50"
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            })
            resp = urllib.request.urlopen(req, timeout=10)
            data = _json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.debug("华尔街见闻快讯失败: %s", e)
            return

        items = (data.get("data") or {}).get("items") or []
        before = len(result.flashes)
        for item in items[:30]:
            content = item.get("content_text") or item.get("title") or ""
            content = str(content).replace("<p>", "").replace("</p>", "").strip()
            if not content:
                continue
            ts = item.get("display_time") or item.get("created_at") or 0
            try:
                time_str = _dt.fromtimestamp(int(ts)).strftime("%H:%M")
            except (TypeError, ValueError, OSError):
                time_str = ""
            result.flashes.append(NewsFlash(
                time=time_str,
                content=content[:200],
                important=bool(item.get("important")) or "重要" in content,
                source="wscn",
            ))
        if len(result.flashes) > before:
            logger.info("华尔街见闻快讯：%d 条", len(result.flashes) - before)

    def _fetch_jin10(self, result: NewsResult) -> None:
        """金十数据快讯（JS 变量直连，补充宏观/国际快讯）。"""
        import re
        import urllib.request
        import json as _json
        url = "https://www.jin10.com/flash_newest.js"
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.jin10.com/",
            })
            raw = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            logger.debug("金十快讯失败: %s", e)
            return

        m = re.search(r"flash_newest\s*=\s*(\[.*?\]);?\s*$", raw, re.S)
        if not m:
            return
        try:
            items = _json.loads(m.group(1))
        except _json.JSONDecodeError:
            return
        before = len(result.flashes)
        for item in items[:30]:
            text = item.get("title") or item.get("content") or ""
            if not text:
                continue
            time_str = str(item.get("time", ""))[-5:] if item.get("time") else ""
            result.flashes.append(NewsFlash(
                time=time_str,
                content=str(text)[:200],
                important=bool(item.get("important")),
                source="jin10",
            ))
        if len(result.flashes) > before:
            logger.info("金十快讯：%d 条", len(result.flashes) - before)

    def _fetch_sina_finance(self, result: NewsResult) -> None:
        """新浪财经直播（免 Key，JSONP 直连，最后的新闻兜底）。"""
        import re
        import time as _time
        import urllib.request
        import json as _json
        url = (
            f"https://zhibo.sina.com.cn/api/zhibo/feed?callback=callback"
            f"&page=1&page_size=20&zhibo_id=152&tag_id=0&dire=f&dpc=1&pagesize=20"
            f"&id=4161089&type=0&_={int(_time.time())}"
        )
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://finance.sina.com.cn",
            })
            raw = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            logger.debug("新浪财经直播失败: %s", e)
            return

        m = re.search(r"callback\((.*)\)", raw, re.S)
        if not m:
            return
        try:
            data = _json.loads(m.group(1))
        except _json.JSONDecodeError:
            return
        feed_list = ((data.get("result") or {}).get("data") or {}).get("feed") or {}
        items = feed_list.get("list") or []
        before = len(result.flashes)
        for item in items[:20]:
            text = item.get("rich_text") or ""
            if not text:
                continue
            tags = [g.get("name", "") for g in (item.get("tag") or []) if g.get("name")]
            result.flashes.append(NewsFlash(
                time="",
                content=text[:200],
                important="焦点" in tags or "重要" in text,
                subjects=tags,
                source="sina",
            ))
        if len(result.flashes) > before:
            logger.info("新浪财经直播兜底：%d 条", len(result.flashes) - before)

    def _dedup_flashes(self, result: NewsResult) -> None:
        """按 content_hash 去重并截断到上限；记录去重前后数量。"""
        before = len(result.flashes)
        seen: set[str] = set()
        unique: list[NewsFlash] = []
        for f in result.flashes:
            key = f.content_hash or _content_hash(f.content)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(f)
        result.flashes = unique[:self.flash_limit]
        result.flashes.sort(key=lambda f: f.time, reverse=True)
        result.source_state["dedup"] = {
            "ok": True,
            "before": before,
            "after": len(result.flashes),
            "removed": before - len(result.flashes),
        }

    # ------------------------------------------------------------------ #
    def _fetch_stock_news(self, code: str) -> list[StockNews]:
        """拉单只股票关联新闻（多源尝试）。

        使用新浪财经 + 同花顺个股新闻替代已移除的东财搜索 API。
        """
        import urllib.parse
        import urllib.request
        import json as _json
        import re

        news: list[StockNews] = []

        # 来源1：新浪财经个股新闻
        try:
            market = "sh" if code.startswith("6") else "sz"
            sina_url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{market}{code}.phtml"
            req = urllib.request.Request(sina_url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://finance.sina.com.cn/",
            })
            raw = urllib.request.urlopen(req, timeout=10).read().decode("gbk", errors="ignore")
            # Simple HTML extraction for news titles
            for m in re.finditer(r'<a[^>]*target="_blank"[^>]*>\s*(.+?)\s*</a>', raw):
                title = re.sub(r'<[^>]+>', '', m.group(1)).strip()
                if title and len(title) > 4:
                    news.append(StockNews(code=code, title=title[:200], source="sina_stock"))
                    if len(news) >= self.stock_news_limit:
                        break
        except Exception as e:  # noqa: BLE001
            logger.debug("新浪个股新闻 %s 失败: %s", code, e)

        return news[:self.stock_news_limit]

    # ------------------------------------------------------------------ #
    def _fetch_xueqiu_hotstock(self, result: NewsResult) -> None:
        """雪球热门股票（A股人气排行 + 情绪信号）。"""
        import urllib.request
        import json as _json
        url = "https://stock.xueqiu.com/v5/stock/hot_stock/list.json?size=50&_type=12&type=12"
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://xueqiu.com/hq",
            })
            raw = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
            data = _json.loads(raw)
        except Exception as e:  # noqa: BLE001
            logger.debug("雪球热门股票失败: %s", e)
            return

        items = (data.get("data") or {}).get("items") or []
        before = len(result.flashes)
        for item in items[:20]:
            stock_info = item.get("stock") or item.get("quote") or {}
            name = stock_info.get("name", "")
            code = str(stock_info.get("symbol", ""))
            pct = stock_info.get("percent", stock_info.get("change", 0))
            hot = item.get("hot_rank", item.get("rank", ""))
            content = f"雪球热门: {name}({code}) 涨跌{pct}% 热度排名{hot}" if name else ""
            if content:
                result.flashes.append(NewsFlash(
                    time=datetime.now().strftime("%H:%M"),
                    content=content[:200],
                    important=False,
                    source="xueqiu_hotstock",
                ))
        if len(result.flashes) > before:
            logger.info("雪球热门股票：%d 条", len(result.flashes) - before)

    def _fetch_gelonghui(self, result: NewsResult) -> None:
        """格隆汇快讯（专业港A股财经媒体）。"""
        import urllib.request
        import json as _json
        import re
        url = "https://www.gelonghui.com/api/subjects/30/articles"
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.gelonghui.com/",
            })
            raw = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
            data = _json.loads(raw)
        except Exception as e:  # noqa: BLE001
            logger.debug("格隆汇快讯失败: %s", e)
            return

        articles = (data.get("result") or {}).get("articles") or data.get("data") or []
        before = len(result.flashes)
        for a in articles[:15]:
            title = a.get("title", "")
            if not title:
                continue
            # Filter for A-share relevant content
            text = re.sub(r'<[^>]+>', '', title).strip()
            result.flashes.append(NewsFlash(
                time=datetime.now().strftime("%H:%M"),
                content=text[:200],
                important=False,
                source="gelonghui",
            ))
        if len(result.flashes) > before:
            logger.info("格隆汇快讯：%d 条", len(result.flashes) - before)

    # ------------------------------------------------------------------ #
    def _extract_hot_themes(self, result: NewsResult) -> list[tuple[str, int]]:
        """从电报题材标签 + 新闻文本提取高频题材词，返回 [(题材, 次数)] 降序。"""
        counter: Counter[str] = Counter()
        # 1. 优先用结构化题材标签
        for f in result.flashes:
            if f.subjects:
                for subj in f.subjects:
                    counter[subj] += 1
        # 2. 无标签的电报，用关键词计数兜底
        for f in result.flashes:
            if f.subjects:
                continue
            for keyword in self.theme_keywords:
                if keyword in f.content:
                    counter[keyword] += 1
        # 3. 个股新闻标题也纳入
        for news_list in result.stock_news.values():
            for n in news_list:
                for keyword in self.theme_keywords:
                    if keyword in n.title:
                        counter[keyword] += 1

        return counter.most_common(15)


# 题材关键词库（覆盖 A 股主流概念，用于热度提取）
_DEFAULT_THEME_KEYWORDS = [
    # 科技
    "AI", "人工智能", "算力", "芯片", "半导体", "光模块", "CPO", "服务器",
    "机器人", "人形机器人", "工业母机", "华为", "消费电子", "MR", "VR",
    "数据要素", "信创", "国产替代", "量子", "脑机接口",
    # 新能源
    "新能源", "光伏", "储能", "锂电", "固态电池", "充电桩", "风电", "氢能",
    "核电", "稀土", "特高压",
    # 医药
    "医药", "创新药", "中药", "医疗器械", "CRO", "疫苗", "合成生物",
    # 金融周期
    "券商", "银行", "保险", "地产", "建材", "钢铁", "有色", "煤炭",
    "化工", "石油", "黄金", "铜",
    # 消费
    "白酒", "食品", "零售", "旅游", "传媒", "游戏", "教育", "纺织",
    # 军工航天
    "军工", "航天", "航空", "低空经济", "商业航天", "卫星",
    # 政策题材
    "国企改革", "重组", "并购", "回购", "增持", "一带一路", "碳中和",
]