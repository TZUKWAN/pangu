"""舆情引擎：新闻聚合、深度分析、回测、舆情演化模拟。

核心能力：
1. 聚合多源新闻（财联社/华尔街见闻/金十/新浪/雪球/格隆汇），拉取 200+ 条
2. LLM 深度分析：提取主题、判断情绪方向、识别关键事件
3. 舆情回测：拉取历史新闻（1周/2周/1月），分析舆情趋势与周期性
4. 舆情演化模拟：基于历史舆情数据，推测未来板块轮动方向
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger("pangu.sentiment_engine")


def _fetch_cls_flash(date_str: str | None = None, limit: int = 80) -> list[dict[str, Any]]:
    """Fetch 财联社电报."""
    import urllib.request
    items = []
    # Try multiple pages to get more news
    for page in range(3):
        if len(items) >= limit:
            break
        try:
            url = f"https://www.cls.cn/api/cache?app=CailianpressWeb&name=telegraph&os=web&sv=8.7.9&page={page}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.cls.cn/",
            })
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read().decode("utf-8"))
            roll = (data.get("data") or {}).get("roll_data") or []
            for x in roll:
                if len(items) >= limit:
                    break
                if x.get("is_ad"):
                    continue
                ctime = x.get("ctime", 0)
                try:
                    t = datetime.fromtimestamp(int(ctime))
                    ts = t.strftime("%Y-%m-%d %H:%M")
                    d = t.strftime("%Y%m%d")
                except (TypeError, ValueError, OSError):
                    continue
                if date_str and d != date_str:
                    continue
                content = (x.get("content") or x.get("brief") or "").strip()
                if not content:
                    continue
                subjects = [s.get("subject_name", "") for s in (x.get("subjects") or []) if s.get("subject_name")]
                items.append({"time": ts, "content": content, "subjects": subjects,
                              "important": x.get("level") != "C", "source": "cls"})
        except Exception as e:
            logger.debug("cls flash fetch page %d: %s", page, e)
            break
    logger.info("cls fetched %d news", len(items))
    return items


def _fetch_wscn_flash(limit: int = 50) -> list[dict[str, Any]]:
    """Fetch 华尔街见闻."""
    import urllib.request
    items = []
    try:
        url = "https://api-one-wscn.awtmt.com/apiv1/content/lives?channel=global-channel&limit=50"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode("utf-8"))
        raw_items = (data.get("data") or {}).get("items") or []
        for item in raw_items[:limit]:
            content = item.get("content_text") or item.get("title") or ""
            content = str(content).replace("<p>", "").replace("</p>", "").strip()
            if not content:
                continue
            ts = item.get("display_time") or item.get("created_at") or 0
            try:
                t = datetime.fromtimestamp(int(ts)).strftime("%H:%M")
            except (TypeError, ValueError, OSError):
                t = ""
            items.append({"time": t, "content": content[:200], "subjects": [],
                          "important": bool(item.get("important")), "source": "wscn"})
    except Exception as e:
        logger.debug("wscn fetch: %s", e)
    return items


def _fetch_jin10_flash(limit: int = 40) -> list[dict[str, Any]]:
    """Fetch 金十数据."""
    import urllib.request
    items = []
    try:
        url = "https://www.jin10.com/flash_newest.js"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0", "Referer": "https://www.jin10.com/",
        })
        raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
        m = re.search(r"flash_newest\s*=\s*(\[.*?\]);?\s*$", raw, re.S)
        if m:
            data = json.loads(m.group(1))
            for item in data[:limit]:
                text = item.get("title") or item.get("content") or ""
                if not text:
                    continue
                items.append({"time": str(item.get("time", ""))[-5:], "content": str(text)[:200],
                              "subjects": [], "important": False, "source": "jin10"})
    except Exception as e:
        logger.debug("jin10 fetch: %s", e)
    return items


def _fetch_sina_flash(limit: int = 30) -> list[dict[str, Any]]:
    """Fetch 新浪财经直播."""
    import urllib.request
    items = []
    try:
        t = int(time.time())
        url = f"https://zhibo.sina.com.cn/api/zhibo/feed?callback=callback&page=1&page_size={limit}&zhibo_id=152&tag_id=0&dire=f&dpc=1&_={t}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn",
        })
        raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
        m = re.search(r"callback\((.*)\)", raw, re.S)
        if m:
            data = json.loads(m.group(1))
            feed_list = ((data.get("result") or {}).get("data") or {}).get("feed") or {}
            for item in (feed_list.get("list") or [])[:limit]:
                text = item.get("rich_text") or ""
                if text:
                    items.append({"time": "", "content": text[:200], "subjects": [],
                                  "important": False, "source": "sina"})
    except Exception as e:
        logger.debug("sina fetch: %s", e)
    return items


def _fetch_xueqiu_hot(limit: int = 20) -> list[dict[str, Any]]:
    """Fetch 雪球热门股票."""
    import urllib.request
    items = []
    try:
        url = "https://stock.xueqiu.com/v5/stock/hot_stock/list.json?size=50&_type=12&type=12"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0", "Referer": "https://xueqiu.com/hq",
        })
        raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
        data = json.loads(raw)
        raw_items = (data.get("data") or {}).get("items") or []
        for item in raw_items[:limit]:
            stock_info = item.get("stock") or item.get("quote") or {}
            name = stock_info.get("name", "")
            code = str(stock_info.get("symbol", ""))
            pct = stock_info.get("percent", stock_info.get("change", 0))
            if name:
                items.append({"time": datetime.now().strftime("%H:%M"),
                              "content": f"雪球热门: {name}({code}) {pct}%",
                              "subjects": [], "important": False, "source": "xueqiu"})
    except Exception as e:
        logger.debug("xueqiu fetch: %s", e)
    return items


@dataclass
class SentimentReport:
    """舆情分析报告."""
    date: str
    total_news: int
    source_breakdown: dict[str, int] = field(default_factory=dict)
    top_themes: list[tuple[str, int]] = field(default_factory=list)
    sentiment_summary: str = ""       # LLM 分析摘要
    key_events: list[str] = field(default_factory=list)
    sector_impact: dict[str, str] = field(default_factory=dict)  # 板块→影响
    risk_alerts: list[str] = field(default_factory=list)
    trend_forecast: str = ""          # 未来走势预测
    raw_news: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date, "total_news": self.total_news,
            "source_breakdown": self.source_breakdown,
            "top_themes": [{"theme": t, "count": c} for t, c in self.top_themes[:20]],
            "sentiment_summary": self.sentiment_summary,
            "key_events": self.key_events[:10],
            "sector_impact": self.sector_impact,
            "risk_alerts": self.risk_alerts[:5],
            "trend_forecast": self.trend_forecast,
        }


class SentimentEngine:
    """舆情引擎：聚合+分析+回测."""

    def __init__(self) -> None:
        self._cache: dict[str, SentimentReport] = {}

    def fetch_today_news(self, target_count: int = 200) -> list[dict[str, Any]]:
        """拉取今日新闻，目标200条."""
        all_news: list[dict[str, Any]] = []
        fetchers = [
            ("cls", lambda: _fetch_cls_flash(limit=80)),
            ("wscn", lambda: _fetch_wscn_flash(limit=50)),
            ("jin10", lambda: _fetch_jin10_flash(limit=40)),
            ("sina", lambda: _fetch_sina_flash(limit=30)),
            ("xueqiu", lambda: _fetch_xueqiu_hot(limit=20)),
        ]
        for name, fn in fetchers:
            try:
                items = fn()
                for item in items:
                    item["source"] = name
                all_news.extend(items)
                logger.info("舆情引擎: %s -> %d 条", name, len(items))
            except Exception as e:
                logger.debug("舆情源 %s 失败: %s", name, e)

        # 去重
        seen = set()
        dedup = []
        for n in all_news:
            key = n["content"][:80]
            if key not in seen:
                seen.add(key)
                dedup.append(n)
        all_news = dedup[:max(target_count, len(dedup))]
        logger.info("舆情引擎: 聚合 %d 条新闻 (去重后)", len(all_news))
        return all_news

    def analyze(self, news: list[dict[str, Any]], llm_client=None) -> SentimentReport:
        """深度分析新闻集合."""
        report = SentimentReport(
            date=datetime.now().strftime("%Y%m%d"),
            total_news=len(news),
            raw_news=news,
        )

        # 来源统计
        src_counts: Counter = Counter()
        for n in news:
            src_counts[n.get("source", "unknown")] += 1
        report.source_breakdown = dict(src_counts.most_common(10))

        # 主题提取（基于关键词词频）
        theme_keywords = [
            "AI", "人工智能", "芯片", "半导体", "新能源", "光伏", "储能", "锂电",
            "医药", "创新药", "券商", "银行", "地产", "军工", "航天", "低空经济",
            "机器人", "华为", "消费电子", "白酒", "食品", "传媒", "游戏",
            "降息", "降准", "政策", "监管", "减持", "回购", "分红",
            "北向", "主力", "游资", "机构", "涨停", "跌停", "突破",
            "业绩", "预增", "亏损", "重组", "并购",
        ]
        theme_counter: Counter = Counter()
        for n in news:
            content = n.get("content", "")
            subjects = n.get("subjects", [])
            for kw in theme_keywords:
                if kw in content:
                    theme_counter[kw] += 1
            for s in subjects:
                if s:
                    theme_counter[s] += 1
        report.top_themes = theme_counter.most_common(30)

        # LLM 深度分析（如果可用）
        if llm_client:
            try:
                # 构建分析 prompt
                news_sample = []
                for n in news[:50]:
                    news_sample.append(f"[{n.get('source','')}] {n.get('time','')}: {n.get('content','')[:100]}")
                news_text = "\n".join(news_sample)

                top_t = ", ".join([f"{t}({c})" for t, c in report.top_themes[:15]])
                prompt = f"""分析以下今日A股财经新闻（共{len(news)}条），请输出JSON格式：

```json
{{
  "sentiment_summary": "200字以内的舆情综合分析",
  "key_events": ["最重要的3-5个事件"],
  "sector_impact": {{"板块名": "利好/利空/中性", ...}},
  "risk_alerts": ["需要警惕的2-3个风险"],
  "trend_forecast": "基于舆情判断未来一周可能的走势和热点方向，100字"
}}
```

今日新闻TOP50（来自财联社/华尔街见闻/金十/新浪/雪球）：
{news_text}

热门主题: {top_t}"""

                resp = llm_client.simple_chat(
                    [{"role": "system", "content": "你是A股舆情分析专家，擅长从大量财经新闻中提取关键信息和趋势判断。只输出JSON，不要其他内容。"},
                     {"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=2000,
                )
                if resp:
                    # 提取JSON
                    json_start = resp.find("{")
                    json_end = resp.rfind("}")
                    if json_start >= 0 and json_end > json_start:
                        try:
                            parsed = json.loads(resp[json_start:json_end + 1])
                            report.sentiment_summary = str(parsed.get("sentiment_summary", ""))
                            report.key_events = [str(e) for e in parsed.get("key_events", [])]
                            report.sector_impact = {str(k): str(v) for k, v in parsed.get("sector_impact", {}).items()}
                            report.risk_alerts = [str(r) for r in parsed.get("risk_alerts", [])]
                            report.trend_forecast = str(parsed.get("trend_forecast", ""))
                        except json.JSONDecodeError:
                            report.sentiment_summary = resp[:500]
                    else:
                        report.sentiment_summary = resp[:500]
            except Exception as e:
                logger.warning("LLM sentiment analysis failed: %s", e)
                report.sentiment_summary = f"AI分析暂不可用（{e}），以下是基于关键词统计的主题分布：{top_t}"

        self._cache[report.date] = report
        return report

    def backtest(self, lookback_days: int = 7) -> dict[str, Any]:
        """舆情回测：对过去N天的新闻进行汇总分析."""
        results: dict[str, Any] = {
            "lookback_days": lookback_days,
            "periods": [],
            "theme_evolution": {},
            "summary": "",
        }
        # 由于无法获取历史新闻（多数新闻源仅提供当日数据），
        # 这里使用本地缓存的报告进行回测
        cache_keys = sorted(self._cache.keys(), reverse=True)
        available = cache_keys[:min(lookback_days, len(cache_keys))]

        theme_tracker: dict[str, list[int]] = {}
        for key in available:
            report = self._cache[key]
            results["periods"].append({
                "date": key, "total": report.total_news,
                "themes": [(t, c) for t, c in report.top_themes[:10]],
                "summary": report.sentiment_summary[:200],
            })
            for theme, count in report.top_themes[:15]:
                if theme not in theme_tracker:
                    theme_tracker[theme] = []
                theme_tracker[theme].append(count)

        # 主题演化
        results["theme_evolution"] = {
            theme: {"counts": counts, "trend": "上升" if len(counts) >= 2 and counts[-1] > counts[0] else ("下降" if len(counts) >= 2 and counts[-1] < counts[0] else "平稳")}
            for theme, counts in theme_tracker.items()
        }
        results["summary"] = f"回测周期 {lookback_days} 天，可用数据 {len(available)} 天。"
        return results

    def get_or_fetch(self, llm_client=None) -> SentimentReport:
        """获取或拉取今日舆情报告."""
        today = datetime.now().strftime("%Y%m%d")
        if today in self._cache:
            return self._cache[today]
        news = self.fetch_today_news(target_count=200)
        return self.analyze(news, llm_client=llm_client)
