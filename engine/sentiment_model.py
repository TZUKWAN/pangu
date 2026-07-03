"""情感分析模型封装 + 舆情演化追踪。

基于 BettaFish 的 WeiboMultilingualSentiment (tabularisai/multilingual-sentiment-analysis)，
提供 5 级情感分类（非常负面/负面/中性/正面/非常正面）。
集成舆情演化：按日存储情感分数，支持时间序列回测与趋势可视化。
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("pangu.sentiment_model")

_SENTIMENT_MODEL = None
_SENTIMENT_TOKENIZER = None
_DATA_DIR = Path("data/sentiment")
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 5 级中文标签映射
LABEL_MAP = {
    "1 star": "非常负面",
    "2 stars": "负面",
    "3 stars": "中性",
    "4 stars": "正面",
    "5 stars": "非常正面",
}


def _load_model():
    """延迟加载情感分析模型（首次调用时加载到内存）。"""
    global _SENTIMENT_MODEL, _SENTIMENT_TOKENIZER
    if _SENTIMENT_MODEL is not None:
        return True
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        model_name = "tabularisai/multilingual-sentiment-analysis"
        _SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _SENTIMENT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        # 使用 CPU（避免 GPU 内存问题）
        _SENTIMENT_MODEL.eval()
        logger.info("情感模型已加载: %s", model_name)
        return True
    except ImportError:
        logger.warning("transformers/torch 未安装，情感分析降级为关键词模式")
        return False
    except Exception as e:
        logger.warning("情感模型加载失败: %s，降级为关键词模式", e)
        return False


@dataclass
class SentimentScore:
    """单条文本的情感分析结果。"""
    text: str
    label: str  # 非常负面/负面/中性/正面/非常正面
    confidence: float  # 0-1
    scores: dict[str, float] = field(default_factory=dict)  # 5 级概率分布

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text[:100], "label": self.label,
                "confidence": round(self.confidence, 3),
                "scores": {k: round(v, 3) for k, v in self.scores.items()}}


@dataclass
class DailySentiment:
    """单日舆情汇总。"""
    date: str
    total: int = 0
    distribution: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_confidence: float = 0.0
    sentiment_index: float = 0.0  # -100(极度悲观) 到 +100(极度乐观)
    top_negative: list[dict[str, Any]] = field(default_factory=list)
    top_positive: list[dict[str, Any]] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date, "total": self.total,
            "distribution": dict(self.distribution),
            "avg_confidence": round(self.avg_confidence, 3),
            "sentiment_index": round(self.sentiment_index, 1),
            "top_negative": self.top_negative[:5],
            "top_positive": self.top_positive[:5],
        }


class SentimentTracker:
    """舆情追踪器：分析 + 存储 + 演化回测。"""

    def __init__(self):
        self._cache: dict[str, DailySentiment] = {}

    def analyze_text(self, text: str) -> Optional[SentimentScore]:
        """对单条文本进行 5 级情感分析。"""
        if not text or not text.strip():
            return None

        # 尝试使用 ML 模型
        if _load_model():
            try:
                import torch
                inputs = _SENTIMENT_TOKENIZER(
                    text[:512], return_tensors="pt", truncation=True, padding=True
                )
                with torch.no_grad():
                    outputs = _SENTIMENT_MODEL(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                idx = torch.argmax(probs).item()

                # Map numeric id to label
                id2label = _SENTIMENT_MODEL.config.id2label
                raw_label = id2label.get(idx, f"{idx} stars")
                # Map both English and star labels to Chinese
                EN_MAP = {"1 star":"非常负面","2 stars":"负面","3 stars":"中性","4 stars":"正面","5 stars":"非常正面",
                         "Very negative":"非常负面","Negative":"负面","Neutral":"中性","Positive":"正面","Very positive":"非常正面"}
                label = EN_MAP.get(raw_label, LABEL_MAP.get(raw_label, raw_label))

                scores = {}
                for i, prob in enumerate(probs.tolist()):
                    raw = id2label.get(i, f"{i} stars")
                    scores[EN_MAP.get(raw, LABEL_MAP.get(raw, raw))] = prob

                return SentimentScore(
                    text=text, label=label,
                    confidence=float(probs[idx]),
                    scores=scores,
                )
            except Exception as e:
                logger.debug("ML sentiment failed: %s", e)

        # 降级：关键词模式
        return self._keyword_sentiment(text)

    def _keyword_sentiment(self, text: str) -> SentimentScore:
        """关键词情感分析降级方案。"""
        pos_words = ["上涨", "涨停", "利好", "突破", "增长", "盈利", "回购", "分红", "反弹",
                     "领涨", "大涨", "强势", "新高", "放量", "买入", "增持", "净流入"]
        neg_words = ["下跌", "跌停", "利空", "暴跌", "亏损", "退市", "减持", "处罚", "诉讼",
                     "风险", "崩盘", "破位", "缩量", "卖出", "净流出", "踩雷", "爆仓"]
        pos = sum(1 for w in pos_words if w in text)
        neg = sum(1 for w in neg_words if w in text)
        total = pos + neg
        if total == 0:
            label, conf, scores = "中性", 0.5, {"非常负面": 0.05, "负面": 0.1, "中性": 0.7, "正面": 0.1, "非常正面": 0.05}
        else:
            ratio = pos / total
            if ratio >= 0.8:
                label = "非常正面"
            elif ratio >= 0.6:
                label = "正面"
            elif ratio >= 0.4:
                label = "中性"
            elif ratio >= 0.2:
                label = "负面"
            else:
                label = "非常负面"
            conf = min(0.85, 0.5 + total * 0.05)
            scores = {
                "非常负面": round((1 - ratio) ** 2, 3),
                "负面": round((1 - ratio) * 0.6, 3),
                "中性": round(0.5 - abs(ratio - 0.5), 3),
                "正面": round(ratio * 0.6, 3),
                "非常正面": round(ratio ** 2, 3),
            }
        return SentimentScore(text=text, label=label, confidence=conf, scores=scores)

    def analyze_batch(self, texts: list[str]) -> list[SentimentScore]:
        """批量情感分析。"""
        results = []
        for text in texts:
            score = self.analyze_text(text)
            if score:
                results.append(score)
        return results

    def daily_report(self, news_items: list[dict[str, Any]],
                     date: Optional[str] = None) -> DailySentiment:
        """对当日新闻做舆情汇总分析。"""
        date = date or datetime.now().strftime("%Y%m%d")
        texts = [n.get("content", "") for n in news_items if n.get("content")]
        results = self.analyze_batch(texts)

        report = DailySentiment(date=date, total=len(results))
        pos_total = neg_total = 0
        for r in results:
            report.distribution[r.label] += 1
            if r.confidence > 0.6:
                report.items.append(r.to_dict())
            if "正面" in r.label:
                pos_total += 1
            elif "负面" in r.label:
                neg_total += 1

        report.avg_confidence = sum(r.confidence for r in results) / max(1, len(results))
        if results:
            report.sentiment_index = round((pos_total - neg_total) / len(results) * 100, 1)
            sorted_by_conf = sorted(results, key=lambda x: x.confidence, reverse=True)
            report.top_positive = [r.to_dict() for r in sorted_by_conf if "正面" in r.label][:5]
            report.top_negative = [r.to_dict() for r in sorted_by_conf if "负面" in r.label][:5]

        # 持久化
        self._save(report)
        self._cache[date] = report
        return report

    def evolution(self, lookback_days: int = 7) -> dict[str, Any]:
        """舆情演化：过去 N 天的情感指数变化趋势。"""
        dates = []
        indices = []
        distributions = []
        details = []
        today = datetime.now()
        for i in range(lookback_days, 0, -1):
            d = (today - timedelta(days=i)).strftime("%Y%m%d")
            report = self._load(d)
            if report:
                dates.append(d)
                indices.append(report.sentiment_index)
                distributions.append(report.distribution)
                details.append(report.to_dict())
        # Trend analysis
        trend = "平稳"
        if len(indices) >= 3:
            recent = indices[-3:]
            if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                trend = "上升"
            elif all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
                trend = "下降"
        return {
            "lookback_days": lookback_days,
            "dates": dates,
            "indices": indices,
            "distributions": distributions,
            "trend": trend,
            "latest_index": indices[-1] if indices else 0,
            "details": details,
        }

    def _save(self, report: DailySentiment):
        path = _DATA_DIR / f"{report.date}.json"
        try:
            path.write_text(json.dumps(report.to_dict(), ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.debug("save sentiment %s: %s", report.date, e)

    def _load(self, date: str) -> Optional[DailySentiment]:
        if date in self._cache:
            return self._cache[date]
        path = _DATA_DIR / f"{date}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return DailySentiment(
                    date=data["date"], total=data["total"],
                    distribution=defaultdict(int, data.get("distribution", {})),
                    avg_confidence=data.get("avg_confidence", 0),
                    sentiment_index=data.get("sentiment_index", 0),
                    top_negative=data.get("top_negative", []),
                    top_positive=data.get("top_positive", []),
                )
            except Exception:
                pass
        return None
