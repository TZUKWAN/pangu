"""数据层：akshare 取数封装。

设计原则：
1. 所有 akshare 调用都带重试 + 异常隔离（akshare 网络脆弱，单点失败不应炸整个 pipeline）。
2. 文件缓存：行情类请求 30 分钟内复用，减少对数据源的压力。
3. 失败降级：返回空 DataFrame + 警告日志，而非抛异常，让上层决定如何处理。
4. 数据源以「不封 IP、免 token」的源为主：
   - 同花顺 stock_fund_flow_individual   全市场实时行情（含资金流，主源）
   - 腾讯 qt.gtimg.cn                     实时行情（含 PE/PB/市值，批量兜底）
   - 新浪 stock_zh_a_daily                单只历史日 K（前复权，主源）
   - 新浪 stock_financial_abstract        财务摘要
   - 同花顺 stock_board_concept_name_ths  概念板块列表
   - adata concept_constituent_ths        概念板块成分股（同花顺源）
   - 同花顺 data.10jqka.com.cn            涨跌停/强势股池（替代东财源）
   - 通达信 mootdx TCP                    日K线兜底（不封IP）

已知限制（已处理）：
- 东方财富系接口已全部从主链路移除（policy）。
- 个股资金流以同花顺 stock_fund_flow_individual 为主（不封 IP，自带主力净流入）。

注意：本模块只做「取数 + 清洗」，不做策略判断。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

# 必须在 import akshare 之前全局禁用 tqdm，否则 akshare 内部会拿到原生 tqdm 引用
from . import _tqdm_patch  # noqa: F401
# 必须在 import akshare 之前 patch 掉系统代理，避免 akshare 的国内数据源请求
# 被 Clash 等代理劫持导致 ProxyError 取数失败
from . import _proxy_patch  # noqa: F401

import pandas as pd

from .errors import DataUnavailableError, MarketClosedError, retry_on_error

logger = logging.getLogger("pangu.data")

# akshare import 失败时给出清晰提示，而不是晦涩的 ModuleNotFoundError
try:
    import akshare as ak
    _AK_VERSION = getattr(ak, "__version__", "unknown")
    logger.debug("akshare %s loaded", _AK_VERSION)
except ImportError:  # pragma: no cover - 环境问题，引导用户装依赖
    ak = None
    _AK_VERSION = None

# 可选的 adata 数据源，作为 akshare 失败时的 fallback
try:
    import adata as _adata
    _ADATA_VERSION = getattr(_adata, "__version__", "unknown")
    logger.debug("adata %s loaded", _ADATA_VERSION)
except ImportError:  # pragma: no cover - 可选依赖
    _adata = None
    _ADATA_VERSION = None


class DataLoadError(RuntimeError):
    """数据加载彻底失败（重试用尽 + 无缓存）。"""


class DataLoader:
    """akshare 取数封装，带重试、缓存、降级。

    用法：
        dl = DataLoader(cache_dir="data/cache", cache_ttl_minutes=30)
        spot = dl.all_spot()              # 全市场实时行情
        zt   = dl.limit_up_pool(date=today)
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        cache_ttl_minutes: int = 30,
        retry_times: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.retry_times = retry_times
        self.backoff = backoff_seconds
        # 进程内内存缓存：财务指标按 symbol 缓存 24h，避免反复调用触发限流
        self._fin_cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
        # 进程内内存缓存：日 K / 全市场快照在同一次扫描中会被反复读取，用内存缓存消除重复 IO
        self._memory_cache: dict[str, tuple[datetime, Any]] = {}
        self._memory_cache_ttl = timedelta(minutes=60)
        if ak is None:
            raise DataLoadError(
                "akshare 未安装。请在项目根目录运行: pip install -r requirements.txt"
            )

    # ------------------------------------------------------------------ #
    # 内部工具：重试 + 缓存
    # ------------------------------------------------------------------ #
    def _cache_path(self, key: str) -> Path:
        # 用 key 的 hash 做文件名，避免非法字符
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{h}.parquet"

    def _cache_get(
        self, key: str, *, allow_stale: bool = False
    ) -> Optional[pd.DataFrame]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
        if age > self.cache_ttl and not allow_stale:
            return None
        try:
            return pd.read_parquet(p)
        except Exception:  # 缓存损坏则忽略
            return None

    def _cache_get_stale(self, key: str) -> Optional[pd.DataFrame]:
        """即使缓存过期也返回，用于网络中断时的最后降级。"""
        return self._cache_get(key, allow_stale=True)

    @staticmethod
    def _ensure_columns(
        df: pd.DataFrame, required: list[str]
    ) -> pd.DataFrame:
        """确保 DataFrame 含所需列，缺失列以空值补齐，避免下游 KeyError。"""
        for col in required:
            if col not in df.columns:
                df[col] = pd.Series(dtype=object)
        return df

    def _call_pool(
        self,
        func_name: str,
        cache_key: str,
        required_cols: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """涨跌停股池统一调用：失败返回空 DataFrame，并确保关键列存在。"""
        df = self._call(func_name, cache_key, *args, **kwargs)
        if len(df) == 0:
            return pd.DataFrame(columns=required_cols)
        return self._ensure_columns(df, required_cols)

    def _cache_put(self, key: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._cache_path(key), index=False)
        except Exception as e:  # 缓存写失败不影响主流程
            logger.debug("cache write failed: %s", e)

    def _mem_get(self, key: str) -> Any:
        cached = self._memory_cache.get(key)
        if cached is None:
            return None
        cached_at, value = cached
        if datetime.now() - cached_at > self._memory_cache_ttl:
            self._memory_cache.pop(key, None)
            return None
        return value

    def _mem_put(self, key: str, value: Any) -> None:
        self._memory_cache[key] = (datetime.now(), value)

    def _call_with_timeout(
        self,
        fn: Callable[[], pd.DataFrame],
        timeout_seconds: float = 6.0,
    ) -> Optional[pd.DataFrame]:
        """在 daemon 线程中执行取数函数，超时返回 None，避免网络 hang 住主链路。"""
        import threading
        result_container: list[Any] = [None]
        exception_container: list[Any] = [None]

        def _target() -> None:
            try:
                result_container[0] = fn()
            except Exception as e:  # noqa: BLE001
                exception_container[0] = e

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            logger.debug("data fetch timeout after %.1fs", timeout_seconds)
            return None
        if exception_container[0] is not None:
            logger.debug("data fetch error: %s", exception_container[0])
            return None
        return result_container[0]

    def _call(
        self,
        func_name: str,
        cache_key: str,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """带重试 + 缓存的 akshare 调用。失败返回空 DataFrame（不抛）。"""
        # 1. 命中缓存
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("cache hit: %s", func_name)
            return cached

        # 2. 重试调用
        func = getattr(ak, func_name, None)
        if func is None:
            logger.error("akshare 无此接口: %s", func_name)
            return pd.DataFrame()

        @retry_on_error(
            exceptions=(Exception,),
            tries=self.retry_times,
            backoff_seconds=self.backoff,
        )
        def _fetch() -> pd.DataFrame:
            df = func(*args, **kwargs)
            if df is None or len(df) == 0:
                raise DataUnavailableError(f"{func_name} 返回空数据")
            return df.reset_index(drop=True) if hasattr(df, "reset_index") else df

        try:
            df = _fetch()
        except Exception as e:  # noqa: BLE001
            logger.error("%s 重试用尽，返回空（%s）", func_name, e)
            return pd.DataFrame()

        self._cache_put(cache_key, df)
        return df

    # ------------------------------------------------------------------ #
    # 行情
    # ------------------------------------------------------------------ #
    def all_spot(self) -> pd.DataFrame:
        """全市场实时行情快照。

        主源同花顺个股资金流（不封IP，5186只，自带资金流），次选腾讯（含PE/PB/市值）。
        不再依赖东方财富。返回列：代码 / 名称 / 最新价 / 涨跌幅 / 成交额 / 换手率 / 主力净流入-净额 等
        """
        # 0. 进程内内存缓存
        mem = self._mem_get("all_spot")
        if mem is not None and len(mem) > 0:
            return mem.copy()

        # 1. 新鲜文件缓存命中
        cached = self._cache_get("all_spot")
        if cached is not None and len(cached) > 0:
            self._mem_put("all_spot", cached)
            return cached.copy()

        # 2. 主源：同花顺全市场行情（不封IP，5186只，自带资金流，13秒）
        if os.environ.get("PANGU_TDX_FALLBACK", "1") != "0":
            try:
                from . import tdx_source
                df = tdx_source.ths_all_spot()
                if len(df) > 0:
                    self._cache_put("all_spot", df)
                    self._mem_put("all_spot", df)
                    return df.copy()
            except Exception as e:  # noqa: BLE001
                logger.warning("同花顺全市场行情失败: %s", e)
            # 2b. 次选：腾讯批量行情（含PE/PB，但慢）
            try:
                df = tdx_source.tencent_all_spot()
                if len(df) > 0:
                    self._cache_put("all_spot", df)
                    self._mem_put("all_spot", df)
                    return df.copy()
            except Exception as e:  # noqa: BLE001
                logger.debug("腾讯全市场行情失败: %s", e)

        # 3. 过期缓存兜底
        stale = self._cache_get_stale("all_spot")
        if stale is not None and len(stale) > 0:
            logger.warning("all_spot 实时取数失败/为空，使用过期缓存（%d 条）", len(stale))
            self._mem_put("all_spot", stale)
            return stale.copy()

        raise DataUnavailableError(
            "全市场实时行情获取失败，且无可用缓存（可能网络中断或非交易日）"
        )

    def daily_kline(
        self,
        symbol: str,
        days: int = 60,
        adjust: str = "qfq",
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """单只股票历史日 K。

        主源新浪 stock_zh_a_daily（不封IP，前复权），失败走腾讯前复权K线/mootdx兜底。
        返回列：日期 / 股票代码 / 开盘 / 收盘 / 最高 / 最低 / 成交量 / 成交额 / 换手率

        扫描阶段会被高频调用，单只设 6s 硬超时、不重试，避免问题股拖垮整个链路。
        """
        end_dt = datetime.strptime(date, "%Y%m%d") if date else datetime.now()
        end = end_dt.strftime("%Y%m%d")
        start = (end_dt - timedelta(days=days * 2)).strftime("%Y%m%d")
        cache_key = f"kline:{symbol}:{adjust}:{start}:{end}"

        # 进程内内存缓存（同一次扫描中反复读取同一 symbol）
        mem = self._mem_get(cache_key)
        if mem is not None:
            return mem.copy()

        # 文件缓存命中
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._mem_put(cache_key, cached)
            return cached.copy()

        # 1. 主源：新浪 stock_zh_a_daily（前复权，不封IP）
        sina_symbol = _guess_sina_symbol(symbol)

        def _fetch_sina() -> pd.DataFrame:
            old_retry = self.retry_times
            self.retry_times = 1
            try:
                return self._call(
                    "stock_zh_a_daily", cache_key + ":sina",
                    symbol=sina_symbol, start_date=start, end_date=end, adjust=adjust,
                )
            finally:
                self.retry_times = old_retry

        df = self._call_with_timeout(_fetch_sina, timeout_seconds=6.0)
        if df is not None and len(df) > 0:
            df = _normalize_sina_kline(df, symbol)
            if len(df) > days:
                df = df.tail(days).reset_index(drop=True)
            self._cache_put(cache_key, df)
            self._mem_put(cache_key, df)
            return df.copy()

        # 2. 兜底：腾讯前复权K线（不封IP）
        if os.environ.get("PANGU_TDX_FALLBACK", "1") != "0":
            try:
                from . import tdx_source
                df = tdx_source.tencent_kline_qfq(symbol, days=days, adjust=adjust)
                if len(df) > 0:
                    self._cache_put(cache_key, df)
                    self._mem_put(cache_key, df)
                    return df.copy()
            except Exception as e:  # noqa: BLE001
                logger.debug("腾讯前复权K线 %s 失败: %s", symbol, e)

        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 涨跌停全家桶（情绪温度计核心数据源）
    # ------------------------------------------------------------------ #
    def limit_up_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        """涨停股池。

        仅使用同花顺涨停揭秘。东财接口在当前环境不稳定，且已按策略移出主链路。
        返回列：序号 / 代码 / 名称 / 涨跌幅 / 最新价 / 成交额 / 流通市值 / 总市值 /
               换手率 / 封板资金 / 首次封板时间 / 最后封板时间 / 炸板次数 /
               涨停统计（连板数）/ 连板数（概念）/ 所属行业
        """
        date = date or datetime.now().strftime("%Y%m%d")
        return _ths_limit_up_pool(date)

    def broke_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        """炸板池（曾涨停又打开）。

        ⚠️ 东财独家数据无等价免 token 替代，已停用。返回空 DataFrame，
        情绪温度计的炸板率改用同花顺涨停揭秘的「封板成功率」近似。
        """
        return pd.DataFrame()

    def limit_down_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        """跌停股池。仅使用同花顺，避免东财接口卡住或返回异常。"""
        date = date or datetime.now().strftime("%Y%m%d")
        return _ths_limit_pool(date, pool_type="down")

    def strong_pool(self, date: Optional[str] = None) -> pd.DataFrame:
        """强势股池。仅使用同花顺强势股，避免东财接口依赖。"""
        date = date or datetime.now().strftime("%Y%m%d")
        return _ths_strong_pool(date)

    # ------------------------------------------------------------------ #
    # 板块
    # ------------------------------------------------------------------ #
    def concept_boards(self) -> pd.DataFrame:
        """概念板块列表 + 当日涨跌幅。

        主源同花顺 stock_board_concept_name_ths（不封 IP）。
        返回列：板块名称 / 板块代码 / 涨跌幅
        """
        # 主源：同花顺概念板块
        df = self._call("stock_board_concept_name_ths", "concept_boards_ths")
        if len(df) > 0:
            col_map = {"name": "板块名称", "code": "板块代码"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "涨跌幅" not in df.columns:
                df["涨跌幅"] = 0.0
            return df
        return df

    def concept_constituents(self, board_symbol: str, board_name: str | None = None) -> pd.DataFrame:
        """概念板块成分股。

        主源 adata concept_constituent_ths（同花顺源，不封IP）。
        返回列：代码 / 名称
        """
        code = str(board_symbol).strip()
        # 主源：adata 同花顺概念成分股
        if _adata is not None:
            try:
                df = _adata.stock.info.concept_constituent_ths(concept_code=code)
                if df is not None and len(df) > 0:
                    col_map = {"stock_code": "代码", "short_name": "名称",
                               "code": "代码", "name": "名称"}
                    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                    return df
            except Exception as e:  # noqa: BLE001
                logger.debug("adata 概念成分股 %s 失败: %s", code, e)

        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 资金流（主力 = 超大单+大单净额）
    # ------------------------------------------------------------------ #
    def individual_fund_flow(self, symbol: str, fast: bool = False) -> pd.DataFrame:
        """个股资金流（近期每日主力净流入）。

        主源同花顺 stock_fund_flow_individual（不封IP），失败回退 all_spot。
        返回列统一：日期 / 主力净流入-净额 / 主力净流入-净占比 等。
        """
        # 代码不合法直接返回空，不浪费重试
        if not symbol or not isinstance(symbol, str) or not symbol.strip().isdigit():
            return pd.DataFrame()
        old_retry = self.retry_times
        if fast:
            self.retry_times = 1
        try:
            # 1. 主源：同花顺（不封IP）
            df = self._call(
                "stock_fund_flow_individual", f"fund_ths:{symbol}",
                symbol="即时",
            )
            if len(df) > 0:
                # 同花顺返回的是当日全市场排名，过滤出本股票
                code_col = find_col(df, ["股票代码", "代码"])
                if code_col:
                    df = df[df[code_col].astype(str).str.strip() == symbol.strip()]
                return df
            # 同花顺stock_fund_flow_individual失败 → 用 all_spot 兜底
            logger.debug("individual_fund_flow THS failed, fallback to all_spot for %s", symbol)
            try:
                spot = self.all_spot()
                code_col = find_col(spot, ["代码"])
                if code_col:
                    row = spot[spot[code_col].astype(str).str.strip() == symbol.strip()]
                    if len(row) > 0:
                        net_col = find_col(row, ["主力净流入-净额"])
                        pct_col = find_col(row, ["主力净流入-净占比"])
                        if net_col:
                            result = pd.DataFrame([{
                                "日期": datetime.now().strftime("%Y%m%d"),
                                "主力净流入-净额": str(row[net_col].iloc[0]),
                                "主力净流入-净占比": str(row[pct_col].iloc[0]) if pct_col else "0",
                                "股票代码": symbol,
                            }])
                            return result
            except Exception:  # noqa: BLE001
                pass
            return pd.DataFrame()
        finally:
            self.retry_times = old_retry

    def sector_fund_flow_rank(self, indicator: str = "今日") -> pd.DataFrame:
        """概念板块资金流排名。

        主源同花顺 stock_fund_flow_concept（不封IP）。
        indicator: "今日"/"5日"/"10日"（同花顺用"即时"/"3日"/"5日"/"10日"）。
        """
        # 同花顺 indicator 映射
        ths_indicator = {"今日": "即时", "5日": "5日", "10日": "10日"}.get(indicator, "即时")
        df = self._call(
            "stock_fund_flow_concept", f"sector_fund_ths:{indicator}",
            symbol=ths_indicator,
        )
        return df

    # ------------------------------------------------------------------ #
    # 财务（量化护栏用）
    # ------------------------------------------------------------------ #
    def financial_indicator(self, symbol: str) -> pd.DataFrame:
        """财务指标（最近若干报告期）。

        主源新浪 stock_financial_abstract（不封IP，财务摘要）。
        返回列含：选项 / 日期 / 加权净资产收益率(ROE)/ 资产负债率 / 净利润(万元) 等。
        用于排雷：亏损 / 高负债 / 退市风险。

        带进程内 24h 内存缓存，避免扫描候选池时逐只重复调用触发 akshare 限流。
        """
        now = datetime.now()
        # 统一内存缓存（与 _mem/_fin 合并，避免双缓存不一致）
        mem = self._mem_get(f"fin:{symbol}")
        if mem is not None:
            return mem.copy()
        cached = self._fin_cache.get(symbol)
        if cached is not None:
            cached_at, cached_df = cached
            if now - cached_at < timedelta(hours=24):
                logger.debug("financial_indicator memory cache hit: %s", symbol)
                self._mem_put(f"fin:{symbol}", cached_df)
                return cached_df.copy()

        old_retry = self.retry_times
        self.retry_times = 1
        try:
            df = self._call(
                "stock_financial_abstract", f"fin_sina:{symbol}",
                symbol=symbol,
            )
        finally:
            self.retry_times = old_retry
        self._fin_cache[symbol] = (now, df.copy())
        self._mem_put(f"fin:{symbol}", df)
        return df.copy()


def _guess_market(symbol: str) -> str:
    """根据股票代码前缀猜市场（akshare 个股资金流接口需要）。

    6/9 开头 → sh（上交所，9 是 B 股）；0/3 开头 → sz（深交所）；4/8 → bj（北交所）。
    """
    if not symbol:
        return "sh"
    head = symbol[0]
    if head in ("6", "9"):
        return "sh"
    if head in ("0", "3"):
        return "sz"
    if head in ("4", "8"):
        return "bj"
    return "sh"


# ---------------------------------------------------------------------- #
# 同花顺涨停/跌停/强势股兜底（不封 IP，data.10jqka.com.cn）
# ---------------------------------------------------------------------- #
def _ths_api_get(path: str, params: dict | None = None) -> Any:
    """同花顺 dataapi HTTP GET（不封 IP）。失败返回 None。"""
    import urllib.parse
    import urllib.request
    base = "https://data.10jqka.com.cn/dataapi" + path
    url = base
    if params:
        url = base + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://data.10jqka.com.cn/",
        })
        import json as _json
        resp = urllib.request.urlopen(req, timeout=10)
        return _json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.debug("同花顺 dataapi %s 失败: %s", path, e)
        return None


def _ths_limit_up_pool(date: str) -> pd.DataFrame:
    """同花顺涨停揭秘池（涨停股 + 连板高度 + 题材标签 + 封板成功率）。"""
    data = _ths_api_get("/limit_up/limit_up_pool", {
        "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
        "field": "code,name,amount,current,zt_status,high_days,reason_type",
    })
    if not data or not isinstance(data, dict):
        return pd.DataFrame()
    rows = data.get("data") or []
    if not rows:
        return pd.DataFrame()
    out = []
    for r in rows:
        out.append({
            "代码": str(r.get("code", "")),
            "名称": str(r.get("name", "")),
            "最新价": safe_float(r.get("current")),
            "成交额": safe_float(r.get("amount")),
            "连板数": safe_float(r.get("high_days"), 1),
            "涨停统计": f"{int(safe_float(r.get('high_days'), 1))}天{int(safe_float(r.get('high_days'), 1))}板",
            "所属行业": str(r.get("reason_type", "")) or "",
        })
    logger.info("同花顺涨停池兜底：%d 只", len(out))
    return pd.DataFrame(out)


def _ths_limit_pool(date: str, pool_type: str = "down") -> pd.DataFrame:
    """同花顺跌停池。pool_type: 'down'。"""
    data = _ths_api_get(f"/limit_{pool_type}/limit_{pool_type}_pool", {
        "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
        "field": "code,name,current,amount",
    })
    if not data or not isinstance(data, dict):
        return pd.DataFrame()
    rows = data.get("data") or []
    if not rows:
        return pd.DataFrame()
    out = [{
        "代码": str(r.get("code", "")),
        "名称": str(r.get("name", "")),
        "最新价": safe_float(r.get("current")),
        "成交额": safe_float(r.get("amount")),
    } for r in rows]
    return pd.DataFrame(out)


def _ths_strong_pool(date: str) -> pd.DataFrame:
    """同花顺强势股池（zx.10jqka.com.cn/getharden）。"""
    import urllib.request
    import json as _json
    url = "https://zx.10jqka.com.cn/getharden?field=code,name,current,chg,reason"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://stock.10jqka.com.cn/",
        })
        resp = urllib.request.urlopen(req, timeout=10)
        data = _json.loads(resp.read().decode("utf-8"))
        rows = data.get("data") or []
        if not rows:
            return pd.DataFrame()
        out = [{
            "代码": str(r.get("code", "")),
            "名称": str(r.get("name", "")),
            "最新价": safe_float(r.get("current")),
            "涨跌幅": safe_float(r.get("chg")),
            "所属行业": str(r.get("reason", "")) or "",
        } for r in rows]
        return pd.DataFrame(out)
    except Exception as e:  # noqa: BLE001
        logger.debug("同花顺强势股失败: %s", e)
        return pd.DataFrame()


def _guess_sina_symbol(symbol: str) -> str:
    """6位代码 → 新浪格式带市场前缀（sz000001 / sh600519）。"""
    return f"{_guess_market(symbol)}{symbol}"


def _normalize_sina_kline(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把新浪 stock_zh_a_daily 的英文列名统一为 akshare 中文风格。

    新浪返回列：date/open/high/low/close/volume/outstanding_share/turnover
    统一为：日期/股票代码/开盘/收盘/最高/最低/成交量/成交额/换手率
    """
    if df is None or len(df) == 0:
        return df
    col_map = {
        "date": "日期", "open": "开盘", "close": "收盘",
        "high": "最高", "low": "最低", "volume": "成交量",
        "amount": "成交额", "turnover": "换手率", "outstanding_share": "流通股本",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y%m%d")
    if "股票代码" not in df.columns:
        df["股票代码"] = symbol
    return df.reset_index(drop=True)


def _china_tz() -> Any:
    """返回北京时间时区（优先 pytz，否则用固定 +08:00）。"""
    try:
        import pytz
        return pytz.timezone("Asia/Shanghai")
    except Exception:  # noqa: BLE001
        from datetime import timezone, timedelta
        return timezone(timedelta(hours=8))


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """判断给定时间是否处于 A 股交易时段。

    A 股交易时间（工作日）：
        上午 09:30 - 11:30
        下午 13:00 - 15:00
    周末及非工作日默认关闭（不处理法定节假日，简单用工作日推断）。
    """
    if dt is None:
        dt = datetime.now(_china_tz())
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=_china_tz())

    if dt.weekday() >= 5:  # 周六、日
        return False

    t = dt.time()
    morning = time(9, 30) <= t <= time(11, 30)
    afternoon = time(13, 0) <= t <= time(15, 0)
    return morning or afternoon


def last_trading_date(reference: Optional[datetime] = None) -> str:
    """返回给定日期（默认今天）的最近交易日（"YYYYMMDD"）。

    简单用工作日推断：周末回退到周五；法定节假日不做精确处理。
    """
    if reference is None:
        reference = datetime.now(_china_tz())
    d = reference.date()
    while d.weekday() >= 5:  # 周六/周日回退
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def safe_float(v: Any, default: float = float("nan")) -> float:
    """安全转 float，处理 akshare 返回的字符串/None/NaN。"""
    try:
        if v is None:
            return default
        f = float(v)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """匹配列名：精确 > 前缀 > 子串（子串时取最短匹配避免误匹配）。

    修复点：原来纯子串匹配会把 '涨跌幅' 误命中 '涨跌幅.1'（领涨股票涨幅），
    导致取错列。现在精确匹配优先，子串匹配时取列名最短（最具体）的那个。
    """
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    for c in candidates:
        # 1. 精确匹配
        if c in cols:
            return c
        # 2. 前缀匹配（处理 "涨跌幅(%)" 等变体）
        for real in cols:
            if str(real).startswith(c):
                return real
        # 3. 子串匹配：取列名最短的（最具体，避免 '涨跌幅.1' 这种）
        matches = [real for real in cols if c in str(real)]
        if matches:
            return min(matches, key=lambda x: len(str(x)))
    return None


# ====================================================================== #
# 多源 fallback：akshare → adata → 本地快照 → 过期缓存 → 空 DataFrame
# ====================================================================== #
def _fetch_all_spot_adata() -> pd.DataFrame:
    """使用 adata 获取全市场实时行情，列名统一为 akshare 风格。"""
    if _adata is None:
        return pd.DataFrame()
    try:
        codes_df = _adata.stock.info.all_code()
        if codes_df is None or codes_df.empty:
            return pd.DataFrame()
        code_col = "stock_code"
        if code_col not in codes_df.columns:
            # 兼容可能的列名变体
            code_col = next(
                (c for c in codes_df.columns if "code" in str(c).lower() or "代码" in str(c)),
                None,
            )
        if code_col is None:
            return pd.DataFrame()
        code_list = codes_df[code_col].astype(str).tolist()
        parts: list[pd.DataFrame] = []
        # adata 批量接口建议单次不超过 500 只
        for i in range(0, len(code_list), 500):
            batch = code_list[i : i + 500]
            df = _adata.stock.market.list_market_current(code_list=batch)
            if df is not None and len(df) > 0:
                parts.append(df)
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        column_map = {
            "stock_code": "代码",
            "short_name": "名称",
            "price": "最新价",
            "change": "涨跌额",
            "change_pct": "涨跌幅",
            "volume": "成交量",
            "amount": "成交额",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        return df
    except Exception as e:  # noqa: BLE001
        logger.warning("adata all_spot 失败: %s", e)
        return pd.DataFrame()


def _fetch_daily_kline_adata(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """使用 adata 获取个股日 K，列名统一为 akshare 风格。"""
    if _adata is None:
        return pd.DataFrame()
    adjust_map = {"qfq": 1, "hfq": 2, "": 0}
    adjust_type = adjust_map.get(adjust, 1)
    try:
        df = _adata.stock.market.get_market(
            stock_code=symbol,
            start_date=start_date,
            end_date=end_date,
            k_type=1,
            adjust_type=adjust_type,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        column_map = {
            "stock_code": "股票代码",
            "trade_date": "日期",
            "open": "开盘",
            "close": "收盘",
            "high": "最高",
            "low": "最低",
            "volume": "成交量",
            "amount": "成交额",
            "change": "涨跌额",
            "change_pct": "涨跌幅",
            "turnover_ratio": "换手率",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        # adata 没有直接返回振幅，根据高低/昨收估算
        if "振幅" not in df.columns and "pre_close" in df.columns:
            if {"最高", "最低"}.issubset(df.columns):
                df["振幅"] = ((df["最高"] - df["最低"]) / df["pre_close"] * 100).round(2)
        # 统一日期格式为 akshare 风格的 YYYYMMDD 字符串
        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y%m%d")
        return df
    except Exception as e:  # noqa: BLE001
        logger.warning("adata daily_kline %s 失败: %s", symbol, e)
        return pd.DataFrame()


class MultiSourceDataLoader(DataLoader):
    """多源降级数据加载器。

    对 ``all_spot`` 和 ``daily_kline`` 实现 fallback：
        akshare → adata → 本地快照 → 过期缓存 → 空 DataFrame

    其余方法直接继承 ``DataLoader``，保持接口兼容。
    """

    def __init__(
        self,
        *args: Any,
        snapshot_dir: str | Path = "data/snapshots",
        **kwargs: Any,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # 允许 akshare 未安装时仍可通过 adata / 本地快照工作
        try:
            super().__init__(*args, **kwargs)
        except DataLoadError:
            if _adata is None:
                logger.warning("akshare 未安装且 adata 不可用，仅依赖本地快照/缓存")
            self.cache_dir = Path(kwargs.get("cache_dir", "data/cache"))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_ttl = timedelta(minutes=kwargs.get("cache_ttl_minutes", 30))
            self.retry_times = kwargs.get("retry_times", 3)
            self.backoff = kwargs.get("backoff_seconds", 2.0)
            self._fin_cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
            self._memory_cache: dict[str, tuple[datetime, Any]] = {}
            self._memory_cache_ttl = timedelta(minutes=60)

    # ------------------------------------------------------------------ #
    def all_spot(self) -> pd.DataFrame:
        """全市场实时行情，带多源 fallback。

        优先级：同花顺全市场(不封IP,5186只) → 腾讯批量(含PE/PB) → adata → 本地快照 → 过期缓存。
        主源同花顺 stock_fund_flow_individual，13秒出全市场，自带资金流，最稳。
        """
        # 0. 进程内内存缓存
        mem = self._mem_get("all_spot")
        if mem is not None and len(mem) > 0:
            return mem.copy()

        # 1. 新鲜文件缓存命中
        cached = self._cache_get("all_spot")
        if cached is not None and len(cached) > 0:
            self._mem_put("all_spot", cached)
            return cached.copy()

        # 2. 主源：同花顺全市场行情（不封IP，5186只，13秒，最稳）
        if os.environ.get("PANGU_TDX_FALLBACK", "1") != "0":
            try:
                from . import tdx_source
                df = tdx_source.ths_all_spot()
                if len(df) > 0:
                    logger.info("all_spot 主源同花顺：%d 条", len(df))
                    self._cache_put("all_spot", df)
                    self._mem_put("all_spot", df)
                    return df.copy()
            except Exception as e:  # noqa: BLE001
                logger.warning("同花顺全市场行情失败: %s", e)
            # 2b. 次选：腾讯批量行情（含PE/PB，但慢）
            try:
                df = tdx_source.tencent_all_spot()
                if len(df) > 0:
                    logger.info("all_spot fallback 腾讯：%d 条", len(df))
                    self._cache_put("all_spot", df)
                    self._mem_put("all_spot", df)
                    return df.copy()
            except Exception as e:  # noqa: BLE001
                logger.debug("腾讯 all_spot 失败: %s", e)

        if _adata is not None and os.environ.get("PANGU_TDX_FALLBACK", "1") != "0":
            df = _fetch_all_spot_adata()
            if len(df) > 0:
                logger.info("all_spot fallback 到 adata：%d 条", len(df))
                self._mem_put("all_spot", df)
                return df.copy()

        df = self._load_snapshot("all_spot")
        if df is not None and len(df) > 0:
            logger.warning("all_spot fallback 到本地快照：%d 条", len(df))
            self._mem_put("all_spot", df)
            return df.copy()

        stale = self._cache_get_stale("all_spot")
        if stale is not None and len(stale) > 0:
            logger.warning("all_spot fallback 到过期缓存：%d 条", len(stale))
            self._mem_put("all_spot", stale)
            return stale.copy()

        logger.error("all_spot 全部数据源不可用，返回空 DataFrame")
        return pd.DataFrame()

    def daily_kline(
        self,
        symbol: str,
        days: int = 60,
        adjust: str = "qfq",
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """个股日 K，带多源 fallback。

        优先级：akshare → mootdx(不封IP,不复权) → adata → 过期缓存。
        ⚠️ mootdx 返回不复权价，跨除权日精确盈亏需注意；趋势/均线/突破判断不受影响。
        """
        try:
            df = super().daily_kline(symbol, days=days, adjust=adjust, date=date)
            if len(df) > 0:
                return df
        except Exception as e:  # noqa: BLE001
            logger.warning("akshare daily_kline %s 失败，尝试 fallback: %s", symbol, e)

        # mootdx（通达信 TCP，不封 IP）——新浪/腾讯失败时的 K 线兜底
        if os.environ.get("PANGU_TDX_FALLBACK", "1") != "0":
            try:
                from . import tdx_source
                df = tdx_source.tdx_daily_kline(symbol, days=days)
                if len(df) > 0:
                    logger.info("daily_kline %s fallback 到 mootdx：%d 条(不复权)", symbol, len(df))
                    return df
            except Exception as e:  # noqa: BLE001
                logger.debug("mootdx daily_kline %s 失败: %s", symbol, e)

        end_dt = datetime.strptime(date, "%Y%m%d") if date else datetime.now()
        start = (end_dt - timedelta(days=days * 2)).strftime("%Y%m%d")
        end = end_dt.strftime("%Y%m%d")

        if _adata is not None and os.environ.get("PANGU_TDX_FALLBACK", "1") != "0":
            df = _fetch_daily_kline_adata(symbol, start, end, adjust)
            if len(df) > 0:
                logger.info("daily_kline %s fallback 到 adata：%d 条", symbol, len(df))
                if len(df) > days:
                    df = df.tail(days).reset_index(drop=True)
                return df

        # 本地快照不含个股 K 线，但可用最新快照日作为参考；继续尝试过期缓存
        _ = self._load_snapshot("all_spot", as_of_date=date)

        cache_key = f"kline:{symbol}:{adjust}:{start}:{end}"
        stale = self._cache_get_stale(cache_key)
        if stale is not None and len(stale) > 0:
            logger.warning("daily_kline %s fallback 到过期缓存：%d 条", symbol, len(stale))
            return stale

        logger.error("daily_kline %s 全部数据源不可用，返回空 DataFrame", symbol)
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    def _load_snapshot(
        self, name: str, as_of_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """读取指定日期或之前最新的本地快照。"""
        if as_of_date is None:
            target = datetime.now()
        else:
            target = (
                datetime.strptime(as_of_date, "%Y%m%d")
                if "-" not in as_of_date
                else datetime.strptime(as_of_date, "%Y-%m-%d")
            )

        latest: Optional[tuple[datetime, Path]] = None
        for d in self.snapshot_dir.iterdir():
            if not d.is_dir():
                continue
            try:
                dt = datetime.strptime(d.name, "%Y-%m-%d")
            except ValueError:
                continue
            if dt <= target and (latest is None or dt > latest[0]):
                latest = (dt, d)

        if latest is None:
            return None
        path = latest[1] / f"{name}.parquet"
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001
            logger.debug("读取快照 %s 失败: %s", path, e)
            return None
