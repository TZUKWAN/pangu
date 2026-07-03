"""第三数据源：通达信(mootdx) + 腾讯财经 直连，彻底绕过东方财富封 IP。

来源：a-stock-data 工具包（github.com/simonlin1212/a-stock-data）的核心能力提取。

为什么需要它
------------
akshare / adata 的很多接口最终也走东方财富(push2.eastmoney.com)，东财对高频
请求会封 IP。本模块用两个**不封 IP**的数据源作第一优先级：
- mootdx（通达信 TCP 7709 二进制协议）：K线、实时报价、逐笔。TCP 协议实测不封。
- 腾讯财经（qt.gtimg.cn HTTP GBK）：PE/PB/市值/换手率/涨跌停。HTTP 实测不封。
- 腾讯前复权 K 线（web.ifzq.gtimg.cn）：用于均线/突破/市场广度计算。

能覆盖选股所需的核心数据：
- 全市场实时行情（腾讯批量拉，含 PE/PB/市值/换手率/涨跌停）→ 替代 all_spot
- 个股日 K 线（mootdx 不复权 / 腾讯前复权）→ 替代 daily_kline
- 指数 K 线（腾讯前复权）→ 替代指数历史接口
- 已去除对东财独家数据（龙虎榜/涨停池/资金流）的依赖，改走同花顺/腾讯/新浪。

设计：所有函数失败返回空，不抛异常，让上层(data_loader)做 fallback。
"""

from __future__ import annotations

import logging
import socket
import time
import random
import urllib.request
from typing import Any, Optional

import pandas as pd

from . import _proxy_patch  # noqa: F401

logger = logging.getLogger("pangu.tdx_source")

# ---------------------------------------------------------------------- #
# mootdx 客户端（规避 0.11.x BESTIP 空串 bug）
# ---------------------------------------------------------------------- #
_TDX_SERVERS = [
    ("119.97.185.59", 7709), ("124.70.133.119", 7709), ("116.205.183.150", 7709),
    ("123.60.73.44", 7709), ("116.205.163.254", 7709), ("121.36.225.169", 7709),
    ("123.60.70.228", 7709), ("124.71.9.153", 7709), ("110.41.147.114", 7709),
    ("124.71.187.122", 7709),
]

_client: Any = None
_client_failed = False  # 标记是否已确认 mootdx 不可用（海外网络等），避免反复探测


def _probe(ip: str, port: int, timeout: float = 1.0) -> bool:
    """TCP 握手探测，判断服务器是否可达。"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False


def get_tdx_client() -> Any:
    """获取 mootdx 客户端单例（规避 BESTIP bug，顺序探测可用服务器）。

    海外网络通常全部超时（TCP 7709），会快速失败并标记不可用。
    为避免阻塞主流程，最多探测前 5 个服务器（每个1秒），4秒内无可用则放弃。
    """
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        return None
    try:
        from mootdx.quotes import Quotes
    except ImportError:
        logger.debug("mootdx 未安装，tdx 数据源不可用")
        _client_failed = True
        return None

    # 最多探测前 5 个服务器，避免弱网/测试环境长时间阻塞
    for ip, port in _TDX_SERVERS[:5]:
        if _probe(ip, port, timeout=1.0):
            try:
                _client = Quotes.factory(market="std", server=(ip, port))
                logger.info("mootdx 已连接服务器 %s:%d", ip, port)
                return _client
            except Exception as e:  # noqa: BLE001
                logger.debug("mootdx 连接 %s 失败: %s", ip, e)
                continue
    # 全部服务器不可达，回退 bestip
    try:
        _client = Quotes.factory(market="std", bestip=True)
        return _client
    except Exception:
        pass
    try:
        _client = Quotes.factory(market="std")
        return _client
    except Exception as e:  # noqa: BLE001
        logger.warning("mootdx 所有服务器不可达（海外网络通常如此）: %s", e)
        _client_failed = True
        return None


# ---------------------------------------------------------------------- #
# mootdx 日 K 线（不复权原始价）
# ---------------------------------------------------------------------- #
def tdx_daily_kline(symbol: str, days: int = 60) -> pd.DataFrame:
    """用 mootdx 取日 K 线。

    ⚠️ 返回【不复权】原始价（通达信原始数据）。跨除权日做估值/回测需自行复权。
       选股场景（看趋势/均线/突破）用不复权价影响不大；精确盈亏用前复权源。
    返回列（统一为 akshare 风格中文列名）：日期/开盘/收盘/最高/最低/成交量/成交额
    """
    client = get_tdx_client()
    if client is None:
        return pd.DataFrame()
    try:
        # frequency=9 是日线（mootdx 0.11.x 实测值表）
        df = client.bars(symbol=symbol, frequency=9, offset=days)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        # mootdx 返回列：open/close/high/low/vol/amount/datetime（英文）
        col_map = {
            "datetime": "日期", "open": "开盘", "close": "收盘",
            "high": "最高", "low": "最低", "vol": "成交量", "amount": "成交额",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y%m%d")
        return df.reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        logger.debug("mootdx K线 %s 失败: %s", symbol, e)
        return pd.DataFrame()


# ---------------------------------------------------------------------- #
# 腾讯财经实时行情（不封 IP，含 PE/PB/市值/换手率/涨跌停）
# ---------------------------------------------------------------------- #
def _tencent_prefix(code: str) -> str:
    """6位代码 → 腾讯市场前缀。"""
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith("8"):
        return f"bj{code}"
    return f"sz{code}"


def tencent_quote(codes: list[str]) -> dict[str, dict[str, Any]]:
    """批量拉腾讯财经实时行情。

    返回 {code: {name, price, pe_ttm, pb, mcap_yi, float_mcap_yi, turnover_pct, ...}}
    字段索引已实测校准（2026-05）：39=PE_TTM, 44=总市值亿, 45=流通市值亿, 46=PB。
    """
    if not codes:
        return {}
    prefixed = [_tencent_prefix(c) for c in codes]
    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        resp = urllib.request.urlopen(req, timeout=10)
        data = resp.read().decode("gbk")
    except Exception as e:  # noqa: BLE001
        logger.debug("腾讯行情取数失败: %s", e)
        return {}

    result: dict[str, dict[str, Any]] = {}
    for line in data.strip().split(";"):
        line = line.strip()
        if not line or "=" not in line or '"' not in line:
            continue
        try:
            key = line.split("=")[0].split("_")[-1]
            vals = line.split('"')[1].split("~")
            if len(vals) < 53:
                continue
            code = key[2:]
            result[code] = {
                "name": vals[1],
                "price": float(vals[3]) if vals[3] else 0,
                "last_close": float(vals[4]) if vals[4] else 0,
                "open": float(vals[5]) if vals[5] else 0,
                "change_amt": float(vals[31]) if vals[31] else 0,
                "change_pct": float(vals[32]) if vals[32] else 0,
                "high": float(vals[33]) if vals[33] else 0,
                "low": float(vals[34]) if vals[34] else 0,
                "amount_wan": float(vals[37]) if vals[37] else 0,
                "turnover_pct": float(vals[38]) if vals[38] else 0,
                "pe_ttm": float(vals[39]) if vals[39] else 0,
                "amplitude_pct": float(vals[43]) if vals[43] else 0,
                "mcap_yi": float(vals[44]) if vals[44] else 0,
                "float_mcap_yi": float(vals[45]) if vals[45] else 0,
                "pb": float(vals[46]) if vals[46] else 0,
                "limit_up": float(vals[47]) if vals[47] else 0,
                "limit_down": float(vals[48]) if vals[48] else 0,
                "vol_ratio": float(vals[49]) if vals[49] else 0,
            }
        except (ValueError, IndexError):
            continue
    return result


def ths_all_spot() -> pd.DataFrame:
    """用同花顺个股资金流接口拼全市场实时行情（替代 all_spot）。

    同花顺 stock_fund_flow_individual 返回 5186 只全市场股票，
    自带 最新价/涨跌幅/换手率/资金流/成交额，是国内最稳的免认证全市场快照。
    缺 PE/PB/市值（如需可后续用腾讯批量补，但选股核心不依赖这些）。
    """
    import akshare as ak
    try:
        df = ak.stock_fund_flow_individual(symbol="即时")
    except Exception as e:  # noqa: BLE001
        logger.debug("同花顺个股资金流失败: %s", e)
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    # 标准化列名为 akshare 中文风格
    col_map = {
        "股票代码": "代码", "股票简称": "名称", "最新价": "最新价",
        "涨跌幅": "涨跌幅", "换手率": "换手率", "成交额": "成交额",
        "净额": "主力净流入-净额",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    # 数值化（涨跌幅/换手率是 "20.09%" 字符串）
    for c in ["涨跌幅", "换手率"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace("%", ""), errors="coerce")
    if "最新价" in df.columns:
        df["最新价"] = pd.to_numeric(df["最新价"], errors="coerce")
    # 代码统一为字符串
    if "代码" in df.columns:
        df["代码"] = df["代码"].astype(str).str.zfill(6).str.strip()
    # 流通市值缺失，用成交额近似占位（选股市值过滤会用，但只是粗筛）
    if "流通市值" not in df.columns:
        df["流通市值"] = 0
    logger.info("同花顺全市场行情：%d 只", len(df))
    return df


def tencent_all_spot(timeout_sec: float = 20.0) -> pd.DataFrame:
    """用腾讯财经拼全市场实时行情（替代 all_spot）。

    需要 adata 提供 全市场代码列表（不封IP），再批量调腾讯行情。
    返回列统一为 akshare 风格：代码/名称/最新价/涨跌幅/换手率/市盈率-动态/市净率/流通市值

    注意：全市场批量拉取较慢（5970只分100只/批）。用 timeout_sec 限制总耗时，
    避免在 mock/测试或弱网环境卡死；超时返回已取到的部分。
    """
    try:
        import adata
    except Exception as e:  # noqa: BLE001
        logger.debug("adata 未安装，腾讯全市场行情不可用: %s", e)
        return pd.DataFrame()

    # 用线程隔离取代码列表，避免 adata 内部联网卡住主流程
    import threading
    codes_box: list[list] = [[]]  # codes_box[0] = code list
    def _get_codes():
        try:
            codes_df = adata.stock.info.all_code()
            if codes_df is None or codes_df.empty:
                return
            code_col = "stock_code" if "stock_code" in codes_df.columns else codes_df.columns[0]
            codes_box[0] = codes_df[code_col].astype(str).tolist()
        except Exception as e:  # noqa: BLE001
            logger.debug("adata 全市场代码列表失败: %s", e)

    t = threading.Thread(target=_get_codes, daemon=True)
    t.start()
    t.join(timeout=8.0)  # 代码列表最多等 8 秒
    all_codes = codes_box[0]
    if not all_codes:
        return pd.DataFrame()

    rows = []
    deadline = time.time() + timeout_sec
    # 腾讯批量接口建议单次不超过 100 只
    for i in range(0, len(all_codes), 100):
        if time.time() > deadline:
            logger.debug("腾讯全市场行情超时，已取 %d 只", len(rows))
            break
        batch = all_codes[i:i + 100]
        quotes = tencent_quote(batch)
        for code, q in quotes.items():
            if q["price"] <= 0:
                continue
            rows.append({
                "代码": code,
                "名称": q["name"],
                "最新价": q["price"],
                "涨跌幅": q["change_pct"],
                "涨跌额": q["change_amt"],
                "成交量": 0,  # 腾讯批量不返回成交量，置0
                "成交额": q["amount_wan"] * 1e4,
                "振幅": q["amplitude_pct"],
                "最高": q["high"],
                "最低": q["low"],
                "今开": q["open"],
                "昨收": q["last_close"],
                "量比": q["vol_ratio"],
                "换手率": q["turnover_pct"],
                "市盈率-动态": q["pe_ttm"],
                "市净率": q["pb"],
                "总市值": q["mcap_yi"] * 1e8,
                "流通市值": q["float_mcap_yi"] * 1e8,
            })
        # 批量间小延迟，礼貌请求
        if i + 100 < len(all_codes):
            time.sleep(0.2)
    if not rows:
        return pd.DataFrame()
    logger.info("腾讯全市场行情：%d 只", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# 腾讯前复权日 K 线（不封 IP，web.ifzq.gtimg.cn）
# ---------------------------------------------------------------------- #
def tencent_kline_qfq(symbol: str, days: int = 60, adjust: str = "qfq") -> pd.DataFrame:
    """腾讯前复权日 K 线。

    走 web.ifzq.gtimg.cn/appstock/app/fqkline/get（不封 IP），返回前复权价，
    比 mootdx 的不复权原始价更适合做均线/突破判断。
    返回列（统一为 akshare 中文风格）：日期/股票代码/开盘/收盘/最高/最低/成交量/成交额
    """
    import urllib.request
    import json as _json
    code = str(symbol).strip()
    # 若已带市场前缀（如指数 sh000001）直接用，否则用 _tencent_prefix 补全
    if code[:2] in ("sh", "sz", "bj"):
        full_code = code
    else:
        full_code = _tencent_prefix(code)
    # fqkline 接口 param 格式：市场前缀代码,周期(=day),起始,结束,数量,复权类型
    # 实测有效格式：param=sz000001,day,,,30,qfq
    fqtype = {"qfq": "qfq", "hfq": "hfq", "": ""}.get(adjust, "qfq")
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={full_code},day,,,{days},{fqtype}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://gu.qq.com/",
        })
        resp = urllib.request.urlopen(req, timeout=10)
        data = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.debug("腾讯前复权K线 %s 失败: %s", code, e)
        return pd.DataFrame()

    # 响应结构：data -> {full_code} -> {qfqday/day: [[date, open, close, high, low, volume], ...]}
    body = data.get("data") or {}
    if not isinstance(body, dict):
        return pd.DataFrame()
    stock_data = body.get(full_code) or {}
    # 复权字段优先 qfqday，其次 day
    rows = stock_data.get("qfqday") or stock_data.get("day") or []
    if not rows:
        return pd.DataFrame()
    out = []
    for r in rows:
        # r = [date, open, close, high, low, volume, ?]
        if len(r) < 6:
            continue
        out.append({
            "日期": str(r[0]),
            "股票代码": code,
            "开盘": safe_parse(r[1]),
            "收盘": safe_parse(r[2]),
            "最高": safe_parse(r[3]),
            "最低": safe_parse(r[4]),
            "成交量": safe_parse(r[5]),
        })
    if not out:
        return pd.DataFrame()
    logger.debug("腾讯前复权K线 %s：%d 条", code, len(out))
    return pd.DataFrame(out)


def safe_parse(v: Any) -> float:
    """安全转 float，失败返回 0。"""
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return 0.0
