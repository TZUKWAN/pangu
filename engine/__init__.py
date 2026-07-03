"""盘古 Pangu — A股短线「情绪+趋势」选股引擎。

模块组成：
- data_loader:    akshare 取数（带重试/降级/缓存）
- sentiment_meter: 情绪温度计 0-100（攻防姿态）
- trend_scanner:  趋势选股（板块 RPS 轮动 + 个股形态 + 资金流入）
- quant_guard:    量化护栏（排雷/估值过滤）
- pipeline:       串联 ①情绪→②趋势→③护栏→候选池
- report:         候选池 → Markdown 简报
- cli:            命令行入口

数据层基于 akshare（免费、无 token、5/5 覆盖 A 股短线所需数据）。
"""

__version__ = "0.1.0"
