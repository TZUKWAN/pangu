"""每日收盘快照：把关键数据表落地为 parquet，便于后续回测与 fallback。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader

logger = logging.getLogger("pangu.snapshot")


@dataclass
class SnapshotResult:
    """一次快照构建的结果。"""

    date_dir: str
    snapshot_dir: Path
    rows: dict[str, int] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date_dir": self.date_dir,
            "snapshot_dir": str(self.snapshot_dir),
            "rows": self.rows,
            "paths": {k: str(v) for k, v in self.paths.items()},
            "errors": self.errors,
        }


class SnapshotBuilder:
    """收盘后抓取关键数据表并保存到 data/snapshots/YYYY-MM-DD/ 下。

    保存内容：
        - all_spot            全市场实时行情
        - concept_boards      概念板块列表 + 涨跌幅
        - limit_up_pool       涨停股池
        - broke_pool          炸板池
        - limit_down_pool     跌停池
        - sector_fund_flow_rank  板块资金流排名（今日）
    """

    def __init__(
        self,
        dl: DataLoader,
        snapshot_dir: str | Path = "data/snapshots",
    ) -> None:
        self.dl = dl
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def build(self, date: Optional[str] = None) -> SnapshotResult:
        """抓取并保存指定日期的快照。date 支持 YYYYMMDD 或 YYYY-MM-DD，默认今天。"""
        yyyymmdd, ymd_dir = self._normalize_date(date)
        day_dir = self.snapshot_dir / ymd_dir
        day_dir.mkdir(parents=True, exist_ok=True)

        result = SnapshotResult(date_dir=ymd_dir, snapshot_dir=self.snapshot_dir)

        # 定义要保存的表：名称 -> 取数函数
        fetchers = {
            "all_spot": lambda: self.dl.all_spot(),
            "concept_boards": lambda: self.dl.concept_boards(),
            "limit_up_pool": lambda: self.dl.limit_up_pool(date=yyyymmdd),
            "broke_pool": lambda: self.dl.broke_pool(date=yyyymmdd),
            "limit_down_pool": lambda: self.dl.limit_down_pool(date=yyyymmdd),
            "sector_fund_flow_rank": lambda: self.dl.sector_fund_flow_rank(indicator="今日"),
        }

        for name, fetcher in fetchers.items():
            path = day_dir / f"{name}.parquet"
            try:
                df = fetcher()
                if df is None:
                    df = pd.DataFrame()
                df.to_parquet(path, index=False)
                result.rows[name] = len(df)
                result.paths[name] = path
                logger.info("快照 %s/%s 已保存：%d 条", ymd_dir, name, len(df))
            except Exception as e:  # noqa: BLE001
                # 单表失败不应阻断其余表的保存
                msg = f"{name}: {e}"
                logger.warning("快照 %s 失败：%s", ymd_dir, msg)
                result.errors.append(msg)
                # 仍写入空文件，保持目录结构一致
                try:
                    pd.DataFrame().to_parquet(path, index=False)
                    result.paths[name] = path
                except Exception as write_err:  # noqa: BLE001
                    result.errors.append(f"{name} 空文件写入失败: {write_err}")

        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_date(date: Optional[str]) -> tuple[str, str]:
        """把输入日期统一为 (YYYYMMDD, YYYY-MM-DD) 元组。"""
        if date is None:
            d = datetime.now()
        elif "-" in date:
            d = datetime.strptime(date, "%Y-%m-%d")
        else:
            d = datetime.strptime(date, "%Y%m%d")
        return d.strftime("%Y%m%d"), d.strftime("%Y-%m-%d")
