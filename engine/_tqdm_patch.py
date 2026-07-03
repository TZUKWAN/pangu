"""全局禁用 tqdm 进度条，避免污染 CLI/REPL 输出。

akshare 内部多处使用 `from tqdm import tqdm` 或 `akshare.utils.tqdm.get_tqdm()`，
且默认 enable=True；仅在 import 前设置 TQDM_DISABLE 不够可靠。

本模块在解释器最早时机（data_loader 导入 akshare 之前）把 `tqdm` 替换为
始终禁用显示的 SilentTqdm，从而覆盖所有入口（CLI、REPL、测试、Notebook）。
"""

from __future__ import annotations

import os
import sys

# 同时保留环境变量，给遵守该变量的库使用
os.environ.setdefault("TQDM_DISABLE", "1")


try:
    import tqdm
    import tqdm.std
except ImportError:  # pragma: no cover
    tqdm = None

if tqdm is not None:
    class SilentTqdm(tqdm.std.tqdm):
        """永远禁用显示的 tqdm 子类。"""

        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    # 替换所有可能被 akshare 内部 `from tqdm import tqdm` 拿到的引用
    tqdm.tqdm = SilentTqdm
    tqdm.std.tqdm = SilentTqdm
    sys.modules["tqdm"].tqdm = SilentTqdm
