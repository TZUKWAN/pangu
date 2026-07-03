"""独立 Agent 的命令行入口。

用法：
    python -m engine.agent.cli "明天A股帮我关注3-5只短线票"
    python -m engine.agent.cli --config config/settings.yaml "000001 平安银行能买吗"
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ..config import load_config
from .core import PanguAgent

console = Console()
logger = logging.getLogger("pangu.agent.cli")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pangu-agent",
        description="盘古独立 Agent — 直接调用 LLM API 选股",
    )
    parser.add_argument("question", nargs="?", default="明天A股帮我关注3-5只短线票", help="向 Agent 提问")
    parser.add_argument("-c", "--config", default=None, help="配置文件路径")
    parser.add_argument("-d", "--date", default=None, help="日期 YYYYMMDD（默认今天）")
    parser.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config(args.config)

    try:
        agent = PanguAgent.from_config(cfg)
    except ValueError as e:
        console.print(Panel(f"[red]配置错误：{e}[/]", title="⚠️ 未配置 LLM", border_style="red"))
        console.print("""
请在 config/settings.yaml 中添加 llm 配置段，例如：

llm:
  api_key: "sk-..."            # 或设置环境变量 PANGU_LLM_API_KEY
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  timeout: 300
""")
        return 1

    console.print(f"\n[bold magenta]🤖 盘古 Agent[/] 正在分析：{args.question}\n")
    try:
        answer = agent.run(args.question, date=args.date)
    except Exception as e:  # noqa: BLE001
        logger.exception("Agent 执行失败")
        console.print(Panel(f"[red]{e}[/]", title="✗ 执行失败", border_style="red"))
        return 1

    console.print(Panel(Markdown(answer), title="🤖 盘古 Agent 选股报告", border_style="magenta"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
