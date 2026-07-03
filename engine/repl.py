"""盘古 Pangu — 交互式 CLI（REPL）。

这是面向用户的主交互入口：rich 美化的菜单驱动终端，无需 HTML。
直接运行：python -m engine.repl

功能菜单：
  1. 📊 今日市场速览      — 情绪温度 + 热门板块 + 一句话建议
  2. 🎯 选股（完整链路）  — 情绪→趋势→护栏→买卖点，输出候选股
  3. 🤖 AI 选股（LLM综合）— 独立 Agent 调工具做深度分析
  4. 💼 持仓管理          — 记录买卖、查实时盈亏、胜率统计
  5. 🔬 回测              — 验证策略历史有效性
  6. 📰 读财经简报        — 最新一期 capitalise-finnews 简报
  7. ⚙️  RPS 预计算       — 盘后跑一次，更新真实 RPS
  0. 退出

设计原则：
- 所有耗时操作带 spinner/进度提示
- 网络错误友好降级（不裸露堆栈）
- 中文界面，A 股术语
- 单只票深度查可随时触发
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box

from .config import build_data_loader, load_config

console = Console()
logger = logging.getLogger("pangu.repl")


# ====================================================================== #
# 渲染辅助
# ====================================================================== #
def _temp_bar(temp: float, width: int = 20) -> str:
    """情绪温度条：用色块直观显示 0-100。"""
    filled = int(temp / 100 * width)
    if temp < 40:
        color = "cyan"
    elif temp > 85:
        color = "red"
    else:
        color = "green"
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] {temp:.1f}"


def show_banner() -> None:
    console.print(Align.center(
        Text("盘 古 PANGU", style="bold magenta", justify="center"),
        vertical="middle",
    ))
    console.print(Align.center(Text("A股短线 · 情绪+趋势 选股系统", style="dim")))
    console.print()


def show_sentiment(s: dict) -> None:
    """渲染情绪温度计。"""
    comp = s.get("components", {})
    temp = s.get("temperature", 0)
    posture = s.get("posture", "?")

    # 温度面板
    panel_body = Text.assemble(
        _temp_bar(temp), "\n",
        Text(f"姿态：{posture}", style="bold yellow"), "\n\n",
        Text(s.get("advice", ""), style="italic"),
    )
    console.print(Panel(panel_body, title="[bold]📊 情绪温度计[/]", border_style="blue"))

    # 分项表
    t = Table(title="情绪分项", box=box.SIMPLE_HEAVY, show_lines=False)
    t.add_column("指标", style="cyan")
    t.add_column("数值", justify="right")
    t.add_column("得分", justify="right", style="green")
    rows = [
        ("涨停家数", comp.get("limit_up_count", 0), comp.get("limit_up_score", 0)),
        ("最高连板", f"{comp.get('consecutive_height', 0)}板", comp.get("consecutive_score", 0)),
        ("炸板率", f"{comp.get('broke_rate', 0)*100:.1f}%", comp.get("broke_rate_score", 0)),
        ("跌停家数", comp.get("limit_down_count", 0), comp.get("limit_down_score", 0)),
        ("涨/跌", f"{comp.get('advance', 0)}/{comp.get('decline', 0)}", comp.get("advance_decline_score", 0)),
    ]
    for name, val, score in rows:
        t.add_row(name, str(val), f"{score:.0f}")
    console.print(t)


def show_candidates(cands: list[dict], title: str = "今日推荐") -> None:
    """渲染推荐股表（含推荐度/上涨概率/预测涨幅/简短理由/买卖点）。"""
    from .recommender import grade_color
    if not cands:
        console.print(Panel("[dim]无推荐股（情绪冰点或无符合趋势的标的）[/]", title=title, border_style="yellow"))
        return
    t = Table(title=title, box=box.ROUNDED, show_lines=True)
    t.add_column("等级", justify="center")
    t.add_column("代码", style="dim")
    t.add_column("名称", style="bold")
    t.add_column("推荐度", justify="right")
    t.add_column("上涨概率", justify="right")
    t.add_column("预测涨幅", justify="right")
    t.add_column("盈亏比", justify="right")
    t.add_column("买点", justify="right", style="yellow")
    t.add_column("止损", justify="right", style="red")
    t.add_column("理由", overflow="fold", style="cyan")

    for c in cands:
        rec = c.get("recommend", {})
        ee = c.get("entry_exit", {})
        grade = rec.get("grade", "?")
        grade_str = Text(grade, style=grade_color(grade))
        tp = rec.get("target_pct", [0, 0])
        buy = rec.get("buy_point", "-")
        stop = rec.get("stop_loss", "-")
        calib = "✓" if rec.get("calibrated") else ""
        t.add_row(
            grade_str,
            c.get("code", ""), c.get("name", ""),
            f"{rec.get('recommend_score', 0):.1f}",
            f"{rec.get('up_prob', 0):.0f}%{calib}",
            f"{tp[0]}-{tp[1]}%",
            f"{rec.get('risk_reward_ratio', 0):.1f}",
            str(buy), str(stop),
            rec.get("tag", ""),
        )
    console.print(t)
    console.print("[dim]✓=概率已校准（无✓为模型估算，参考用）  等级：S优/A佳/B可/C慎[/]")


def show_entry_exit_detail(ee: dict, code: str, name: str) -> None:
    """渲染单只票的买卖点详情。"""
    lines = [Text(f"  代码 {code}  名称 {name}", style="bold")]
    # 买点
    lines.append(Text("\n  买点：", style="yellow"))
    for bp in ee.get("buy_points", []):
        mark = "★" if bp.get("is_primary") else " "
        lines.append(Text(f"    {mark} {bp['price']}  ({bp['type']})  {bp.get('condition', '')}"))
    # 止损
    sl = ee.get("stop_loss", {})
    lines.append(Text(f"\n  止损：{sl.get('price', '-')}  ({sl.get('method', '-')})", style="red"))
    # 止盈
    lines.append(Text("\n  止盈：", style="green"))
    for tp in ee.get("take_profit", []):
        lines.append(Text(f"    {tp['price']}  ({tp['method']})"))
    # 仓位
    pos = ee.get("position", {})
    lines.append(Text(
        f"\n  仓位建议：{pos.get('shares', 0)}股  风险{pos.get('risk_pct', 0)}%  "
        f"情绪系数{pos.get('emotion_factor', 0)}", style="cyan"))
    lines.append(Text(f"  盈亏比：{ee.get('risk_reward_ratio', '-')}", style="bold magenta"))
    console.print(Panel(Text.join("\n", lines) if hasattr(Text, "join") else Text("\n".join(str(x) for x in lines)),
                        title=f"🎯 {name} 交易计划", border_style="green"))


# ====================================================================== #
# 命令实现
# ====================================================================== #
def cmd_sentiment(cfg: dict) -> None:
    """今日市场速览。"""
    console.print()
    with console.status("[bold cyan]正在计算市场情绪…", spinner="dots"):
        try:
            from .sentiment_meter import SentimentMeter
            dl = build_data_loader(cfg)
            bd = SentimentMeter(dl, cfg.get("sentiment", {})).measure()
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]✗ 情绪计算失败：{e}[/]")
            return
    show_sentiment(bd.to_dict())
    console.print()


def cmd_scan(cfg: dict) -> None:
    """完整选股链路。"""
    console.print()
    with console.status("[bold cyan]选股中（情绪→趋势→护栏→买卖点，约2-4分钟）…", spinner="moon"):
        try:
            from .pipeline import Pipeline
            pipe = Pipeline(
                dl=build_data_loader(cfg),
                sentiment_cfg=cfg.get("sentiment", {}),
                trend_cfg=cfg.get("trend", {}),
                guard_cfg=cfg.get("guard", {}),
                entry_exit_cfg=cfg.get("entry_exit", cfg),
                pick_count=cfg.get("output", {}).get("pick_count", 5),
                db_path=cfg.get("output", {}).get("db_path", "data/pangu.db"),
            )
            result = pipe.run()
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]✗ 选股失败：{e}[/]")
            return

    s = result.to_dict()["sentiment"]
    show_sentiment(s)
    console.print()
    # 热门板块
    if result.boards:
        bt = Table(title="🔥 热门板块", box=box.SIMPLE)
        bt.add_column("板块", style="cyan")
        bt.add_column("涨跌%", justify="right")
        bt.add_column("主力净流入(万)", justify="right")
        for b in result.boards[:6]:
            bt.add_row(b["name"], str(b["pct"]), str(b.get("fund_net_wan", "-")))
        console.print(bt)
        console.print()
    show_candidates(result.to_dict()["candidates"])
    console.print()

    # 提示深度查看
    if result.candidates:
        ans = Prompt.ask("\n[dim]输入代码深度查看某只票买卖点（回车跳过）", default="")
        if ans.strip():
            _show_one_detail(result.to_dict()["candidates"], ans.strip())


def _show_one_detail(cands: list[dict], code: str) -> None:
    for c in cands:
        if c.get("code") == code:
            ee = c.get("entry_exit", {})
            if ee:
                show_entry_exit_detail(ee, c["code"], c["name"])
            else:
                console.print("[yellow]该股无买卖点数据[/]")
            return
    console.print(f"[yellow]未在候选池找到 {code}[/]")


def cmd_ai_pick(cfg: dict) -> None:
    """AI 选股（独立 Agent）。"""
    console.print()
    q = Prompt.ask("[bold]向盘古AI提问[/]", default="今天A股帮我选3-5只短线票")
    console.print(f"\n[dim]🤖 盘古 Agent 调用中（请确保 config/settings.yaml 已配置 llm）…[/]")
    with console.status("[bold magenta]AI 深度分析中…", spinner="aesthetic"):
        try:
            from .agent.core import PanguAgent
            agent = PanguAgent.from_config(cfg)
            output = agent.run(q)
        except ValueError as e:
            console.print(f"[red]✗ 配置错误：{e}[/]")
            console.print("""
[dim]请在 config/settings.yaml 中添加 llm 配置段：[/]
[dim]
llm:
  api_key: "sk-..."
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  timeout: 300
""")
            return
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]✗ AI 调用失败：{e}[/]")
            return
    console.print(Panel(output or "(无输出)", title="[bold]🤖 盘古AI 选股报告[/]",
                        border_style="magenta"))
    console.print()


def cmd_portfolio(cfg: dict) -> None:
    """持仓管理。"""
    try:
        from .portfolio import PortfolioTracker
    except ImportError:
        console.print("[red]✗ 持仓模块未安装[/]")
        return
    tracker = PortfolioTracker(cfg.get("output", {}).get("db_path", "data/pangu.db"))
    _portfolio_menu(tracker, cfg)


def _portfolio_menu(tracker, cfg: dict) -> None:
    while True:
        console.print(Panel(
            "[1] 查看持仓(实时盈亏)  [2] 买入  [3] 卖出  [4] 交易明细  [5] 总览统计  [0] 返回",
            title="💼 持仓管理", border_style="cyan"))
        choice = Prompt.ask("选择", default="0")
        try:
            if choice == "0":
                return
            elif choice == "1":
                with console.status("查实时盈亏…", spinner="dots"):
                    holdings = tracker.current_holdings()
                _render_holdings(holdings)
            elif choice == "2":
                code = Prompt.ask("代码")
                name = Prompt.ask("名称", default=code)
                shares = IntPrompt.ask("数量")
                price = float(Prompt.ask("买入价"))
                reason = Prompt.ask("理由", default="")
                tracker.record_buy(code, name, shares, price, datetime.now().strftime("%Y%m%d"), reason)
                console.print(f"[green]✓ 已记录买入 {name} {shares}股@{price}[/]")
            elif choice == "3":
                code = Prompt.ask("代码")
                shares = IntPrompt.ask("数量")
                price = float(Prompt.ask("卖出价"))
                pnl = tracker.record_sell(code, shares, price, datetime.now().strftime("%Y%m%d"))
                console.print(f"[green]✓ 已卖出，这笔盈亏：{pnl:+.2f}[/]")
            elif choice == "4":
                txns = tracker.transactions()
                _render_transactions(txns)
            elif choice == "5":
                summ = tracker.summary()
                _render_summary(summ)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]操作失败：{e}[/]")
        console.print()


def _render_holdings(holdings) -> None:
    """渲染持仓列表（dataclass 列表 → rich Table）。"""
    if not holdings:
        console.print("[dim]暂无持仓[/]")
        return
    t = Table(title="💼 当前持仓", box=box.ROUNDED, show_lines=True)
    t.add_column("代码", style="dim")
    t.add_column("名称", style="bold")
    t.add_column("持仓", justify="right")
    t.add_column("成本", justify="right")
    t.add_column("现价", justify="right")
    t.add_column("市值", justify="right")
    t.add_column("浮盈亏", justify="right")
    t.add_column("盈亏%", justify="right")
    for h in holdings:
        d = h.to_dict() if hasattr(h, "to_dict") else (h if isinstance(h, dict) else {})
        pnl = d.get("unrealized_pnl", d.get("pnl", 0))
        pnl_pct = d.get("pnl_pct", d.get("profit_pct", 0))
        color = "red" if pnl < 0 else "green"
        t.add_row(
            str(d.get("code", "")), str(d.get("name", "")),
            str(d.get("shares", 0)), f"{d.get('avg_cost', 0):.2f}",
            f"{d.get('current_price', d.get('price', 0)):.2f}",
            f"{d.get('market_value', 0):.0f}",
            f"[{color}]{pnl:+.0f}[/{color}]",
            f"[{color}]{pnl_pct:+.1f}%[/{color}]",
        )
    console.print(t)


def _render_transactions(txns) -> None:
    if not txns:
        console.print("[dim]暂无交易记录[/]")
        return
    t = Table(title="📜 交易明细", box=box.SIMPLE)
    t.add_column("日期", style="dim")
    t.add_column("代码")
    t.add_column("名称")
    t.add_column("操作")
    t.add_column("数量", justify="right")
    t.add_column("价格", justify="right")
    t.add_column("盈亏", justify="right")
    for tx in txns[-20:]:  # 最近20条
        d = tx.to_dict() if hasattr(tx, "to_dict") else (tx if isinstance(tx, dict) else {})
        pnl = d.get("pnl", 0)
        pnl_str = f"{pnl:+.0f}" if pnl else "-"
        action = d.get("action", "")
        a_color = "green" if action == "buy" else "yellow"
        t.add_row(
            str(d.get("date", "")), str(d.get("code", "")), str(d.get("name", "")),
            f"[{a_color}]{action}[/{a_color}]", str(d.get("shares", 0)),
            f"{d.get('price', 0):.2f}", pnl_str,
        )
    console.print(t)


def _render_summary(summ: dict) -> None:
    """渲染总览统计（dict → rich Panel）。"""
    if not isinstance(summ, dict):
        console.print(summ)
        return
    pnl = summ.get("total_pnl", summ.get("realized_pnl", 0))
    color = "red" if pnl < 0 else "green"
    lines = [
        f"总投入：{summ.get('total_cost', summ.get('total_invested', 0)):.0f} 元",
        f"当前市值：{summ.get('current_value', summ.get('market_value', 0)):.0f} 元",
        f"已实现盈亏：[{color}]{pnl:+.0f}[/{color}] 元",
    ]
    if "win_rate" in summ:
        lines.append(f"胜率：{summ['win_rate']*100:.0f}%   交易次数：{summ.get('trade_count', summ.get('closed_trades', 0))}")
    if "avg_hold_days" in summ:
        lines.append(f"平均持仓天数：{summ['avg_hold_days']:.1f}")
    console.print(Panel("\n".join(lines), title="📊 持仓总览", border_style="green"))


def cmd_backtest(cfg: dict) -> None:
    """回测。"""
    try:
        from .backtest import Backtester, BacktestConfig
    except ImportError:
        console.print("[red]✗ 回测模块未安装[/]")
        return
    console.print(Panel("验证策略在历史区间的有效性", title="🔬 回测", border_style="blue"))
    start = Prompt.ask("开始日期 YYYYMMDD", default="20260301")
    end = Prompt.ask("结束日期 YYYYMMDD", default=datetime.now().strftime("%Y%m%d"))
    # watchlist：留空则用全市场候选（慢），建议先小范围验证
    wl = Prompt.ask("标的（逗号分隔代码，留空=自动选热门）", default="")
    watchlist = [c.strip() for c in wl.split(",") if c.strip()] if wl.strip() else []
    with console.status("[bold cyan]回测中（历史数据量大，请耐心）…", spinner="aesthetic"):
        try:
            bt_cfg = BacktestConfig(
                start_date=start, end_date=end, watchlist=watchlist,
                sentiment_cfg=cfg.get("sentiment", {}),
                trend_cfg=cfg.get("trend", {}),
                entry_exit_cfg=cfg,
            )
            bt = Backtester(bt_cfg, build_data_loader(cfg))
            result = bt.run()
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]✗ 回测失败：{e}[/]")
            return
    rep = result.to_report() if hasattr(result, "to_report") else str(result)
    console.print(Panel(rep, title="📊 回测报告", border_style="green"))


def cmd_news(cfg: dict) -> None:
    """读财经简报。"""
    d = PROJECT_ROOT / "data" / "reports"
    try:
        files = sorted([f for f in d.iterdir() if f.suffix == ".md"], reverse=True)
    except Exception:
        files = []
    if not files:
        console.print("[yellow]暂无简报。先运行 capitalise-finnews 技能生成，或运行 report 命令。[/]")
        return
    console.print(f"[dim]最新简报：{files[0].name}[/]\n")
    console.print(Panel(files[0].read_text(encoding="utf-8")[:4000],
                        title="📰 财经简报", border_style="blue"))


def cmd_rps_build(cfg: dict) -> None:
    """RPS 预计算。"""
    from . import rps as rps_mod
    console.print()
    with console.status("[bold cyan]预计算全市场RPS（约1-2分钟）…", spinner="moon"):
        try:
            res = rps_mod.compute_all_rps(
                build_data_loader(cfg),
                db_path=cfg.get("output", {}).get("db_path", "data/pangu.db"),
                workers=10,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]✗ RPS 计算失败：{e}[/]")
            return
    console.print(Panel(
        f"成功 {res['ok']}/{res['total']} 只  耗时 {res['elapsed']}s\n"
        f"日期 {res['date']}",
        title="✓ RPS 预计算完成", border_style="green"))


# ====================================================================== #
# 主循环
# ====================================================================== #
MENU = """[bold cyan]
╭──────────────────────────────────────╮
│   1. 📊 今日市场速览                   │
│   2. 🎯 今日推荐（选股+排序+概率）     │
│   3. 🤖 AI选股（LLM综合分析）          │
│   4. 💼 持仓管理                       │
│   5. 🔬 回测                           │
│   6. 📰 读财经简报                     │
│   7. ⚙️  RPS预计算                     │
│   0. 退出                              │
╰──────────────────────────────────────╯[/bold cyan]"""

HANDLERS = {
    "1": cmd_sentiment, "2": cmd_scan, "3": cmd_ai_pick,
    "4": cmd_portfolio, "5": cmd_backtest, "6": cmd_news, "7": cmd_rps_build,
}


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    cfg = load_config()
    show_banner()
    console.print("[dim]提示：每个交易日收盘后建议先跑 [7] 更新RPS。输入 0 退出。[/]\n")

    while True:
        console.print(MENU)
        choice = Prompt.ask("[bold]选择功能[/]", default="0", choices=list(HANDLERS.keys()) + ["0"])
        if choice == "0":
            console.print("[bold magenta]再见。投资有风险，盈亏自负。[/]")
            return 0
        try:
            HANDLERS[choice](cfg)
        except KeyboardInterrupt:
            console.print("\n[yellow]已中断[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]✗ 出错：{e}[/]")
            logger.exception("REPL 命令失败")
        console.print()


if __name__ == "__main__":
    sys.exit(main())
