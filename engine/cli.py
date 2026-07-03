"""命令行入口。

用法：
    python -m engine.cli sentiment [--date YYYYMMDD]     # 只看情绪温度
    python -m engine.cli market-phase [--date YYYYMMDD]  # 识别市场阶段/情绪周期
    python -m engine.cli pools [--date YYYYMMDD]         # 运行七大策略池
    python -m engine.cli scan [--date YYYYMMDD]          # 跑完整选股链路，输出 JSON
    python -m engine.cli report [--date YYYYMMDD]        # 跑链路 + 生成 Markdown 简报

所有命令都打印结果到 stdout（JSON 或 Markdown），report 还会写文件。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import build_data_loader, load_config
from .market_phase import MarketPhaseAnalyzer
from .pipeline import Pipeline
from .report import save_report
from .sentiment_meter import SentimentMeter
from .snapshot import SnapshotBuilder
from .strategy_pools import run_all_pools


def cmd_agent(args: argparse.Namespace, cfg: dict) -> int:
    """独立 Agent 入口。"""
    from .agent.cli import main as agent_main
    return agent_main([
        args.question,
        *(["--config", args.config] if args.config else []),
        *(["--date", args.date] if args.date else []),
        *(["--verbose"] if args.verbose else []),
    ])


def cmd_rank(args: argparse.Namespace, cfg: dict) -> int:
    """全市场排行榜。"""
    from .ranking import MarketRanking, RANK_TYPES

    mr = MarketRanking()
    exclude_st = not args.no_st_filter

    if args.type == "all":
        breadth = mr.get_market_breadth()
        print(f"\n📊 市场宽度  {breadth['updated']}")
        print(f"   总数 {breadth['total']:>5}  |  上涨 {breadth['up']:>4} ({breadth['up_pct']}%)  |  "
              f"下跌 {breadth['down']:>4} ({breadth['down_pct']}%)")
        print(f"   涨停 {breadth['limit_up']:>4}  |  跌停 {breadth['limit_down']:>4}  |  "
              f"总成交 {breadth['total_amount_yi']:.0f}亿")

        for rt in ["gainers", "losers", "volume", "turnover", "net_inflow"]:
            cfg_r = RANK_TYPES[rt]
            r = mr.get_rank(rt, top_n=args.top, exclude_st=exclude_st)
            print(f"\n{'─'*60}")
            print(f"  {cfg_r['label']} TOP{args.top}")
            print(f"{'─'*60}")
            for row in r.rows:
                code = row.get('代码', '')
                name = row.get('名称', '')
                price = row.get('最新价', '')
                pct = row.get('涨跌幅', '')
                amt = row.get('成交额', '')
                inflow = row.get('主力净流入-净额', '')
                print(f"  #{row['rank']:>3} {code} {name:<8} "
                      f"{str(price):>8}  {str(pct):>8}%  {str(amt):>12}  {str(inflow):>12}")
    else:
        r = mr.get_rank(args.type, top_n=args.top, exclude_st=exclude_st)
        cfg_r = RANK_TYPES[args.type]
        print(f"\n  {cfg_r['label']} TOP{args.top}  ({r.updated} · 共{r.total_stocks}只)")
        print(f"  {'─'*55}")
        for row in r.rows:
            code = row.get('代码', '')
            name = row.get('名称', '')
            pct = row.get('涨跌幅', '')
            amt = row.get('成交额', '')
            inflow = row.get('主力净流入-净额', '')
            sort_val = row.get('_sort_value', '')
            print(f"  #{row['rank']:>3}  {code}  {name:<10}  "
                  f"涨跌:{str(pct):>8}%  "
                  f"成交:{str(amt):>12}  "
                  f"主力:{str(inflow):>12}")
    return 0


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_sentiment(args: argparse.Namespace, cfg: dict) -> int:
    dl = build_data_loader(cfg)
    meter = SentimentMeter(dl, cfg.get("sentiment", {}))
    bd = meter.measure(args.date)
    print(json.dumps(bd.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_market_phase(args: argparse.Namespace, cfg: dict) -> int:
    dl = build_data_loader(cfg)
    phase = MarketPhaseAnalyzer(dl, cfg).analyze(args.date)
    print(json.dumps(phase.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_pools(args: argparse.Namespace, cfg: dict) -> int:
    dl = build_data_loader(cfg)
    results = run_all_pools(dl, cfg, args.date)
    out = {name: [s.to_dict() for s in sigs] for name, sigs in results.items()}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _build_pipeline(cfg: dict) -> Pipeline:
    """从配置统一构造 Pipeline（含 entry_exit / db_path / full_cfg）。"""
    return Pipeline(
        dl=build_data_loader(cfg),
        sentiment_cfg=cfg.get("sentiment", {}),
        trend_cfg=cfg.get("trend", {}),
        guard_cfg=cfg.get("guard", {}),
        entry_exit_cfg=cfg.get("entry_exit", cfg),  # kimi 的 engine 期望整个 cfg
        pick_count=cfg.get("output", {}).get("pick_count", 5),
        db_path=cfg.get("output", {}).get("db_path", "data/pangu.db"),
        full_cfg=cfg,
    )


def cmd_rps_build(args: argparse.Namespace, cfg: dict) -> int:
    """离线预计算全市场 20 日 RPS，存 SQLite。盘后跑一次即可。"""
    from . import rps as rps_mod
    dl = build_data_loader(cfg)
    workers = getattr(args, "workers", 10)
    result = rps_mod.compute_all_rps(
        dl, date=args.date,
        db_path=cfg.get("output", {}).get("db_path", "data/pangu.db"),
        workers=workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_calibrate(args: argparse.Namespace, cfg: dict) -> int:
    """离线校准上涨概率模型（用历史数据统计），存 SQLite。首次慢（10-30分钟）。"""
    from . import probability_calibrator as pc
    dl = build_data_loader(cfg)
    result = pc.calibrate(
        dl, end_date=args.date, months=args.months,
        workers=getattr(args, "workers", 8),
        db_path=cfg.get("output", {}).get("db_path", "data/pangu.db"),
        max_codes=args.max_codes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_snapshot_build(args: argparse.Namespace, cfg: dict) -> int:
    """每日收盘后保存关键数据快照到 data/snapshots/YYYY-MM-DD/。"""
    dl = build_data_loader(cfg)
    snapshot_dir = cfg.get("data", {}).get("snapshot_dir", "data/snapshots")
    builder = SnapshotBuilder(dl, snapshot_dir=snapshot_dir)
    result = builder.build(args.date)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_scan(args: argparse.Namespace, cfg: dict) -> int:
    pipe = _build_pipeline(cfg)
    result = pipe.run(args.date)
    data = result.to_dict()
    report_dir = cfg.get("output", {}).get("report_dir", "data/reports")
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    (out_dir / f"{result.date}_p0.json").write_text(payload, encoding="utf-8")
    (out_dir / f"{result.date}.json").write_text(payload, encoding="utf-8")
    print(result.to_json())
    return 0


def cmd_report(args: argparse.Namespace, cfg: dict) -> int:
    pipe = _build_pipeline(cfg)
    result = pipe.run(args.date)
    report_dir = cfg.get("output", {}).get("report_dir", "data/reports")
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 同时保存 P0 完整 JSON 报告与 Markdown 简报
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    (out_dir / f"{result.date}_p0.json").write_text(payload, encoding="utf-8")
    (out_dir / f"{result.date}.json").write_text(payload, encoding="utf-8")
    path = save_report(result, report_dir)
    print(f"# 报告已保存: {path} 与 {out_dir / (result.date + '_p0.json')}\n")
    from .report import render_markdown
    print(render_markdown(result))
    return 0


def cmd_daily(args: argparse.Namespace, cfg: dict) -> int:
    """每日盘后调度入口。"""
    from .scheduler import DailyScheduler
    scheduler = DailyScheduler(
        cfg=cfg,
        date=args.date,
        skip_rps=args.skip_rps,
        skip_snapshot=args.skip_snapshot,
        skip_notify=args.skip_notify,
        dry_run=args.dry_run,
        workers=args.workers,
    )
    summary = scheduler.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["overall_status"] == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pangu",
        description="盘古 — A股短线「情绪+趋势」选股引擎",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    parser.add_argument("-c", "--config", default=None, help="配置文件路径")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn in [("sentiment", cmd_sentiment), ("scan", cmd_scan), ("report", cmd_report)]:
        p = sub.add_parser(name, help=f"{name} 命令")
        p.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
        p.set_defaults(func=fn)

    # market-phase：市场阶段识别
    p_phase = sub.add_parser("market-phase", help="识别当前市场阶段/情绪周期")
    p_phase.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    p_phase.set_defaults(func=cmd_market_phase)

    # pools：运行七大策略池
    p_pools = sub.add_parser("pools", help="运行七大策略池并输出原始信号")
    p_pools.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    p_pools.set_defaults(func=cmd_pools)

    # daily：每日盘后调度（RPS -> 快照 -> 扫描 -> 报告 -> 通知）
    p_daily = sub.add_parser("daily", help="每日盘后调度链路")
    p_daily.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    p_daily.add_argument("--skip-rps", action="store_true", help="跳过 RPS 预计算")
    p_daily.add_argument("--skip-snapshot", action="store_true", help="跳过收盘快照")
    p_daily.add_argument("--skip-notify", action="store_true", help="跳过通知")
    p_daily.add_argument("--dry-run", action="store_true", help="只检查配置与通知，不执行耗时取数")
    p_daily.add_argument("--workers", type=int, default=10, help="RPS 预计算并发数")
    p_daily.set_defaults(func=cmd_daily)

    # rps-build：离线预计算全市场 RPS（盘后跑一次）
    p_rps = sub.add_parser("rps-build", help="预计算全市场20日RPS存库")
    p_rps.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    p_rps.add_argument("--workers", type=int, default=10, help="并发线程数")
    p_rps.set_defaults(func=cmd_rps_build)

    # calibrate：离线校准上涨概率（首次慢）
    p_cal = sub.add_parser("calibrate", help="校准上涨概率模型（历史统计）")
    p_cal.add_argument("--date", default=None, help="基准日 YYYYMMDD")
    p_cal.add_argument("--months", type=int, default=6, help="回溯月数")
    p_cal.add_argument("--max-codes", type=int, default=None, help="最大股票数（调试用）")
    p_cal.add_argument("--workers", type=int, default=8, help="并发线程数")
    p_cal.set_defaults(func=cmd_calibrate)

    # snapshot-build：每日收盘后保存关键数据快照
    p_snap = sub.add_parser("snapshot-build", help="保存每日关键数据快照")
    p_snap.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    p_snap.set_defaults(func=cmd_snapshot_build)

    # agent：独立 LLM Agent
    p_agent = sub.add_parser("agent", help="独立 Agent 选股问答")
    p_agent.add_argument("question", nargs="?", default="今天A股帮我选3-5只短线票", help="向 Agent 提问")
    p_agent.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    p_agent.set_defaults(func=cmd_agent)

    # repl：交互式终端（主入口，推荐日常用这个）
    p_repl = sub.add_parser("repl", help="交互式CLI（菜单驱动，rich美化）")
    p_repl.set_defaults(func=lambda a, c: _cmd_repl(c))

    # rank：全市场排行榜
    p_rank = sub.add_parser("rank", help="全市场排行榜（涨幅/跌幅/成交额/换手率/资金流）")
    p_rank.add_argument("type", nargs="?", default="gainers",
                        choices=["gainers","losers","volume","turnover","net_inflow","net_outflow","cap","all"],
                        help="排行榜类型（all=全部）")
    p_rank.add_argument("--top", type=int, default=20, help="显示前N只")
    p_rank.add_argument("--no-st-filter", action="store_true", help="不过滤ST")
    p_rank.set_defaults(func=cmd_rank)

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = load_config(args.config)

    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:
        print("\n中断", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001
        logging.getLogger("pangu").exception("命令执行失败")
        print(f"错误: {e}", file=sys.stderr)
        return 1


def _cmd_repl(cfg: dict) -> int:
    from .repl import main as repl_main
    return repl_main()


if __name__ == "__main__":
    sys.exit(main())
