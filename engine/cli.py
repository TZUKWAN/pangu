"""命令行入口。

用法：
    python -m engine.cli sentiment [--date YYYYMMDD]     # 只看情绪温度
    python -m engine.cli market-phase [--date YYYYMMDD]  # 识别市场阶段/情绪周期
    python -m engine.cli pools [--date YYYYMMDD]         # 运行七大策略池
    python -m engine.cli doctor [--date YYYYMMDD]        # 数据源与系统健康检查
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

import pandas as pd

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
    report_dir = cfg.get("output", {}).get("report_dir", "data/reports")
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date = args.date

    # 默认行为：渲染已有 JSON 报告，不重跑 Pipeline（避免 180-300s 超时）。
    # 只有显式 --rerun 才触发完整 Pipeline 重跑。
    if not getattr(args, "rerun", False):
        loaded = _load_existing_report(out_dir, date)
        if loaded is not None:
            result, json_path = loaded
            from .report import render_markdown
            print(f"# 渲染已有报告: {json_path}（用 --rerun 强制重跑 Pipeline）\n")
            print(render_markdown(result))
            return 0
        # 没有找到已有报告 → 回退到重跑（并提示）
        print(f"# 未找到 {date or '今天'} 的已有报告，回退到重跑 Pipeline\n")

    pipe = _build_pipeline(cfg)
    result = pipe.run(date)
    # 同时保存 P0 完整 JSON 报告与 Markdown 简报
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    (out_dir / f"{result.date}_p0.json").write_text(payload, encoding="utf-8")
    (out_dir / f"{result.date}.json").write_text(payload, encoding="utf-8")
    path = save_report(result, report_dir)
    print(f"# 报告已保存: {path} 与 {out_dir / (result.date + '_p0.json')}\n")
    from .report import render_markdown
    print(render_markdown(result))
    return 0


def _load_existing_report(report_dir: Path, date: str | None) -> tuple[PipelineResult, Path] | None:
    """尝试加载已有 JSON 报告并重建 PipelineResult。失败返回 None。"""
    from .pipeline import PipelineResult
    # 候选文件名优先级：<date>.json > <date>_p0.json > latest_ok.json
    candidates: list[Path] = []
    if date:
        candidates.append(report_dir / f"{date}.json")
        candidates.append(report_dir / f"{date}_p0.json")
    else:
        latest_ok = report_dir / "latest_ok.json"
        if latest_ok.exists():
            candidates.append(latest_ok)
        # 取目录下最新的 *.json（排除 _p0、latest_ok、degraded）
        jsons = sorted(
            (p for p in report_dir.glob("*.json") if not p.name.endswith("_p0.json") and p.name != "latest_ok.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(jsons[:1])
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "date" not in data:
                continue
            return PipelineResult.from_dict(data), path
        except Exception:
            continue
    return None


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


def cmd_doctor(args: argparse.Namespace, cfg: dict) -> int:
    """数据源与系统健康检查。"""
    import os

    dl = build_data_loader(cfg)
    date = args.date or pd.Timestamp.now().strftime("%Y%m%d")
    report: dict[str, Any] = {"date": date, "checks": {}, "overall_status": "ok", "warnings": []}

    def _is_trading_day(date_str: str) -> Optional[bool]:
        try:
            import akshare as ak
            df = ak.tool_trade_date_hist_sina()
            if df is not None and not df.empty:
                df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")
                return date_str in set(df["trade_date"])
        except Exception:  # noqa: BLE001
            pass
        # fallback: 周末为非交易日
        try:
            return pd.Timestamp(date_str).weekday() < 5
        except Exception:  # noqa: BLE001
            return None

    # 1. all_spot
    try:
        spot = dl.all_spot()
        rows = len(spot) if spot is not None else 0
        report["checks"]["all_spot"] = {"status": "ok" if rows > 3000 else "failed", "rows": rows}
    except Exception as e:  # noqa: BLE001
        report["checks"]["all_spot"] = {"status": "failed", "error": str(e)}

    # 2. daily_kline
    try:
        k = dl.daily_kline("000001", days=60, date=date)
        klen = len(k) if k is not None else 0
        report["checks"]["daily_kline"] = {"status": "ok" if klen >= 30 else "failed", "rows": klen}
    except Exception as e:  # noqa: BLE001
        report["checks"]["daily_kline"] = {"status": "failed", "error": str(e)}

    # 3. limit_up_pool
    try:
        zt = dl.limit_up_pool(date)
        zt_len = len(zt) if zt is not None else 0
        report["checks"]["limit_up_pool"] = {"status": "ok" if zt_len > 0 else "degraded", "rows": zt_len}
    except Exception as e:  # noqa: BLE001
        report["checks"]["limit_up_pool"] = {"status": "failed", "error": str(e)}

    # 4. limit_down_pool
    try:
        dt = dl.limit_down_pool(date)
        dt_len = len(dt) if dt is not None else 0
        report["checks"]["limit_down_pool"] = {"status": "ok", "rows": dt_len}
    except Exception as e:  # noqa: BLE001
        report["checks"]["limit_down_pool"] = {"status": "failed", "error": str(e)}

    # 5. fund_flow
    try:
        ff = dl.all_fund_flow_snapshot(fast=True)
        ff_len = len(ff) if ff is not None else 0
        report["checks"]["fund_flow"] = {"status": "ok" if ff_len > 0 else "failed", "rows": ff_len}
    except Exception as e:  # noqa: BLE001
        report["checks"]["fund_flow"] = {"status": "failed", "error": str(e)}

    # 6. RPS 表
    try:
        from . import rps as rps_mod
        rps_map = rps_mod.load_rps_map(date, cfg.get("output", {}).get("db_path", "data/pangu.db"))
        rps_count = len(rps_map) if rps_map else 0
        report["checks"]["rps_table"] = {"status": "ok" if rps_count > 3000 else "failed", "count": rps_count}
    except Exception as e:  # noqa: BLE001
        report["checks"]["rps_table"] = {"status": "failed", "error": str(e)}

    # 7. LLM 配置
    llm_cfg = cfg.get("llm", {})
    api_key = os.environ.get(llm_cfg.get("api_key_env", "PANGU_LLM_API_KEY"))
    base_url = os.environ.get(llm_cfg.get("base_url_env", "PANGU_LLM_BASE_URL"))
    model = os.environ.get(llm_cfg.get("model_env", "PANGU_LLM_MODEL"))
    report["checks"]["llm_config"] = {
        "status": "ok" if api_key and base_url and model else "degraded",
        "has_api_key": bool(api_key),
        "has_base_url": bool(base_url),
        "has_model": bool(model),
    }

    # 8. 配置文件
    report["checks"]["config"] = {"status": "ok", "path": args.config or "config/settings.yaml"}

    # 9. 交易日判断
    is_trade = _is_trading_day(date)
    if is_trade is None:
        report["checks"]["trading_day"] = {"status": "degraded", "is_trading_day": None, "note": "无法判断"}
    else:
        report["checks"]["trading_day"] = {"status": "ok", "is_trading_day": bool(is_trade)}

    # 10. 策略池快速健康
    try:
        pools = run_all_pools(dl, cfg, date)
        pool_counts = {name: len(sigs) for name, sigs in pools.items()}
        report["checks"]["strategy_pools"] = {"status": "ok", "counts": pool_counts}
        if sum(pool_counts.values()) == 0:
            report["checks"]["strategy_pools"]["status"] = "degraded"
            report["warnings"].append("所有策略池均未产出信号")
    except Exception as e:  # noqa: BLE001
        report["checks"]["strategy_pools"] = {"status": "failed", "error": str(e)}

    # 汇总
    for v in report["checks"].values():
        if v.get("status") == "failed":
            report["overall_status"] = "failed"
            break
        elif v.get("status") == "degraded" and report["overall_status"] == "ok":
            report["overall_status"] = "degraded"

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pangu",
        description="盘古 — A股短线「情绪+趋势」选股引擎",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    parser.add_argument("-c", "--config", default=None, help="配置文件路径")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn in [("sentiment", cmd_sentiment), ("scan", cmd_scan)]:
        p = sub.add_parser(name, help=f"{name} 命令")
        p.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
        p.set_defaults(func=fn)

    # report：默认只渲染已有 JSON 报告，--rerun 才重跑完整 Pipeline
    p_report = sub.add_parser("report", help="渲染或重跑选股报告")
    p_report.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    p_report.add_argument("--rerun", action="store_true", help="强制重跑完整 Pipeline（默认只渲染已有 JSON）")
    p_report.set_defaults(func=cmd_report)

    # market-phase：市场阶段识别
    p_phase = sub.add_parser("market-phase", help="识别当前市场阶段/情绪周期")
    p_phase.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    p_phase.set_defaults(func=cmd_market_phase)

    # pools：运行七大策略池
    p_pools = sub.add_parser("pools", help="运行七大策略池并输出原始信号")
    p_pools.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    p_pools.set_defaults(func=cmd_pools)

    # doctor：数据源与系统健康检查
    p_doc = sub.add_parser("doctor", help="数据源与系统健康检查")
    p_doc.add_argument("--date", default=None, help="日期 YYYYMMDD（默认今天）")
    p_doc.set_defaults(func=cmd_doctor)

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
