"""Pangu 3.0 CLI（Phase 10）。

用法：
  python -m engine.decision.cli recommend [--asof ISO] [--limit 20] [--refresh]
  python -m engine.decision.cli analyze 600519
  python -m engine.decision.cli why <run_id> <code>
  python -m engine.decision.cli status
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pangu")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("recommend", help="下一交易日 Top20 决策候选")
    p_rec.add_argument("--asof", default=None)
    p_rec.add_argument("--limit", type=int, default=20)
    p_rec.add_argument("--refresh", action="store_true")

    p_ana = sub.add_parser("analyze", help="单票完整分析")
    p_ana.add_argument("code")
    p_ana.add_argument("--asof", default=None)

    p_why = sub.add_parser("why", help="按 run_id 展开完整证据链")
    p_why.add_argument("run_id")
    p_why.add_argument("code")

    sub.add_parser("status", help="数据源状态与 freshness")

    args = ap.parse_args(argv)

    from engine.decision.service import PanguDecisionService
    from engine.decision.render import (render_top20, render_single,
                                        render_why, render_status)

    svc = PanguDecisionService()
    if args.cmd == "recommend":
        run = svc.recommend_next_session(asof=args.asof, limit=args.limit,
                                         force_refresh=args.refresh)
        print(render_top20(run))
    elif args.cmd == "analyze":
        run = svc.analyze_stock(args.code, asof=args.asof)
        print(render_single(run, args.code))
    elif args.cmd == "why":
        out = svc.explain(args.run_id, args.code)
        if out is None:
            print("未找到对应 run 或代码", file=sys.stderr)
            return 2
        print(render_why(out["run"], out["decision"]))
    elif args.cmd == "status":
        print(render_status(svc.status()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
