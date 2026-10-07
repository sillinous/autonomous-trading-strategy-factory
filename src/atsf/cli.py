"""``atsf`` command line: fetch data, backtest archetypes, and run the research factory.

Examples::

    atsf archetypes
    atsf data SPY --start 2005-01-01 --out spy.csv
    atsf backtest SPY --archetype donchian_breakout
    atsf research SPY --start 2005-01-01 --json report.json

Nothing here can submit orders; promotion only ever reaches the paper stage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, run_long_signal_backtest
from .factory import FactoryReport, run_factory, summarize_backtest
from .generator import ARCHETYPES, archetype_strategy
from .data import validate_market_data
from .signals import strategy_position


def load_market_data(symbol: str, *, source: str = "yahoo", start: str | None = None,
                     end: str | None = None, csv: str | None = None) -> pd.DataFrame:
    """Load validated daily OHLCV from a CSV file or an external source."""
    if csv:
        frame = pd.read_csv(csv)
        frame.columns = [column.lower() for column in frame.columns]
        key = "date" if "date" in frame.columns else "timestamp"
        frame[key] = pd.to_datetime(frame[key])
        frame = frame.set_index(key).sort_index()
        frame.index.name = "date"
        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        frame = validate_market_data(frame[["open", "high", "low", "close", "volume"]])
    else:
        from .external_data import ExternalDataGateway

        envelope = ExternalDataGateway().market_daily(
            symbol, source=source,
            start=pd.Timestamp(start) if start else None,
            end=pd.Timestamp(end) if end else None,
        )
        records = envelope.payload["records"]
        frame = pd.DataFrame(records)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.set_index("timestamp")
        frame.index.name = "date"
    if start:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end:
        frame = frame.loc[frame.index < pd.Timestamp(end)]
    if frame.empty:
        raise ValueError("no market data in the requested range")
    return frame


def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def format_report(report: FactoryReport) -> str:
    lines = [
        f"{report.symbol}  {report.start[:10]} → {report.end[:10]}  ({report.bars} bars, "
        f"data {report.dataset_version})",
        f"trials {report.n_trials}   PBO "
        + ("n/a" if report.pbo is None else f"{report.pbo:.2f}")
        + (("  ✗ population overfit (strict: no promotions)" if report.strict_pbo
            else "  ⚠ IS ranking unreliable; promotions rest on absolute gates")
           if report.population_overfit else ""),
        "",
        f"{'family':<20}{'OOS ret':>9}{'Sharpe':>8}{'MaxDD':>8}{'trades':>8}{'DSR':>7}{'IS':>4}  status",
    ]
    for row in report.rows:
        status = "PROMOTE → paper" if row.promoted else "reject"
        lines.append(
            f"{row.family:<20}{_pct(row.oos_return):>9}{row.oos_sharpe:>8.2f}"
            f"{_pct(-row.oos_drawdown):>8}{row.oos_trades:>8}{row.deflated_sharpe:>7.2f}"
            f"{'✓' if row.train_passed else '✗':>4}  {status}"
        )
        if not row.promoted and row.reasons:
            more = f"  (+{len(row.reasons) - 1} more)" if len(row.reasons) > 1 else ""
            lines.append(f"{'':<20}└ {row.reasons[0]}{more}")
    promoted = report.promoted
    lines += ["", f"{len(promoted)} of {len(report.rows)} strategies promoted to paper."]
    if report.benchmark_sharpe is not None:
        lines.append(f"buy & hold Sharpe over the same OOS years: {report.benchmark_sharpe:.2f}")
    return "\n".join(lines)


def _cmd_archetypes(_: argparse.Namespace) -> int:
    for name, (style, indicators, *_rest) in ARCHETYPES.items():
        kinds = ", ".join(f"{i.kind}({i.period})" for i in indicators)
        print(f"{name:<22}{style:<16}{kinds}")
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    frame = load_market_data(args.symbol, source=args.source, start=args.start, end=args.end)
    if args.out:
        frame.to_csv(args.out)
        print(f"wrote {len(frame)} bars to {args.out}")
    else:
        print(frame.tail(10).to_string())
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    data = load_market_data(args.symbol, source=args.source, start=args.start, end=args.end,
                            csv=args.csv)
    strategy = archetype_strategy(args.archetype, [args.symbol.upper()])
    config = BacktestConfig(commission_bps=args.commission_bps, slippage_bps=args.slippage_bps)
    result = run_long_signal_backtest(data, strategy_position(data, strategy), strategy, config)
    stats = summarize_backtest(result.equity)
    buy_hold = summarize_backtest(data["close"])
    print(f"{args.archetype} on {args.symbol.upper()}  {data.index[0].date()} → {data.index[-1].date()}")
    print(f"{'':<14}{'strategy':>10}{'buy&hold':>10}")
    for key in ("total_return", "cagr", "max_drawdown"):
        print(f"{key:<14}{_pct(stats[key]):>10}{_pct(buy_hold[key]):>10}")
    print(f"{'sharpe':<14}{stats['sharpe']:>10.2f}{buy_hold['sharpe']:>10.2f}")
    trips = result.round_trips
    print(f"round trips {len(trips)}   win rate "
          f"{(trips['pnl'] > 0).mean() * 100 if len(trips) else 0:.0f}%   exposure "
          f"{stats['exposure'] * 100:.0f}%")
    return 0


def _cmd_research(args: argparse.Namespace) -> int:
    data = load_market_data(args.symbol, source=args.source, start=args.start, end=args.end,
                            csv=args.csv)
    symbol = args.symbol.upper()
    strategies = None
    if args.propose:
        from .factory import default_strategies
        from .proposer import ClaudeProposer

        history = None
        if args.history:
            prior = json.loads(Path(args.history).read_text())
            history = [{"family": r["family"], "oos_sharpe": round(r["oos_sharpe"], 2),
                        "oos_trades": r["oos_trades"], "promoted": r["promoted"],
                        "reasons": r["reasons"][:2]} for r in prior.get("rows", [])]
        batch = ClaudeProposer().propose(symbol, data, args.propose, history)
        for proposal in batch.proposals:
            mark = "✓" if proposal.accepted else "✗"
            name = proposal.strategy.name if proposal.strategy else "?"
            print(f"{mark} {name}: {proposal.rationale or proposal.reason}"
                  + ("" if proposal.accepted else f"  [{proposal.reason}]"))
        print(f"{len(batch.accepted)} of {len(batch.proposals)} proposals admitted ({batch.model})\n")
        strategies = default_strategies(symbol) + batch.accepted
    report = run_factory(data, symbol, strategies=strategies, seed=args.seed,
                         perturbation_samples=args.perturbations, strict_pbo=args.strict_pbo,
                         require_benchmark=not args.allow_below_benchmark)
    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"report written to {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atsf", description="Autonomous Trading Strategy Factory")
    sub = parser.add_subparsers(dest="command", required=True)

    def market(p: argparse.ArgumentParser, csv: bool = True) -> None:
        p.add_argument("symbol")
        p.add_argument("--source", default="yahoo", choices=["yahoo", "stooq", "alphavantage"])
        p.add_argument("--start")
        p.add_argument("--end")
        if csv:
            p.add_argument("--csv", help="load bars from a CSV instead of the network")

    sub.add_parser("archetypes", help="list built-in strategy archetypes").set_defaults(func=_cmd_archetypes)

    data = sub.add_parser("data", help="fetch adjusted daily bars")
    market(data, csv=False)
    data.add_argument("--out")
    data.set_defaults(func=_cmd_data)

    backtest = sub.add_parser("backtest", help="backtest one archetype")
    market(backtest)
    backtest.add_argument("--archetype", required=True, choices=sorted(ARCHETYPES))
    backtest.add_argument("--commission-bps", type=float, default=1.0)
    backtest.add_argument("--slippage-bps", type=float, default=2.0)
    backtest.set_defaults(func=_cmd_backtest)

    research = sub.add_parser("research", help="run the full gated research factory")
    market(research)
    research.add_argument("--seed", type=int, default=0)
    research.add_argument("--perturbations", type=int, default=12)
    research.add_argument("--json", help="write the full report as JSON")
    research.add_argument("--strict-pbo", action="store_true",
                          help="block all promotions when population PBO exceeds 0.5")
    research.add_argument("--propose", type=int, default=0, metavar="N",
                          help="also test N strategies proposed by Claude (needs ANTHROPIC_API_KEY)")
    research.add_argument("--history", help="prior --json report to show the proposer")
    research.add_argument("--allow-below-benchmark", action="store_true",
                          help="promote strategies whose OOS Sharpe trails buy-and-hold")
    research.set_defaults(func=_cmd_research)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
