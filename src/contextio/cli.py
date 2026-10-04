"""Command-line interface: ``contextio generate``, ``contextio run``, ``contextio strategies``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from contextio.bench import Dataset, generate_dataset
from contextio.eval import (
    DEFAULT_BUDGETS,
    BenchmarkConfig,
    run_benchmark,
    standard_strategies,
    summarize,
    to_markdown,
    tokens_to_reach,
    write_records_csv,
    write_summaries_json,
)
from contextio.pricing import PricingModel


def _int_list(value: str) -> list[int]:
    try:
        return [int(v) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc


def _name_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _load_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset:
        return Dataset.load_jsonl(args.dataset)
    return generate_dataset(n_users=args.users, seed=args.seed)


def cmd_generate(args: argparse.Namespace) -> int:
    dataset = generate_dataset(n_users=args.users, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_jsonl(out)
    print(json.dumps(dataset.stats(), indent=2))
    print(f"wrote {out}", file=sys.stderr)
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    for strategy in standard_strategies():
        print(strategy.name)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    available = {s.name: s for s in standard_strategies()}
    names = args.strategies or list(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        print(f"unknown strategies: {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(available)}", file=sys.stderr)
        return 2

    dataset = _load_dataset(args)
    config = BenchmarkConfig(
        budgets=args.budgets,
        pricing=PricingModel(
            input_per_mtok=args.input_price,
            output_per_mtok=args.output_price,
            cached_input_per_mtok=args.cached_price,
        ),
        output_tokens=args.output_tokens,
        simulate_cache=not args.no_cache,
    )

    answerer = None
    if args.llm_model:
        from contextio.eval.llm import KeywordAnswerer, OpenAICompatibleClient

        answerer = KeywordAnswerer(
            OpenAICompatibleClient(args.llm_model, base_url=args.llm_base_url)
        )

    def progress(done: int, total: int) -> None:
        if not args.quiet:
            print(f"\r  users {done}/{total}", end="" if done < total else "\n", file=sys.stderr)

    stats = dataset.stats()
    print(
        f"{stats['name']}: {stats['users']} users, {stats['queries']} queries, "
        f"{stats['avg_tokens_per_user']:,.0f} avg tokens/user",
        file=sys.stderr,
    )
    started = time.perf_counter()
    records = run_benchmark(
        dataset, [available[n] for n in names], config, answerer=answerer, progress=progress
    )
    elapsed = time.perf_counter() - started

    summaries = summarize(records, by_kind=args.by_kind)
    overall = [s for s in summaries if s.query_kind == "all"]
    columns = list(_columns(args, answerer is not None))
    table = to_markdown(summaries if args.by_kind else overall, columns)
    frontier = tokens_to_reach(overall, args.target)

    print(table)
    print(f"\nFewest mean context tokens to reach {args.target:.0%} answerable rate:")
    for name, tokens in frontier.items():
        print(f"  {name:<20} {'not reached' if tokens is None else f'{tokens:,.0f}'}")
    print(f"\n{len(records):,} records in {elapsed:.1f}s", file=sys.stderr)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_records_csv(records, out / "records.csv")
        write_summaries_json(summaries, out / "summary.json")
        (out / "summary.md").write_text(table + "\n")
        print(f"wrote results to {out}/", file=sys.stderr)
    return 0


def _columns(args: argparse.Namespace, with_accuracy: bool) -> Sequence[str]:
    columns = ["strategy", "budget"]
    if args.by_kind:
        columns.append("query_kind")
    columns.append("answerable_rate")
    if with_accuracy:
        columns.append("accuracy")
    columns += [
        "recall",
        "precision",
        "utilization",
        "stale_rate",
        "mean_tokens",
        "cache_hit_rate",
        "p50_latency_ms",
        "mean_cost_usd",
        "answerable_per_1k_tokens",
    ]
    return columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextio",
        description="Benchmark context selection, compression, and caching under token budgets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a synthetic benchmark dataset as JSONL")
    gen.add_argument("--users", type=int, default=100)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--out", default="data/synthetic.jsonl")
    gen.set_defaults(func=cmd_generate)

    strategies = sub.add_parser("strategies", help="list available strategies")
    strategies.set_defaults(func=cmd_strategies)

    run = sub.add_parser("run", help="run strategies over a dataset across token budgets")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--dataset", help="JSONL dataset; defaults to a generated one")
    source.add_argument("--users", type=int, default=50, help="users to generate")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--budgets", type=_int_list, default=list(DEFAULT_BUDGETS))
    run.add_argument("--strategies", type=_name_list, help="comma-separated strategy names")
    run.add_argument("--by-kind", action="store_true", help="break results down by query kind")
    run.add_argument("--target", type=float, default=0.9, help="answerable rate for the frontier")
    run.add_argument("--out", help="directory for records.csv, summary.json, and summary.md")
    run.add_argument("--input-price", type=float, default=3.0, help="$ per 1M input tokens")
    run.add_argument("--output-price", type=float, default=15.0, help="$ per 1M output tokens")
    run.add_argument("--cached-price", type=float, default=0.3, help="$ per 1M cached tokens")
    run.add_argument("--output-tokens", type=int, default=300, help="assumed answer length")
    run.add_argument("--no-cache", action="store_true", help="disable prefix cache simulation")
    run.add_argument("--llm-model", help="grade answers with this model (OpenAI-compatible API)")
    run.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
