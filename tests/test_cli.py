import json

import pytest

from contextio.bench import Dataset
from contextio.cli import build_parser, main


def test_generate_writes_dataset(tmp_path, capsys):
    out = tmp_path / "bench.jsonl"
    assert main(["generate", "--users", "2", "--seed", "4", "--out", str(out)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["users"] == 2
    assert len(Dataset.load_jsonl(out)) == 2


def test_strategies_lists_names(capsys):
    assert main(["strategies"]) == 0
    names = capsys.readouterr().out.split()
    assert "full" in names and "hybrid" in names and "compressed-hybrid" in names


def test_run_prints_table_and_writes_outputs(tmp_path, capsys):
    out = tmp_path / "results"
    code = main(
        [
            "run",
            "--users",
            "2",
            "--budgets",
            "250,1000",
            "--strategies",
            "full,bm25",
            "--out",
            str(out),
            "--quiet",
        ]
    )
    assert code == 0
    stdout = capsys.readouterr().out
    assert "| full | none |" in stdout
    assert "| bm25 | 250 |" in stdout
    assert "answerable rate" in stdout
    assert {p.name for p in out.iterdir()} == {"records.csv", "summary.json", "summary.md"}


def test_run_from_dataset_by_kind(tmp_path, capsys):
    path = tmp_path / "d.jsonl"
    main(["generate", "--users", "1", "--out", str(path)])
    capsys.readouterr()
    assert main(["run", "--dataset", str(path), "--strategies", "full", "--by-kind"]) == 0
    assert "query_kind" in capsys.readouterr().out


def test_run_rejects_unknown_strategy(capsys):
    assert main(["run", "--users", "1", "--strategies", "nope"]) == 2
    assert "unknown strategies: nope" in capsys.readouterr().err


def test_budget_parsing():
    args = build_parser().parse_args(["run", "--budgets", "100, 200"])
    assert args.budgets == [100, 200]
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--budgets", "a,b"])
