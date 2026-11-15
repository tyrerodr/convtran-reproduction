"""Checks for the recorded-result validation used by notebooks 04-06.

Uses the tracked results CSVs (read only) and temporary log files; no training.
Run from the repository root: python tests/test_recorded_runs.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recorded_runs as rr  # noqa: E402
from multiscale_frontend import frontend_tag  # noqa: E402
from pooling_variants import pooling_tag  # noqa: E402

SEEDS = [1234, 2345, 3456, 4567, 5678]
CONFIG = {"epochs": 100, "batch_size": 16, "emb_size": 16, "num_heads": 8}
POOLING = ROOT / "results" / "pooling_runs.csv"
MULTISCALE = ROOT / "results" / "multiscale_frontend_runs.csv"
POSITIONAL = ROOT / "results" / "positional_encoding_runs.csv"


def plan_for(results, variants):
    datasets = sorted(set(results.loc[results["epochs"] == 100, "dataset"]))
    return [(d, v, s) for d in datasets for s in SEEDS for v in variants]


def pooling(results=None):
    results = pd.read_csv(POOLING) if results is None else results
    plan = plan_for(pd.read_csv(POOLING), ["average", "avg_max"])
    return plan, rr.select_records(
        results, plan, **CONFIG, tag_column="pooling", tag_of=pooling_tag
    )


def test_tracked_csvs_are_valid():
    plan, (valid, invalid, ignored) = pooling()
    assert len(plan) == len(valid) == 280 and not invalid and ignored == 2
    results = pd.read_csv(MULTISCALE)
    plan = plan_for(results, ["original", "multiscale"])
    valid, invalid, ignored = rr.select_records(
        results, plan, **CONFIG, tag_column="frontend", tag_of=frontend_tag
    )
    assert len(plan) == len(valid) == 30 and not invalid and ignored == 2
    results = pd.read_csv(POSITIONAL)
    plan = plan_for(results, ["tape_erpe", "sinusoidal", "rope"])
    valid, invalid, ignored = rr.select_records(results, plan, **CONFIG)
    assert len(plan) == len(valid) == 420 and not invalid


def test_missing_row():
    results = pd.read_csv(POOLING)
    target = (results["dataset"] == "Libras") & (results["variant"] == "avg_max") \
        & (results["seed"] == 2345) & (results["epochs"] == 100)
    assert target.sum() == 1
    plan, (valid, invalid, _) = pooling(results[~target])
    assert ("Libras", "avg_max", 2345) not in valid and not invalid
    assert len(valid) == 279


def test_duplicate_row():
    results = pd.read_csv(POOLING)
    duplicated = pd.concat([results, results[results["epochs"] == 100].head(1)])
    try:
        pooling(duplicated)
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("duplicate rows were accepted")


def test_mismatched_configuration():
    results = pd.read_csv(POOLING)
    first = results.index[results["epochs"] == 100][0]
    key = tuple(results.loc[first, ["dataset", "variant", "seed"]])

    wrong_tag = results.copy()
    wrong_tag.loc[first, "pooling"] = "something-else"
    _, (valid, invalid, _) = pooling(wrong_tag)
    assert key not in valid and "pooling is" in invalid[key]

    wrong_heads = results.copy()
    wrong_heads.loc[first, "num_heads"] = 4
    _, (valid, invalid, ignored) = pooling(wrong_heads)
    assert key not in valid and key not in invalid and ignored == 3


def test_invalid_metrics():
    results = pd.read_csv(POOLING)
    rows = results.index[results["epochs"] == 100][:4]
    broken = results.copy()
    broken["test_loss"] = broken["test_loss"].astype(object)
    broken.loc[rows[0], "accuracy"] = 1.2
    broken.loc[rows[1], "test_loss"] = float("nan")
    broken.loc[rows[2], "train_epoch_times_s"] = "0.1;0.2"
    broken.loc[rows[3], "peak_cuda_allocated_train_val_mib"] = float("nan")
    _, (valid, invalid, _) = pooling(broken)
    assert len(invalid) == 4 and len(valid) == 276
    reasons = " | ".join(invalid.values())
    for text in ("accuracy", "test loss", "epoch times", "peak CUDA"):
        assert text in reasons, text


def test_local_log_states_are_separate():
    results = pd.read_csv(POOLING)
    plan, (valid, invalid, _) = pooling(results)
    small_plan = plan[:3]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        logs = {key: root / f"{i}.log" for i, key in enumerate(small_plan)}
        logs[small_plan[0]].write_text("complete")
        logs[small_plan[1]].write_text("partial training output")
        table = rr.run_table(
            small_plan, valid, invalid, lambda d, v, s: logs[(d, v, s)],
            lambda log: log.read_text() == "complete", root,
        )
    assert list(table["local_log"]) == ["complete", "incomplete", "missing"]
    assert list(table["result"]) == ["recorded"] * 3
    assert table["test_accuracy_pct"].notna().all()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
