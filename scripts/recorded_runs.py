"""Recorded results and local run logs for the experiment notebooks.

The notebooks summarize the tracked results CSVs. Logs and checkpoints under
``outputs/`` are local and not tracked, so a fresh clone has none. Their state
is reported next to each recorded result but is not used as evidence for it:
a CSV row does not verify that a log or checkpoint exists, and a local log
does not replace a missing or invalid CSV row.
"""

import math

import pandas as pd

KEY = ["dataset", "variant", "seed"]
RUN_COLUMNS = [
    "dataset", "variant", "seed", "result", "issue", "test_accuracy_pct",
    "test_loss", "first_epoch_s", "later_epochs_mean_s", "timed_epochs",
    "peak_cuda_alloc_train_val_mib", "device", "local_log", "log",
]
NUMERIC_COLUMNS = [
    "test_accuracy_pct", "test_loss", "first_epoch_s", "later_epochs_mean_s",
    "peak_cuda_alloc_train_val_mib",
]


def finite(value):
    """Return value as a finite float, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def epoch_times(value):
    """Parse the semicolon-separated epoch times written by the runners."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [float(part) for part in str(value).split(";") if part]


def row_problems(row, epochs, tag_column=None, expected_tag=None):
    """List the reasons why a result row cannot be used."""
    problems = []
    if tag_column is not None and str(row[tag_column]) != expected_tag:
        problems.append(f"{tag_column} is {row[tag_column]!r}, expected {expected_tag!r}")
    accuracy = finite(row["accuracy"])
    if accuracy is None or not 0 <= accuracy <= 1:
        problems.append(f"accuracy {row['accuracy']!r} is not in [0, 1]")
    loss = finite(row["test_loss"])
    if loss is None or loss < 0:
        problems.append(f"test loss {row['test_loss']!r} is not finite and non-negative")
    try:
        times = epoch_times(row["train_epoch_times_s"])
    except ValueError:
        times = None
    if times is None or len(times) != epochs or not all(
        math.isfinite(time) and time > 0 for time in times
    ):
        problems.append(f"expected {epochs} finite positive epoch times")
    device = row["device"]
    if device == "cuda":
        peak = finite(row["peak_cuda_allocated_train_val_mib"])
        if peak is None or peak <= 0:
            problems.append("peak CUDA memory is missing or not positive")
    elif device != "cpu":
        problems.append(f"unknown device {device!r}")
    return problems


def select_records(results, plan, *, epochs, batch_size, emb_size, num_heads,
                   tag_column=None, tag_of=None, extra_check=None):
    """Split the result rows of the planned runs into valid and invalid ones.

    A row belongs to a planned run when its dataset, variant and seed are in
    ``plan`` and its epochs, batch size, embedding size and number of heads
    match exactly. Other rows, such as the two-epoch pilot, are ignored and
    counted. Duplicate rows for one planned run raise ``ValueError``. Rows
    that belong to a planned run but have a different configuration tag or an
    invalid metric are returned as invalid with the reason.

    Returns ``(valid, invalid, ignored)``: ``valid`` maps each run key to its
    row, ``invalid`` maps each run key to the reason, and ``ignored`` is the
    number of rows for other configurations.
    """
    plan_keys = set(plan)
    seeds = pd.to_numeric(results["seed"], errors="coerce")
    keys = [
        (dataset, variant, int(seed) if pd.notna(seed) else None)
        for dataset, variant, seed in zip(results["dataset"], results["variant"], seeds)
    ]
    in_plan = pd.Series([key in plan_keys for key in keys], index=results.index, dtype=bool)
    same_config = (
        (pd.to_numeric(results["epochs"], errors="coerce") == epochs)
        & (pd.to_numeric(results["batch_size"], errors="coerce") == batch_size)
        & (pd.to_numeric(results["emb_size"], errors="coerce") == emb_size)
        & (pd.to_numeric(results["num_heads"], errors="coerce") == num_heads)
    )
    candidates = results[in_plan & same_config].assign(seed=seeds[in_plan & same_config])
    duplicates = candidates[candidates.duplicated(KEY, keep=False)]
    if not duplicates.empty:
        raise ValueError(
            "Duplicate result rows for one planned run; the CSV must be "
            "corrected before it can be summarized:\n"
            + duplicates.sort_values(KEY).to_string()
        )

    valid, invalid = {}, {}
    for _, row in candidates.iterrows():
        key = (row["dataset"], row["variant"], int(row["seed"]))
        expected_tag = tag_of(row["variant"]) if tag_of is not None else None
        problems = row_problems(row, epochs, tag_column, expected_tag)
        if extra_check is not None:
            problems += extra_check(row)
        if problems:
            invalid[key] = "; ".join(problems)
        else:
            valid[key] = row
    return valid, invalid, len(results) - len(candidates)


def local_log_state(log, log_complete):
    if not log.is_file():
        return "missing"
    return "complete" if log_complete(log) else "incomplete"


def run_table(plan, valid, invalid, log_of, log_complete, root, extra_columns=()):
    """One row per planned run: recorded result and local log state."""
    rows = []
    for dataset, variant, seed in plan:
        key = (dataset, variant, seed)
        log = log_of(dataset, variant, seed)
        row = {
            "dataset": dataset, "variant": variant, "seed": seed,
            "result": (
                "recorded" if key in valid
                else "invalid" if key in invalid else "missing"
            ),
            "issue": invalid.get(key),
            "local_log": local_log_state(log, log_complete),
            "log": str(log.relative_to(root)),
        }
        record = valid.get(key)
        if record is not None:
            times = epoch_times(record["train_epoch_times_s"])
            row.update(
                test_accuracy_pct=float(record["accuracy"]) * 100,
                test_loss=float(record["test_loss"]),
                first_epoch_s=times[0],
                later_epochs_mean_s=(
                    sum(times[1:]) / len(times[1:]) if len(times) > 1 else None
                ),
                timed_epochs=len(times),
                peak_cuda_alloc_train_val_mib=finite(
                    record["peak_cuda_allocated_train_val_mib"]
                ),
                device=record["device"],
                **{column: record[column] for column in extra_columns},
            )
        rows.append(row)
    columns = RUN_COLUMNS[:-2] + list(extra_columns) + RUN_COLUMNS[-2:]
    table = pd.DataFrame(rows).reindex(columns=columns)
    table[NUMERIC_COLUMNS] = table[NUMERIC_COLUMNS].apply(pd.to_numeric)
    table["timed_epochs"] = pd.to_numeric(table["timed_epochs"]).astype("Int64")
    return table


def print_status(table, ignored, retry_command):
    """Print recorded-result and local-log counts, problems and retry hints."""
    results = table["result"].value_counts()
    logs = table["local_log"].value_counts()
    print(f"Planned runs: {len(table)}")
    print(
        "Recorded results (CSV): "
        + ", ".join(f"{state} {results.get(state, 0)}"
                    for state in ("recorded", "missing", "invalid"))
    )
    print(
        "Local logs (outputs/, not tracked): "
        + ", ".join(f"{state} {logs.get(state, 0)}"
                    for state in ("complete", "incomplete", "missing"))
    )
    print(f"CSV rows for other configurations, ignored: {ignored}")
    print("A recorded result does not verify that a local log or checkpoint exists.")

    for run in table[table["result"] == "invalid"].itertuples(index=False):
        print(f"Invalid result row: {run.dataset}, {run.variant}, seed {run.seed}: {run.issue}")
    incomplete = table[table["local_log"] == "incomplete"]
    if not incomplete.empty:
        print("\nIncomplete local logs are kept. Inspect each log, then retry from the repository root:")
        for run in incomplete.itertuples(index=False):
            print(" ", retry_command(run.dataset, run.variant, run.seed))
