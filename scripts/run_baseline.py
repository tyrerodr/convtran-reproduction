import argparse
import csv
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "ConvTran"
SEEDS = [1234, 2345, 3456, 4567, 5678]
PATTERN = re.compile(
    r"Best Model Test Summary:.*?loss:\s*([0-9.eE+-]+)"
    r".*?accuracy:\s*([0-9.eE+-]+)"
)
FIELDS = ["dataset", "seed", "epochs", "batch_size", "accuracy", "test_loss"]


def read_result(log):
    if not log.is_file():
        return None
    matches = PATTERN.findall(log.read_text(errors="replace"))
    return tuple(map(float, matches[-1])) if matches else None


def worker(dataset, data, seed, output):
    sys.path.insert(0, str(UPSTREAM))
    import utils

    with tempfile.TemporaryDirectory(prefix="convtran-") as temporary:
        selected = Path(temporary)
        (selected / dataset).symlink_to(data, target_is_directory=True)

        def select_dataset(config):
            config["data_path"] = str(selected)

        utils.Data_Verifier = select_dataset
        sys.argv = [
            str(UPSTREAM / "main.py"),
            "--epochs", "100",
            "--gpu", "0",
            "--batch_size", "16",
            "--seed", str(seed),
            "--output_dir", str(output / f"seed{seed}"),
        ]
        runpy.run_path(str(UPSTREAM / "main.py"), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(args.dataset).name != args.dataset or args.dataset in (".", ".."):
        parser.error("Use a dataset name, not a path.")
    if not (UPSTREAM / "main.py").is_file():
        parser.error("Initialize the upstream/ConvTran submodule first.")

    data = ROOT / "Dataset" / "UEA" / "Multivariate_ts" / args.dataset
    for split in ("TRAIN", "TEST"):
        required = data / f"{args.dataset}_{split}.ts"
        if not required.is_file():
            parser.error(f"Missing dataset file: {required}")

    output = ROOT / "outputs" / "baseline" / args.dataset.lower()
    if args.worker is not None:
        worker(args.dataset, data, args.worker, output)
        return

    rows = []
    for seed in SEEDS:
        directory = output / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / "run.log"
        result = read_result(log)
        if result is None:
            if log.exists():
                raise SystemExit(
                    f"Incomplete log: {log}. Inspect and rename it before retrying."
                )
            print(f"Training {args.dataset}, seed={seed}. Log: {log}", flush=True)
            with log.open("x") as stream:
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()),
                     "--dataset", args.dataset, "--worker", str(seed)],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result = read_result(log)
            if result is None:
                raise SystemExit(f"No final result found in {log}")
        else:
            print(f"Existing result: {args.dataset}, seed={seed}")
        loss, accuracy = result
        rows.append(dict(zip(FIELDS, [args.dataset, seed, 100, 16, accuracy, loss])))

    destination = ROOT / "results" / "baseline_runs.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = []
    if destination.exists():
        with destination.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise SystemExit("Existing CSV schema differs; leaving it unchanged.")
            previous = [row for row in reader if row["dataset"] != args.dataset]

    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=destination.parent, delete=False, suffix=".tmp"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(previous + rows)
        temporary_csv = Path(stream.name)
    temporary_csv.replace(destination)
    print(f"Saved results to: {destination}")


if __name__ == "__main__":
    main()
