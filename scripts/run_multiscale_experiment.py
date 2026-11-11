"""Run one ConvTran temporal front-end variant with the upstream pipeline.

Each run executes the pinned upstream ``main.py`` in its own process with
tAPE + eRPE (same loader, split, preprocessing, RAdam optimizer,
lowest-validation-loss checkpoint and test evaluation as the baseline). Only
the first temporal convolution changes. Timing and CUDA memory are measured
as in ``run_positional_experiment.py``; the submodule is not modified.
"""

import argparse
import csv
import runpy
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from multiscale_frontend import FRONTENDS, frontend_tag
from run_baseline import ROOT, UPSTREAM, positive_int, read_result
from run_positional_experiment import install_measurements, read_metrics

VARIANT_NAMES = tuple(FRONTENDS)
EXPERIMENT_ROOT = ROOT / "outputs" / "experiments" / "multiscale_frontend"
RESULTS_PATH = ROOT / "results" / "multiscale_frontend_runs.csv"
CONFIG_FIELDS = [
    "variant", "frontend", "dataset", "seed", "epochs", "batch_size",
    "emb_size", "num_heads",
]
FIELDS = CONFIG_FIELDS + [
    "accuracy", "test_loss", "mean_train_epoch_s", "train_epoch_times_s",
    "peak_cuda_allocated_train_val_mib", "device",
]


def run_directory(variant, dataset, seed, epochs, batch_size, emb_size, num_heads):
    """Directory that identifies one run by its full configuration."""
    return (
        EXPERIMENT_ROOT / variant / dataset.lower()
        / (f"seed{seed}_ep{epochs}_bs{batch_size}_emb{emb_size}_h{num_heads}"
           f"_fe-{frontend_tag(variant)}")
    )


def worker(args, data, output):
    import multiscale_frontend
    import utils
    from Models import model as upstream_model

    original_setup = utils.Setup

    def setup_variant(parsed):
        # Recorded in configuration.json next to the upstream settings.
        parsed.frontend_variant = args.variant
        parsed.frontend = frontend_tag(args.variant)
        return original_setup(parsed)

    def build(config):
        return multiscale_frontend.model_factory(config, args.variant)

    with tempfile.TemporaryDirectory(prefix="convtran-") as temporary:
        selected = Path(temporary)
        (selected / args.dataset).symlink_to(data, target_is_directory=True)

        def select_dataset(config):
            config["data_path"] = str(selected)

        utils.Data_Verifier = select_dataset
        utils.Setup = setup_variant
        upstream_model.model_factory = build
        install_measurements()
        sys.argv = [
            str(UPSTREAM / "main.py"),
            "--epochs", str(args.epochs),
            "--gpu", "0",
            "--batch_size", str(args.batch_size),
            "--emb_size", str(args.emb_size),
            "--num_heads", str(args.num_heads),
            "--Fix_pos_encode", "tAPE",
            "--Rel_pos_encode", "eRPE",
            "--seed", str(args.seed),
            "--output_dir", str(output),
        ]
        runpy.run_path(str(UPSTREAM / "main.py"), run_name="__main__")


def save_row(row):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    previous = []
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise SystemExit("Existing CSV schema differs; leaving it unchanged.")
            key = [str(row[field]) for field in CONFIG_FIELDS]
            previous = [
                old for old in reader
                if [old[field] for field in CONFIG_FIELDS] != key
            ]
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=RESULTS_PATH.parent, delete=False, suffix=".tmp"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(previous + [row])
        temporary_csv = Path(stream.name)
    temporary_csv.replace(RESULTS_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, choices=VARIANT_NAMES)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=positive_int, required=True)
    parser.add_argument("--batch-size", type=positive_int, required=True)
    parser.add_argument("--emb-size", type=positive_int, required=True)
    parser.add_argument("--num-heads", type=positive_int, required=True)
    parser.add_argument(
        "--retry", action="store_true",
        help="Keep an incomplete run.log as run.failed-<UTC time>.log and rerun.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(args.dataset).name != args.dataset or args.dataset in (".", ".."):
        parser.error("Use a dataset name, not a path.")
    if args.emb_size % args.num_heads:
        parser.error("--emb-size must be divisible by --num-heads.")
    if sum(filters for filters, _ in FRONTENDS["multiscale"]) != 4 * args.emb_size:
        parser.error("The multiscale branches are defined for --emb-size 16.")
    if not (UPSTREAM / "main.py").is_file():
        parser.error("Initialize the upstream/ConvTran submodule first.")

    data = ROOT / "Dataset" / "UEA" / "Multivariate_ts" / args.dataset
    for split in ("TRAIN", "TEST"):
        required = data / f"{args.dataset}_{split}.ts"
        if not required.is_file():
            parser.error(f"Missing dataset file: {required}")

    config = [args.variant, args.dataset, args.seed, args.epochs,
              args.batch_size, args.emb_size, args.num_heads]
    directory = run_directory(*config)
    if args.worker:
        worker(args, data, directory)
        return

    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "run.log"
    retry = shlex.join([
        "python", str(Path(__file__).resolve().relative_to(ROOT)),
        *[arg for arg in sys.argv[1:] if arg != "--retry"], "--retry",
    ])
    result, metrics = read_result(log), read_metrics(log)
    if result is not None and metrics is not None:
        print(f"Existing complete run: {log}")
    else:
        if log.exists():
            if not args.retry:
                raise SystemExit(
                    f"Incomplete log: {log}\n"
                    f"Inspect it, then retry with: {retry}\n"
                    "(--retry keeps it as run.failed-<UTC time>.log)"
                )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            kept = log.with_name(f"run.failed-{stamp}.log")
            log.rename(kept)
            print(f"Kept incomplete log as: {kept}")
        print(f"Training {args.variant} on {args.dataset}. Log: {log}", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
        command += [
            "--variant", args.variant, "--dataset", args.dataset,
            "--seed", str(args.seed), "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--emb-size", str(args.emb_size), "--num-heads", str(args.num_heads),
        ]
        with log.open("x") as stream:
            process = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            )
        result, metrics = read_result(log), read_metrics(log)
        if process.returncode or result is None or metrics is None:
            raise SystemExit(
                f"Run failed (exit code {process.returncode}). Log: {log}\n"
                f"Inspect it, then retry from the repository root with: {retry}"
            )

    loss, accuracy = result
    row = {
        "variant": args.variant,
        "frontend": frontend_tag(args.variant),
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "emb_size": args.emb_size,
        "num_heads": args.num_heads,
        "accuracy": accuracy,
        "test_loss": loss,
        "mean_train_epoch_s": metrics["mean_train_epoch_s"],
        "train_epoch_times_s": ";".join(
            f"{value:.6f}" for value in metrics["train_epoch_times_s"]
        ),
        "peak_cuda_allocated_train_val_mib": metrics["peak_cuda_allocated_train_val_mib"],
        "device": metrics["device"],
    }
    save_row(row)
    print(f"Saved result to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
