"""Run one ConvTran positional-encoding variant with the upstream pipeline.

Each run executes the pinned upstream ``main.py`` in its own process (same
loader, split, preprocessing, RAdam optimizer, lowest-validation-loss
checkpoint, and test evaluation as the baseline). Only the positional
encoding changes. Timing and CUDA memory are measured by wrapping upstream
functions in the worker process; the submodule is not modified.
"""

import argparse
import csv
import json
import re
import runpy
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from run_baseline import ROOT, UPSTREAM, positive_int, read_result

VARIANT_NAMES = ("tape_erpe", "sinusoidal", "rope")
EXPERIMENT_ROOT = ROOT / "outputs" / "experiments" / "positional_encoding"
RESULTS_PATH = ROOT / "results" / "positional_encoding_runs.csv"
METRICS_MARKER = "EXPERIMENT_METRICS "
METRICS_PATTERN = re.compile(r"^EXPERIMENT_METRICS (\{.*\})$", re.MULTILINE)
CONFIG_FIELDS = [
    "variant", "dataset", "seed", "epochs", "batch_size", "emb_size", "num_heads",
]
FIELDS = CONFIG_FIELDS + [
    "accuracy", "test_loss", "mean_train_epoch_s", "train_epoch_times_s",
    "peak_cuda_allocated_train_val_mib", "device",
]


def run_directory(variant, dataset, seed, epochs, batch_size, emb_size, num_heads):
    """Directory that identifies one run by its full configuration."""
    return (
        EXPERIMENT_ROOT / variant / dataset.lower()
        / f"seed{seed}_ep{epochs}_bs{batch_size}_emb{emb_size}_h{num_heads}"
    )


def read_metrics(log):
    if not log.is_file():
        return None
    matches = METRICS_PATTERN.findall(log.read_text(errors="replace"))
    return json.loads(matches[-1]) if matches else None


def install_measurements():
    """Wrap upstream training to record epoch times and peak CUDA memory.

    Training-epoch time: CUDA is synchronized before and after each call to
    ``SupervisedTrainer.train_epoch`` (data loading, forward, backward, clip,
    optimizer step). Validation, checkpoint saving, and logging are excluded.

    Peak memory: ``train_runner`` is called after ``model.to(device)``. CUDA is
    synchronized and the peak counter reset when it starts; the peak is read
    after it returns, before the best checkpoint is loaded for testing.
    Upstream validates and saves a checkpoint after every training epoch, so
    the peak covers training and validation.
    """
    import torch
    import Training

    epoch_times = []
    original_epoch = Training.SupervisedTrainer.train_epoch
    original_runner = Training.train_runner

    def synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed_train_epoch(self, epoch_num=None):
        device = next(self.model.parameters()).device
        synchronize(device)
        start = time.perf_counter()
        result = original_epoch(self, epoch_num)
        synchronize(device)
        epoch_times.append(time.perf_counter() - start)
        return result

    def measured_train_runner(config, model, trainer, val_evaluator, path):
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        original_runner(config, model, trainer, val_evaluator, path)
        peak = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 2**20
        metrics = {
            "device": device.type,
            "train_epoch_times_s": epoch_times,
            "mean_train_epoch_s": (
                sum(epoch_times) / len(epoch_times) if epoch_times else None
            ),
            "peak_cuda_allocated_train_val_mib": peak,
        }
        print(METRICS_MARKER + json.dumps(metrics), flush=True)

    Training.SupervisedTrainer.train_epoch = timed_train_epoch
    Training.train_runner = measured_train_runner


def worker(args, data, output):
    import positional_encodings
    import utils
    from Models import model as upstream_model

    settings = positional_encodings.VARIANTS[args.variant]
    original_setup = utils.Setup

    def setup_variant(parsed):
        # Upstream argparse does not accept "Sin" or "RoPE"; set them before
        # Setup so configuration.json records the variant actually trained.
        for key, value in settings.items():
            setattr(parsed, key, value)
        return original_setup(parsed)

    with tempfile.TemporaryDirectory(prefix="convtran-") as temporary:
        selected = Path(temporary)
        (selected / args.dataset).symlink_to(data, target_is_directory=True)

        def select_dataset(config):
            config["data_path"] = str(selected)

        utils.Data_Verifier = select_dataset
        utils.Setup = setup_variant
        upstream_model.model_factory = positional_encodings.model_factory
        install_measurements()
        sys.argv = [
            str(UPSTREAM / "main.py"),
            "--epochs", str(args.epochs),
            "--gpu", "0",
            "--batch_size", str(args.batch_size),
            "--emb_size", str(args.emb_size),
            "--num_heads", str(args.num_heads),
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
    if args.variant == "rope" and (args.emb_size // args.num_heads) % 2:
        parser.error("RoPE needs an even head dimension (emb-size / num-heads).")
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
    row = dict(zip(CONFIG_FIELDS, config))
    row.update(
        accuracy=accuracy,
        test_loss=loss,
        mean_train_epoch_s=metrics["mean_train_epoch_s"],
        train_epoch_times_s=";".join(
            f"{value:.6f}" for value in metrics["train_epoch_times_s"]
        ),
        peak_cuda_allocated_train_val_mib=metrics["peak_cuda_allocated_train_val_mib"],
        device=metrics["device"],
    )
    save_row(row)
    print(f"Saved result to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
