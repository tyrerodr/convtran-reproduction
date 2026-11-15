# ConvTran reproduction

We reproduce the released ConvTran implementation on the 30 UEA multivariate time series datasets and evaluate changes to positional encoding, the convolutional front end, and final pooling. The released-code baseline completed 28 datasets. MotorImagery and EigenWorms exceeded GPU memory at batch size 16 and were evaluated separately in feasibility runs.

[ConvTran paper](https://doi.org/10.1007/s10618-023-00948-2) · [Official implementation](https://github.com/Navidfoumani/ConvTran/tree/148afb6ca549915b7e78b05e2ec3b4ba6e052341)

## Environment

From a Linux shell with Git, uv, and an NVIDIA driver compatible with CUDA 11.8:

```bash
git clone --recurse-submodules https://github.com/tyrerodr/convtran-reproduction.git
cd convtran-reproduction
uv python install 3.10
uv sync --locked
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

For an existing checkout, initialise the upstream submodule with `git submodule update --init --recursive`. The benchmarks used an RTX 4070 Ti with 12 GB VRAM and PyTorch 2.0.1 with CUDA 11.8. In Jupyter or VS Code, select the repository's `.venv/bin/python` as the notebook kernel.

## Data and baseline

Open [notebook 02](notebooks/02_uea_benchmark.ipynb) and run all cells. It downloads [Multivariate2018_ts.zip](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip) to `Dataset/archives/`, extracts the TRAIN and TEST files under `Dataset/UEA/Multivariate_ts/<dataset>/`, and checks the split files and dataset dimensions. The published evaluation also includes Actitracker HAR and Ford. This reproduction covers UEA.

Run the initial BasicMotions check and the two memory feasibility runs from the repository root:

```bash
uv run python scripts/run_baseline.py --dataset BasicMotions
uv run python scripts/run_baseline.py --dataset MotorImagery --batch-size 8 --seed 1234
uv run python scripts/run_baseline.py --dataset EigenWorms --batch-size 1 --seed 1234 --temporal-pool-factor 4
```

[Notebook 01](notebooks/01_reproduction.ipynb) describes BasicMotions. [Notebook 03](notebooks/03_baseline_training.ipynb) documents the baseline and memory pilots. To rerun the 28 completed baseline datasets using the names in the tracked baseline CSV, run:

```bash
cut -d, -f1 results/baseline_runs.csv | tail -n +2 | sort -u | while IFS= read -r dataset; do
  uv run python scripts/run_baseline.py --dataset "$dataset" || exit 1
done
```

The baseline uses 100 epochs, batch size 16, embedding dimension 16, eight heads, tAPE + eRPE, RAdam at 0.001, and seeds 1234, 2345, 3456, 4567 and 5678. Seeds share the upstream validation split. The checkpoint with lowest validation loss is evaluated on the official test split. The paper describes embedding size 64, Adam, GELU in the feed-forward block and early stopping. The released configuration uses embedding size 16, RAdam, ReLU in that block and a fixed epoch budget.

## Extension experiments

Set the mode below and use **Run All** in the corresponding notebook. Set `RUN_TRAINING = False` before opening the results tables without training. Set it to `True` to launch the selected plan. Output directories contain logs and checkpoints. The experimental runners reuse complete logs for the same configuration and print a `--retry` command for incomplete runs.

| Notebook | Mode | Comparison | Recorded results |
|---|---|---|---|
| [04 Positional encoding](notebooks/04_positional_encoding_experiments.ipynb) | benchmark | tAPE + eRPE, sinusoidal and RoPE across 28 datasets (420 runs) | [CSV](results/positional_encoding_runs.csv) |
| [05 Multiscale front end](notebooks/05_multiscale_frontend_experiments.ipynb) | comparison | Original and multiscale stems on Handwriting, FaceDetection and PEMS-SF (30 runs) | [CSV](results/multiscale_frontend_runs.csv) |
| [06 Pooling](notebooks/06_pooling_experiments.ipynb) | benchmark | Global average and learned average + maximum across 28 datasets (280 runs) | [CSV](results/pooling_runs.csv) |

Each comparison uses five seeds and 100 epochs per dataset and variant. Each notebook also has a two-epoch BasicMotions pilot. Example commands for one seed of each extension:

```bash
uv run python scripts/run_positional_experiment.py --variant rope --dataset Handwriting --seed 1234 --epochs 100 --batch-size 16 --emb-size 16 --num-heads 8
uv run python scripts/run_multiscale_experiment.py --variant multiscale --dataset Handwriting --seed 1234 --epochs 100 --batch-size 16 --emb-size 16 --num-heads 8
uv run python scripts/run_pooling_experiment.py --variant avg_max --dataset Handwriting --seed 1234 --epochs 100 --batch-size 16 --emb-size 16 --num-heads 8
```

## Reproduction results

Published accuracies come from Table 2 of the ConvTran paper. Our baseline values are five-seed means from [baseline_runs.csv](results/baseline_runs.csv). The difference is our result minus the published value, in percentage points.

| Dataset | Published accuracy (%) | Our baseline (%) | Difference (pp) |
|---|---:|---:|---:|
| ArticularyWordRecognition | 98.33 | 96.80 | -1.53 |
| AtrialFibrillation | 40.00 | 32.00 | -8.00 |
| BasicMotions | 100.00 | 100.00 | 0.00 |
| CharacterTrajectories | 99.22 | 99.16 | -0.06 |
| Cricket | 100.00 | 98.33 | -1.67 |
| DuckDuckGeese | 62.00 | 61.20 | -0.80 |
| EigenWorms | 59.34 | OOM | --- |
| Epilepsy | 98.55 | 96.38 | -2.17 |
| ERing | 96.29 | 84.15 | -12.14 |
| EthanolConcentration | 36.12 | 29.20 | -6.92 |
| FaceDetection | 67.22 | 65.42 | -1.80 |
| FingerMovements | 56.00 | 52.80 | -3.20 |
| HandMovementDirection | 40.54 | 34.05 | -6.49 |
| Handwriting | 37.52 | 23.58 | -13.94 |
| Heartbeat | 78.53 | 73.27 | -5.26 |
| InsectWingbeat | 71.32 | 70.83 | -0.49 |
| JapaneseVowels | 98.91 | 98.49 | -0.42 |
| Libras | 92.77 | 83.78 | -8.99 |
| LSST | 61.56 | 61.09 | -0.47 |
| MotorImagery | 56.00 | OOM | --- |
| NATOPS | 94.44 | 94.22 | -0.22 |
| PEMS-SF | 82.84 | 72.83 | -10.01 |
| PenDigits | 98.71 | 98.25 | -0.46 |
| PhonemeSpectra | 30.62 | 26.69 | -3.93 |
| RacketSports | 86.18 | 82.24 | -3.94 |
| SelfRegulationSCP1 | 91.80 | 87.17 | -4.63 |
| SelfRegulationSCP2 | 58.33 | 53.44 | -4.89 |
| SpokenArabicDigits | 99.45 | 99.05 | -0.40 |
| StandWalkJump | 33.33 | 32.00 | -1.33 |
| UWaveGestureLibrary | 89.06 | 85.69 | -3.37 |

MotorImagery reached 54.00% at batch size 8. EigenWorms reached 48.85% at batch size 1 with factor-4 temporal average pooling. These runs used seed 1234 and are recorded in [adapted_runs.csv](results/adapted_runs.csv) and [adapted_runs_batch1_pool4.csv](results/adapted_runs_batch1_pool4.csv), respectively.

## Extension results

| Experiment | Mean test accuracy | Scope |
|---|---|---|
| Positional encoding | tAPE + eRPE 70.95%, sinusoidal 69.73%, RoPE 69.61% | 28 datasets |
| Multiscale front end | Original 53.77%, multiscale 53.40% | 3 datasets |
| Final pooling comparison | Global average 70.96%, average + maximum 70.97% | 28 datasets |

Each row uses its own matched control. Dataset means receive equal weight. Combined pooling improved accuracy on 12 datasets, tied on 6 and scored lower on 10. Mean training time per epoch after the first increased by 2.1%, with nearly identical mean peak CUDA allocation.

