# Learning to Rank Hotel Search Results - A Two-Stage Retrieval and Ranking Pipeline on the Expedia Dataset

End-to-end learning-to-rank pipeline on ~1M real hotel search logs, covering data ingestion, two-tower retrieval, and neural and gradient-boosted rankers. See [report/report.md](report/report.md) for full methodology, results, and analysis.

## TL;DR
- Built an end-to-end LTR pipeline (data ingestion, retrieval, ranking, evaluation) on ~1M real hotel search logs with graded relevance labels
- LightGBM LambdaRank baseline: `NDCG@5 = 0.379`, stable across an 18-run hyperparameter sweep (range 0.012), establishing that the bottleneck is features, not tuning
- Two-tower retrieval (TFRS, in-batch negatives): `Recall@20 = 0.781` after 30 epochs; loss plateaus at epoch ~10, indicating the ceiling is feature expressiveness and negative quality, not training duration. 
- Neural listwise ranker (TF, ApproxNDCG): `NDCG@5 = 0.110`; val loss better than train loss rules out overfitting; the gap versus LightGBM is from model architecture, not a tuning failure

## Project Structure

```
configs/          # YAML configuration files for each model and the data pipeline
data/             # Raw CSV data and processed parquet splits
report/           # Full report with methodology, results, and figures
runs/             # Saved model artefacts and metrics from training runs
src/expedia_ltr/
  pipeline.py         # Query-grouped train/val/test split
  features.py         # Feature definitions and preprocessing
  config.py           # Typed config dataclasses
  lgbm_train.py       # LightGBM LambdaRank training and evaluation
  lgbm_sweep.py       # Hyperparameter sweep over LightGBM configs
  two_tower.py        # Two-tower model definition (TFRS)
  two_tower_train.py  # Two-tower training
  two_tower_eval.py   # Retrieval evaluation (Recall@k)
  ranker.py           # Neural ranker model definition (ApproxNDCG)
  ranker_train.py     # Neural ranker training and evaluation
  tf_data.py          # TensorFlow dataset pipeline
  metrics.py          # NDCG, MRR, Recall metric implementations
  data_stats.py       # Dataset statistics and EDA utilities
```

## Setup

```bash
pip install -e .
```

## Usage

**1. Process data** (train/val/test split):
```bash
python -m expedia_ltr.pipeline --config configs/data_v1.yaml
```

**2. Train LightGBM ranker** (baseline):
```bash
python -m expedia_ltr.lgbm_train --config configs/lgbm_v1.yaml
```

**3. Train two-tower retrieval model**:
```bash
python -m expedia_ltr.two_tower_train --config configs/twotower_v1.yaml
```

**4. Train neural listwise ranker**:
```bash
python -m expedia_ltr.ranker_train --config configs/ranker_v1.yaml
```

Each training run saves a config snapshot, metrics JSON, and model artifacts to the `output_dir` specified in the config.
