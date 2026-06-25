"""src/pipeline.py - query-grouped train/val/test split."""
import os
import pandas as pd
import numpy as np
from .config import PipelineConfig

def grouped_split(df: pd.DataFrame, group_col: str, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    rng.shuffle(groups)

    n = len(groups)
    n_test = int(n * test_frac)
    n_val = int(n* val_frac)

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test+n_val])
    train_groups = set(groups[n_test+n_val:])

    return (
        df[df[group_col].isin(train_groups)],
        df[df[group_col].isin(val_groups)],
        df[df[group_col].isin(test_groups)],
    )

def main(config_path: str, nrows: int | None = None):
    cfg = PipelineConfig.from_yaml(config_path)
    df = pd.read_csv(cfg.data.raw_path, nrows=nrows)

    train, val, test = grouped_split(df, cfg.data.group_col, cfg.split.val_frac, cfg.split.test_frac, cfg.split.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    train.to_parquet(f"{cfg.output_dir}/train.parquet")
    val.to_parquet(f"{cfg.output_dir}/val.parquet")
    test.to_parquet(f"{cfg.output_dir}/test.parquet")

    print(f"Train groups: {train[cfg.data.group_col].nunique():,} ({len(train):,} rows)")
    print(f"Val groups: {val[cfg.data.group_col].nunique():,} ({len(val):,} rows)")
    print(f"Test groups: {test[cfg.data.group_col].nunique():,} ({len(test):,} rows)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_v1.yaml")
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.nrows)