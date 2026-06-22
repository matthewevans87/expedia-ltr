"""src/lgbm_train.py - LightGBM LambdaRank baseline training."""

import json
import os
import shutil
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import LGBMRunConfig
from features import make_label, get_feature_matrix, get_group_sizes
from metrics import evaluate_groups

def load_split(processed_dir: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(f"{processed_dir}/{split}.parquet")

def train(config_path: str):
    cfg = LGBMRunConfig.from_yaml(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    shutil.copy(config_path, f"{cfg.output_dir}/config_snapshot.yaml")

    feature_cols = (
        cfg.data["feature_cols"]["query_level"]
        + cfg.data["feature_cols"]["item_level"]
        + cfg.data["feature_cols"]["interaction_level"]
    )

    # --- Load splits (sorted by group for LightGBM) ---
    print("Loading data...")
    group_col = cfg.data["group_col"]
    label_col = cfg.data["label_col"]
    booking_col = cfg.data["booking_col"]

    train_df = load_split(cfg.data["processed_dir"], "train").sort_values(group_col)
    val_df = load_split(cfg.data["processed_dir"], "val").sort_values(group_col)

    X_train = get_feature_matrix(train_df, feature_cols)
    y_train = make_label(train_df, label_col, booking_col)
    g_train = get_group_sizes(train_df, group_col)

    X_val = get_feature_matrix(val_df, feature_cols)
    y_val = make_label(val_df, label_col, booking_col)
    g_val = get_group_sizes(val_df, group_col)

    train_data = lgb.Dataset(X_train, label=y_train, group=g_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_data)

    params = {
        "objective": cfg.lgbm.objective,
        "metric": cfg.lgbm.metric,
        "ndcg_eval_at": cfg.lgbm.ndcg_eval_at,
        "num_leaves": cfg.lgbm.num_leaves,
        "learning_rate": cfg.lgbm.learning_rate,
        "label_gain": cfg.lgbm.label_gain,
        "seed": cfg.lgbm.seed,
        "verbose": -1
    }

    print("Training LightGBM LambdaRank...")
    t0 = time.time()
    model = lgb.train(
        params, 
        train_data,
        num_boost_round=cfg.lgbm.n_estimators,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(cfg.lgbm.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s. Best iteration: {model.best_iteration}")

    model.save_model(f"{cfg.output_dir}/model.txt")

    # --- Evaluate on val and test ---
    for split_name, split_df in [("val", val_df), ("test", load_split(cfg.data["processed_dir"], "test"))]:
        X = get_feature_matrix(split_df, feature_cols)
        scores = model.predict(X)
        split_df = split_df.copy()
        split_df["score"] = scores
        split_df["label"] = make_label(split_df, label_col, booking_col)
        metrics = evaluate_groups(split_df, score_col="score", label_col="label", group_col=group_col, k_values=cfg.eval.k_values)
        print(f"\n{split_name.upper()} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
    metrics["best_iteration"] = model.best_iteration
    metrics["train_time_s"] = round(elapsed, 1)
    with open(f"{cfg.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {cfg.output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lgbm_v1.yaml")
    args = parser.parse_args()
    train(args.config)