"""src/two_tower_train.py - two tower training and retrieval evaluation."""

import os

# tf-keras (Keras 2 compatibility layer) is installed; activate it before any TF
# import so that tensorflow-recommenders (which requires Keras 2) loads correctly.
# This is the approach recommended by TF for TF >= 2.16 + TFRS:
# https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/keras_version.md
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import json, shutil, time
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from config import TwoTowerRunConfig
from tf_data import make_twotower_dataset_fast, fill_missing
from two_tower import TwoTowerModel, build_tower
from metrics import recall_at_k

def build_item_dataset(df: pd.DataFrame, cfg: dict,
                        batch_size: int = 512) -> tf.data.Dataset:
    """Dataset of item features only — used to build the item index."""
    i_cont = cfg["item_features"]["continuous"]
    i_cat  = list(cfg["item_features"]["categorical"].keys())
    df = fill_missing(df, i_cont, 0.0)
    df = fill_missing(df, i_cat,  0)
    tensors = {f: tf.constant(df[f].values.astype(np.float32)) for f in i_cont}
    tensors |= {f: tf.constant(df[f].values.astype(np.int32))  for f in i_cat}
    return tf.data.Dataset.from_tensor_slices(tensors)


def retrieval_recall_at_k(query_tower, item_tower,
                           val_df: pd.DataFrame, cfg: dict,
                           k_values: list[int]) -> dict:
    """
    Brute-force Recall@k: embed all queries and all items,
    score via dot product, evaluate recall.
    Only feasible at val-set scale; production would use ANN (ScaNN/Faiss).
    """
    q_cont = cfg["query_features"]["continuous"]
    q_cat  = list(cfg["query_features"]["categorical"].keys())
    i_cont = cfg["item_features"]["continuous"]
    i_cat  = list(cfg["item_features"]["categorical"].keys())

    val_df = fill_missing(val_df.copy(), q_cont + i_cont, 0.0)
    val_df = fill_missing(val_df, q_cat + i_cat, 0)
    val_df = val_df.reset_index(drop=True)

    # Embed all items in val set
    item_inputs = {f: tf.constant(val_df[f].values.astype(np.float32)) for f in i_cont}
    item_inputs |= {f: tf.constant(val_df[f].values.astype(np.int32))  for f in i_cat}
    item_embs = item_tower(item_inputs, training=False).numpy()   # (N, 64)

    results = {f"recall@{k}": [] for k in k_values}

    for srch_id, group in val_df.groupby(cfg["group_col"], sort=False):
        idx = group.index.values   # now these are 0-based positions into item_embs

        q_inputs = {f: tf.constant([group[f].iloc[0]], dtype=tf.float32) for f in q_cont}
        q_inputs |= {f: tf.constant([group[f].iloc[0]], dtype=tf.int32)  for f in q_cat}
        q_emb = query_tower(q_inputs, training=False).numpy()[0]   # (64,)

        # Score all items in the group (in real deployment, this is ANN over full corpus)
        scores = item_embs[idx] @ q_emb   # dot product = cosine (embeddings are L2-normed)
        y_true = (group[cfg["label_col"]].values > 0).astype(float)

        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(y_true, scores, k))

    return {m: float(np.mean(v)) for m, v in results.items()}


def train(config_path: str):
    cfg = TwoTowerRunConfig.from_yaml(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    shutil.copy(config_path, f"{cfg.output_dir}/config_snapshot.yaml")

    tf.random.set_seed(cfg.training.seed)

    print("Loading data...")
    train_df = pd.read_parquet(f"{cfg.data['processed_dir']}/train.parquet")
    val_df   = pd.read_parquet(f"{cfg.data['processed_dir']}/val.parquet")

    train_ds = make_twotower_dataset_fast(
        train_df, cfg.data, cfg.training.batch_size, shuffle=True, seed=cfg.training.seed
    )
    item_ds = build_item_dataset(train_df, cfg.data)

    q_cfg = cfg.data["query_features"]
    i_cfg = cfg.data["item_features"]

    query_tower = build_tower(
        q_cfg["continuous"], q_cfg["categorical"],
        cfg.model.tower_dims, cfg.model.dropout
    )
    item_tower = build_tower(
        i_cfg["continuous"], i_cfg["categorical"],
        cfg.model.tower_dims, cfg.model.dropout
    )

    model = TwoTowerModel(query_tower, item_tower, item_ds)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.training.learning_rate))

    print("Training two-tower model...")
    t0 = time.time()
    history = model.fit(train_ds, epochs=cfg.training.epochs, verbose=1)
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    print("\nEvaluating retrieval (Recall@k on val set)...")
    metrics = retrieval_recall_at_k(
        query_tower, item_tower, val_df, cfg.data, cfg.eval.k_values
    )
    print("Retrieval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    metrics["train_time_s"] = round(elapsed, 1)
    with open(f"{cfg.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    query_tower.save(f"{cfg.output_dir}/query_tower")
    item_tower.save(f"{cfg.output_dir}/item_tower")
    print(f"\nSaved to {cfg.output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/twotower_v1.yaml")
    args = parser.parse_args()
    train(args.config)