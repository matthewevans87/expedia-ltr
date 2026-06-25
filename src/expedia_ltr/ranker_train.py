"""src/ranker_train.py — neural ranker training and evaluation."""
import json, os, shutil, time
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from .tf_data import make_ranker_dataset
from .ranker import build_ranker
from .metrics import evaluate_groups


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    shutil.copy(config_path, f"{cfg['output_dir']}/config_snapshot.yaml")
    tf.random.set_seed(cfg["training"]["seed"])

    all_feature_cols = (
        cfg["data"]["feature_cols"]["query_level"]
        + cfg["data"]["feature_cols"]["item_level"]
        + cfg["data"]["feature_cols"]["interaction_level"]
    )
    n_features = len(all_feature_cols)

    print("Loading data...")
    train_df = pd.read_parquet(f"{cfg['data']['processed_dir']}/train.parquet")
    val_df   = pd.read_parquet(f"{cfg['data']['processed_dir']}/val.parquet")
    test_df  = pd.read_parquet(f"{cfg['data']['processed_dir']}/test.parquet")

    train_ds = make_ranker_dataset(
        train_df, cfg["data"],
        batch_size=cfg["training"]["batch_size"],
        list_size=cfg["training"]["list_size"],
        shuffle=True, seed=cfg["training"]["seed"]
    )
    val_ds = make_ranker_dataset(
        val_df, cfg["data"],
        batch_size=cfg["training"]["batch_size"],
        list_size=cfg["training"]["list_size"],
        shuffle=False
    )

    model = build_ranker(
        n_features=n_features,
        hidden_dims=cfg["model"]["hidden_dims"],
        dropout=cfg["model"]["dropout"],
        use_batch_norm=cfg["model"]["use_batch_norm"],
        learning_rate=cfg["training"]["learning_rate"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_ndcg@5", patience=5, mode="max",
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_ndcg@5", factor=0.5, patience=3,
            mode="max", verbose=1
        ),
    ]

    print("\nTraining neural ranker...")
    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg["training"]["epochs"],
        callbacks=callbacks, verbose=1
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s")

    # Save before evaluating (lesson from Day 3)
    model.save(f"{cfg['output_dir']}/model.keras")

    # --- Evaluate using shared eval harness (same code as LightGBM) ---
    # Group-wise scoring: keeps list_size consistent with training,
    # avoiding the BatchNorm single-item statistics bug.
    all_metrics = {}
    list_size = cfg["training"]["list_size"]

    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        split_df = split_df.copy().reset_index(drop=True)
        split_df[all_feature_cols] = split_df[all_feature_cols].fillna(0.0)

        # Graded label for eval (unscaled — evaluate_groups uses label > 0 for relevance)
        split_df["label"] = split_df[cfg["data"]["label_col"]].astype(float)
        split_df.loc[split_df[cfg["data"]["booking_col"]] == 1, "label"] = 5.0

        all_scores = np.zeros(len(split_df), dtype=np.float32)

        for _, group in split_df.groupby(cfg["data"]["group_col"], sort=False):
            idx = group.index.values
            feats = group[all_feature_cols].values.astype(np.float32)  # (n, F)
            n, F = feats.shape

            if n <= list_size:
                # Pad to list_size
                if n < list_size:
                    pad = np.zeros((list_size - n, F), dtype=np.float32)
                    feats_padded = np.vstack([feats, pad])
                else:
                    feats_padded = feats

                scores = model(
                    feats_padded[np.newaxis, :, :], training=False
                ).numpy()[0, :n]

            else:
                # Score in list_size chunks, concatenate
                chunk_scores = []
                for start in range(0, n, list_size):
                    chunk = feats[start:start + list_size]
                    c = len(chunk)
                    if c < list_size:
                        pad = np.zeros((list_size - c, F), dtype=np.float32)
                        chunk = np.vstack([chunk, pad])
                    s = model(
                        chunk[np.newaxis, :, :], training=False
                    ).numpy()[0, :c]
                    chunk_scores.append(s)
                scores = np.concatenate(chunk_scores)

            all_scores[idx] = scores

        split_df["score"] = all_scores

        metrics = evaluate_groups(
            split_df, score_col="score", label_col="label",
            group_col=cfg["data"]["group_col"],
            k_values=cfg["eval"]["k_values"]
        )
        print(f"\n{split_name.upper()} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        all_metrics[split_name] = metrics

    all_metrics["train_time_s"] = round(elapsed, 1)
    all_metrics["history"] = {k: [float(v) for v in vals]
                               for k, vals in history.history.items()}
    with open(f"{cfg['output_dir']}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved to {cfg['output_dir']}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ranker_v1.yaml")
    args = parser.parse_args()
    train(args.config)