"""src/tf_data.py - tf.data pipeline for two-tower ranker models."""
import numpy as np
import pandas as pd
import tensorflow as tf

def fill_missing(df: pd.DataFrame, cols: list[str], fill_value: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    df[cols] = df[cols].fillna(fill_value)
    return df

def make_twotower_dataset(df: pd.DataFrame, cfg: dict, batch_size: int, shuffle: bool = True, seed: int = 42) -> tf.data.Dataset:
    """
    Yields dicts: {"query": {feature: tensor}, "item": {feature: tensor}, "label": tensor}
    Label is the graded relevance (0/1/5) used as a positive signal.
    """

    q_cont = cfg["query_features"]["continuous"]
    q_cat = list(cfg["query_features"]["categorical"].keys())
    i_cont = cfg["item_features"]["continuous"]
    i_cat = list(cfg["item_features"]["categorical"].keys())
    all_cols = q_cont + q_cat + i_cont + i_cat + [cfg["label_col"], cfg["booking_col"]]

    df = fill_missing(df, q_cont + i_cont, fill_value=0.0)
    df = fill_missing(df, q_cat + i_cat, fill_value=0)

    # Graded Label: 0, 1, or 5
    label = df[cfg["label_col"]].values.astype(np.float32)
    label[df[cfg["booking_col"]].values == 1] = 5.0

    def gen():
        for i in range(len(df)):
            row = df.iloc[i]
            yield {
                "query": {f: np.float32(row[f]) for f in q_cont} | {f: np.int32(row[f]) for f in q_cat},
                "item": {f: np.float32(row[f]) for f in i_cont} | {f: np.int32(row[f]) for f in i_cat},
                "label": np.float32(label[i])
            }
    
    # Build output signature for tf.data
    q_sig = {f: tf.TensorSpec(shape=(), dtype=tf.float32) for f in q_cont}
    q_sig |= {f: tf.TensorSpec(shape=(), dtype=tf.int32) for f in q_cat}

    i_sig = {f: tf.TensorSpec(shape=(), dtype=tf.float32) for f in i_cont}
    i_sig |= {f: tf.TensorSpec(shape=(), dtype=tf.int32) for f in i_cat}

    sig = ({"query": q_sig, "item": i_sig, "label": tf.TensorSpec(shape=(), dtype=tf.float32)})

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    if shuffle: 
        ds = ds.shuffle(buffer_size=50_000, seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def make_twotower_dataset_fast(df, cfg, batch_size, shuffle=True, seed=42):
    q_cont = cfg["query_features"]["continuous"]
    q_cat  = list(cfg["query_features"]["categorical"].keys())
    i_cont = cfg["item_features"]["continuous"]
    i_cat  = list(cfg["item_features"]["categorical"].keys())

    df = fill_missing(df, q_cont + i_cont, 0.0)
    df = fill_missing(df, q_cat + i_cat,   0)

    label = df[cfg["label_col"]].values.astype(np.float32)
    label[df[cfg["booking_col"]].values == 1] = 5.0

    tensors = {
        "query": {f: tf.constant(df[f].values.astype(np.float32)) for f in q_cont}
               | {f: tf.constant(df[f].values.astype(np.int32))   for f in q_cat},
        "item":  {f: tf.constant(df[f].values.astype(np.float32)) for f in i_cont}
               | {f: tf.constant(df[f].values.astype(np.int32))   for f in i_cat},
        "label": tf.constant(label),
    }
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def make_ranker_dataset(df: pd.DataFrame, cfg: dict, batch_size: int, list_size: int, shuffle: bool = True, seed: int = 42) -> tf.data.Dataset:
    """
    Yields padded query-grouped tensors for TF-Ranking.
    Shape per batch: features (batch, list_size, n_features), labels (batch, list_size)
    Padding value: 0.0 for features, -1.0 for labels (TF-Ranking ignores negatives labels)
    """

    all_feature_cols = (
        cfg["feature_cols"]["query_level"]
        + cfg["feature_cols"]["item_level"]
        + cfg["feature_cols"]["interaction_level"]
    )
    label_col = cfg["label_col"]
    booking_col = cfg["booking_col"]
    group_col = cfg["group_col"]

    df = df.copy()
    df[all_feature_cols] = df[all_feature_cols].fillna(0.0)

    # Graded label: 0, 1, 5
    df["_label"] = df[label_col].astype(float)
    df.loc[df[booking_col] == 1, "_label"] = 5.0
    df["_label"] = df["_label"].map({0.0: 0.0, 1.0: 0.2, 5.0: 1.0})

    groups = []
    for _, group in df.groupby(group_col, sort=False):
        feats  = group[all_feature_cols].values.astype(np.float32)  # (n, F)
        labels = group["_label"].values.astype(np.float32)           # (n,)
        n, F   = feats.shape

        # Pad or truncate to exactly (list_size, F)
        if n >= list_size:
            feats  = feats[:list_size, :]          # explicit second dim
            labels = labels[:list_size]
        else:
            pad_rows = list_size - n
            feats  = np.vstack([
                feats,
                np.zeros((pad_rows, F), dtype=np.float32)
            ])
            labels = np.concatenate([
                labels,
                np.full(pad_rows, -1.0, dtype=np.float32)
            ])

        assert feats.shape == (list_size, F), f"Shape mismatch: {feats.shape}"
        groups.append((feats, labels))

    if shuffle: 
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(groups))
        groups = [groups[i] for i in idx]

    feat_array = np.stack([g[0] for g in groups])
    label_array = np.stack([g[1] for g in groups])

    ds = tf.data.Dataset.from_tensor_slices((feat_array, label_array))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)



# import pandas as pd, yaml
# # from tf_data import make_twotower_dataset_fast

# with open("configs/twotower_v1.yaml") as f:
#     cfg = yaml.safe_load(f)

# df = pd.read_parquet("data/processed/v1/train.parquet").head(10_000)
# ds = make_twotower_dataset_fast(df, cfg["data"], batch_size=32)
# batch = next(iter(ds))
# print("Query keys:", list(batch["query"].keys()))
# print("Item keys:", list(batch["item"].keys()))
# print("Label shape:", batch["label"].shape)
# print("Label sample:", batch["label"][:8].numpy())
# # Should see 0s, 1s, and occasional 5s

# import pandas as pd, yaml, numpy as np
# import tensorflow as tf
# from .tf_data import make_ranker_dataset

# with open("configs/ranker_v1.yaml") as f:
#     cfg = yaml.safe_load(f)

# df = pd.read_parquet("data/processed/v1/train.parquet").head(50_000)
# ds = make_ranker_dataset(df, cfg["data"], batch_size=64, list_size=25)
# feats, labels = next(iter(ds))
# print("Features shape:", feats.shape)   # expect (64, 25, 16)
# print("Labels shape:",   labels.shape)  # expect (64, 25)
# print("Label sample (first query):", labels[0].numpy())
# # Should see mix of 0s, some 1s, occasional 5s, and -1s (padding)