# Quick eval script — run this instead of retraining
import json, yaml
import pandas as pd
import tensorflow as tf
from tf_data import fill_missing
from two_tower_train import retrieval_recall_at_k  # after patching

with open("configs/twotower_v1.yaml") as f:
    cfg = yaml.safe_load(f)

val_df = pd.read_parquet("data/processed/v1/val.parquet")
query_tower = tf.saved_model.load("runs/twotower_v1/query_tower")
item_tower  = tf.saved_model.load("runs/twotower_v1/item_tower")

metrics = retrieval_recall_at_k(
    query_tower, item_tower, val_df, cfg["data"], k_values=[5, 10, 20]
)
print(metrics)
with open("runs/twotower_v1/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)