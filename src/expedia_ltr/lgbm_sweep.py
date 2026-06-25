"""src/lgbm_sweep.py - ablation over num_leaves and learning_rate."""
import itertools
import json
import os
import copy
import yaml
from .lgbm_train import train
from .config import LGBMRunConfig

SWEEP = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.02, 0.05],
    "seed": [42, 7, 123]
}

base_config_path = "configs/lgbm_v1.yaml"
with open(base_config_path) as f:
    base = yaml.safe_load(f)

keys, values = zip(*SWEEP.items())
summary = []

for combo in itertools.product(*values):
    params = dict(zip(keys, combo))
    run_name = "_".join(f"{k}{v}" for k, v in params.items())
    run_dir = f"runs/lgbm_sweep/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    cfg = copy.deepcopy(base)
    cfg["lgbm"].update(params)
    cfg["output_dir"] = run_dir

    config_path = f"{run_dir}/config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    train(config_path)

    with open(f"{run_dir}/metrics.json") as f:
        metrics = json.load(f)
    summary.append({"run": run_name, **params, **metrics})

with open(f"runs/lgbm_sweep/summary.json", "w") as f:
    json.dump(summary, f)

import pandas as pd
df = pd.DataFrame(summary).sort_values("ndcg@5", ascending=False)
print("\nSweep results (sorted by NDGC@5):")
print(df[["run", "num_leaves", "learning_rate", "seed", "ndcg@5", "ndcg@10", "mrr"]].to_string(index=False))