"""src/data_stats.py - dataset summary report"""

import argparse
import pandas as pd

def summarize(df: pd.DataFrame, group_col: str, label_cols: list[str], position_col: str | None):
    print(f"Rows: {len(df):,}")
    print(f"Unique Groups ({group_col}): {df[group_col].nunique():,}")

    group_sizes = df.groupby(group_col).size()
    print("Group size distribution:")
    print(group_sizes.describe())

    print("Label base rates:")
    print(df[label_cols].mean())

    if position_col and position_col in df.columns:
        print("Click/Booking rate by position (top 10):")
        rate_by_position = df.groupby(position_col)[label_cols].mean().head(10)
        print(rate_by_position)
    
    print("Missingness (top 15 columns by % missing):")
    missing = df.isna().mean().sort_values(ascending=False).head(15)
    print(missing[missing > 0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/raw/train.csv")
    parser.add_argument("--nrows", type=int, default=500_000)
    args = parser.parse_args()

    df = pd.read_csv(args.path, nrows=args.nrows)
    summarize(
        df,
        group_col="srch_id",
        label_cols=["click_bool", "booking_bool"],
        position_col="position",
    )