"""src/features.py - feature prep and label construction."""
import numpy as np
import pandas as pd

def make_label(df: pd.DataFrame, click_col: str = "click_bool", booking_col: str = "booking_bool") -> pd.Series:
    """
        Graded Relevance: 
        0 = no interaction
        1 = click
        2 = booking
    """
    label = df[click_col].astype(int).copy()
    # boost bookings to 5
    label[df[booking_col] == 1] = 2
    return label

def get_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Fill missing values with -1 (LightGBM handles -1 natively)"""
    X = df[feature_cols].copy()
    X = X.fillna(-1)
    return X

def get_group_sizes(df: pd.DataFrame, group_col: str) -> np.ndarray:
    """
    LightGBM's ranking API requires a 1-D array of group sizes in order.
    IMPORTANT: The dataframe must be sorted by group_col before calling this.
    """
    return df.groupby(group_col, sort=False).size().to_numpy()
