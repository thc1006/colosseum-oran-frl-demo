# 檔案：src/colosseum_oran_frl_demo/data/validate.py
"""
Light-weight integrity checks for KPI parquet files.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def integrity_check(parquet_path: str | Path, timestamp_col: str = "timestamp") -> None:
    """
    Performs light-weight integrity checks on a KPI Parquet file.

    Args:
        parquet_path: The path to the Parquet file.
        timestamp_col: The name of the timestamp column (default: "timestamp").

    Raises:
        ValueError: If the dataset is empty.
    """
    df: pd.DataFrame = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError("Dataset is empty")

    # timestamp monotonic?
    if timestamp_col in df.columns:
        ts_sorted: bool = df[timestamp_col].is_monotonic_increasing
        if not ts_sorted:
            print("⚠️  Timestamp column is NOT sorted")

    # null %
    null_ratio: float = df.isna().mean().mean()
    print(f"🔍 Null ratio = {null_ratio:.3%}")
    if null_ratio > 0.05:
        print("⚠️  >5 % missing values – you may want to clean")

    print("✅  Basic integrity check done")
