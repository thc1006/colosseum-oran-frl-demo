import pandas as pd
import pytest
from pathlib import Path
from colosseum_oran_frl_demo.data.validate import integrity_check


def test_integrity_check_empty_dataframe(tmp_path: Path) -> None:
    """
    Tests that integrity_check raises ValueError for an empty DataFrame.
    """
    empty_parquet = tmp_path / "empty.parquet"
    pd.DataFrame().to_parquet(empty_parquet)
    with pytest.raises(ValueError, match="Dataset is empty"):
        integrity_check(empty_parquet)

def test_integrity_check_monotonic_timestamp(tmp_path: Path) -> None:
    """
    Tests integrity_check with monotonic and non-monotonic timestamps.
    """
    # Monotonic timestamp
    monotonic_data = {"timestamp": [1, 2, 3], "value": [10, 20, 30]}
    monotonic_parquet = tmp_path / "monotonic.parquet"
    pd.DataFrame(monotonic_data).to_parquet(monotonic_parquet)
    integrity_check(monotonic_parquet)  # Should not raise an error

    # Non-monotonic timestamp
    non_monotonic_data = {"timestamp": [1, 3, 2], "value": [10, 20, 30]}
    non_monotonic_parquet = tmp_path / "non_monotonic.parquet"
    pd.DataFrame(non_monotonic_data).to_parquet(non_monotonic_parquet)
    # We expect a print statement for non-monotonic, not an error
    integrity_check(non_monotonic_parquet)

def test_integrity_check_null_ratio(tmp_path: Path) -> None:
    """
    Tests integrity_check with different null ratios.
    """
    # Low null ratio
    low_null_data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    low_null_parquet = tmp_path / "low_null.parquet"
    pd.DataFrame(low_null_data).to_parquet(low_null_parquet)
    integrity_check(low_null_parquet)  # Should not print warning

    # High null ratio
    high_null_data = {"col1": [1, None, 3], "col2": [4, 5, None]}
    high_null_parquet = tmp_path / "high_null.parquet"
    pd.DataFrame(high_null_data).to_parquet(high_null_parquet)
    # We expect a print statement for high null ratio, not an error
    integrity_check(high_null_parquet)

def test_integrity_check_no_timestamp_column(tmp_path: Path) -> None:
    """
    Tests integrity_check when no timestamp column is present.
    """
    no_ts_data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    no_ts_parquet = tmp_path / "no_ts.parquet"
    pd.DataFrame(no_ts_data).to_parquet(no_ts_parquet)
    integrity_check(no_ts_parquet)  # Should not raise an error
