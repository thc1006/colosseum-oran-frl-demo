# ─── tests/test_dataset.py ───
import pandas as pd
from pathlib import Path
from colosseum_oran_frl_demo.data.dataset import make_parquet
import pytest

def test_make_parquet(tmp_path: Path) -> None:
    """
    Tests the make_parquet function for correct conversion and data integrity.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_file = raw_dir / "schedA_tr0_exp0_bs0_kpi.csv"
    
    # Create a dummy CSV file
    dummy_data = {
        "timestamp": [0, 1, 2],
        "Throughput_DL_Mbps": [10, 20, 30],
        "Latency_proxy_ms": [5, 8, 12],
        "BS_ID": [1, 1, 1],
        "Slice_ID": [0, 2, 0]
    }
    pd.DataFrame(dummy_data).to_csv(csv_file, index=False)

    out_dir = tmp_path / "processed"
    make_parquet(raw_dir, out_dir, n_jobs=1)

    files = list(out_dir.glob("*.parquet"))
    assert len(files) == 1
    
    df = pd.read_parquet(files[0])
    assert {"sched", "tr", "exp", "bs"}.issubset(df.columns)
    assert not df.empty
    assert len(df) == len(dummy_data["timestamp"])
    assert df["Throughput_DL_Mbps"].iloc[0] == 10

def test_make_parquet_no_csv_found(tmp_path: Path) -> None:
    """
    Tests make_parquet when no CSV files are found.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    
    with pytest.raises(FileNotFoundError, match=f"No CSV found under {raw_dir}"):
        make_parquet(raw_dir, out_dir)
