import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import pytest
from colosseum_oran_frl_demo.data.dataset import make_parquet, _parse_common_info, _convert_single

@pytest.fixture
def dummy_csv_data():
    return {
        "timestamp": [0, 1, 2],
        "Throughput_DL_Mbps": [10, 20, 30],
        "Latency_proxy_ms": [5, 8, 12],
        "BS_ID": [1, 1, 1],
        "Slice_ID": [0, 2, 0]
    }

@pytest.fixture
def create_dummy_csv(tmp_path, dummy_csv_data):
    def _creator(filename="kpi.csv", sub_dirs=None):
        path = tmp_path
        if sub_dirs:
            for sd in sub_dirs:
                path = path / sd
                path.mkdir(exist_ok=True)
        csv_file = path / filename
        pd.DataFrame(dummy_csv_data).to_csv(csv_file, index=False)
        return csv_file
    return _creator

# --- Unit tests for _parse_common_info ---
def test_parse_common_info_standard_path():
    csv_path = Path("/path/to/raw/schedA/tr0/exp0/bs0/kpi.csv")
    info = _parse_common_info(csv_path)
    assert info == {"sched": "schedA", "tr": "tr0", "exp": "exp0", "bs": "bs0"}

def test_parse_common_info_different_filename():
    csv_path = Path("/data/raw/schedulerB/traffic1/experimentX/basestationY/another_kpi_file.csv")
    info = _parse_common_info(csv_path)
    assert info == {"sched": "schedulerB", "tr": "traffic1", "exp": "experimentX", "bs": "basestationY"}

def test_parse_common_info_short_path():
    csv_path = Path("kpi.csv")
    info = _parse_common_info(csv_path)
    assert info == {"sched": "na", "tr": "na", "exp": "na", "bs": "na"}

# --- Unit tests for _convert_single ---
def test_convert_single(tmp_path, create_dummy_csv, dummy_csv_data):
    raw_dir = tmp_path / "raw" / "schedA" / "tr0" / "exp0" / "bs0"
    csv_file = create_dummy_csv(sub_dirs=["raw", "schedA", "tr0", "exp0", "bs0"])
    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    _convert_single(csv_file, out_dir)

    parquet_file = out_dir / "kpi.parquet"
    assert parquet_file.exists()

    df_converted = pd.read_parquet(parquet_file)
    assert not df_converted.empty
    assert len(df_converted) == len(dummy_csv_data["timestamp"])
    assert df_converted["Throughput_DL_Mbps"].iloc[0] == 10
    assert "sched" in df_converted.columns
    assert df_converted["sched"].iloc[0] == "schedA"
    assert df_converted["tr"].iloc[0] == "tr0"
    assert df_converted["exp"].iloc[0] == "exp0"
    assert df_converted["bs"].iloc[0] == "bs0"

def test_convert_single_empty_csv(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_file = raw_dir / "empty.csv"
    pd.DataFrame(columns=["timestamp", "col1"]).to_csv(csv_file, index=False)

    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    _convert_single(csv_file, out_dir)

    parquet_file = out_dir / "empty.parquet"
    assert parquet_file.exists()
    df_converted = pd.read_parquet(parquet_file)
    assert df_converted.empty
    assert "sched" in df_converted.columns # Metadata columns should still be added

# --- Integration tests for make_parquet ---
def test_make_parquet_single_file(tmp_path, create_dummy_csv, dummy_csv_data):
    raw_dir = tmp_path / "raw"
    create_dummy_csv(sub_dirs=["raw", "s1", "t1", "e1", "b1"], filename="file1.csv")
    out_dir = tmp_path / "processed"
    make_parquet(raw_dir, out_dir, n_jobs=1)

    files = list(out_dir.glob("*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert not df.empty
    assert df["sched"].iloc[0] == "s1"
    assert df["tr"].iloc[0] == "t1"

def test_make_parquet_multiple_files(tmp_path, create_dummy_csv):
    raw_dir = tmp_path / "raw"
    create_dummy_csv(sub_dirs=["raw", "s1", "t1", "e1", "b1"], filename="file1.csv")
    create_dummy_csv(sub_dirs=["raw", "s2", "t2", "e2", "b2"], filename="file2.csv")
    out_dir = tmp_path / "processed"
    make_parquet(raw_dir, out_dir, n_jobs=2)

    files = list(out_dir.glob("*.parquet"))
    assert len(files) == 2

    df1 = pd.read_parquet(out_dir / "file1.parquet")
    assert df1["sched"].iloc[0] == "s1"
    df2 = pd.read_parquet(out_dir / "file2.parquet")
    assert df2["sched"].iloc[0] == "s2"

def test_make_parquet_different_pattern(tmp_path, create_dummy_csv):
    raw_dir = tmp_path / "raw"
    create_dummy_csv(sub_dirs=["raw"], filename="data.txt") # Not a csv
    create_dummy_csv(sub_dirs=["raw"], filename="kpi.csv")
    out_dir = tmp_path / "processed"
    make_parquet(raw_dir, out_dir, pattern="*.csv", n_jobs=1)

    files = list(out_dir.glob("*.parquet"))
    assert len(files) == 1
    assert (out_dir / "kpi.parquet").exists()

def test_make_parquet_no_csv_found(tmp_path: Path) -> None:
    """
    Tests make_parquet when no CSV files are found.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    
    with pytest.raises(FileNotFoundError, match=r"No CSV found under .*"):
        make_parquet(raw_dir, out_dir)

def test_make_parquet_output_directory_creation(tmp_path, create_dummy_csv):
    raw_dir = tmp_path / "input_data"
    raw_dir.mkdir()
    create_dummy_csv(filename="test.csv", sub_dirs=["input_data"])
    out_dir = tmp_path / "new_processed_data" / "sub_dir"
    
    # out_dir does not exist initially
    assert not out_dir.exists()
    
    make_parquet(raw_dir, out_dir, n_jobs=1)
    
    # out_dir should be created after make_parquet runs
    assert out_dir.exists()
    assert len(list(out_dir.glob("*.parquet"))) == 1