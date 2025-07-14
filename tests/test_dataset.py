# ─── tests/test_dataset.py ───
import pandas as pd
from pathlib import Path
from colosseum_oran_frl_demo.data.dataset import make_parquet


def test_make_parquet(tmp_path: Path):
    raw_dir = tmp_path / """raw"""
    raw_dir.mkdir()
    csv = raw_dir / """schedA_tr0_exp0_bs0_kpi.csv"""
    pd.DataFrame({"""timestamp""": [0, 1], """Throughput_DL_Mbps""": [10, 20]}).to_csv(
        csv, index=False
    )

    out_dir = tmp_path / """processed"""
    make_parquet(raw_dir, out_dir, n_jobs=1)

    files = list(out_dir.glob("""*.parquet"""))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert {"""sched""", """tr""", """exp""", """bs"""}.issubset(df.columns)
