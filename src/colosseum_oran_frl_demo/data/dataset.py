# ─── src/colosseum_oran_frl_demo/data/dataset.py ───
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import itertools
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _parse_common_info(csv_path: Path) -> dict:
    parts = csv_path.as_posix().split("/")
    sched, tr, exp, bs = parts[-5:-1] if len(parts) >= 5 else ("na",) * 4
    return dict(sched=sched, tr=tr, exp=exp, bs=bs)


def _convert_single(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    info = _parse_common_info(csv_path)
    for k, v in info.items():
        df[k] = v
    pq.write_table(
        pa.Table.from_pandas(df),
        out_dir / f"{csv_path.stem}.parquet",
        compression="zstd",
    )


def make_parquet(
    raw_root: Path, out_dir: Path, pattern: str = "*.csv", n_jobs: int = 4
):
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list(raw_root.rglob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found under {raw_root}")

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        ex.map(_convert_single, csv_files, itertools.repeat(out_dir))
    print(f"✅  {len(csv_files)} CSV ➜ {out_dir} (Parquet)")
