# 檔案：scripts/make_dataset.py
import argparse
import colosseum_oran_frl_demo.data.dataset as ds


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("""--raw""", required=True, help="""Raw KPI CSV root""")
    p.add_argument("""--out""", required=True, help="""Parquet output dir""")
    p.add_argument("""--jobs""", type=int, default=4)
    args = p.parse_args()
    ds.make_parquet(args.raw, args.out, n_jobs=args.jobs)


if __name__ == """__main__""":
    cli()
