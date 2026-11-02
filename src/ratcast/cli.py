"""`ratcast <step>` runs one stage of the pipeline; `ratcast all` runs them in order."""

import argparse
import time

import pandas as pd

from ratcast import config

STEPS = ["download", "clean", "features", "train", "evaluate", "maps", "site"]


def run_download(force: bool) -> None:
    from ratcast.download import download, download_boroughs

    download(force=force)
    download_boroughs(force=force)


def run_clean(_: bool) -> None:
    from ratcast.clean import clean

    events = clean(pd.read_parquet(config.RAW_PARQUET))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(config.EVENTS_PARQUET, index=False)


def run_features(_: bool) -> None:
    from ratcast.features import build_forecast_dataset

    feats = build_forecast_dataset(pd.read_parquet(config.EVENTS_PARQUET))
    feats.to_parquet(config.FEATURES_PARQUET, index=False)
    print(f"Built {feats['forecast_date'].nunique()} yearly forecasts, {len(feats):,} rows -> {config.FEATURES_PARQUET}")


def run_train(_: bool) -> None:
    from ratcast.train import train

    train(pd.read_parquet(config.FEATURES_PARQUET))


def run_evaluate(_: bool) -> None:
    from ratcast.evaluate import evaluate

    evaluate()


def run_maps(_: bool) -> None:
    from ratcast.maps import make_maps

    make_maps()


def run_site(_: bool) -> None:
    from ratcast.site import build_site

    build_site()


RUNNERS = {
    "download": run_download,
    "clean": run_clean,
    "features": run_features,
    "train": run_train,
    "evaluate": run_evaluate,
    "maps": run_maps,
    "site": run_site,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="ratcast", description=__doc__)
    parser.add_argument("step", choices=STEPS + ["all"])
    parser.add_argument("--force", action="store_true", help="re-download even if the raw data exists")
    args = parser.parse_args()

    for step in STEPS if args.step == "all" else [args.step]:
        start = time.perf_counter()
        print(f"\n== {step} ==")
        RUNNERS[step](args.force)
        print(f"   ({time.perf_counter() - start:.1f}s)")


if __name__ == "__main__":
    main()
