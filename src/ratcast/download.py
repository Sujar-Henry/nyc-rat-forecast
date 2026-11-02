"""Download the DOHMH Rodent Inspection dataset from NYC Open Data.

The Socrata API is paged so a dropped connection only costs one chunk; finished
chunks are kept on disk and skipped on the next run.
"""

import io
import time

import pandas as pd
import requests

from ratcast import config

PAGE_SIZE = 250_000


def _fetch_page(offset: int, retries: int = 5) -> pd.DataFrame:
    params = {
        "$select": ",".join(config.RAW_COLUMNS),
        "$order": ":id",
        "$limit": PAGE_SIZE,
        "$offset": offset,
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(config.SODA_ENDPOINT, params=params, timeout=300)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text), dtype=str)
        except (requests.RequestException, pd.errors.ParserError) as exc:
            if attempt == retries:
                raise
            wait = 2**attempt
            print(f"  page at offset {offset:,} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download(force: bool = False) -> pd.DataFrame:
    if config.RAW_PARQUET.exists() and not force:
        print(f"Raw data already at {config.RAW_PARQUET}; use --force to re-download.")
        return pd.read_parquet(config.RAW_PARQUET)

    parts_dir = config.RAW_DIR / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    while True:
        part = parts_dir / f"part_{offset // PAGE_SIZE:04d}.parquet"
        if part.exists() and not force:
            n = len(pd.read_parquet(part, columns=["job_id"]))
        else:
            print(f"Fetching rows {offset:,} - {offset + PAGE_SIZE:,} ...")
            page = _fetch_page(offset)
            n = len(page)
            if n:
                page.to_parquet(part, index=False)
        if n < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    frames = [pd.read_parquet(p) for p in sorted(parts_dir.glob("part_*.parquet"))]
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(config.RAW_PARQUET, index=False)
    print(f"Saved {len(raw):,} records to {config.RAW_PARQUET}")
    return raw


BOROUGHS_URL = "https://data.cityofnewyork.us/resource/gthc-hcne.geojson"  # clipped to shoreline


def download_boroughs(force: bool = False) -> None:
    if config.BOROUGHS_GEOJSON.exists() and not force:
        return
    resp = requests.get(BOROUGHS_URL, timeout=120)
    resp.raise_for_status()
    config.BOROUGHS_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    config.BOROUGHS_GEOJSON.write_bytes(resp.content)
    print(f"Saved borough boundaries to {config.BOROUGHS_GEOJSON}")
