"""Turn the raw export into a tidy table of dated, geolocated events.

Every row in the raw data is a visit to a property. Only Initial and Compliance
visits are inspections with a pass/fail outcome; baiting, stoppage and clean-up
visits are kept because they are useful history, but they are never prediction
targets.
"""

import h3
import numpy as np
import pandas as pd

from ratcast import config

INSPECTION_TYPES = {"Initial", "Compliance"}
EVENT_KIND = {
    "Initial": "inspection",
    "Compliance": "inspection",
    "Treatments": "treatment",
    "Stoppage": "stoppage",
    "Clean Ups": "cleanup",
}


def _bbl_key(raw: pd.DataFrame) -> pd.Series:
    """Borough-Block-Lot uniquely identifies a tax lot; rebuild it when missing."""
    bbl = pd.to_numeric(raw["bbl"], errors="coerce").astype("Int64").astype("string")
    boro = pd.to_numeric(raw["boro_code"], errors="coerce").astype("Int64")
    block = pd.to_numeric(raw["block"], errors="coerce").astype("Int64")
    lot = pd.to_numeric(raw["lot"], errors="coerce").astype("Int64")
    rebuilt = (
        boro.astype("string")
        + block.astype("string").str.zfill(5)
        + lot.astype("string").str.zfill(4)
    )
    return bbl.fillna(rebuilt)


def _h3_cells(lat: pd.Series, lon: pd.Series, resolution: int) -> pd.Series:
    coords = pd.DataFrame({"lat": lat.round(6), "lon": lon.round(6)})
    unique = coords.drop_duplicates()
    cells = [h3.latlng_to_cell(a, b, resolution) for a, b in zip(unique["lat"], unique["lon"])]
    lookup = unique.assign(cell=cells)
    return coords.merge(lookup, on=["lat", "lon"], how="left")["cell"].to_numpy()


def clean(raw: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    as_of = as_of or pd.Timestamp.today().normalize()
    df = raw.copy()
    n_raw = len(df)

    df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
    df["date"] = df["inspection_date"].dt.normalize()
    df = df[df["date"].between(pd.Timestamp(config.MIN_DATE), as_of)]

    df["kind"] = df["inspection_type"].map(EVENT_KIND)
    df = df[df["kind"].notna()]

    df["bbl"] = _bbl_key(df)
    df = df[df["bbl"].notna()]

    # Geocodes: coerce, discard anything outside the city, then borrow the
    # coordinates of other visits to the same lot where they are missing.
    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    lat_ok = df["latitude"].between(*config.NYC_BOUNDS["lat"])
    lon_ok = df["longitude"].between(*config.NYC_BOUNDS["lon"])
    df.loc[~(lat_ok & lon_ok), ["latitude", "longitude"]] = np.nan
    by_lot = df.groupby("bbl")[["latitude", "longitude"]].transform("median")
    df[["latitude", "longitude"]] = df[["latitude", "longitude"]].fillna(by_lot)
    df = df[df["latitude"].notna() & df["longitude"].notna()]

    result = df["result"].fillna("")
    df["rat_activity"] = result.str.contains("Rat Activity", regex=False).astype("int8")
    df["other_failure"] = result.str.contains("Other Reason", regex=False).astype("int8")
    df["baited"] = (result == "Bait applied").astype("int8")
    df["is_compliance"] = (df["inspection_type"] == "Compliance").astype("int8")

    df["borough"] = df["borough"].str.strip().str.title()
    df["zip_code"] = df["zip_code"].str.extract(r"(\d{5})", expand=False)
    df["nta"] = df["nta"].str.strip()
    df["community_board"] = df["community_board"].str.strip()

    df["h3_8"] = _h3_cells(df["latitude"], df["longitude"], config.H3_RESOLUTION)
    df["h3_7"] = [h3.cell_to_parent(c, config.H3_RESOLUTION - 1) for c in df["h3_8"]]

    df = df.drop_duplicates(subset=["job_id", "inspection_date", "inspection_type", "result"])
    df = df.sort_values(["date", "inspection_date"], kind="stable").reset_index(drop=True)

    keep = [
        "job_id", "inspection_date", "date", "inspection_type", "kind", "result",
        "rat_activity", "other_failure", "baited", "is_compliance",
        "bbl", "house_number", "street_name", "zip_code", "borough", "nta",
        "community_board", "council_district", "latitude", "longitude", "h3_8", "h3_7",
    ]
    df = df[keep]

    n_insp = (df["kind"] == "inspection").sum()
    print(
        f"Cleaned {n_raw:,} raw records -> {len(df):,} events "
        f"({n_insp:,} inspections, {df.loc[df.kind == 'inspection', 'rat_activity'].mean():.1%} "
        f"found rat activity)"
    )
    return df
