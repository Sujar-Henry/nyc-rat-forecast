"""Paths, dataset identifiers and modeling constants shared across the pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MAP_DIR = OUTPUT_DIR / "maps"
MODEL_DIR = OUTPUT_DIR / "models"

RAW_PARQUET = RAW_DIR / "rodent_inspections.parquet"
EVENTS_PARQUET = PROCESSED_DIR / "events.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
PREDICTIONS_PARQUET = PROCESSED_DIR / "test_predictions.parquet"
BOROUGHS_GEOJSON = RAW_DIR / "borough_boundaries.geojson"
SITE_DIR = PROJECT_ROOT / "site"

# Where the published page lives (used for link-preview tags).
SITE_URL = "https://sujar-henry.github.io/nyc-rat-forecast/"

# NYC Open Data: "Rodent Inspection" (DOHMH)
# https://data.cityofnewyork.us/Health/Rodent-Inspection/p937-wjvj
DATASET_ID = "p937-wjvj"
SODA_ENDPOINT = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.csv"

RAW_COLUMNS = [
    "job_id",
    "inspection_date",
    "inspection_type",
    "result",
    "letter_type",
    "observations",
    "bbl",
    "boro_code",
    "block",
    "lot",
    "house_number",
    "street_name",
    "zip_code",
    "borough",
    "latitude",
    "longitude",
    "community_board",
    "council_district",
    "census_tract",
    "nta",
]

# Records outside this window are data-entry errors (the raw file has a handful
# of inspections dated 1918 or 2045).
MIN_DATE = "2009-01-01"

# Rough bounding box around the five boroughs, used to drop bad geocodes.
NYC_BOUNDS = {"lat": (40.47, 40.93), "lon": (-74.27, -73.68)}

# Every split is a forecast made on a January 1st with history frozen there.
# train: forecasts made each Jan 1 from FIRST_FORECAST_YEAR, each looking up
#        to FORECAST_HORIZON_DAYS ahead but never past TRAIN_END
# valid: forecast made on TRAIN_END for the following year (early stopping)
# test:  forecast made on VALID_END for every inspection since
FIRST_FORECAST_YEAR = 2010
FORECAST_HORIZON_DAYS = 730
TRAIN_END = "2024-01-01"
VALID_END = "2025-01-01"

# H3 resolution 8 cells are ~0.7 km^2, roughly a few city blocks across.
H3_RESOLUTION = 8
