"""Turn inspection history into forecasts made on a fixed date.

Every modeling row answers one question: "Standing on the morning of the
forecast date, knowing only what the city knew then, will this first-time
inspection find rats?" History is frozen at the forecast date for every row,
however far ahead the inspection is, so nothing that happens after that
morning (including the results of other inspections being predicted) can leak
into a prediction.

Only first visits ("Initial" inspections) are targets. Follow-up compliance
visits exist only because an earlier inspection failed, so knowing one is
coming would give away a result from the forecast period. They still count as
history before the forecast date.
"""

import numpy as np
import pandas as pd

from ratcast import config
from ratcast.history import days_since_prior, last_prior_value, prior_sums

EPOCH = pd.Timestamp("1970-01-01")

TIME_FEATURES = ["days_ahead", "month", "day_of_week", "doy_sin", "doy_cos"]
LOCATION_FEATURES = ["latitude", "longitude", "borough", "nta"]
PROPERTY_FEATURES = [
    "bbl_prior_inspections",
    "bbl_prior_rat_hits",
    "bbl_prior_other_failures",
    "bbl_prior_treatments",
    "bbl_prior_stoppages",
    "bbl_rat_rate",
    "bbl_inspections_365d",
    "bbl_rat_hits_365d",
    "bbl_treatments_365d",
    "bbl_days_since_inspection",
    "bbl_days_since_rats",
    "bbl_days_since_treatment",
    "bbl_last_found_rats",
]
# "cell" is the H3 res-8 hexagon around the lot (a few blocks); "area" is its
# res-7 parent (roughly a neighbourhood). Windows end at the forecast date.
CELL_FEATURES = [
    "cell_inspections_90d",
    "cell_rat_hits_90d",
    "cell_rat_rate_90d",
    "cell_inspections_365d",
    "cell_rat_hits_365d",
    "cell_rat_rate_365d",
    "cell_treatments_365d",
    "cell_rat_rate_all",
    "area_inspections_365d",
    "area_rat_hits_365d",
    "area_rat_rate_365d",
]

FEATURE_COLUMNS = TIME_FEATURES + LOCATION_FEATURES + PROPERTY_FEATURES + CELL_FEATURES
CATEGORICAL_COLUMNS = ["borough", "nta"]

META_COLUMNS = [
    "job_id", "date", "inspection_type", "bbl", "house_number", "street_name",
    "zip_code", "h3_8",
]


def _rate(hits: np.ndarray, total: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, hits / total, np.nan)


def _day(ts) -> int:
    return (pd.Timestamp(ts) - EPOCH).days


def _prepare(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev["day"] = (ev["date"] - EPOCH).dt.days
    is_insp = ev["kind"].eq("inspection")
    ev["n_insp"] = is_insp.astype(np.int8)
    ev["n_rat"] = (is_insp & ev["rat_activity"].eq(1)).astype(np.int8)
    ev["n_other"] = (is_insp & ev["other_failure"].eq(1)).astype(np.int8)
    ev["n_treat"] = ev["kind"].eq("treatment").astype(np.int8)
    ev["n_stop"] = ev["kind"].isin(["stoppage", "cleanup"]).astype(np.int8)
    # One shared vocabulary per categorical, so every forecast frame lines up.
    for col in CATEGORICAL_COLUMNS:
        ev[col] = pd.Categorical(ev[col], categories=sorted(ev[col].dropna().unique()))
    return ev


def _forecast(ev: pd.DataFrame, forecast_date, until=None) -> pd.DataFrame:
    cutoff = _day(forecast_date)
    end = _day(until) if until is not None else np.iinfo(np.int64).max
    h = ev[ev["day"] < cutoff]
    q = ev[ev["kind"].eq("inspection") & ev["inspection_type"].eq("Initial")
           & (ev["day"] >= cutoff) & (ev["day"] < end)]
    insp = h[h["kind"] == "inspection"]
    rats = h[h["n_rat"] == 1]
    treats = h[h["n_treat"] == 1]
    as_of = np.full(len(q), cutoff)  # every lookup is "as of the forecast morning"

    meta = [c for c in META_COLUMNS if c in ev.columns]
    out = q[meta].copy()
    out["forecast_date"] = pd.Timestamp(forecast_date)
    out["label"] = q["rat_activity"].astype(np.int8).to_numpy()

    # When the inspection falls, relative to the forecast and the calendar
    date = q["date"]
    doy = date.dt.dayofyear.to_numpy()
    out["days_ahead"] = q["day"].to_numpy() - cutoff
    out["month"] = date.dt.month.to_numpy()
    out["day_of_week"] = date.dt.dayofweek.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Where
    out["latitude"] = q["latitude"].to_numpy()
    out["longitude"] = q["longitude"].to_numpy()
    for col in CATEGORICAL_COLUMNS:
        out[col] = q[col]

    # This property's own history
    cols = ["n_insp", "n_rat", "n_other", "n_treat", "n_stop"]
    allt = prior_sums(h["bbl"], h["day"], h[cols], q["bbl"], as_of)
    out["bbl_prior_inspections"] = allt[:, 0]
    out["bbl_prior_rat_hits"] = allt[:, 1]
    out["bbl_prior_other_failures"] = allt[:, 2]
    out["bbl_prior_treatments"] = allt[:, 3]
    out["bbl_prior_stoppages"] = allt[:, 4]
    out["bbl_rat_rate"] = _rate(allt[:, 1], allt[:, 0])

    yr = prior_sums(h["bbl"], h["day"], h[["n_insp", "n_rat", "n_treat"]], q["bbl"], as_of, window_days=365)
    out["bbl_inspections_365d"] = yr[:, 0]
    out["bbl_rat_hits_365d"] = yr[:, 1]
    out["bbl_treatments_365d"] = yr[:, 2]

    out["bbl_days_since_inspection"] = days_since_prior(insp["bbl"], insp["day"], q["bbl"], as_of)
    out["bbl_days_since_rats"] = days_since_prior(rats["bbl"], rats["day"], q["bbl"], as_of)
    out["bbl_days_since_treatment"] = days_since_prior(treats["bbl"], treats["day"], q["bbl"], as_of)
    out["bbl_last_found_rats"] = last_prior_value(insp["bbl"], insp["day"], insp["n_rat"], q["bbl"], as_of)

    # The surrounding blocks
    c90 = prior_sums(h["h3_8"], h["day"], h[["n_insp", "n_rat"]], q["h3_8"], as_of, window_days=90)
    c365 = prior_sums(h["h3_8"], h["day"], h[["n_insp", "n_rat", "n_treat"]], q["h3_8"], as_of, window_days=365)
    call = prior_sums(h["h3_8"], h["day"], h[["n_insp", "n_rat"]], q["h3_8"], as_of)
    out["cell_inspections_90d"] = c90[:, 0]
    out["cell_rat_hits_90d"] = c90[:, 1]
    out["cell_rat_rate_90d"] = _rate(c90[:, 1], c90[:, 0])
    out["cell_inspections_365d"] = c365[:, 0]
    out["cell_rat_hits_365d"] = c365[:, 1]
    out["cell_rat_rate_365d"] = _rate(c365[:, 1], c365[:, 0])
    out["cell_treatments_365d"] = c365[:, 2]
    out["cell_rat_rate_all"] = _rate(call[:, 1], call[:, 0])

    a365 = prior_sums(h["h3_7"], h["day"], h[["n_insp", "n_rat"]], q["h3_7"], as_of, window_days=365)
    out["area_inspections_365d"] = a365[:, 0]
    out["area_rat_hits_365d"] = a365[:, 1]
    out["area_rat_rate_365d"] = _rate(a365[:, 1], a365[:, 0])

    return out.reset_index(drop=True)


def build_features(events: pd.DataFrame, forecast_date, until=None) -> pd.DataFrame:
    """Forecast rows for first visits on or after ``forecast_date`` (and before
    ``until``), using only events from before ``forecast_date``."""
    return _forecast(_prepare(events), forecast_date, until)


def build_forecast_dataset(events: pd.DataFrame,
                           first_year: int = config.FIRST_FORECAST_YEAR,
                           train_end=config.TRAIN_END,
                           valid_end=config.VALID_END,
                           horizon_days: int = config.FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    """Re-run history as a forecast made every January 1st.

    train: one forecast per year from ``first_year``, each covering up to
           ``horizon_days`` ahead but never reaching ``train_end``
    valid: a forecast made on ``train_end`` for the year up to ``valid_end``
    test:  a forecast made on ``valid_end`` for everything after it
    """
    ev = _prepare(events)
    train_end, valid_end = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    frames = []
    for year in range(first_year, train_end.year):
        start = pd.Timestamp(year=year, month=1, day=1)
        until = min(start + pd.Timedelta(days=horizon_days), train_end)
        frames.append(_forecast(ev, start, until).assign(split="train"))
    frames.append(_forecast(ev, train_end, valid_end).assign(split="valid"))
    frames.append(_forecast(ev, valid_end).assign(split="test"))
    return pd.concat(frames, ignore_index=True)
