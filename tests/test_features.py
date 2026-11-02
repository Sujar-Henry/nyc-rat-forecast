import numpy as np
import pandas as pd
import pytest

from ratcast.features import FEATURE_COLUMNS, build_features, build_forecast_dataset

FORECAST = pd.Timestamp("2022-06-01")


@pytest.fixture
def events() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 900
    lots = [f"30{i:08d}" for i in range(15)]
    bbl = rng.choice(lots, n)
    kind = rng.choice(["inspection"] * 4 + ["treatment", "stoppage"], n)
    is_insp = kind == "inspection"
    return pd.DataFrame({
        "job_id": [f"PC{i}" for i in range(n)],
        "date": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D"),
        "kind": kind,
        "inspection_type": np.where(is_insp, rng.choice(["Initial", "Compliance"], n), "Treatments"),
        "rat_activity": (is_insp & (rng.random(n) < 0.3)).astype("int8"),
        "other_failure": (is_insp & (rng.random(n) < 0.1)).astype("int8"),
        "baited": (kind == "treatment").astype("int8"),
        "bbl": bbl,
        "h3_8": [f"c8_{int(b) % 5}" for b in bbl],
        "h3_7": [f"c7_{int(b) % 2}" for b in bbl],
        "latitude": 40.7,
        "longitude": -73.9,
        "borough": rng.choice(["Brooklyn", "Queens"], n),
        "nta": rng.choice(["Bushwick (West)", "Astoria (Central)", "Flatbush"], n),
        "zip_code": "11237",
    }).assign(is_compliance=lambda d: (d.inspection_type == "Compliance").astype("int8"))


def _features(frame: pd.DataFrame, **kw) -> pd.DataFrame:
    feats = build_features(frame, FORECAST, **kw)
    return feats.sort_values("job_id").set_index("job_id")[FEATURE_COLUMNS]


def test_rewriting_everything_after_the_forecast_date_changes_nothing(events):
    # The forecast is made on the morning of FORECAST: no outcome from that day
    # on may reach any feature, however far ahead the inspection is.
    after = events["date"] >= FORECAST
    rewritten = events.copy()
    rewritten.loc[after, "rat_activity"] = 1 - rewritten.loc[after, "rat_activity"]
    rewritten.loc[after, "other_failure"] = 1 - rewritten.loc[after, "other_failure"]
    rewritten.loc[after & (events.kind == "treatment"), "kind"] = "stoppage"

    pd.testing.assert_frame_equal(_features(events), _features(rewritten))


def test_targets_are_first_visits_inside_the_window(events):
    until = pd.Timestamp("2023-01-01")
    feats = build_features(events, FORECAST, until=until)
    expected = events[(events.inspection_type == "Initial") & (events.date >= FORECAST) & (events.date < until)]
    assert sorted(feats["job_id"]) == sorted(expected["job_id"])
    merged = feats.merge(expected[["job_id", "rat_activity"]], on="job_id")
    assert (merged["label"] == merged["rat_activity"]).all()


def test_days_ahead_counts_from_the_forecast_date(events):
    feats = build_features(events, FORECAST)
    assert (feats["days_ahead"] == (feats["date"] - FORECAST).dt.days).all()
    assert feats["days_ahead"].min() >= 0


def test_history_counts_are_correct_for_a_known_lot():
    base = dict(h3_8="c", h3_7="p", latitude=40.7, longitude=-73.9, borough="Queens",
                nta="Astoria (Central)", zip_code="11103", other_failure=0, baited=0,
                is_compliance=0, inspection_type="Initial", kind="inspection")
    rows = [
        dict(job_id="a", date="2023-01-01", bbl="1", rat_activity=1),
        dict(job_id="b", date="2023-03-01", bbl="1", rat_activity=0),
        dict(job_id="c", date="2023-03-01", bbl="2", rat_activity=1),
        dict(job_id="d", date="2023-03-21", bbl="1", rat_activity=0),
        dict(job_id="e", date="2023-03-25", bbl="2", rat_activity=1),
    ]
    events = pd.DataFrame([{**base, **r} for r in rows]).assign(date=lambda d: pd.to_datetime(d.date))
    f = build_features(events, pd.Timestamp("2023-03-11")).set_index("job_id")

    assert sorted(f.index) == ["d", "e"]
    # Everything is measured as of the forecast date (03-11), not the visit date.
    assert f.loc["d", "bbl_prior_inspections"] == 2
    assert f.loc["d", "bbl_prior_rat_hits"] == 1
    assert f.loc["d", "bbl_rat_rate"] == 0.5
    assert f.loc["d", "bbl_days_since_inspection"] == 10
    assert f.loc["d", "bbl_last_found_rats"] == 0
    assert f.loc["d", "bbl_days_since_rats"] == 69
    assert f.loc["d", "days_ahead"] == 10
    assert f.loc["d", "cell_rat_hits_365d"] == 2
    assert f.loc["e", "bbl_prior_rat_hits"] == 1  # c, but not d (after the forecast)


def test_dataset_splits_never_cross_their_boundaries(events):
    data = build_forecast_dataset(events, first_year=2020, train_end="2022-01-01", valid_end="2023-01-01")
    train, valid, test = (data[data["split"] == s] for s in ("train", "valid", "test"))
    assert train["date"].max() < pd.Timestamp("2022-01-01")
    assert valid["date"].between("2022-01-01", "2022-12-31").all()
    assert (valid["forecast_date"] == pd.Timestamp("2022-01-01")).all()
    assert (test["forecast_date"] == pd.Timestamp("2023-01-01")).all()
    assert test["date"].min() >= pd.Timestamp("2023-01-01")
    # Categories line up across every forecast frame, so the model sees one vocabulary.
    assert data["nta"].dtype == "category"


def test_test_set_is_blind_to_test_period_outcomes(events):
    # The regression test for the original map leak: rewriting every outcome in
    # the test period must not move a single test-set feature.
    kw = dict(first_year=2020, train_end="2022-01-01", valid_end="2023-01-01")
    after = events["date"] >= pd.Timestamp("2023-01-01")
    rewritten = events.copy()
    rewritten.loc[after, "rat_activity"] = 1 - rewritten.loc[after, "rat_activity"]

    def test_feats(frame):
        d = build_forecast_dataset(frame, **kw)
        return d[d["split"] == "test"].sort_values("job_id").set_index("job_id")[FEATURE_COLUMNS]

    pd.testing.assert_frame_equal(test_feats(events), test_feats(rewritten))
