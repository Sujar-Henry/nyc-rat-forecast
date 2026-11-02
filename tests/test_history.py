import numpy as np
import pandas as pd

from ratcast.history import days_since_prior, last_prior_value, prior_sums


def test_same_day_events_are_not_counted():
    # An inspection's own outcome (and anything else logged that day) is not
    # known before the inspector arrives.
    out = prior_sums(
        event_keys=np.array(["A", "A"]),
        event_days=np.array([10, 20]),
        event_values=np.array([[1], [1]]),
        query_keys=np.array(["A"]),
        query_days=np.array([20]),
    )
    assert out.tolist() == [[1]]


def test_window_includes_lower_boundary_and_excludes_older():
    out = prior_sums(
        event_keys=np.array(["A", "A", "A"]),
        event_days=np.array([69, 70, 99]),
        event_values=np.array([[1], [1], [1]]),
        query_keys=np.array(["A"]),
        query_days=np.array([100]),
        window_days=30,
    )
    assert out.tolist() == [[2]]


def test_keys_do_not_mix():
    out = prior_sums(
        event_keys=np.array(["A", "B", "B"]),
        event_days=np.array([1, 1, 2]),
        event_values=np.array([[1], [1], [1]]),
        query_keys=np.array(["A", "B", "C"]),
        query_days=np.array([5, 5, 5]),
    )
    assert out.tolist() == [[1], [2], [0]]


def test_multiple_value_columns_sum_independently():
    out = prior_sums(
        event_keys=np.array(["A", "A"]),
        event_days=np.array([1, 2]),
        event_values=np.array([[1, 0], [1, 1]]),
        query_keys=np.array(["A"]),
        query_days=np.array([3]),
    )
    assert out.tolist() == [[2, 1]]


def test_unsorted_inputs_give_same_answer():
    keys = np.array(["B", "A", "B", "A"])
    days = np.array([5, 3, 1, 9])
    vals = np.array([[1], [1], [1], [1]])
    out = prior_sums(keys, days, vals, np.array(["A", "B"]), np.array([10, 4]))
    assert out.tolist() == [[2], [1]]


def test_days_since_prior_picks_most_recent_strictly_earlier_event():
    out = days_since_prior(
        event_keys=np.array(["A", "A", "A", "B"]),
        event_days=np.array([2, 7, 10, 1]),
        query_keys=np.array(["A", "A", "B", "C"]),
        query_days=np.array([10, 2, 4, 4]),
    )
    assert out[0] == 3  # day 7, not the same-day event on day 10
    assert np.isnan(out[1])  # nothing before the first event
    assert out[2] == 3
    assert np.isnan(out[3])  # unseen key


def test_accepts_pandas_inputs():
    events = pd.DataFrame({"k": ["A", "A"], "d": [1, 2], "v": [1, 1]})
    out = prior_sums(events["k"], events["d"], events[["v"]], pd.Series(["A"]), pd.Series([3]))
    assert out.tolist() == [[2]]


def test_last_prior_value_uses_latest_earlier_day_and_max_on_ties():
    out = last_prior_value(
        event_keys=np.array(["A", "A", "A", "A"]),
        event_days=np.array([1, 5, 5, 9]),
        event_values=np.array([1, 0, 1, 0]),
        query_keys=np.array(["A", "A", "A"]),
        query_days=np.array([9, 5, 1]),
    )
    assert out[0] == 1  # day 5 had a pass and a fail; the fail wins
    assert out[1] == 1  # day 1
    assert np.isnan(out[2])
