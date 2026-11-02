"""Point-in-time history lookups.

Every feature that summarises the past has to answer "what was known the
morning of this inspection?". These helpers do that for millions of
(key, day) queries at once: events are sorted by (key, day), and each query
binary-searches for the slice of events with the same key that happened
strictly before its day. Same-day events are always excluded, so an
inspection can never see its own result.
"""

import numpy as np
import pandas as pd

# Days are stored as integers since 1970; packing (key, day) into a single
# int64 as key * _STRIDE + day keeps every key's days in their own range.
_STRIDE = 1 << 20


def _encode_keys(event_keys, query_keys) -> tuple[np.ndarray, np.ndarray]:
    ev = pd.Series(np.asarray(event_keys, dtype=object))
    qu = pd.Series(np.asarray(query_keys, dtype=object))
    codes, _ = pd.factorize(pd.concat([ev, qu], ignore_index=True))
    codes = codes.astype(np.int64)
    return codes[: len(ev)], codes[len(ev) :]


def _as_days(days) -> np.ndarray:
    d = np.asarray(days, dtype=np.int64)
    if d.size and (d.min() < 0 or d.max() >= _STRIDE):
        raise ValueError("days must be in [0, 2**20)")
    return d


def _sorted_events(event_keys, event_days, query_keys, *values):
    ek, qk = _encode_keys(event_keys, query_keys)
    ed = _as_days(event_days)
    keep = ek >= 0  # missing keys never match anything
    ek, ed = ek[keep], ed[keep]
    values = [np.asarray(v)[keep] for v in values]
    packed = ek * _STRIDE + ed
    order = np.argsort(packed, kind="stable")
    return qk, packed[order], ed[order], [v[order] for v in values]


def prior_sums(event_keys, event_days, event_values, query_keys, query_days,
               window_days: int | None = None) -> np.ndarray:
    """Sum event values per query over events with the same key and
    ``query_day - window_days <= event_day < query_day`` (all history when
    ``window_days`` is None). Returns an array of shape (n_queries, n_values).
    """
    vals = np.asarray(event_values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    qk, packed, _, (vals,) = _sorted_events(event_keys, event_days, query_keys, vals)
    qd = _as_days(query_days)

    csum = np.vstack([np.zeros((1, vals.shape[1])), np.cumsum(vals, axis=0)])
    hi = np.searchsorted(packed, qk * _STRIDE + qd, side="left")
    lo_day = 0 if window_days is None else np.clip(qd - window_days, 0, None)
    lo = np.searchsorted(packed, qk * _STRIDE + lo_day, side="left")
    return csum[hi] - csum[lo]


def days_since_prior(event_keys, event_days, query_keys, query_days) -> np.ndarray:
    """Days between each query and the latest strictly-earlier event with the
    same key; NaN when there is none."""
    qk, packed, ed, _ = _sorted_events(event_keys, event_days, query_keys)
    qd = _as_days(query_days)

    hi = np.searchsorted(packed, qk * _STRIDE + qd, side="left")
    start = np.searchsorted(packed, qk * _STRIDE, side="left")
    has_prior = hi > start
    prev_day = ed[np.maximum(hi - 1, 0)] if len(ed) else np.zeros_like(qd)
    return np.where(has_prior, qd - prev_day, np.nan)


def last_prior_value(event_keys, event_days, event_values, query_keys, query_days) -> np.ndarray:
    """Value of the latest strictly-earlier event with the same key (the max
    if several share that day); NaN when there is none."""
    frame = pd.DataFrame({
        "k": np.asarray(event_keys, dtype=object),
        "d": _as_days(event_days),
        "v": np.asarray(event_values, dtype=np.float64),
    })
    daily = frame.groupby(["k", "d"], sort=False, dropna=True)["v"].max().reset_index()
    qk, packed, _, (vals,) = _sorted_events(daily["k"], daily["d"], query_keys, daily["v"].to_numpy())
    qd = _as_days(query_days)

    hi = np.searchsorted(packed, qk * _STRIDE + qd, side="left")
    start = np.searchsorted(packed, qk * _STRIDE, side="left")
    has_prior = hi > start
    prev = vals[np.maximum(hi - 1, 0)] if len(vals) else np.zeros(len(qd))
    return np.where(has_prior, prev, np.nan)
