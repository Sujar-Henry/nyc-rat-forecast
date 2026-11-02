"""Score the held-out (2025+) predictions and draw the report figures."""

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

from ratcast import config, style
from ratcast.train import MODEL_PATH

METRICS_PATH = config.OUTPUT_DIR / "metrics.json"

FEATURE_LABELS = {
    "days_ahead": "Days between forecast and inspection",
    "bbl_last_found_rats": "Last visit to this lot found rats",
    "bbl_rat_rate": "Share of past visits that found rats",
    "bbl_days_since_rats": "Days since rats were last found here",
    "bbl_days_since_inspection": "Days since this lot was last inspected",
    "bbl_days_since_treatment": "Days since this lot was last baited",
    "bbl_prior_inspections": "Past inspections at this lot",
    "bbl_prior_rat_hits": "Past rat findings at this lot",
    "bbl_prior_other_failures": "Past non-rat failures at this lot",
    "bbl_prior_treatments": "Past baiting visits at this lot",
    "bbl_prior_stoppages": "Past burrow stoppages / clean-ups",
    "bbl_inspections_365d": "Inspections at this lot, past year",
    "bbl_rat_hits_365d": "Rat findings at this lot, past year",
    "bbl_treatments_365d": "Baiting at this lot, past year",
    "cell_inspections_90d": "Inspections nearby, past 90 days",
    "cell_rat_hits_90d": "Rat findings nearby, past 90 days",
    "cell_rat_rate_90d": "Nearby rat-find rate, past 90 days",
    "cell_inspections_365d": "Inspections nearby, past year",
    "cell_rat_hits_365d": "Rat findings nearby, past year",
    "cell_rat_rate_365d": "Nearby rat-find rate, past year",
    "cell_treatments_365d": "Baiting nearby, past year",
    "cell_rat_rate_all": "Nearby rat-find rate, all time",
    "area_inspections_365d": "Inspections in the neighbourhood, past year",
    "area_rat_hits_365d": "Rat findings in the neighbourhood, past year",
    "area_rat_rate_365d": "Neighbourhood rat-find rate, past year",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "borough": "Borough",
    "nta": "Neighbourhood (NTA)",
    "month": "Month",
    "day_of_week": "Day of week",
    "doy_sin": "Time of year (sin)",
    "doy_cos": "Time of year (cos)",
}


def _baseline_scores(preds: pd.DataFrame) -> dict[str, pd.Series]:
    """Simple rules an analyst could use instead of a model, with the same
    knowledge cut-off (both are measured as of the forecast date)."""
    nearby = preds["cell_rat_rate_365d"].fillna(preds["cell_rat_rate_365d"].median())
    return {
        "Last year's rate in this hexagon": nearby,
        "This lot's own track record": preds["bbl_rat_rate"].fillna(nearby),
    }


def _map_agreement(preds: pd.DataFrame, min_inspections: int = 25) -> dict:
    """How closely the hexagon maps line up, with two yardsticks: simply
    re-using last year's map, and two random halves of the real outcomes
    (as close as any forecast could hope to get)."""
    g = preds.groupby("h3_8").agg(n=("label", "size"), actual=("label", "mean"),
                                  model=("p_rats", "mean"), last_year=("cell_rat_rate_365d", "mean"))
    g = g[g["n"] >= min_inspections]
    half = np.random.default_rng(0).random(len(preds)) < 0.5
    a = preds[half].groupby("h3_8")["label"].agg(["size", "mean"])
    b = preds[~half].groupby("h3_8")["label"].agg(["size", "mean"])
    both = a.join(b, lsuffix="_a", rsuffix="_b").reindex(g.index)
    return {
        "hexagons": int(len(g)),
        "model_r": float(g["actual"].corr(g["model"])),
        "last_years_map_r": float(g["actual"].corr(g["last_year"])),
        "split_half_r": float(both["mean_a"].corr(both["mean_b"])),
    }


def _lift_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(-p, kind="stable")
    bins = np.empty(len(p), dtype=int)
    bins[order] = np.arange(len(p)) * n_bins // len(p)  # 0 = riskiest tenth
    df = pd.DataFrame({"bin": bins, "y": y, "p": p})
    out = df.groupby("bin").agg(inspections=("y", "size"), found=("y", "sum"),
                                hit_rate=("y", "mean"), predicted=("p", "mean"))
    out["share_of_all_finds"] = out["found"] / out["found"].sum()
    return out


def compute_metrics(preds: pd.DataFrame) -> dict:
    y, p = preds["label"].to_numpy(), preds["p_rats"].to_numpy()
    lift = _lift_table(y, p)
    base = float(y.mean())
    metrics = {
        "test_period": f"{preds['date'].min():%Y-%m-%d} to {preds['date'].max():%Y-%m-%d}",
        "test_inspections": int(len(y)),
        "found_rat_activity": int(y.sum()),
        "base_rate": base,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "top_10pct_hit_rate": float(lift.loc[0, "hit_rate"]),
        "top_10pct_lift": float(lift.loc[0, "hit_rate"] / base),
        "top_20pct_share_of_finds": float(lift.loc[[0, 1], "found"].sum() / y.sum()),
        "bottom_50pct_hit_rate": float(lift.loc[5:, "found"].sum() / lift.loc[5:, "inspections"].sum()),
        "baselines_roc_auc": {
            name: float(roc_auc_score(y, s)) for name, s in _baseline_scores(preds).items()
        },
        "by_borough": {
            b: {"inspections": int(len(g)), "actual": float(g["label"].mean()),
                "predicted": float(g["p_rats"].mean()), "roc_auc": float(roc_auc_score(g["label"], g["p_rats"]))}
            for b, g in preds.groupby("borough", observed=True) if g["label"].nunique() == 2
        },
        "mean_predicted": float(p.mean()),
        "map": _map_agreement(preds),
        "deciles": lift.reset_index().to_dict(orient="records"),
    }
    return metrics


def _plot_deciles(metrics: dict) -> None:
    dec = pd.DataFrame(metrics["deciles"])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(1, len(dec) + 1)
    ax.bar(x, dec["hit_rate"], width=0.72, color=style.SERIES[0], zorder=2)
    ax.hlines(metrics["base_rate"], 0.5, len(dec) + 0.5, color=style.INK_2, linewidth=1, zorder=3)
    ax.text(len(dec) + 0.6, metrics["base_rate"], f"city-wide\naverage {metrics['base_rate']:.0%}",
            va="center", ha="left", color=style.INK_2, fontsize=10, linespacing=1.3)
    for xi, rate in zip(x[:2], dec["hit_rate"][:2]):
        ax.text(xi, rate + 0.01, f"{rate:.0%}", ha="center", va="bottom", color=style.INK, fontsize=11, weight="bold")
    ax.set_xticks(x, [f"{(i - 1) * 10}–{i * 10}%" for i in x])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("First visits that found rat activity")
    ax.set_xlabel("First visits ranked by the model's predicted risk (riskiest first)")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(0.4, len(dec) + 1.9)
    ax.set_title("The riskiest 10% of first visits found rats "
                 f"{metrics['top_10pct_lift']:.1f}× as often as average")
    style.subtitle(ax, f"Forecast made Jan 1, 2025 · first visits {metrics['test_period']} · "
                       f"{metrics['test_inspections']:,} inspections")
    style.save(fig, config.FIGURE_DIR / "hit_rate_by_risk_decile.png")


def _plot_calibration(metrics: dict) -> None:
    dec = pd.DataFrame(metrics["deciles"])
    fig, ax = plt.subplots(figsize=(6.2, 6))
    lim = max(dec["hit_rate"].max(), dec["predicted"].max()) * 1.08
    ax.plot([0, lim], [0, lim], color=style.AXIS, linewidth=1, zorder=1)
    ax.scatter(dec["predicted"], dec["hit_rate"], s=64, color=style.SERIES[0],
               edgecolor=style.SURFACE, linewidth=2, zorder=3)
    ax.text(lim * 0.97, lim * 0.97, "perfect calibration", rotation=45, ha="right", va="bottom",
            color=style.MUTED, fontsize=9, rotation_mode="anchor")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted chance of finding rats")
    ax.set_ylabel("Actual share that found rats")
    gap = float((dec["predicted"] - dec["hit_rate"]).mean())
    if gap > 0.02:
        ax.set_title("Right order, but it ran high")
        style.subtitle(ax, f"Each dot is a tenth of the test inspections · predictions ran {gap * 100:.0f} pts high")
    elif gap < -0.02:
        ax.set_title("Right order, but it ran low")
        style.subtitle(ax, f"Each dot is a tenth of the test inspections · predictions ran {-gap * 100:.0f} pts low")
    else:
        ax.set_title("Predicted chances match what happened")
        style.subtitle(ax, "Each dot is a tenth of the test inspections")
    style.save(fig, config.FIGURE_DIR / "calibration.png")


def _plot_monthly(preds: pd.DataFrame) -> None:
    m = preds.assign(month=preds["date"].dt.to_period("M").dt.to_timestamp()).groupby("month").agg(
        actual=("label", "mean"), predicted=("p_rats", "mean"), n=("label", "size"))
    m = m[m["n"] >= 500]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(m.index, m["actual"], color=style.SERIES[1], marker="o", markersize=5, label="Actually found")
    ax.plot(m.index, m["predicted"], color=style.SERIES[0], marker="o", markersize=5, label="Model expected")
    last = m.index[-1]
    ax.text(last, m["actual"].iloc[-1], "  actual", color=style.INK_2, va="center", fontsize=10)
    ax.text(last, m["predicted"].iloc[-1], "  predicted", color=style.INK_2, va="center", fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_ylim(0, max(m["actual"].max(), m["predicted"].max()) * 1.25)
    ax.set_ylabel("First visits that found rat activity")
    ax.legend(loc="upper left", ncols=2)
    ax.set_title(f"Inspectors found rats on {preds['label'].mean():.0%} of first visits; "
                 f"the model expected {preds['p_rats'].mean():.0%}")
    style.subtitle(ax, "Forecast made Jan 1, 2025, knowing nothing after that morning")
    style.save(fig, config.FIGURE_DIR / "monthly_predicted_vs_actual.png")


def _plot_roc(preds: pd.DataFrame, metrics: dict) -> None:
    y = preds["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot([0, 1], [0, 1], color=style.AXIS, linewidth=1)
    curves = [("XGBoost model", preds["p_rats"], metrics["roc_auc"])]
    curves += [(k, s, metrics["baselines_roc_auc"][k]) for k, s in _baseline_scores(preds).items()]
    for (name, score, auc), color in zip(curves, style.SERIES):
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, color=color, label=f"{name}  (AUC {auc:.3f})")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Clean inspections flagged (false positive rate)")
    ax.set_ylabel("Rat findings caught (true positive rate)")
    ax.legend(loc="lower right")
    ax.set_title("Better than simple rules of thumb")
    style.subtitle(ax, "ROC curves, first visits since Jan 1, 2025")
    style.save(fig, config.FIGURE_DIR / "roc_vs_baselines.png")


def _plot_importance() -> None:
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    gain = pd.Series(booster.get_score(importance_type="total_gain")).sort_values(ascending=False)
    top = (gain / gain.sum()).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([FEATURE_LABELS.get(f, f) for f in top.index], top.to_numpy(), color=style.SERIES[0],
            height=0.68, zorder=2)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=style.INK_2, labelsize=10.5)
    ax.set_xlabel("Share of the model's total split gain")
    ax.set_title("What the model leans on")
    style.subtitle(ax, "Top 12 features by total gain")
    style.save(fig, config.FIGURE_DIR / "feature_importance.png")


def evaluate(preds: pd.DataFrame | None = None) -> dict:
    preds = pd.read_parquet(config.PREDICTIONS_PARQUET) if preds is None else preds
    preds["date"] = pd.to_datetime(preds["date"])
    metrics = compute_metrics(preds)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Test period {metrics['test_period']}: {metrics['test_inspections']:,} inspections, "
          f"{metrics['base_rate']:.1%} found rat activity")
    print(f"  ROC AUC {metrics['roc_auc']:.3f} | avg precision {metrics['average_precision']:.3f} | "
          f"log loss {metrics['log_loss']:.4f} | Brier {metrics['brier']:.4f}")
    for name, auc in metrics["baselines_roc_auc"].items():
        print(f"  baseline '{name}': ROC AUC {auc:.3f}")
    mp = metrics["map"]
    print(f"  map ({mp['hexagons']} hexagons): model r {mp['model_r']:.3f} | last year's map r {mp['last_years_map_r']:.3f} "
          f"| two halves of the real data r {mp['split_half_r']:.3f}")
    print(f"  mean predicted {metrics['mean_predicted']:.1%} vs actual {metrics['base_rate']:.1%}")
    print(f"  top 10% riskiest: {metrics['top_10pct_hit_rate']:.1%} found rats ({metrics['top_10pct_lift']:.1f}x); "
          f"top 20% hold {metrics['top_20pct_share_of_finds']:.0%} of all finds; "
          f"bottom half: {metrics['bottom_50pct_hit_rate']:.1%}")

    style.apply()
    _plot_deciles(metrics)
    _plot_calibration(metrics)
    _plot_monthly(preds)
    _plot_roc(preds, metrics)
    _plot_importance()
    return metrics
