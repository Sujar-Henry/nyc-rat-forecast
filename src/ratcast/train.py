"""Fit the XGBoost model on past forecasts and score the newest one.

Training rows are forecasts made every January 1st from 2010 to 2023. The
model is early-stopped on the forecast made on 2024-01-01, and only then asked
to forecast everything since 2025-01-01, knowing nothing after that morning.
"""

import json

import pandas as pd
import xgboost as xgb

from ratcast import config
from ratcast.features import FEATURE_COLUMNS

PARAMS = dict(
    n_estimators=4000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=1,
    eval_metric=["auc", "logloss"],  # early stopping watches the last one
    early_stopping_rounds=150,
    n_jobs=-1,
    random_state=42,
)

MODEL_PATH = config.MODEL_DIR / "xgb_rat_activity.json"
MODEL_CARD = config.MODEL_DIR / "model_card.json"
KEEP_FOR_EVALUATION = ["latitude", "longitude", "borough", "nta", "days_ahead",
                       "bbl_rat_rate", "cell_rat_rate_365d"]


def split(feats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: feats[feats["split"] == name] for name in ("train", "valid", "test")}


def train(feats: pd.DataFrame) -> tuple[xgb.XGBClassifier, pd.DataFrame]:
    parts = split(feats)
    for name, part in parts.items():
        span = f"{part['date'].min():%Y-%m-%d} to {part['date'].max():%Y-%m-%d}"
        made = part["forecast_date"].nunique()
        print(f"{name:>5}: {len(part):>9,} rows from {made:>2} forecast(s)  {span}  rat rate {part['label'].mean():.1%}")

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(
        parts["train"][FEATURE_COLUMNS], parts["train"]["label"],
        eval_set=[(parts["valid"][FEATURE_COLUMNS], parts["valid"]["label"])],
        verbose=250,
    )
    print(f"Best iteration: {model.best_iteration} (valid log loss {model.best_score:.4f})")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    MODEL_CARD.write_text(json.dumps({
        "question": "Knowing only what was known on the forecast date, will this first-time "
                    "(Initial) inspection find rat activity?",
        "target": "result mentions 'Rat Activity'",
        "features": FEATURE_COLUMNS,
        "train": {"forecast_dates": sorted(f"{d:%Y-%m-%d}" for d in parts["train"]["forecast_date"].unique()),
                  "horizon_days": config.FORECAST_HORIZON_DAYS, "labels_before": config.TRAIN_END,
                  "rows": len(parts["train"])},
        "valid": {"forecast_date": config.TRAIN_END, "until": config.VALID_END, "rows": len(parts["valid"])},
        "test": {"forecast_date": config.VALID_END, "rows": len(parts["test"]),
                 "last_date": f"{parts['test']['date'].max():%Y-%m-%d}"},
        "best_iteration": int(model.best_iteration),
        "params": {k: v for k, v in PARAMS.items() if k != "eval_metric"},
    }, indent=2))

    test = parts["test"].copy()
    test["p_rats"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    keep = [c for c in test.columns if c not in FEATURE_COLUMNS] + KEEP_FOR_EVALUATION
    test[keep].to_parquet(config.PREDICTIONS_PARQUET, index=False)
    print(f"Saved model to {MODEL_PATH} and {len(test):,} test predictions to {config.PREDICTIONS_PARQUET}")
    return model, test
