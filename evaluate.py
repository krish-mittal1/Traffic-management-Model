"""Full forecaster evaluation on the held-out window: regression, ranking and
top-50 classification metrics. Output: metrics_full.json."""

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)

from forecast import FEATS, TEST_DAYS, add_features, build_panel, precision_at_k

HERE = pathlib.Path(__file__).parent
TOP_K_HOTSPOT = 50  # a zone counts as a "hotspot" if it's in the actual top-50 for the window


def main():
    df = pd.read_parquet(HERE / "violations_clean.parquet")
    panel = add_features(build_panel(df))
    cutoff = panel.d.max() - pd.Timedelta(days=TEST_DAYS)
    train = panel[panel.d <= cutoff].dropna(subset=FEATS)
    test = panel[panel.d > cutoff].dropna(subset=FEATS).copy()

    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=400, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, random_state=42,
    )
    model.fit(train[FEATS], train["y"])
    test["pred"] = model.predict(test[FEATS]).clip(min=0)

    y, p = test["y"].to_numpy(), test["pred"].to_numpy()

    # point-forecast (regression)
    mae = mean_absolute_error(y, p)
    rmse = mean_squared_error(y, p) ** 0.5
    wape = np.abs(y - p).sum() / y.sum()
    r2 = r2_score(y, p)
    pearson = float(np.corrcoef(y, p)[0, 1])
    bias = float((p - y).mean())
    base_mae = mean_absolute_error(y, test["roll7"])

    # ranking by per-zone totals over the window
    actual = test.groupby("h3")["y"].sum()
    pred_z = test.groupby("h3")["pred"].sum()
    spearman = float(actual.rank().corr(pred_z.rank()))
    p10 = precision_at_k(actual, pred_z, 10)
    p20 = precision_at_k(actual, pred_z, 20)
    p50 = precision_at_k(actual, pred_z, 50)

    # classification: is each zone in the actual top-50 next period?
    zones = actual.index
    y_true = (actual.rank(ascending=False) <= TOP_K_HOTSPOT).astype(int).loc[zones].to_numpy()
    score = pred_z.loc[zones].to_numpy()
    y_hat = (pred_z.rank(ascending=False) <= TOP_K_HOTSPOT).astype(int).loc[zones].to_numpy()
    acc = accuracy_score(y_true, y_hat)
    prec = precision_score(y_true, y_hat)
    rec = recall_score(y_true, y_hat)
    f1 = f1_score(y_true, y_hat)
    auc = roc_auc_score(y_true, score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()

    metrics = {
        "holdout_days": TEST_DAYS, "test_rows": int(len(test)), "n_zones": int(len(zones)),
        "regression": {
            "MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "WAPE_pct": round(100 * wape, 1), "R2": round(r2, 3),
            "pearson_r": round(pearson, 3), "bias": round(bias, 3),
            "baseline_roll7_MAE": round(base_mae, 3),
        },
        "ranking": {
            "spearman_rho": round(spearman, 3),
            "precision_at_10": round(p10, 3),
            "precision_at_20": round(p20, 3),
            "precision_at_50": round(p50, 3),
        },
        "classification_top%d_hotspot" % TOP_K_HOTSPOT: {
            "accuracy": round(acc, 3), "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3), "roc_auc": round(auc, 3),
            "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
        },
    }
    (HERE / "metrics_full.json").write_text(json.dumps(metrics, indent=2))

    print("=" * 56)
    print("FORECASTER — FULL EVALUATION (held-out last 21 days)")
    print("=" * 56)
    print(f"\n1) POINT FORECAST (per zone-day, {len(test):,} cases)")
    print(f"   MAE         {mae:6.3f}   (naive baseline {base_mae:.3f})")
    print(f"   RMSE        {rmse:6.3f}")
    print(f"   WAPE        {100*wape:6.1f}%   (total volume error)")
    print(f"   R2          {r2:6.3f}")
    print(f"   Pearson r   {pearson:6.3f}")
    print(f"   Bias        {bias:+6.3f}   (≈0 = unbiased)")
    print(f"\n2) ZONE RANKING ({len(zones)} zones)")
    print(f"   Spearman ρ  {spearman:6.3f}")
    print(f"   Precision@10/20/50   {p10:.0%} / {p20:.0%} / {p50:.0%}")
    print(f"\n3) HOTSPOT CLASSIFICATION (top-{TOP_K_HOTSPOT} = hotspot, yes/no)")
    print(f"   Accuracy    {acc:6.1%}")
    print(f"   Precision   {prec:6.1%}")
    print(f"   Recall      {rec:6.1%}")
    print(f"   F1          {f1:6.3f}")
    print(f"   ROC-AUC     {auc:6.3f}")
    print(f"   Confusion   TP={tp} FP={fp} FN={fn} TN={tn}")
    print("\nSaved -> metrics_full.json")


if __name__ == "__main__":
    main()
