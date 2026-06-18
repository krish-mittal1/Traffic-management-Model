"""Forecast next-7-day violations per zone with a gradient-boosted model.

Features go beyond pure history: each zone also carries leak-free structural context
(road class, junction proximity, metro/market vicinity), which makes the ranking more
robust across time windows. Validated with a rolling-origin backtest against a naive
persistence baseline. Writes forecast_eval.json, forecast_*.parquet and
zone_time_profile.parquet."""

import json
import pathlib

import h3
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

HERE = pathlib.Path(__file__).parent
SRC = HERE / "violations_clean.parquet"
ZONES = HERE / "zones_h3.parquet"
H3_RES = 9
TEST_DAYS = 21  # final 3 weeks held out for the headline validation
N_FOLDS = 3     # rolling-origin backtest folds
FOLD_DAYS = 14  # length of each backtest window

# structural context per zone — exogenous (not derived from the counts we predict),
# so it is safe to feed the forecaster without leaking the target.
STRUCT = ["road_factor", "lanes_factor", "junction_factor",
          "junction_dist_m", "near_metro", "near_market"]
TEMPORAL = ["lag1", "lag2", "lag7", "roll7", "roll14", "roll28", "zmean",
            "dow", "is_weekend", "dom"]
FEATS = TEMPORAL + STRUCT


def build_panel(df):
    # dense (zone, date) panel of daily counts, zero-filled on quiet days
    df = df.copy()
    df["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(df.latitude, df.longitude)]
    df["d"] = pd.to_datetime(df["date"])
    daily = df.groupby(["h3", "d"]).size().rename("y").reset_index()

    keep = daily.groupby("h3")["y"].sum()
    keep = keep[keep >= 30].index  # only zones with enough history to forecast
    daily = daily[daily.h3.isin(keep)]

    full_dates = pd.date_range(daily.d.min(), daily.d.max(), freq="D")
    idx = pd.MultiIndex.from_product([keep, full_dates], names=["h3", "d"])
    panel = daily.set_index(["h3", "d"]).reindex(idx, fill_value=0).reset_index()

    # attach leak-free structural context
    z = pd.read_parquet(ZONES)[["h3"] + STRUCT].copy()
    z["near_metro"] = z["near_metro"].astype(int)
    z["near_market"] = z["near_market"].astype(int)
    panel = panel.merge(z, on="h3", how="left")
    panel[STRUCT] = panel[STRUCT].fillna(0)
    return panel


def add_features(panel):
    panel = panel.sort_values(["h3", "d"]).copy()
    g = panel.groupby("h3")["y"]
    panel["lag1"] = g.shift(1)
    panel["lag2"] = g.shift(2)
    panel["lag7"] = g.shift(7)
    panel["roll7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    panel["roll14"] = g.shift(1).rolling(14).mean().reset_index(level=0, drop=True)
    panel["roll28"] = g.shift(1).rolling(28).mean().reset_index(level=0, drop=True)
    panel["zmean"] = g.shift(1).expanding().mean().reset_index(level=0, drop=True)  # leak-free zone rate
    panel["dow"] = panel.d.dt.dayofweek
    panel["is_weekend"] = (panel.dow >= 5).astype(int)
    panel["dom"] = panel.d.dt.day
    return panel


def precision_at_k(actual, pred, k):
    return len(set(actual.nlargest(k).index) & set(pred.nlargest(k).index)) / k


def fit_model():
    # Poisson loss — correct for count data; lifts hotspot precision.
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=400, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, random_state=42,
    )


def evaluate_window(panel, cutoff, horizon):
    """Train on data up to cutoff, score the next `horizon` days. Returns a metrics dict."""
    train = panel[panel.d <= cutoff].dropna(subset=FEATS)
    test = panel[(panel.d > cutoff) & (panel.d <= cutoff + pd.Timedelta(days=horizon))].dropna(subset=FEATS).copy()
    if test.empty:
        return None
    model = fit_model()
    model.fit(train[FEATS], train["y"])
    test["pred"] = model.predict(test[FEATS]).clip(min=0)

    actual = test.groupby("h3")["y"].sum()
    pred_z = test.groupby("h3")["pred"].sum()
    naive = train.groupby("h3")["y"].sum().reindex(actual.index).fillna(0)
    return {
        "model_mae": mean_absolute_error(test["y"], test["pred"]),
        "base_mae": mean_absolute_error(test["y"], test["roll7"]),
        "spearman": float(actual.rank().corr(pred_z.rank())),
        "p20": precision_at_k(actual, pred_z, 20),
        "p50": precision_at_k(actual, pred_z, 50),
        "base_p20": precision_at_k(actual, naive, 20),
        "base_p50": precision_at_k(actual, naive, 50),
        "actual": actual, "pred_z": pred_z, "naive": naive,
    }


def main():
    df = pd.read_parquet(SRC)
    panel = add_features(build_panel(df))
    print(f"Panel: {panel.h3.nunique():,} zones x {panel.d.nunique()} days = {len(panel):,} rows")
    print(f"Features: {len(FEATS)} ({', '.join(FEATS)})")

    last = panel.d.max()

    # headline validation on the final TEST_DAYS window
    cutoff = last - pd.Timedelta(days=TEST_DAYS)
    r = evaluate_window(panel, cutoff, TEST_DAYS)

    # on the zones that rose most vs history, does the model still beat persistence?
    risers = (r["naive"].rank(ascending=False) - r["actual"].rank(ascending=False)) \
        .sort_values(ascending=False).head(30).index
    ml_risers = float(r["actual"][risers].rank().corr(r["pred_z"][risers].rank()))
    naive_risers = float(r["actual"][risers].rank().corr(r["naive"][risers].rank()))

    # rolling-origin backtest: repeat over several earlier windows for a robust read
    cv = []
    for k in range(N_FOLDS):
        c = last - pd.Timedelta(days=TEST_DAYS + k * FOLD_DAYS)
        rr = evaluate_window(panel, c, FOLD_DAYS)
        if rr:
            cv.append(rr)
    cv_p20 = float(np.mean([c["p20"] for c in cv]))
    cv_p50 = float(np.mean([c["p50"] for c in cv]))
    cv_sp = float(np.mean([c["spearman"] for c in cv]))
    mase = r["model_mae"] / r["base_mae"]  # <1 means we beat the naive roll-7 baseline

    metrics = {
        "test_window_days": TEST_DAYS,
        "n_features": len(FEATS),
        "model_mae_per_zone_day": round(r["model_mae"], 3),
        "baseline_roll7_mae": round(r["base_mae"], 3),
        "mase_vs_roll7": round(mase, 3),
        "rank_spearman": round(r["spearman"], 3),
        "precision_at_20": round(r["p20"], 3),
        "precision_at_50": round(r["p50"], 3),
        "baseline_precision_at_20": round(r["base_p20"], 3),
        "baseline_precision_at_50": round(r["base_p50"], 3),
        "emerging_spearman_ml": round(ml_risers, 3),
        "emerging_spearman_naive": round(naive_risers, 3),
        "backtest_folds": len(cv),
        "cv_precision_at_20_mean": round(cv_p20, 3),
        "cv_precision_at_50_mean": round(cv_p50, 3),
        "cv_rank_spearman_mean": round(cv_sp, 3),
    }
    (HERE / "forecast_eval.json").write_text(json.dumps(metrics, indent=2))
    print("\nHeadline validation (held-out last 21 days):")
    print(f"  point forecast MAE : {r['model_mae']:.3f}/zone/day (baseline {r['base_mae']:.3f}, MASE {mase:.2f})")
    print(f"  zone-ranking Spearman : {r['spearman']:.3f}")
    print(f"  precision@20 : {r['p20']:.0%} (naive {r['base_p20']:.0%})   "
          f"precision@50 : {r['p50']:.0%} (naive {r['base_p50']:.0%})")
    print(f"  emerging-hotspot skill (Spearman on risers): ML {ml_risers:.2f} vs naive {naive_risers:.2f}")
    print(f"\nRolling-origin backtest ({len(cv)} folds x {FOLD_DAYS}d):")
    print(f"  precision@20 {cv_p20:.0%}   precision@50 {cv_p50:.0%}   Spearman {cv_sp:.3f}")

    # retrain on everything, then roll the forecast forward 7 days
    full = panel.dropna(subset=FEATS)
    model = fit_model()
    model.fit(full[FEATS], full["y"])
    hist = {z: g.sort_values("d")["y"].tolist() for z, g in panel.groupby("h3")}
    struct_map = panel.groupby("h3")[STRUCT].first().to_dict("index")
    rows = []
    for step in range(1, 8):
        day = last + pd.Timedelta(days=step)
        zones = list(hist.keys())
        feats = []
        for z in zones:
            s = hist[z]
            sm = struct_map.get(z, dict.fromkeys(STRUCT, 0))
            feats.append([
                s[-1], s[-2] if len(s) >= 2 else np.nan, s[-7] if len(s) >= 7 else np.nan,
                np.mean(s[-7:]), np.mean(s[-14:]), np.mean(s[-28:]), np.mean(s),
                day.dayofweek, int(day.dayofweek >= 5), day.day,
                sm["road_factor"], sm["lanes_factor"], sm["junction_factor"],
                sm["junction_dist_m"], sm["near_metro"], sm["near_market"],
            ])
        yhat = model.predict(pd.DataFrame(feats, columns=FEATS)).clip(min=0)
        for z, yh in zip(zones, yhat):
            hist[z].append(yh)
            rows.append({"h3": z, "date": day.date(), "pred": round(float(yh), 2)})
    fc = pd.DataFrame(rows)
    fc.to_parquet(HERE / "forecast_next7.parquet", index=False)

    nxt = fc.groupby("h3")["pred"].sum().rename("pred_next7").reset_index()
    cent = nxt["h3"].apply(lambda c: pd.Series(h3.cell_to_latlng(c), index=["lat", "lon"]))
    nxt = pd.concat([nxt, cent], axis=1).sort_values("pred_next7", ascending=False)
    nxt.to_parquet(HERE / "forecast_zones.parquet", index=False)
    print(f"\nForecast next 7 days written for {fc.h3.nunique():,} zones.")

    # per-zone day-of-week x hour profile, used by the playbook for the "when"
    dfp = df.copy()
    dfp["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(dfp.latitude, dfp.longitude)]
    prof = dfp.groupby(["h3", "dow", "hour"]).size().rename("n").reset_index()
    prof.to_parquet(HERE / "zone_time_profile.parquet", index=False)
    print(f"Wrote per-zone time profiles ({len(prof):,} rows).")


if __name__ == "__main__":
    main()
