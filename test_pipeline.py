"""
Sanity tests for the artifacts. Run: .venv/Scripts/python -m pytest -q
(or just .venv/Scripts/python test_pipeline.py)
"""

import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).parent


def test_clean_data_quality():
    df = pd.read_parquet(HERE / "violations_clean.parquet")
    assert len(df) > 200_000, "expected the bulk of records to survive cleaning"
    # geo must be inside Bengaluru
    assert df.latitude.between(12.7, 13.2).all()
    assert df.longitude.between(77.3, 77.9).all()
    # rejected/duplicate rows must be gone
    assert not df.validation_status.isin(["rejected", "duplicate"]).any()
    assert df.hour.between(0, 23).all()


def test_zones_scored_and_ranked():
    z = pd.read_parquet(HERE / "zones_h3.parquet")
    assert {"pics", "road_class", "junction_dist_m", "rank"}.issubset(z.columns)
    assert z.pics.between(0, 100).all()
    assert (z.junction_dist_m >= 0).all()
    # rank is dense 1..N and ordered by PICS
    assert z["rank"].min() == 1 and z["rank"].max() == len(z)
    assert z.sort_values("rank").pics.is_monotonic_decreasing


def test_patrol_route_valid():
    r = pd.read_parquet(HERE / "patrol_route.parquet")
    assert len(r) >= 8
    # every stop visited once, cumulative distance monotonic non-decreasing
    assert r["stop"].is_unique
    assert r["cum_km"].is_monotonic_increasing


def test_model_beats_baseline():
    ev = json.loads((HERE / "forecast_eval.json").read_text())
    # the ML must beat naive persistence, else the forecast adds nothing
    assert ev["precision_at_50"] > ev["baseline_precision_at_50"]
    assert ev["emerging_spearman_ml"] > ev["emerging_spearman_naive"]


def test_poi_context_tagged():
    z = pd.read_parquet(HERE / "zones_h3.parquet")
    assert {"metro_dist_m", "market_dist_m", "context"}.issubset(z.columns)
    assert z["context"].notna().all()


def test_emerging_and_roi():
    e = pd.read_parquet(HERE / "emerging.parquet")
    assert len(e) > 0 and (e["momentum"] > 0).all()
    cb = json.loads((HERE / "cost_benefit.json").read_text())
    assert 0 < cb["top_50_zones_cover_pct"] <= 100
    assert cb["officer_hours_saved_per_week"] > 0


def test_causal_outputs():
    c = json.loads((HERE / "causal_impact.json").read_text())
    assert "did_estimate" in c and "hotspot_rank_persistence_spearman" in c
    assert "did_caveat" in c  # honesty: regression-to-mean caveat must be present
    assert 0 <= c["top20_retention"] <= 1


def test_forecast_is_useful():
    ev = json.loads((HERE / "forecast_eval.json").read_text())
    # the operational claim: we identify next-period top zones well
    assert ev["precision_at_20"] >= 0.7, "forecast should reliably pre-identify hotspots"
    assert ev["rank_spearman"] >= 0.7
    fc = pd.read_parquet(HERE / "forecast_zones.parquet")
    assert (fc.pred_next7 >= 0).all()


def test_playbook_actionable():
    pb = pd.read_parquet(HERE / "playbook.parquet")
    assert len(pb) >= 25
    for col in ["location", "station", "road", "deploy_when", "target_violation"]:
        assert col in pb.columns
        assert pb[col].notna().all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll sanity tests passed.")
