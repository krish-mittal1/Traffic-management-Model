"""Export all parquet/json data to dashboard_data.json for the inline HTML dashboard."""
import json, pathlib
import pandas as pd

HERE = pathlib.Path(".")
g = lambda f: pd.read_parquet(HERE / f)
j = lambda f: json.loads((HERE / f).read_text())

print("Loading data...")
df   = g("violations_clean.parquet")
zones = g("zones_h3.parquet")
playbook = g("playbook.parquet")
forecast = g("forecast_zones.parquet")
ev   = j("forecast_eval.json")
route = g("patrol_route.parquet")
causal = j("causal_impact.json")
es   = g("causal_eventstudy.parquet")
emerging = g("emerging.parquet")
cb   = j("cost_benefit.json")
curve = g("coverage_curve.parquet")

# ── Zones (top 500 for map perf) ──
z_cols = ["h3","rank","lat","lon","pics","violations","road_class",
          "top_location","top_station","top_violation","context",
          "junction_dist_m","peak_hour"]
zones_out = zones.head(500)[z_cols].fillna("").round({"pics":1,"lat":6,"lon":6}).to_dict("records")

# ── Forecast ──
fz = forecast.head(120).merge(
    zones[["h3","top_location","top_station","road_class","context"]],
    on="h3", how="left")
fz["top_location"] = fz["top_location"].fillna("Zone")
fz["pred_next7"] = fz["pred_next7"].round(1)
fz_out = fz[["h3","lat","lon","pred_next7","top_location","top_station","road_class","context"]].fillna("").to_dict("records")

# ── Patrol route ──
route["peak_hour"] = route["peak_hour"].astype(int)
route_out = route[["stop","pics","top_location","top_station","road_class","peak_hour","lat","lon","leg_km","cum_km"]].fillna("").round({"leg_km":2,"cum_km":2}).to_dict("records")

# ── Playbook ──
pb = playbook.fillna("")
pb_out = pb.to_dict("records")

# ── Emerging ──
em = emerging.head(20)[["top_location","top_station","recent_daily","pred_daily","momentum_pct"]].fillna("")
em_out = em.round({"recent_daily":1,"pred_daily":1,"momentum_pct":1}).to_dict("records")

# ── Coverage curve ──
curve_out = curve[["zone_pct","viol_pct"]].fillna(0).round(2).to_dict("records")

# ── Causal event study ──
es_out = es[["rel_week","norm_v"]].fillna(0).round(3).to_dict("records")

# ── Violation patterns (pre-aggregated, no raw rows needed in browser) ──
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
piv = df.groupby(["dow_name","hour"]).size().reset_index(name="n")
piv_out = piv.to_dict("records")

vc = df["primary_violation"].value_counts().head(10).reset_index()
vc.columns = ["violation","count"]

sc = df["police_station"].value_counts().head(10).reset_index()
sc.columns = ["station","count"]

vh = df["vehicle_type"].value_counts().head(8).reset_index()
vh.columns = ["vehicle","count"]

# ── Filter options ──
stations = sorted(df["police_station"].dropna().unique().tolist())
vtypes   = sorted(df["primary_violation"].dropna().unique().tolist())
vehicles = sorted(df["vehicle_type"].dropna().unique().tolist())

# ── Zone-level filter aggregates: filtered patterns by station ──
# Pre-compute violation patterns for top 20 stations so the filter can work in-browser
station_patterns = {}
for st in df["police_station"].value_counts().head(20).index:
    sf = df[df["police_station"] == st]
    pv = sf.groupby(["dow_name","hour"]).size().reset_index(name="n")
    vc2 = sf["primary_violation"].value_counts().head(10).reset_index()
    vc2.columns = ["violation","count"]
    station_patterns[st] = {
        "dow_hour": pv.to_dict("records"),
        "violations": vc2.to_dict("records"),
    }

# ── Combine ──
data = {
    "meta": {
        "total_violations": int(len(df)),
        "date_min": str(df["created_ist"].min().date()),
        "date_max": str(df["created_ist"].max().date()),
        "n_zones": int(len(zones)),
        "n_stations": int(df["police_station"].nunique()),
    },
    "zones": zones_out,
    "forecast": fz_out,
    "route": route_out,
    "playbook": pb_out,
    "emerging": em_out,
    "curve": curve_out,
    "eventstudy": es_out,
    "eval": ev,
    "causal": causal,
    "costbenefit": cb,
    "patterns": {
        "dow_hour": piv_out,
        "violations": vc.to_dict("records"),
        "stations": sc.to_dict("records"),
        "vehicles": vh.to_dict("records"),
    },
    "station_patterns": station_patterns,
    "filters": {
        "stations": stations,
        "vtypes": vtypes,
        "vehicles": vehicles,
    },
}

out_path = HERE / "dashboard_data.json"
with open(out_path, "w") as f:
    json.dump(data, f, default=str)

size_kb = out_path.stat().st_size / 1024
print(f"OK  Exported dashboard_data.json  ({size_kb:.0f} KB)")
print(f"    {len(zones_out)} zones, {len(fz_out)} forecast, {len(route_out)} stops, {len(pb_out)} playbook rows")
