"""Flag zones whose forecast is rising fastest vs their recent rate — emerging hotspots
worth enforcing before they peak. Output: emerging.parquet."""

import pathlib

import pandas as pd

from forecast import add_features, build_panel

HERE = pathlib.Path(__file__).parent
TRAIL_DAYS = 28


def main():
    df = pd.read_parquet(HERE / "violations_clean.parquet")
    panel = add_features(build_panel(df))
    fc = pd.read_parquet(HERE / "forecast_zones.parquet")
    zones = pd.read_parquet(HERE / "zones_h3.parquet")

    last = panel.d.max()
    trail = panel[panel.d > last - pd.Timedelta(days=TRAIL_DAYS)].groupby("h3")["y"].mean()

    e = fc.merge(trail.rename("recent_daily").reset_index(), on="h3", how="left")
    e["pred_daily"] = e["pred_next7"] / 7
    e["momentum"] = e["pred_daily"] - e["recent_daily"]
    e["momentum_pct"] = (100 * e["momentum"] / e["recent_daily"].clip(lower=0.3)).round(0)

    e = e[(e["recent_daily"] >= 0.5) & (e["momentum"] > 0)]
    e = e.merge(zones[["h3", "top_location", "top_station", "road_class", "context", "rank"]],
                on="h3", how="left").rename(columns={"rank": "pics_rank"})
    e = e.sort_values("momentum", ascending=False).reset_index(drop=True)
    e["emerge_rank"] = e.index + 1

    keep = ["emerge_rank", "h3", "lat", "lon", "top_location", "top_station", "context",
            "recent_daily", "pred_daily", "momentum", "momentum_pct", "pics_rank"]
    e[keep].to_parquet(HERE / "emerging.parquet", index=False)

    print(f"Detected {len(e):,} rising zones -> emerging.parquet")
    print("\nTop 10 EMERGING hotspots (forecast rising fastest):")
    show = e.head(10)[["emerge_rank", "top_location", "top_station",
                       "recent_daily", "pred_daily", "momentum_pct", "pics_rank"]].copy()
    show["recent_daily"] = show["recent_daily"].round(1)
    show["pred_daily"] = show["pred_daily"].round(1)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
