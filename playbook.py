"""Combine PICS (where), the day/hour profile (when) and the forecast (what's coming)
into a ranked patrol schedule. Output: playbook.parquet / playbook.csv."""

import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).parent
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def best_window(prof_zone):
    # busiest contiguous 3-hour window for a zone
    by_hour = prof_zone.groupby("hour")["n"].sum()
    by_hour = by_hour.reindex(range(24), fill_value=0)
    best_h, best_sum = 0, -1
    for h in range(22):
        s = by_hour.iloc[h:h + 3].sum()
        if s > best_sum:
            best_sum, best_h = s, h
    return best_h, best_h + 3


def main():
    zones = pd.read_parquet(HERE / "zones_h3.parquet")
    prof = pd.read_parquet(HERE / "zone_time_profile.parquet")
    fc = pd.read_parquet(HERE / "forecast_zones.parquet")[["h3", "pred_next7"]]

    zones = zones.merge(fc, on="h3", how="left")

    rows = []
    for _, z in zones.head(50).iterrows():
        pz = prof[prof.h3 == z.h3]
        if len(pz):
            wd = pz[pz.dow < 5]["n"].sum()
            we = pz[pz.dow >= 5]["n"].sum()
            day_lean = "Weekdays" if wd >= we * 1.4 else "Weekends" if we >= wd * 1.4 else "All week"
            busiest = DAYS[int(pz.groupby("dow")["n"].sum().idxmax())]
            h0, h1 = best_window(pz)
        else:
            day_lean, busiest, h0, h1 = "All week", "-", int(z.peak_hour), int(z.peak_hour) + 2
        rows.append({
            "rank": int(z["rank"]),
            "pics": z.pics,
            "location": z.top_location,
            "station": z.top_station,
            "road": z.road_class,
            "violations_5mo": int(z.violations),
            "pred_next7": round(float(z.pred_next7), 0) if pd.notna(z.pred_next7) else None,
            "deploy_when": f"{day_lean}, {h0:02d}:00-{h1:02d}:00",
            "busiest_day": busiest,
            "target_violation": z.top_violation,
            "driver": z.get("context", ""),
        })

    pb = pd.DataFrame(rows)
    pb.to_parquet(HERE / "playbook.parquet", index=False)
    pb.to_csv(HERE / "playbook.csv", index=False)
    print(f"Wrote enforcement playbook for top {len(pb)} zones -> playbook.parquet / .csv\n")
    show = pb.head(15)[
        ["rank", "pics", "location", "station", "road",
         "deploy_when", "target_violation", "pred_next7"]
    ]
    with pd.option_context("display.max_colwidth", 32, "display.width", 200):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
