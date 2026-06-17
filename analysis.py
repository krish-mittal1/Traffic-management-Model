"""Group violations into H3 hotspot zones and compute a base severity/persistence score."""

import pathlib

import h3
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
SRC = HERE / "violations_clean.parquet"
OUT = HERE / "zones_h3.parquet"

H3_RES = 9  # ~174 m hexagons (roughly street-block sized)

# Weight per violation type: lane/footpath/junction blockers hurt flow most,
# admin offences (number plate etc.) barely affect congestion.
SEVERITY = {
    "PARKING ON FOOTPATH": 3.0,
    "PARKING NEAR ROAD CROSSING": 3.0,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 3.0,
    "PARKING IN A MAIN ROAD": 2.5,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 2.5,
    "DOUBLE PARKING": 2.5,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 2.0,
    "WRONG PARKING": 1.5,
    "NO PARKING": 1.5,
    "DEFECTIVE NUMBER PLATE": 0.2,
    "USING BLACK FILM/OTHER MATERIALS": 0.2,
    "REFUSE TO GO FOR HIRE": 0.2,
}
DEFAULT_SEVERITY = 1.0


def short_loc(s):
    m = s.mode()
    if m.empty:
        return "NA"
    return ", ".join(str(m.iat[0]).split(",")[:2]).strip()


def main():
    df = pd.read_parquet(SRC)
    print(f"Loaded {len(df):,} violations")

    df["h3"] = [h3.latlng_to_cell(lat, lon, H3_RES)
                for lat, lon in zip(df["latitude"].to_numpy(), df["longitude"].to_numpy())]
    df["severity"] = df["primary_violation"].map(SEVERITY).fillna(DEFAULT_SEVERITY)
    span_days = (df["created_ist"].max() - df["created_ist"].min()).days or 1

    mode_or_na = lambda s: s.mode().iat[0] if not s.mode().empty else "NA"
    g = df.groupby("h3")
    zones = g.agg(
        violations=("id", "count"),
        severity_sum=("severity", "sum"),
        active_days=("date", "nunique"),
        top_station=("police_station", mode_or_na),
        top_location=("location", short_loc),
    ).reset_index()

    centres = zones["h3"].apply(lambda c: pd.Series(h3.cell_to_latlng(c), index=["lat", "lon"]))
    zones = pd.concat([zones, centres], axis=1)
    zones["top_violation"] = g["primary_violation"].agg(mode_or_na).values
    zones["peak_hour"] = g["hour"].agg(
        lambda s: int(s.mode().iat[0]) if not s.mode().empty else -1).values

    # persistence rewards chronic chokepoints over one-off spikes
    zones["persistence"] = zones["active_days"] / span_days
    raw = zones["severity_sum"] * (0.5 + 0.5 * zones["persistence"])
    z = np.log1p(raw)
    zones["impact_score"] = (100 * (z - z.min()) / (z.max() - z.min())).round(1)
    zones = zones.sort_values("impact_score", ascending=False).reset_index(drop=True)
    zones["rank"] = zones.index + 1

    zones.to_parquet(OUT, index=False)
    print(f"Wrote {len(zones):,} hotspot zones -> {OUT.name} (H3 res {H3_RES})")


if __name__ == "__main__":
    main()
