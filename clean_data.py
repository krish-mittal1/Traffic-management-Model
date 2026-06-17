"""Clean the raw BTP violations CSV into a tidy parquet for the rest of the pipeline."""

import ast
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
RAW_CSV = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "violations_raw.csv"
OUT = HERE / "violations_clean.parquet"

# Bengaluru bounding box, used to throw out junk coordinates
LAT_MIN, LAT_MAX = 12.7, 13.2
LON_MIN, LON_MAX = 77.3, 77.9


def parse_list(val):
    if val is None or (isinstance(val, float) and np.isnan(val)) or val in ("", "NULL"):
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return [val]


def main():
    print(f"Reading {RAW_CSV.name} ...")
    df = pd.read_csv(RAW_CSV, dtype=str, encoding="utf-8", encoding_errors="replace")
    print(f"  raw rows: {len(df):,}")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["created_dt"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df["created_ist"] = df["created_dt"].dt.tz_convert("Asia/Kolkata")
    df["hour"] = df["created_ist"].dt.hour
    df["dow"] = df["created_ist"].dt.dayofweek
    df["dow_name"] = df["created_ist"].dt.day_name()
    df["date"] = df["created_ist"].dt.date
    df["month"] = df["created_ist"].dt.to_period("M").astype(str)

    # Excel sometimes renames the duplicate "offence_code" header to "violation_type.1"
    off_col = next((c for c in ("offence_code", "violation_type.1") if c in df.columns), None)
    df["violations"] = df["violation_type"].apply(parse_list)
    df["offences"] = df[off_col].apply(parse_list) if off_col else [[]] * len(df)
    df["n_violations"] = df["violations"].apply(len)
    df["primary_violation"] = df["violations"].apply(lambda x: x[0] if x else "UNKNOWN")

    before = len(df)
    df = df[df["latitude"].between(LAT_MIN, LAT_MAX) & df["longitude"].between(LON_MIN, LON_MAX)]
    df = df[df["created_dt"].notna()]
    df = df[~df["validation_status"].fillna("none").isin(["rejected", "duplicate"])]
    print(f"  dropped {before - len(df):,} rows (bad geo / time / rejected / duplicate)")

    keep = [
        "id", "latitude", "longitude", "location", "vehicle_type",
        "primary_violation", "violations", "offences", "n_violations",
        "police_station", "junction_name", "validation_status",
        "created_ist", "hour", "dow", "dow_name", "date", "month",
    ]
    out = df[keep].copy()
    out["violations"] = out["violations"].apply(json.dumps)
    out["offences"] = out["offences"].apply(json.dumps)

    out.to_parquet(OUT, index=False)
    print(f"\nWrote {len(out):,} clean rows -> {OUT.name}")
    print(f"  date range: {out['created_ist'].min()} -> {out['created_ist'].max()}")
    print(f"  police stations: {out['police_station'].nunique()}")


if __name__ == "__main__":
    main()
