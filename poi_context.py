"""Tag each hotspot with distance to the nearest metro station / marketplace,
so we can explain the demand driver behind it (metro spillover, market, junction)."""

import pathlib

import numpy as np
import osmnx as ox
import pandas as pd
from sklearn.neighbors import BallTree

HERE = pathlib.Path(__file__).parent
ZONES = HERE / "zones_h3.parquet"
POI_CACHE = HERE / "pois.parquet"
EARTH_M = 6_371_000


def fetch_pois(bbox):
    if POI_CACHE.exists():
        print(f"Loading cached POIs -> {POI_CACHE.name}")
        return pd.read_parquet(POI_CACHE)
    print("Fetching metro stations + marketplaces from OpenStreetMap ...")
    rows = []
    for kind, tags in [("metro", {"station": "subway"}), ("market", {"amenity": "marketplace"})]:
        g = ox.features_from_bbox(bbox=bbox, tags=tags)
        cent = g.geometry.to_crs(3857).centroid.to_crs(4326)
        for pt in cent:
            rows.append({"kind": kind, "lat": pt.y, "lon": pt.x})
    pois = pd.DataFrame(rows)
    pois.to_parquet(POI_CACHE, index=False)
    print(f"  cached {len(pois)} POIs -> {POI_CACHE.name}")
    return pois


def nearest_m(zones, pts):
    if len(pts) == 0:
        return np.full(len(zones), np.nan)
    tree = BallTree(np.radians(pts[["lat", "lon"]].values), metric="haversine")
    d, _ = tree.query(np.radians(zones[["lat", "lon"]].values), k=1)
    return (d[:, 0] * EARTH_M).round(1)


def main():
    zones = pd.read_parquet(ZONES)
    pad = 0.01
    bbox = (zones.lon.min() - pad, zones.lat.min() - pad, zones.lon.max() + pad, zones.lat.max() + pad)
    pois = fetch_pois(bbox)

    zones["metro_dist_m"] = nearest_m(zones, pois[pois.kind == "metro"])
    zones["market_dist_m"] = nearest_m(zones, pois[pois.kind == "market"])
    zones["near_metro"] = zones["metro_dist_m"] < 500
    zones["near_market"] = zones["market_dist_m"] < 300

    def label(r):
        tags = []
        if r.near_metro: tags.append("Metro spillover")
        if r.near_market: tags.append("Market load/unload")
        if r.junction_dist_m < 40: tags.append("Junction chokepoint")
        return " · ".join(tags) if tags else "General commercial"

    zones["context"] = zones.apply(label, axis=1)
    zones.to_parquet(ZONES, index=False)

    print(f"\nTagged {len(zones):,} zones with POI context")
    print(f"  near a metro station (<500 m): {zones.near_metro.sum():,} zones")
    print(f"  near a marketplace   (<300 m): {zones.near_market.sum():,} zones")
    print("\nContext mix of the top-50 hotspots:")
    print(zones.head(50)["context"].value_counts().to_string())


if __name__ == "__main__":
    main()
