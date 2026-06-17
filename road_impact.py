"""PICS — Parking-Induced Congestion Score, grounded in the real OSM road network.

PICS = 100 * norm(log1p(severity * persistence * road_criticality * junction_proximity)),
so a violation on a busy arterial near a junction outranks one on a quiet lane.
The drive network is downloaded once and cached to road_graph.graphml.
"""

import pathlib

import numpy as np
import osmnx as ox
import pandas as pd
from sklearn.neighbors import BallTree

HERE = pathlib.Path(__file__).parent
ZONES = HERE / "zones_h3.parquet"
GRAPHML = HERE / "road_graph.graphml"
EARTH_M = 6_371_000

# How much traffic each road class carries (= how much blocking it hurts).
ROAD_WEIGHT = {
    "motorway": 3.0, "motorway_link": 3.0, "trunk": 3.0, "trunk_link": 3.0,
    "primary": 2.5, "primary_link": 2.5, "secondary": 2.0, "secondary_link": 2.0,
    "tertiary": 1.5, "tertiary_link": 1.5, "unclassified": 1.1,
    "residential": 1.0, "living_street": 0.9, "service": 0.8,
}
DEFAULT_ROAD_WEIGHT = 1.0


def first(x):
    return x[0] if isinstance(x, list) and x else (None if isinstance(x, list) else x)


def road_weight(hwy):
    return ROAD_WEIGHT.get(first(hwy), DEFAULT_ROAD_WEIGHT)


def lanes_factor(lanes):
    # fewer lanes -> a parked car blocks a bigger share of the road
    try:
        n = float(first(lanes))
    except (TypeError, ValueError):
        return 1.0
    return 1.3 if n <= 1 else 1.15 if n == 2 else 0.95 if n >= 4 else 1.0


def load_graph(north, south, east, west):
    if GRAPHML.exists():
        print(f"Loading cached road graph -> {GRAPHML.name}")
        return ox.load_graphml(GRAPHML)
    print("Downloading Bengaluru drive network from OpenStreetMap ...")
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
    ox.save_graphml(G, GRAPHML)
    print(f"  cached {G.number_of_edges():,} edges -> {GRAPHML.name}")
    return G


def main():
    zones = pd.read_parquet(ZONES)
    pad = 0.01
    G = load_graph(zones.lat.max() + pad, zones.lat.min() - pad,
                   zones.lon.max() + pad, zones.lon.min() - pad)
    edges = ox.graph_to_gdfs(G, nodes=False)
    nodes = ox.graph_to_gdfs(G, edges=False)

    # snap each hotspot to its nearest road edge for road class + lane count
    print(f"Snapping {len(zones):,} hotspot zones to nearest road ...")
    ne = ox.distance.nearest_edges(G, X=zones.lon.values, Y=zones.lat.values)
    snapped = edges.loc[[tuple(e) for e in ne]]
    zones["road_class"] = [str(first(h)) for h in snapped["highway"].values]
    lanes_col = snapped["lanes"] if "lanes" in snapped.columns else pd.Series([None] * len(zones))
    zones["road_lanes"] = [first(v) for v in lanes_col.values]
    zones["road_weight"] = snapped["highway"].apply(road_weight).values
    zones["lanes_factor"] = lanes_col.apply(lanes_factor).values
    zones["road_factor"] = zones["road_weight"] * zones["lanes_factor"]

    # distance to nearest real intersection (a junction = node with 3+ streets)
    inter = nodes[nodes["street_count"] >= 3]
    tree = BallTree(np.radians(inter[["y", "x"]].values), metric="haversine")
    d, _ = tree.query(np.radians(zones[["lat", "lon"]].values), k=1)
    zones["junction_dist_m"] = (d[:, 0] * EARTH_M).round(1)
    zones["junction_factor"] = (1 + 0.5 * np.exp(-zones["junction_dist_m"] / 40)).round(3)

    raw = (zones["severity_sum"] * (0.5 + 0.5 * zones["persistence"])
           * zones["road_factor"] * zones["junction_factor"])
    z = np.log1p(raw)
    zones["pics"] = (100 * (z - z.min()) / (z.max() - z.min())).round(1)
    zones["flow_impact_score"] = zones["pics"]  # alias kept for backward compatibility
    zones = zones.sort_values("pics", ascending=False).reset_index(drop=True)
    zones["rank"] = zones.index + 1

    zones.to_parquet(ZONES, index=False)
    print(f"\nUpdated {ZONES.name} with PICS (road + junction grounded)")
    print(f"  intersections used: {len(inter):,} | median junction dist: {zones.junction_dist_m.median():.0f} m")
    print("\nTop 10 by PICS:")
    cols = ["rank", "pics", "violations", "top_station", "road_class", "junction_dist_m", "top_violation", "peak_hour"]
    print(zones[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
