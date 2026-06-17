"""Shortest patrol loop over the top-PICS hotspots (nearest-neighbour + 2-opt TSP).
Output: patrol_route.parquet."""

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
TOP_K = 12          # stops in one patrol shift
AVG_KMPH = 18       # assumed Bengaluru patrol speed
MINS_PER_STOP = 15  # time spent enforcing at each stop


def haversine_matrix(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))   # km


def nearest_neighbour(D, start=0):
    n = len(D)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: D[last, j])
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour


def two_opt(tour, D):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 1):
            for k in range(i + 1, len(tour)):
                a, b = tour[i - 1], tour[i]
                c = tour[k]
                d = tour[k + 1] if k + 1 < len(tour) else None
                if d is None:
                    continue
                delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                if delta < -1e-9:
                    tour[i:k + 1] = tour[i:k + 1][::-1]
                    improved = True
    return tour


def main():
    zones = pd.read_parquet(HERE / "zones_h3.parquet").head(TOP_K).reset_index(drop=True)
    D = haversine_matrix(zones.lat.values, zones.lon.values)

    tour = two_opt(nearest_neighbour(D, start=0), D)

    route = zones.iloc[tour].reset_index(drop=True)
    legs = [0.0] + [D[tour[i], tour[i + 1]] for i in range(len(tour) - 1)]
    route["stop"] = range(1, len(route) + 1)
    route["leg_km"] = np.round(legs, 2)
    route["cum_km"] = np.round(np.cumsum(legs), 2)
    drive_min = route["cum_km"].iloc[-1] / AVG_KMPH * 60
    total_min = drive_min + MINS_PER_STOP * len(route)

    keep = ["stop", "h3", "lat", "lon", "pics", "top_location", "top_station",
            "road_class", "peak_hour", "leg_km", "cum_km"]
    route[keep].to_parquet(HERE / "patrol_route.parquet", index=False)

    naive_km = sum(D[i, i + 1] for i in range(len(zones) - 1))  # visit in PICS order
    opt_km = route["cum_km"].iloc[-1]
    print(f"Optimized patrol route over top {TOP_K} hotspots:")
    print(f"  total distance : {opt_km:.1f} km  (vs {naive_km:.1f} km visiting in rank order "
          f"-> {100*(naive_km-opt_km)/naive_km:.0f}% shorter)")
    print(f"  est. shift time: {total_min:.0f} min "
          f"({drive_min:.0f} driving + {MINS_PER_STOP*len(route)} enforcing)\n")
    print(route[["stop", "pics", "top_location", "top_station", "leg_km", "cum_km"]].to_string(index=False))


if __name__ == "__main__":
    main()
