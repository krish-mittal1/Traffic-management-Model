"""Run the whole pipeline end to end: raw CSV -> every artifact the dashboard needs.

Usage:  python run_pipeline.py "path/to/violations.csv"
Then:   streamlit run app.py
"""

import runpy
import time

STEPS = [
    ("clean_data", "Clean & validate raw violations"),
    ("analysis", "Detect H3 hotspots + base scoring"),
    ("road_impact", "PICS: OSM road + junction grounding"),
    ("poi_context", "Tag metro/market/junction context"),
    ("forecast", "Forecast next-7-day hotspots"),
    ("emerging", "Detect emerging (rising) hotspots"),
    ("playbook", "Build enforcement playbook"),
    ("route_optimizer", "Optimize patrol route (TSP)"),
    ("cost_benefit", "Coverage & officer-hour ROI"),
    ("causal_impact", "Enforcement impact / persistence study"),
    ("evaluate", "Full model metrics report"),
]

if __name__ == "__main__":
    t0 = time.time()
    for i, (mod, desc) in enumerate(STEPS, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(STEPS)}] {desc}  ({mod}.py)\n{'=' * 60}")
        runpy.run_module(mod, run_name="__main__")
    print(f"\nPipeline complete in {time.time() - t0:.0f}s. Launch: streamlit run app.py")
