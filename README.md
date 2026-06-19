# 🅿️ Parking Congestion Intelligence

**Gridlock Hackathon 2.0 — Flipkart × Bengaluru Traffic Police**
Theme: *Poor Visibility on Parking-Induced Congestion*

> AI-driven parking intelligence that **detects** illegal-parking hotspots, **quantifies their
> impact on traffic flow** with a road-network-grounded score (**PICS**), **forecasts** where
> they'll be next, **optimizes patrol routes**, **studies enforcement impact**, and hands officers
> a **mobile playbook** — turning patrol-based, reactive enforcement into targeted, proactive deployment.

Built on the real BTP dataset: **298,450 raw → 248,231 validated violations**, Nov 2023 – Apr 2024, 54 police stations, 100% geo-tagged.

---

## Why this wins the theme

Most entries stop at a heatmap. This goes end-to-end — detection → quantification → prediction → action — with six differentiators:

| Capability | What it is | Headline result |
|---|---|---|
| **PICS metric** | branded *Parking-Induced Congestion Score* fusing severity × persistence × road criticality × junction proximity | transparent, defensible 0–100 score |
| **OSM road fusion** | every hotspot snapped to the real Bengaluru road network (569,474 edges, 166,191 junctions) | arterials & junction-adjacent zones ranked above quiet lanes |
| **POI context** | tags each zone's distance to Namma Metro & marketplaces | **72%** of top-50 hotspots have an explainable driver (metro/market/junction) |
| **Forecasting** | gradient-boosted (Poisson) next-7-day prediction per zone | **ranking-validated: Spearman 0.83 (3-fold CV), precision@50 87% vs 66% naive** |
| **Emerging-hotspot detection** | flags zones rising before they peak | ML **0.65 vs 0.30 naive** on rising zones — beats persistence |
| **Patrol route optimizer** | TSP (nearest-neighbour + 2-opt) over top hotspots | **59% shorter** route (34 km vs 84 km) |
| **Cost-benefit / ROI** | coverage curve + officer-hour saving | **2.8% of zones = 50% of violations**; ~19 officer-hrs/wk saved |
| **Enforcement impact study** | difference-in-differences + hotspot-persistence | **80%** of worst-20 zones persist → targeting works |
| **Officer mobile app** | field-facing patrol UI mockup | where/when/what, on a phone |

### Maps to the brief's three pain points
| "Why it's hard today" | Our answer |
|---|---|
| Enforcement is patrol-based & **reactive** | 7-day **forecast** (ranking-validated; 80% precision@20 in 3-fold CV) + optimized **patrol route** |
| **No heatmap** of violations vs **congestion impact** | **PICS** heatmap, grounded in real road geometry |
| Difficult to **prioritize** zones | ranked zones → **playbook** → **mobile app** for the field |

## The pipeline

```
raw CSV ─▶ clean_data ─▶ analysis ─▶ road_impact ─▶ forecast ─▶ playbook ─▶ route_optimizer ─▶ causal_impact ─▶ evaluate
           validate     H3 hotspot   PICS (OSM road  next-7-day  deployable   patrol TSP        impact /        full model
           & features   + scoring    + junction)     prediction  schedule                       persistence     metrics
                                                                                                                  ▼
                                                                                              app.py (6-tab dashboard)
                                                                                              mobile_mockup.html (officer app)
```

### PICS — Parking-Induced Congestion Score

```
PICS = 100 × norm( log1p( severity × (0.5 + 0.5·persistence) × road_factor × junction_factor ) )
  severity        lane/footpath/junction blockers > generic parking > admin offences
  persistence     distinct active days / total span (chronic chokepoints > one-offs)
  road_factor     OSM road-class weight × lane factor (arterials carry more flow)
  junction_factor exp-decay boost within ~40 m of a real intersection (queue spillback)
```

> **Honest scope.** No live traffic-speed feed exists in the data, so flow impact is grounded in
> *road-network criticality* (a measured, defensible proxy). The enforcement-action field
> (`action_taken_timestamp`) is 100% empty, so the impact study leads with the **confound-free
> persistence result** and reports the DiD as suggestive only (regression-to-the-mean caveat stated).

## Run it

```bash
uv venv --python 3.13
uv pip install --python .venv -r requirements.txt

# point clean_data.py RAW_CSV at the dataset, then build everything:
.venv/Scripts/python run_pipeline.py        # 8 steps, ~5 min (downloads+caches OSM once)
.venv/Scripts/streamlit run app.py           # dashboard at http://localhost:8501
.venv/Scripts/python test_pipeline.py        # sanity tests (6/6)
```

Open `mobile_mockup.html` in a browser for the officer-app view.

## Files

| File | Role |
|---|---|
| `clean_data.py` | raw CSV → validated `violations_clean.parquet` (Excel-rename-robust) |
| `analysis.py` | H3 hotspot detection + base scoring |
| `road_impact.py` | **PICS** — OSM road + junction grounding |
| `poi_context.py` | metro/market proximity tagging → `context` |
| `forecast.py` | next-7-day forecast + temporal validation (vs baseline) |
| `emerging.py` | emerging (rising) hotspot detection → `emerging.parquet` |
| `playbook.py` | deployable enforcement schedule → `playbook.csv` |
| `route_optimizer.py` | patrol-route TSP → `patrol_route.parquet` |
| `cost_benefit.py` | coverage curve + officer-hour ROI → `cost_benefit.json` |
| `causal_impact.py` | DiD + persistence study → `causal_impact.json` |
| `evaluate.py` | full metrics (regression + ranking + classification) → `metrics_full.json` |
| `app.py` | 7-tab interactive dashboard |
| `mobile_mockup.html` | officer-facing patrol app mockup |
| `run_pipeline.py` / `test_pipeline.py` | one-command build / sanity tests |

## Model metrics (rolling-origin backtest)

For dispatch, *which* zones are worst next week matters more than the exact count — so we optimise and report **ranking** quality, and report it averaged across folds rather than from a single lucky window.

- **Ranking (the decision metric): Spearman 0.83 across 3 folds** (0.89 in the best single window); precision@50 **87% CV vs 66% naive**; precision@20 **80% CV — 95% in the best window — vs 80% naive @20**.
- Emerging (rising) zones: ML Spearman **0.65 vs 0.30 naive** — catches zones before they peak.
- Hotspot classification (top-50 yes/no): **accuracy 98.7%, ROC-AUC 0.998** (class-imbalanced; precision/recall in `metrics_full.json`).
- **Point forecast: MASE 1.02 (MAE 2.26 vs naive 2.21) — essentially level with the naive baseline.** Daily per-zone counts are inherently noisy, so absolute point accuracy is *not* the right lens for dispatch decisions. The model's value is in *ranking* zones for patrol allocation, which is exactly what the metrics above measure — and where it clearly beats naive.

## Roadmap
- Replace road-criticality proxy with **measured traffic speed/flow** for true delay quantification.
- Live **camera/ANPR** ingestion for real-time detection.
- Multi-vehicle patrol routing (VRP) across divisions and shifts.
