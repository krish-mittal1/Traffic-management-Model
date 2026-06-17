"""Streamlit dashboard for the parking congestion intelligence project.
Run: streamlit run app.py"""

import json
import pathlib

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

MAP_STYLE = "open-street-map"   # detailed streets/landmarks → real feel when zooming in

HERE = pathlib.Path(__file__).parent
st.set_page_config(page_title="Parking Congestion Intelligence", layout="wide", page_icon="🅿️")


@st.cache_data
def load():
    g = lambda f: pd.read_parquet(HERE / f)
    j = lambda f: json.loads((HERE / f).read_text())
    return (g("violations_clean.parquet"), g("zones_h3.parquet"), g("playbook.parquet"),
            g("forecast_zones.parquet"), j("forecast_eval.json"), g("patrol_route.parquet"),
            j("causal_impact.json"), g("causal_eventstudy.parquet"), g("emerging.parquet"),
            j("cost_benefit.json"), g("coverage_curve.parquet"))


(df, zones, playbook, forecast, ev, route, causal, es, emerging, cb, curve) = load()


def map_layout(fig, zoom=10.5, height=480, center=(12.97, 77.59)):
    fig.update_layout(mapbox_style=MAP_STYLE, mapbox_zoom=zoom,
                      mapbox_center=dict(lat=center[0], lon=center[1]),
                      height=height, margin=dict(l=0, r=0, t=0, b=0))
    return fig


st.title("🅿️ Parking Congestion Intelligence — Bengaluru")
st.caption(
    "Gridlock 2.0 · Flipkart × Bengaluru Traffic Police · "
    f"{len(df):,} validated violations · {df['created_ist'].min().date()} → {df['created_ist'].max().date()} · "
    "PICS scored on the real OSM road network")

with st.sidebar:
    st.header("Filters")
    station = st.selectbox("Police station", ["All"] + sorted(df["police_station"].dropna().unique()))
    vtype = st.selectbox("Violation type", ["All"] + sorted(df["primary_violation"].dropna().unique()))
    vehicle = st.selectbox("Vehicle type", ["All"] + sorted(df["vehicle_type"].dropna().unique()))
    hr = st.slider("Hour of day", 0, 23, (0, 23))
    top_n = st.slider("Hotspots to map", 10, 500, 150, step=10)
    st.divider()
    st.caption("**PICS** = severity × persistence × road criticality × junction proximity → 0–100.")

f = df.copy()
if station != "All": f = f[f["police_station"] == station]
if vtype != "All": f = f[f["primary_violation"] == vtype]
if vehicle != "All": f = f[f["vehicle_type"] == vehicle]
f = f[f["hour"].between(hr[0], hr[1])]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Violations (filtered)", f"{len(f):,}")
c2.metric("Critical zones (PICS ≥75)", f"{(zones['pics'] >= 75).sum():,}")
c3.metric("Forecast precision@20", f"{ev['precision_at_20']:.0%}", f"+{ev['precision_at_20']-ev['baseline_precision_at_20']:.0%} vs naive")
c4.metric("Zones = 50% of violations", f"{cb['zones_to_cover_50pct']}", f"{cb['zones_to_cover_50pct_share']}% of zones")

tabs = st.tabs(["📖 Overview", "🗺️ Hotspots & PICS", "🔮 Forecast & Emerging", "🚓 Patrol Route",
                "📋 Playbook", "💰 Impact & ROI", "📊 Patterns"])

# Tab: overview
with tabs[0]:
    st.subheader("From reactive patrols to predictive, flow-aware enforcement")
    st.markdown(
        "This system turns **248,231 real police violations** into targeted enforcement. "
        "It answers the theme's three gaps directly:")
    a, b, c = st.columns(3)
    a.info("**Detect** illegal-parking hotspots\n\nH3 hexagon zones from 100%-geotagged data.")
    b.warning("**Quantify** traffic-flow impact\n\n**PICS**, grounded in the real OSM road network.")
    c.success("**Act** — proactively\n\nForecast + optimized patrol route + officer playbook.")
    st.divider()
    st.markdown("##### How it works")
    st.markdown(
        "`clean → H3 hotspots → PICS (road+junction) → POI context → forecast & emerging → "
        "playbook → patrol-route TSP → ROI → impact study`")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecast ROC-AUC", "0.998")
    m2.metric("Emerging-zone skill", f"{ev['emerging_spearman_ml']:.2f}", f"vs {ev['emerging_spearman_naive']:.2f} naive")
    m3.metric("Patrol route saved", f"{100*(cb['route_km_naive']-cb['route_km_optimized'])/cb['route_km_naive']:.0f}%")
    m4.metric("Officer-hrs saved/wk", f"~{cb['officer_hours_saved_per_week']:.0f}")
    st.caption("Built entirely on the provided dataset + free OpenStreetMap data — no paid APIs, fully reproducible.")
    with st.expander("📖 Plain-English glossary (what every term means)"):
        st.markdown(
            "- **PICS** — *Parking-Induced Congestion Score* (0–100). How badly a spot chokes traffic. "
            "Higher = block more flow. Combines how many violations, how often, how busy the road is, and how close to a junction.\n"
            "- **Hotspot zone** — a ~170 m square of the city we group violations into (so we get *areas*, not dots).\n"
            "- **Forecast precision@20** — of the 20 zones we predict will be worst next week, how many actually are. "
            "We get 19/20 right (95%).\n"
            "- **Emerging hotspot** — a zone that is *rising fast* — getting worse before it's an obvious problem. Catch it early.\n"
            "- **Persistence** — bad spots stay bad: 80% of the worst-20 zones are still worst-20 months later. So targeting them pays off.\n"
            "- **Officer-hours saved** — smarter patrol routes cover the same hotspots in far less driving.")

# Tab: hotspots & PICS
with tabs[1]:
    st.subheader("Heatmap + PICS-ranked hotspots")
    st.caption("🖱️ **Click a dot on the map** (turn off Brave Shields for localhost if the map is blank) "
               "**or** click a row in the table → details appear below.")
    top = zones.head(top_n).reset_index(drop=True)
    left, right = st.columns([3, 2])
    with left:
        pts = f[["latitude", "longitude"]].dropna()
        if len(pts) > 6000:
            pts = pts.sample(6000, random_state=0)
        fm = folium.Map(location=[12.97, 77.59], zoom_start=11, tiles="OpenStreetMap")
        HeatMap(pts.values.tolist(), radius=9, blur=12, min_opacity=0.3).add_to(fm)
        for _, r in top.iterrows():
            col = "#d7191c" if r.pics >= 75 else "#fdae61" if r.pics >= 50 else "#1a9641"
            folium.CircleMarker(
                [r.lat, r.lon], radius=4 + r.pics / 16, color=col, fill=True, fill_opacity=0.75, weight=1,
                tooltip=f"#{int(r['rank'])} · {r.top_location} · PICS {r.pics} (click)").add_to(fm)
        mstate = st_folium(fm, height=500, use_container_width=True,
                           returned_objects=["last_object_clicked"], key="hotfol")
    with right:
        st.markdown("**Top zones — click a row**")
        show = zones.head(25)[["rank", "pics", "violations", "top_location", "road_class", "context"]].rename(
            columns={"top_location": "location", "road_class": "road"})
        tbl = st.dataframe(show, hide_index=True, use_container_width=True, height=470,
                           on_select="rerun", selection_mode="single-row", key="ztable")

    # detail card (full width) — driven by a MAP-DOT click or a TABLE-ROW click
    chosen_rank = None
    clicked = (mstate or {}).get("last_object_clicked")
    if clicked and clicked.get("lat") is not None:
        d2 = (top["lat"] - clicked["lat"]) ** 2 + (top["lon"] - clicked["lng"]) ** 2
        chosen_rank = int(top.loc[d2.idxmin(), "rank"])
    if chosen_rank is None:
        try:
            rows = tbl.selection.rows
            if rows:
                chosen_rank = int(show.iloc[rows[0]]["rank"])
        except (AttributeError, KeyError, IndexError):
            pass

    if chosen_rank is not None:
        z = zones[zones["rank"] == chosen_rank].iloc[0]
        st.success(f"**#{int(z['rank'])} · {z.top_location}**  ·  PICS **{z.pics}**")
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Violations (5mo)", f"{int(z.violations):,}")
        x2.metric("Road", f"{z.road_class}")
        x3.metric("To junction", f"{int(z.junction_dist_m)} m")
        x4.metric("Peak hour", f"{int(z.peak_hour)}:00")
        st.markdown(f"**Station:** {z.top_station}  |  **Top violation:** {z.top_violation}  |  **Driver:** {z.context}")
        st.link_button("📍 Open exact location in Google Maps",
                       f"https://www.google.com/maps/search/?api=1&query={z.lat},{z.lon}")
    st.caption("Blocking a primary arterial near a junction outranks a quiet residential lane even at lower counts. "
               "‘Context’ explains the demand driver (metro spillover, market load/unload, junction).")

# Tab: forecast & emerging
with tabs[2]:
    st.subheader("Predicted hotspots — next 7 days (proactive deployment)")
    a, b, c, d = st.columns(4)
    a.metric("Precision@20", f"{ev['precision_at_20']:.0%}", f"naive {ev['baseline_precision_at_20']:.0%}")
    b.metric("Precision@50", f"{ev['precision_at_50']:.0%}", f"naive {ev['baseline_precision_at_50']:.0%}")
    c.metric("Ranking Spearman", f"{ev['rank_spearman']:.2f}")
    d.metric("Emerging skill", f"{ev['emerging_spearman_ml']:.2f}", f"vs {ev['emerging_spearman_naive']:.2f} naive")
    st.caption("The model beats pure persistence — most decisively on EMERGING hotspots (zones rising before they peak), "
               "which a 'just-look-at-history' baseline misses.")
    fz = forecast.head(120)
    fmax = forecast["pred_next7"].max()
    fig = go.Figure(go.Scattermapbox(
        lat=fz.lat, lon=fz.lon, mode="markers",
        marker=dict(size=(6 + 18 * fz.pred_next7 / fmax), color=fz.pred_next7,
                    colorscale="Plasma", showscale=True, opacity=0.82,
                    colorbar=dict(title="pred 7d", x=1.0)),
        text=[f"predicted {v:.0f} violations (next 7 days)" for v in fz.pred_next7],
        hoverinfo="text"))
    fig.update_layout(mapbox_style=MAP_STYLE, mapbox_zoom=10.5,
                      mapbox_center=dict(lat=12.97, lon=77.59),
                      height=460, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### 🔺 Fastest-rising (emerging) zones — catch these before they peak")
    em = emerging.head(12)[["top_location", "top_station", "recent_daily", "pred_daily", "momentum_pct"]].rename(
        columns={"top_location": "location", "recent_daily": "now/day", "pred_daily": "forecast/day",
                 "momentum_pct": "rise %"})
    em["now/day"] = em["now/day"].round(1)
    em["forecast/day"] = em["forecast/day"].round(1)
    st.dataframe(em, hide_index=True, use_container_width=True, height=320)

# Tab: patrol route
with tabs[3]:
    st.subheader("Optimized patrol route (TSP over top hotspots)")
    opt = route["cum_km"].iloc[-1]
    a, b, c = st.columns(3)
    a.metric("Stops", f"{len(route)}")
    b.metric("Route length", f"{opt:.1f} km", f"-{100*(cb['route_km_naive']-opt)/cb['route_km_naive']:.0f}% vs naive")
    c.metric("Est. shift", f"~{opt/18*60 + 15*len(route):.0f} min")
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_scattermapbox(lat=route.lat, lon=route.lon, mode="lines",
                              line=dict(width=3, color="#185fa5"), hoverinfo="skip")
        fig.add_scattermapbox(
            lat=route.lat, lon=route.lon, mode="markers+text",
            marker=dict(size=22, color="#185fa5"),
            text=[str(int(s)) for s in route.stop], textfont=dict(color="white", size=11),
            hovertext=[f"Stop {int(s.stop)}: {s.top_location} · PICS {s.pics}" for _, s in route.iterrows()],
            hoverinfo="text")
        st.plotly_chart(map_layout(fig, zoom=11, height=480,
                        center=(route.lat.mean(), route.lon.mean())), use_container_width=True)
    with right:
        st.dataframe(route[["stop", "pics", "top_location", "leg_km", "cum_km"]].rename(
            columns={"top_location": "location"}), hide_index=True, use_container_width=True, height=480)
    st.caption("Nearest-neighbour + 2-opt heuristic on haversine distances.")

# Tab: playbook
with tabs[4]:
    st.subheader("Deployable enforcement playbook — where, when, what, why")
    pb = playbook.rename(columns={"violations_5mo": "violations (5mo)", "pred_next7": "forecast (7d)",
                                  "deploy_when": "deploy when", "target_violation": "target"})
    st.dataframe(pb, hide_index=True, use_container_width=True, height=520)
    st.download_button("⬇️ Download playbook (CSV)", playbook.to_csv(index=False),
                       "enforcement_playbook.csv", "text/csv")

# Tab: impact & ROI
with tabs[5]:
    st.subheader("Why targeting works — coverage, ROI, and enforcement impact")
    a, b, c = st.columns(3)
    a.metric("Top 50 zones cover", f"{cb['top_50_zones_cover_pct']:.0f}%", "of all violations")
    b.metric("Officer-hrs saved/wk", f"~{cb['officer_hours_saved_per_week']:.0f}")
    c.metric("Hotspot persistence", f"{causal['hotspot_rank_persistence_spearman']:.2f}", f"{causal['top20_retention']:.0%} retained")
    st.markdown(f"**{cb['headline']}**")
    n_zones = len(zones)
    pick = st.slider("👮 If we patrol the top N hotspot zones…", 1, min(300, n_zones), 50, step=1)
    covered = float(zones.head(pick)["violations"].sum()) / zones["violations"].sum()
    s1, s2 = st.columns(2)
    s1.metric("Violations covered", f"{covered:.0%}", f"{pick} zones = {100*pick/n_zones:.1f}% of all zones")
    s2.metric("Plain English", f"{pick} spots", f"catch {covered:.0%} of the problem")
    fig = px.area(curve, x="zone_pct", y="viol_pct",
                  labels={"zone_pct": "% of zones patrolled (ranked by PICS)", "viol_pct": "% of violations covered"})
    fig.add_vline(x=100 * pick / n_zones, line_dash="dash", line_color="#185fa5")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.markdown("##### Does enforcement disperse hotspots?")
    st.markdown(f"{causal['headline']}")
    fig2 = px.line(es, x="rel_week", y="norm_v", markers=True,
                   labels={"rel_week": "weeks relative to peak enforcement", "norm_v": "violations (× zone average)"})
    fig2.add_vline(x=0, line_dash="dash", line_color="grey")
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.warning(f"⚠️ Methodology caveat: {causal['did_caveat']}")

# Tab: patterns
with tabs[6]:
    a, b = st.columns(2)
    with a:
        st.subheader("When violations happen (day × hour)")
        piv = f.groupby(["dow_name", "hour"]).size().reset_index(name="n")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if len(piv):
            fig = px.density_heatmap(piv, x="hour", y="dow_name", z="n", category_orders={"dow_name": order},
                                     color_continuous_scale="OrRd")
            fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="", xaxis_title="hour")
            st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Violation mix")
        vc = f["primary_violation"].value_counts().head(10).reset_index(); vc.columns = ["violation", "count"]
        st.plotly_chart(px.bar(vc, x="count", y="violation", orientation="h").update_layout(
            height=330, yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=0)), use_container_width=True)
    c, d = st.columns(2)
    with c:
        st.subheader("Top stations by volume")
        sc = f["police_station"].value_counts().head(10).reset_index(); sc.columns = ["station", "count"]
        st.plotly_chart(px.bar(sc, x="count", y="station", orientation="h").update_layout(
            height=330, yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=0)), use_container_width=True)
    with d:
        st.subheader("Vehicle types")
        vh = f["vehicle_type"].value_counts().head(8).reset_index(); vh.columns = ["vehicle", "count"]
        st.plotly_chart(px.bar(vh, x="count", y="vehicle", orientation="h").update_layout(
            height=330, yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=0)), use_container_width=True)
