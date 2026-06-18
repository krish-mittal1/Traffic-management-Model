"""Streamlit dashboard for the parking congestion intelligence project.
Run: streamlit run app.py"""

import json
import pathlib

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# MapmyIndia (Mappls) Web SDK key — partner mapping tech for Gridlock 2.0.
# Static map keys are client-side by design; lock it to your domains in the Mappls console.
try:
    MAPPLS_KEY = st.secrets["MAPPLS_KEY"]
except Exception:
    MAPPLS_KEY = "7748ecab0e753c215194a09e01debb77"

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


def mappls_map(markers, route=None, center=(12.97, 77.59), zoom=11, height=500, dom="map"):
    """Interactive MapmyIndia (Mappls) map: hover a pin for a quick tooltip, click it for full details."""
    for m in markers:
        m["tip"] = str(m.get("tip", "")).replace('"', "'")
    html = """
<style>
  #__DOM___tip{position:fixed;z-index:99999;pointer-events:none;display:none;
    background:rgba(17,17,17,.93);color:#fff;font:12px/1.4 system-ui,sans-serif;
    padding:6px 10px;border-radius:6px;max-width:260px;box-shadow:0 3px 10px rgba(0,0,0,.45)}
  .pin{cursor:pointer;transition:transform .08s}
  .pin:hover{transform:scale(1.18)}
</style>
<div id="__DOM__" style="width:100%;height:__Hpx__;border-radius:8px;overflow:hidden"></div>
<div id="__DOM___tip"></div>
<script>
var TIP=null;
function tipEl(){ if(!TIP){TIP=document.getElementById('__DOM___tip');} return TIP; }
function showTip(el,e){ var t=tipEl(); t.innerHTML=el.getAttribute('data-tip'); t.style.display='block'; moveTip(e); }
function moveTip(e){ var t=tipEl(); t.style.left=(e.clientX+14)+'px'; t.style.top=(e.clientY+14)+'px'; }
function hideTip(){ tipEl().style.display='none'; }
function draw(map){
  var pts = __PTS__;
  pts.forEach(function(p){
    var el = '<div class="pin" data-tip="'+p.tip+'" onmouseenter="showTip(this,event)" '
           + 'onmousemove="moveTip(event)" onmouseleave="hideTip()" '
           + 'style="display:flex;align-items:center;justify-content:center;width:'+p.size+'px;'
           + 'height:'+p.size+'px;background:'+p.color+';border:1.5px solid #fff;border-radius:50%;'
           + 'color:#fff;font:bold 11px sans-serif;box-shadow:0 0 4px rgba(0,0,0,.45)">'+(p.label||'')+'</div>';
    new mappls.Marker({map:map, position:{lat:p.lat,lng:p.lng}, html:el,
                       popupHtml:p.popup, popupOptions:{maxWidth:300}});
  });
  var rt = __LINE__;
  if (rt.length > 1) {
    new mappls.Polyline({map:map, path:rt, strokeColor:'#185fa5', strokeWidth:4, strokeOpacity:0.85});
  }
}
function start(){
  var map = new mappls.Map('__DOM__', {center:{lat:__LAT__,lng:__LNG__}, zoom:__ZOOM__});
  if (map.on) { map.on('load', function(){draw(map);}); }
  else if (map.addListener) { map.addListener('load', function(){draw(map);}); }
  else { setTimeout(function(){draw(map);}, 1200); }
}
</script>
<script src="https://apis.mappls.com/advancedmaps/api/__KEY__/map_sdk?v=3.0&layer=vector" onload="start()"></script>
"""
    html = (html.replace("__DOM__", dom).replace("__Hpx__", f"{height}px")
                .replace("__PTS__", json.dumps(markers)).replace("__LINE__", json.dumps(route or []))
                .replace("__LAT__", str(center[0])).replace("__LNG__", str(center[1]))
                .replace("__ZOOM__", str(zoom)).replace("__KEY__", MAPPLS_KEY))
    components.html(html, height=height + 12)


def popup_card(title, rows, lat=None, lon=None):
    """Build a styled HTML popup: bold title, key/value rows, optional Google Maps link."""
    body = "".join(f"<tr><td style='color:#888;padding-right:8px'>{k}</td>"
                   f"<td style='text-align:right'><b>{v}</b></td></tr>" for k, v in rows)
    link = ("" if lat is None else
            f"<a href='https://www.google.com/maps/search/?api=1&query={lat},{lon}' "
            f"target='_blank' style='display:block;margin-top:6px;color:#185fa5'>📍 Open in Google Maps</a>")
    return (f"<div style='font:13px system-ui,sans-serif;min-width:190px'><b>{title}</b>"
            f"<hr style='margin:5px 0;border:none;border-top:1px solid #ddd'>"
            f"<table style='width:100%'>{body}</table>{link}</div>")


def congestion_tier(pics):
    """Map the PICS score to a discrete congestion class (matches map pin colours)."""
    return "🔴 Critical" if pics >= 75 else "🟠 High" if pics >= 50 else "🟢 Moderate"


def legend(items):
    """Small inline colour legend, e.g. legend([('Critical (≥75)', '#d7191c'), ...])."""
    chips = "".join(
        f"<span style='margin-right:16px;white-space:nowrap'>"
        f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
        f"background:{c};margin-right:5px;vertical-align:middle'></span>{lbl}</span>"
        for lbl, c in items)
    st.markdown(f"<div style='font:12px system-ui,sans-serif;color:#888;padding:2px 0 8px'>{chips}</div>",
                unsafe_allow_html=True)


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
    m1.metric("Backtested precision@20", f"{ev.get('cv_precision_at_20_mean', ev['precision_at_20']):.0%}",
              f"{ev.get('backtest_folds', 1)} rolling windows")
    m2.metric("Emerging-zone skill", f"{ev['emerging_spearman_ml']:.2f}", f"vs {ev['emerging_spearman_naive']:.2f} naive")
    m3.metric("Patrol route saved", f"{100*(cb['route_km_naive']-cb['route_km_optimized'])/cb['route_km_naive']:.0f}%")
    m4.metric("Officer-hrs saved/wk", f"~{cb['officer_hours_saved_per_week']:.0f}")
    st.caption("Built on the provided BTP dataset + OpenStreetMap road network, visualised on MapmyIndia (Mappls) "
               "interactive maps — partner technology, free tier, fully reproducible.")
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
    st.subheader("PICS-ranked parking hotspots")
    st.caption("🗺️ Interactive **MapmyIndia** map — pin colour = severity (red ≥75, amber ≥50, green below). "
               "Click a pin for a quick popup, or click a table row for the full detail card.")
    top = zones.head(top_n).reset_index(drop=True)
    left, right = st.columns([3, 2])
    with left:
        pc = lambda p: "#d7191c" if p >= 75 else "#fdae61" if p >= 50 else "#1a9641"
        mk = [{
            "lat": float(r.lat), "lng": float(r.lon), "size": int(9 + r.pics / 5),
            "color": pc(r.pics), "label": "",
            "tip": f"#{int(r['rank'])} · {r.top_location} — PICS {r.pics}",
            "popup": popup_card(f"#{int(r['rank'])} · {r.top_location}", [
                ("PICS", r.pics), ("Violations (5mo)", f"{int(r.violations):,}"),
                ("Road", r.road_class), ("To junction", f"{int(r.junction_dist_m)} m"),
                ("Peak hour", f"{int(r.peak_hour)}:00"), ("Station", r.top_station),
                ("Top violation", r.top_violation), ("Driver", r.context),
            ], lat=float(r.lat), lon=float(r.lon)),
        } for _, r in top.iterrows()]
        mappls_map(mk, height=500, dom="hotmap")
        legend([("Critical (PICS ≥75)", "#d7191c"), ("High (50–74)", "#fdae61"), ("Moderate (<50)", "#1a9641")])
    with right:
        st.markdown("**Top zones — click a row**")
        hz = zones.head(25).copy()
        hz["tier"] = hz["pics"].apply(congestion_tier)
        show = hz[["rank", "pics", "tier", "violations", "top_location", "road_class", "context"]].rename(
            columns={"top_location": "location", "road_class": "road"})
        tbl = st.dataframe(show, hide_index=True, use_container_width=True, height=470,
                           on_select="rerun", selection_mode="single-row", key="ztable")

    chosen_rank = None
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
    if "cv_precision_at_20_mean" in ev:
        st.info(f"✅ **Validated, not lucky:** across {ev['backtest_folds']} rolling-origin backtest windows the model "
                f"holds **{ev['cv_precision_at_20_mean']:.0%} precision@20** and **{ev['cv_rank_spearman_mean']:.2f}** "
                f"rank-Spearman — the 95% above is a single window, this is the average over several.")
    fz = forecast.head(120).merge(
        zones[["h3", "top_location", "top_station", "road_class", "context"]], on="h3", how="left")
    fz["top_location"] = fz["top_location"].fillna("Zone")
    fmax = forecast["pred_next7"].max()
    fcol = lambda v: "#7a0177" if v / fmax >= 0.66 else "#d7191c" if v / fmax >= 0.33 else "#fdae61"
    mk = [{
        "lat": float(r.lat), "lng": float(r.lon), "size": int(8 + 20 * r.pred_next7 / fmax),
        "color": fcol(r.pred_next7), "label": "",
        "tip": f"{r.top_location} — predicted {r.pred_next7:.0f} (next 7d)",
        "popup": popup_card(r.top_location, [
            ("Predicted (next 7d)", f"{r.pred_next7:.0f}"), ("Station", r.top_station),
            ("Road", r.road_class), ("Driver", r.context),
        ], lat=float(r.lat), lon=float(r.lon)),
    } for _, r in fz.iterrows()]
    mappls_map(mk, height=460, zoom=10.5, dom="fcmap")
    legend([("Highest predicted", "#7a0177"), ("High", "#d7191c"), ("Lower", "#fdae61")])
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
        mk = [{
            "lat": float(r.lat), "lng": float(r.lon), "size": 26, "color": "#185fa5",
            "label": str(int(r.stop)),
            "tip": f"Stop {int(r.stop)}: {r.top_location}",
            "popup": popup_card(f"Stop {int(r.stop)}: {r.top_location}", [
                ("PICS", r.pics), ("Station", r.top_station), ("Road", r.road_class),
                ("Peak hour", f"{int(r.peak_hour)}:00"), ("Leg", f"{r.leg_km:.1f} km"),
                ("Cumulative", f"{r.cum_km:.1f} km"),
            ], lat=float(r.lat), lon=float(r.lon)),
        } for _, r in route.iterrows()]
        line = [{"lat": float(r.lat), "lng": float(r.lon)} for _, r in route.iterrows()]
        mappls_map(mk, route=line, height=480, zoom=11,
                   center=(route.lat.mean(), route.lon.mean()), dom="ptmap")
        legend([("Patrol stop (numbered in visit order)", "#185fa5")])
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
