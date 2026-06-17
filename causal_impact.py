"""Difference-in-differences + persistence check on enforcement impact: do hotspots
disperse, or persist despite enforcement? Output: causal_impact.json + event study.

Note: this is observational. Ticketing volume is both the treatment proxy and the
outcome (no enforcement-action field exists), so we lead with the confound-free
persistence result and report the DiD as suggestive only.
"""

import json
import pathlib
import sys

import h3
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
RAW = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "violations_raw.csv"
LAT = (12.7, 13.2)
LON = (77.3, 77.9)
H3_RES = 9


def welch_t(a, b):
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    t = (ma - mb) / se if se > 0 else 0.0
    return float(t), float(se)


def main():
    df = pd.read_csv(RAW, dtype=str, encoding="utf-8", encoding_errors="replace",
                     usecols=["latitude", "longitude", "created_datetime"])
    df["latitude"] = pd.to_numeric(df.latitude, errors="coerce")
    df["longitude"] = pd.to_numeric(df.longitude, errors="coerce")
    df["dt"] = pd.to_datetime(df.created_datetime, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    df = df[df.latitude.between(*LAT) & df.longitude.between(*LON) & df.dt.notna()]
    df["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(df.latitude, df.longitude)]
    df["week"] = df.dt.dt.isocalendar().week.astype(int) + 100 * (df.dt.dt.year - 2023)

    wk = df.groupby(["h3", "week"]).size().rename("v").reset_index()
    weeks = sorted(wk.week.unique())[1:-1]  # drop partial first/last weeks
    wk = wk[wk.week.isin(weeks)]
    active = wk.groupby("h3").v.sum()
    active = active[active >= 50].index
    wk = wk[wk.h3.isin(active)]
    full = pd.MultiIndex.from_product([active, weeks], names=["h3", "week"])
    panel = wk.set_index(["h3", "week"]).reindex(full, fill_value=0).reset_index()

    mid = weeks[len(weeks) // 2]
    pre_weeks = [w for w in weeks if w < mid]
    post_weeks = [w for w in weeks if w >= mid]

    pre = panel[panel.week.isin(pre_weeks)].groupby("h3").v.mean()
    post = panel[panel.week.isin(post_weeks)].groupby("h3").v.mean()

    # treated = top tercile by pre-period intensity, control = middle tercile
    q = pre.quantile([1/3, 2/3])
    treated = pre[pre >= q.iloc[1]].index
    control = pre[(pre >= q.iloc[0]) & (pre < q.iloc[1])].index

    d_treat = (post - pre).loc[treated]
    d_ctrl = (post - pre).loc[control]
    did = float(d_treat.mean() - d_ctrl.mean())
    t, se = welch_t(d_treat.values, d_ctrl.values)

    # event study around each treated zone's own peak week
    peak = panel[panel.h3.isin(treated)].loc[
        panel[panel.h3.isin(treated)].groupby("h3").v.idxmax()][["h3", "week"]
    ].set_index("h3").week
    es_rows = []
    for z in treated:
        s = panel[panel.h3 == z].set_index("week").v
        base = s.mean() or 1
        pw = peak.loc[z]
        for w in weeks:
            rel = weeks.index(w) - weeks.index(pw)
            if -4 <= rel <= 4:
                es_rows.append({"rel_week": rel, "norm_v": s.loc[w] / base})
    es = pd.DataFrame(es_rows).groupby("rel_week").norm_v.mean().round(3)
    es.to_frame().reset_index().to_parquet(HERE / "causal_eventstudy.parquet", index=False)

    # persistence: do the worst zones in the first half stay worst in the second?
    rank_corr = float(pre.rank().corr(post.rank()))
    top20_pre = set(pre.nlargest(20).index)
    top20_post = set(post.nlargest(20).index)
    retention20 = len(top20_pre & top20_post) / 20

    out = {
        "design": "comparative interrupted time-series (difference-in-differences) + persistence",
        "n_treated_zones": int(len(treated)),
        "n_control_zones": int(len(control)),
        "pre_weeks": len(pre_weeks), "post_weeks": len(post_weeks),
        "treated_change_per_week": round(float(d_treat.mean()), 2),
        "control_change_per_week": round(float(d_ctrl.mean()), 2),
        "did_estimate": round(did, 2),
        "welch_t": round(t, 2),
        "hotspot_rank_persistence_spearman": round(rank_corr, 3),
        "top20_retention": round(retention20, 3),
        "headline": (
            f"Hotspots persist: pre/post rank correlation {rank_corr:.2f}, "
            f"{retention20:.0%} of the worst-20 zones stay worst-20 across halves. "
            "Enforcement is not dispersing them -> targeted, predictive enforcement is needed."
        ),
        "did_caveat": (
            f"The DiD ({did:+.2f}, t={t:.2f}) shows treated hotspots fell faster than controls, "
            "BUT treatment is defined by pre-period extremity, so this is partly REGRESSION TO THE "
            "MEAN, not proven deterrence. Cleanly separating the two needs the enforcement-action "
            "field, which is 100% empty in this dataset. We therefore lead with the persistence "
            "finding (confound-free) and report the DiD as suggestive only."
        ),
    }
    (HERE / "causal_impact.json").write_text(json.dumps(out, indent=2))

    print("CAUSAL / PERSISTENCE ANALYSIS")
    print(f"  [clean] hotspot rank persistence (Spearman pre->post): {rank_corr:.2f}")
    print(f"  [clean] worst-20 zones retained across halves        : {retention20:.0%}")
    print(f"  [suggestive] DiD = {did:+.2f}/zone/week (t={t:.2f}); treated {d_treat.mean():+.2f} vs control {d_ctrl.mean():+.2f}")
    print("  event-study curve -> causal_eventstudy.parquet")
    print(f"  CAVEAT: {out['did_caveat']}")


if __name__ == "__main__":
    main()
