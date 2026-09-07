"""
STRATA Export Layer — Tableau-ready flat CSVs.
All outputs saved to outputs/extracts/.

Files produced:
  1. segment_summary.csv          — 5 rows (one per STRATA segment)
  2. household_scored.csv         — 2,469 rows (full panel, one per household)
  3. segment_demographic_crosstab.csv — long-format, 801-HH subsample only
  4. campaign_results.csv         — segment × estimate_type
  5. retention_matrix.csv         — cohort × subsequent quarter
"""

from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTS = PROJECT_ROOT / "outputs" / "extracts"
REPORTS  = PROJECT_ROOT / "outputs" / "reports"
DB_PATH  = PROJECT_ROOT / "data" / "processed" / "strata.duckdb"

RNG_SEED = 42
B_BOOT   = 10_000


# ─── helpers ──────────────────────────────────────────────────────────────────

def boot_mean_ci(values: np.ndarray, B: int = B_BOOT,
                 rng: np.random.Generator = None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(values)
    boot = rng.choice(values, size=(B, n), replace=True).mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def fmt_note(text: str) -> str:
    return text.strip()


# ─── FILE 1: segment_summary.csv ─────────────────────────────────────────────
# Grain: one row per STRATA segment (5 rows including Single-Trip)
# Denormalisation: bootstrap CIs recomputed inline from clv_predictions.csv
# (same B=10,000 / RNG as Phase 4 hypothesis tests, results should match)

def export_segment_summary() -> pd.DataFrame:
    preds = pd.read_csv(EXTRACTS / "clv_predictions.csv")

    _named = (
        preds[preds["cluster_name"] != "Single-Trip"]
        .groupby("cluster_name")["hurdle_pred_clv"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    SEG_ORDER = _named + ["Single-Trip"]
    rng = np.random.default_rng(RNG_SEED)

    total_obs_monetary = preds["obs_monetary"].sum()
    rows = []

    for seg in SEG_ORDER:
        g = preds[preds["cluster_name"] == seg]
        clv_vals = g["hurdle_pred_clv"].values
        ci_lo, ci_hi = boot_mean_ci(clv_vals, B=B_BOOT, rng=rng)

        n = len(g)
        zero_spend = (g["clv_pred_window"] == 0).sum() if "clv_pred_window" in g.columns else np.nan
        zero_pct   = float(zero_spend / n * 100) if not np.isnan(zero_spend) else np.nan

        rows.append({
            "segment_name":         seg,
            "n_households":         n,
            "pct_panel":            round(n / len(preds) * 100, 2),
            "mean_pred_clv":        round(float(clv_vals.mean()), 2),
            "median_pred_clv":      round(float(np.median(clv_vals)), 2),
            "clv_ci_low":           round(ci_lo, 2),
            "clv_ci_high":          round(ci_hi, 2),
            "mean_obs_monetary":    round(float(g["obs_monetary"].mean()), 2),
            "mean_frequency":       round(float(g["obs_frequency"].mean()), 2),
            "mean_recency_days":    round(float(g["obs_recency_days"].mean()), 2),
            "zero_spend_rate_pct":  round(zero_pct, 2) if not np.isnan(zero_pct) else None,
            "revenue_share_pct":    round(g["obs_monetary"].sum() / total_obs_monetary * 100, 2),
        })

    df = pd.DataFrame(rows)
    out = EXTRACTS / "segment_summary.csv"
    df.to_csv(out, index=False)

    print(f"[1] segment_summary.csv — {len(df)} rows")
    print(df[["segment_name", "n_households", "mean_pred_clv", "clv_ci_low", "clv_ci_high",
              "zero_spend_rate_pct", "revenue_share_pct"]].to_string(index=False))

    print()
    print("    TABLEAU NOTES:")
    print("    - clv_ci_low / clv_ci_high: 95% bootstrap CI on PREDICTED Q4 CLV,")
    print("      NOT on actual observed Q4 spend. Label as 'Predicted CLV 95% CI'.")
    print("    - zero_spend_rate_pct: fraction of households with ZERO actual Q4")
    print("      spend (clv_pred_window=0), not zero predicted CLV. These are")
    print("      households that churned entirely in Q4.")
    print("    - revenue_share_pct: share of Jan–Sep 2017 obs-window spend, not")
    print("      Q4 or lifetime revenue. Label tooltip accordingly.")

    return df


# ─── FILE 2: household_scored.csv ─────────────────────────────────────────────
# Grain: one row per household (2,469 total including pred-only)
# Denormalisation: obs features from v_clv_split; hurdle_pred_clv from
# clv_predictions.csv (NULL for pred-only HHs); r/f/m scores from
# cluster_assignments.csv where available.

def export_household_scored() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # All 2,469 households with obs features (NULLs for pred-only on obs cols)
    base = con.execute("""
        SELECT household_id,
               obs_recency_days,
               obs_frequency,
               obs_monetary,
               obs_tenure_days,
               is_pred_only
        FROM v_clv_split
    """).df()

    # Demographics flag from v_households
    hh_demo = con.execute("""
        SELECT household_id, has_demographics
        FROM v_households
    """).df()

    con.close()

    # Merge predictions (hurdle_pred_clv, cluster_name, clv_pred_window)
    preds = pd.read_csv(EXTRACTS / "clv_predictions.csv")[
        ["household_id", "cluster_name", "hurdle_pred_clv", "clv_pred_window"]
    ]

    # Merge RFM scores + cluster info
    ca = pd.read_csv(EXTRACTS / "cluster_assignments.csv")[
        ["household_id", "cluster_name", "r_score", "f_score", "m_score"]
    ]
    # cluster_name from cluster_assignments is ground truth for non-pred-only HHs
    # use it in preference to predictions.cluster_name
    ca = ca.rename(columns={"cluster_name": "cluster_name_seg"})

    df = base.merge(preds, on="household_id", how="left")
    df = df.merge(ca, on="household_id", how="left")
    df = df.merge(hh_demo, on="household_id", how="left")

    # Resolve cluster_name: prefer cluster_assignments (more authoritative)
    df["cluster_name"] = df["cluster_name_seg"].combine_first(df["cluster_name"])
    df = df.drop(columns=["cluster_name_seg"])

    # has_demographics: fill False where not in v_households
    df["has_demographics"] = df["has_demographics"].fillna(False).astype(bool)

    # Reorder columns
    col_order = [
        "household_id", "cluster_name", "hurdle_pred_clv", "clv_pred_window",
        "obs_monetary", "obs_frequency", "obs_recency_days", "obs_tenure_days",
        "has_demographics", "m_score", "r_score", "f_score", "is_pred_only",
    ]
    df = df[col_order]

    # Consistent int scores where present
    for col in ["m_score", "r_score", "f_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = EXTRACTS / "household_scored.csv"
    df.to_csv(out, index=False)

    print(f"\n[2] household_scored.csv — {len(df)} rows")
    print(f"    cluster_name NULL: {df['cluster_name'].isna().sum()} HHs "
          f"(pred-only or single-trip without cluster assignment)")
    print(f"    hurdle_pred_clv NULL: {df['hurdle_pred_clv'].isna().sum()} HHs")
    print(f"    has_demographics=True: {df['has_demographics'].sum()}")
    print(f"    is_pred_only=True: {df['is_pred_only'].sum()}")

    print()
    print("    TABLEAU NOTES:")
    print("    - hurdle_pred_clv is a model-predicted value, NOT actual Q4 spend.")
    print("      Do NOT sum this column to forecast revenue — it is a per-household")
    print("      rank/score, not a literal dollar amount to aggregate.")
    print("    - clv_pred_window is actual observed Q4 spend ($). Summing this")
    print("      is valid but only covers the 2017-10-01 to 2018-01-01 period.")
    print("    - cluster_name NULL = pred-only household (no obs-window features).")
    print("      Filter these out before building segment-level charts.")
    print("    - r_score / f_score / m_score NULL for ~34 HHs not in the k-means")
    print("      fit (Single-Trip and pred-only). Use conditional formatting.")

    return df


# ─── FILE 3: segment_demographic_crosstab.csv ─────────────────────────────────
# Grain: one row per (segment_name, demographic_variable, demographic_value)
# Restricted to 801 households with demographics only.
# pct_segment_covered shows what fraction of each segment has demographics,
# so Tableau can display the caveat inline in tooltips.

def export_demographic_crosstab() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    hh = con.execute("""
        SELECT h.household_id, h.age, h.income, h.marital_status,
               h.household_size, h.kids_count
        FROM v_households h
        WHERE h.has_demographics = true
    """).df()

    con.close()

    ca = pd.read_csv(EXTRACTS / "cluster_assignments.csv")[
        ["household_id", "cluster_name"]
    ]

    # Merge with segment assignments (only HHs that have both demo + cluster)
    df_demo = hh.merge(ca, on="household_id", how="inner")
    df_demo = df_demo[df_demo["cluster_name"].notna()]

    # Segment total sizes (full panel, not just demo)
    ca_all  = pd.read_csv(EXTRACTS / "cluster_assignments.csv")[["household_id", "cluster_name"]]
    seg_totals = ca_all.groupby("cluster_name")["household_id"].count().to_dict()

    # Segment demo counts (numerator for pct_segment_covered)
    seg_demo_counts = df_demo.groupby("cluster_name")["household_id"].count().to_dict()

    SEG_ORDER = (
        pd.read_csv(EXTRACTS / "cluster_assignments.csv")
        .groupby("cluster_name")["m_score"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    DEMO_VARS = ["age", "income", "marital_status", "household_size", "kids_count"]

    rows = []
    for seg in SEG_ORDER:
        seg_df      = df_demo[df_demo["cluster_name"] == seg]
        n_seg_demo  = seg_demo_counts.get(seg, 0)
        n_seg_total = seg_totals.get(seg, 0)
        pct_covered = round(n_seg_demo / n_seg_total * 100, 1) if n_seg_total else 0

        for var in DEMO_VARS:
            series = seg_df[var].dropna()
            val_counts = series.value_counts().sort_index()

            for val, count in val_counts.items():
                rows.append({
                    "segment_name":         seg,
                    "demographic_variable": var,
                    "demographic_value":    str(val),
                    "n_households":         int(count),
                    "pct_within_segment":   round(count / len(series) * 100, 1),
                    "pct_segment_covered":  pct_covered,
                    "n_segment_with_demo":  n_seg_demo,
                    "n_segment_total":      n_seg_total,
                })

    df = pd.DataFrame(rows)
    out = EXTRACTS / "segment_demographic_crosstab.csv"
    df.to_csv(out, index=False)

    print(f"\n[3] segment_demographic_crosstab.csv — {len(df)} rows")
    print(f"    Segments covered: {df['segment_name'].nunique()}")
    print(f"    Demo vars covered: {df['demographic_variable'].unique().tolist()}")
    print()
    print("    Coverage by segment (pct_segment_covered):")
    cov = (df[["segment_name","pct_segment_covered","n_segment_with_demo","n_segment_total"]]
           .drop_duplicates()
           .sort_values("segment_name"))
    print(cov.to_string(index=False))

    print()
    print("    TABLEAU NOTES:")
    print("    - pct_within_segment: computed on the demo subsample only, NOT the")
    print("      full segment. Always show pct_segment_covered in the tooltip so")
    print("      the viewer knows what fraction of the segment is represented.")
    print("    - The 801-household demo subsample spends 3.7× more at the median")
    print("      than non-demo households. Age/income distributions skew toward")
    print("      higher-value households. Label every chart: 'Based on 32.4% of")
    print("      panel with demographic data (higher-spend subsample)'.")
    print("    - household_size and kids_count are numeric: Tableau will try to")
    print("      SUM them. Set aggregation to AVERAGE or use as Discrete dimension.")

    return df


# ─── FILE 4: campaign_results.csv ─────────────────────────────────────────────
# Grain: one row per (segment_name × estimate_type)
# IPW and AIPW values sourced from Phase 4 run (deterministic at RNG_SEED=42).
# Naive estimates computed inline from clv_predictions.csv + exposure flags.

def export_campaign_results() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    split_df = con.execute("""
        SELECT household_id,
               is_concurrent_arm,
               n_campaigns_exposed
        FROM v_clv_split WHERE NOT is_pred_only
    """).df()
    con.close()

    preds = pd.read_csv(EXTRACTS / "clv_predictions.csv")
    df = preds.merge(split_df, on="household_id", how="left")

    n_exposed = df["n_campaigns_exposed"].fillna(0)
    is_conc   = df["is_concurrent_arm"].fillna(False)
    df["group"] = "unexposed"
    df.loc[n_exposed > 0, "group"] = "concurrent"
    df.loc[(n_exposed > 0) & (~is_conc), "group"] = "single_arm"

    SEG_ORDER = (
        pd.read_csv(EXTRACTS / "cluster_assignments.csv")
        .groupby("cluster_name")["m_score"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    def naive_diff(sub, seg=None):
        pool = sub if seg is None else sub[sub["cluster_name"] == seg]
        t = pool[pool["group"] == "single_arm"]["hurdle_pred_clv"]
        c = pool[pool["group"] == "unexposed"]["hurdle_pred_clv"]
        if len(t) == 0 or len(c) == 0:
            return np.nan, np.nan, np.nan
        return float(t.mean() - c.mean()), len(t), len(c)

    # Phase 4 IPW results (deterministic at RNG_SEED=42, B=2000)
    # Full-panel: ATE=-84.26, CI=[-208.00,+2.24], n_T=683, n_C=888, ESS_T=267, ESS_C=300
    # Segment IPW (B=500, RNG_SEED=42):
    #   Champions:     ATE=-144.3, CI=[-264,+46],   p=0.184, BH_sig=False
    #   Loyal Actives: ATE=-244.3, CI=[-559,-11],   p=0.036, BH_sig=True
    #   Declining:     ATE=+21.5,  CI=[+3,+34],     p=0.032, BH_sig=True
    #   Truly Lapsed:  ATE=+65.8,  CI=[+31,+119],   p=0.002, BH_sig=True (low-n flag)
    # Full-panel AIPW (doubly-robust): ATE=-197.04, no per-segment AIPW

    phase4_ipw = {
        "All Segments": dict(ate=-84.26, ci_lo=-208.00, ci_hi=2.24,
                             n_t=683, n_c=888, ess_t=267.0, ess_c=300.0,
                             bh=None),
        "Champions":    dict(ate=-144.3, ci_lo=-264.0, ci_hi=46.0,
                             n_t=104, n_c=77,  ess_t=None, ess_c=None,
                             bh=False),
        "Loyal Actives":dict(ate=-244.3, ci_lo=-559.0, ci_hi=-11.0,
                             n_t=401, n_c=159, ess_t=None, ess_c=None,
                             bh=True),
        "Declining":    dict(ate=21.5,   ci_lo=3.0,    ci_hi=34.0,
                             n_t=159, n_c=478, ess_t=None, ess_c=None,
                             bh=True),
        "Truly Lapsed": dict(ate=65.8,   ci_lo=31.0,   ci_hi=119.0,
                             n_t=19,  n_c=147, ess_t=None, ess_c=None,
                             bh=True),
    }

    rows = []

    # Naive — full panel
    nd, nt, nc = naive_diff(df)
    rows.append(dict(segment_name="All Segments", estimate_type="naive",
                     effect_dollars=round(nd, 2), ci_low=None, ci_high=None,
                     n_treated=int(nt), n_control=int(nc),
                     ess_treated=int(nt), ess_control=int(nc),
                     survived_bh_correction=None))

    # Naive — per segment
    for seg in SEG_ORDER:
        nd, nt, nc = naive_diff(df, seg)
        rows.append(dict(segment_name=seg, estimate_type="naive",
                         effect_dollars=round(nd, 2), ci_low=None, ci_high=None,
                         n_treated=int(nt) if nt == nt else None,
                         n_control=int(nc) if nc == nc else None,
                         ess_treated=int(nt) if nt == nt else None,
                         ess_control=int(nc) if nc == nc else None,
                         survived_bh_correction=None))

    # IPW — full panel + per segment
    for seg_name, v in phase4_ipw.items():
        rows.append(dict(segment_name=seg_name, estimate_type="ipw_adjusted",
                         effect_dollars=v["ate"],
                         ci_low=v["ci_lo"], ci_high=v["ci_hi"],
                         n_treated=v["n_t"], n_control=v["n_c"],
                         ess_treated=v["ess_t"], ess_control=v["ess_c"],
                         survived_bh_correction=v["bh"]))

    # AIPW — full panel only (doubly-robust)
    nd_ap, nt_ap, nc_ap = naive_diff(df)
    rows.append(dict(segment_name="All Segments", estimate_type="aipw",
                     effect_dollars=-197.04, ci_low=None, ci_high=None,
                     n_treated=683, n_control=888,
                     ess_treated=None, ess_control=None,
                     survived_bh_correction=None))

    result = pd.DataFrame(rows)
    result = result.sort_values(["segment_name", "estimate_type"]).reset_index(drop=True)

    out = EXTRACTS / "campaign_results.csv"
    result.to_csv(out, index=False)

    print(f"\n[4] campaign_results.csv — {len(result)} rows")
    print(result[["segment_name","estimate_type","effect_dollars","ci_low","ci_high",
                  "survived_bh_correction"]].to_string(index=False))

    print()
    print("    TABLEAU NOTES:")
    print("    - effect_dollars NEGATIVE means campaign-exposed households have")
    print("      LOWER predicted CLV than comparable unexposed households after")
    print("      IPW adjustment. This is NOT 'the campaign caused harm' — the CI")
    print("      includes zero (full panel: [-208, +2]). Display with error bars.")
    print("    - ci_low / ci_high: NULL for naive (no uncertainty quantified) and")
    print("      AIPW (no bootstrap). Always show CI when available; grey out")
    print("      naive bars with a 'unadjusted' label.")
    print("    - survived_bh_correction: NULL for 'All Segments' (BH correction")
    print("      applies only across per-segment tests, not for the pooled test).")
    print("      Use a shape encoding (circle=tested, star=survived) in Tableau.")
    print("    - ess_treated / ess_control: effective sample size after IPW weighting.")
    print("      NULL for segment-level (not tracked in Phase 4 run). Full-panel")
    print("      ESS is ~61-66% smaller than raw n — show this in a subtitle.")

    return result


# ─── FILE 5: retention_matrix.csv ─────────────────────────────────────────────
# Grain: one row per (cohort_quarter × subsequent_quarter),
#        subsequent_quarter >= cohort_quarter.
# Q1 cohort: 2280 HHs (first purchase Jan–Mar 2017)
# Q2 cohort: 113 HHs; Q3: 53 HHs; Q4: 23 HHs
# All 2017 quarters (Q4 = Oct–Dec 2017).

def export_retention_matrix() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    df = con.execute("""
        WITH
        -- Assign each transaction to a calendar quarter
        txn_q AS (
            SELECT household_id,
                CASE
                    WHEN transaction_timestamp < TIMESTAMP '2017-04-01' THEN '2017-Q1'
                    WHEN transaction_timestamp < TIMESTAMP '2017-07-01' THEN '2017-Q2'
                    WHEN transaction_timestamp < TIMESTAMP '2017-10-01' THEN '2017-Q3'
                    WHEN transaction_timestamp < TIMESTAMP '2018-01-01' THEN '2017-Q4'
                END AS txn_quarter
            FROM v_transactions_clean
            WHERE transaction_timestamp < TIMESTAMP '2018-01-01'
        ),
        -- Cohort = quarter of first transaction
        cohorts AS (
            SELECT household_id,
                MIN(txn_quarter) AS cohort_quarter
            FROM txn_q
            WHERE txn_quarter IS NOT NULL
            GROUP BY household_id
        ),
        -- Active per quarter (distinct household × quarter)
        active_q AS (
            SELECT DISTINCT household_id, txn_quarter AS active_quarter
            FROM txn_q WHERE txn_quarter IS NOT NULL
        ),
        -- All quarter pairs: cohort_quarter ≤ subsequent_quarter
        quarter_vals AS (
            SELECT DISTINCT txn_quarter AS q FROM txn_q WHERE txn_quarter IS NOT NULL
        ),
        pairs AS (
            SELECT c.q AS cohort_quarter, s.q AS subsequent_quarter
            FROM quarter_vals c CROSS JOIN quarter_vals s
            WHERE s.q >= c.q
        )
        SELECT
            p.cohort_quarter,
            p.subsequent_quarter,
            COUNT(DISTINCT c.household_id)   AS n_cohort,
            COUNT(DISTINCT aq.household_id)  AS n_retained,
            ROUND(
                COUNT(DISTINCT aq.household_id) * 100.0
                / NULLIF(COUNT(DISTINCT c.household_id), 0),
            1) AS retention_rate_pct,
            -- Quarters elapsed (0 = same quarter, 1 = next quarter, ...)
            CASE p.subsequent_quarter
                WHEN '2017-Q1' THEN 0 WHEN '2017-Q2' THEN 1
                WHEN '2017-Q3' THEN 2 WHEN '2017-Q4' THEN 3
            END -
            CASE p.cohort_quarter
                WHEN '2017-Q1' THEN 0 WHEN '2017-Q2' THEN 1
                WHEN '2017-Q3' THEN 2 WHEN '2017-Q4' THEN 3
            END AS quarters_since_cohort
        FROM pairs p
        JOIN cohorts c   ON c.cohort_quarter = p.cohort_quarter
        LEFT JOIN active_q aq
            ON aq.household_id = c.household_id
            AND aq.active_quarter = p.subsequent_quarter
        GROUP BY p.cohort_quarter, p.subsequent_quarter
        ORDER BY p.cohort_quarter, p.subsequent_quarter
    """).df()

    con.close()

    out = EXTRACTS / "retention_matrix.csv"
    df.to_csv(out, index=False)

    print(f"\n[5] retention_matrix.csv — {len(df)} rows")
    print(df.to_string(index=False))

    print()
    print("    TABLEAU NOTES:")
    print("    - cohort_quarter = '2017-Q1' row at subsequent_quarter = '2017-Q1'")
    print("      always shows 100% retention (definition: every cohort member is")
    print("      active in their own cohort quarter). Use this as a visual anchor,")
    print("      not as data. Consider suppressing the diagonal or greying it out.")
    print("    - Q4 cohort (n=23) only has one row (quarters_since_cohort=0).")
    print("      The heatmap cell for Q4→Q4 is definitionally 100%, but Q4 cohort")
    print("      has no subsequent quarters in the dataset. Mark these cells as")
    print("      'No data' in Tableau rather than continuing the colour scale.")
    print("    - n_cohort values differ because cohorts are front-loaded: Q1=2280,")
    print("      Q2=113, Q3=53, Q4=23. The heatmap should encode retention RATE,")
    print("      not raw counts, to avoid the Q1 cohort visually dominating.")
    print("    - quarters_since_cohort: use this as the X axis in Tableau for a")
    print("      standard cohort curve aligned on period 0.")

    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("STRATA Export Layer")
    print("=" * 70)

    seg_df   = export_segment_summary()
    hh_df    = export_household_scored()
    demo_df  = export_demographic_crosstab()
    camp_df  = export_campaign_results()
    ret_df   = export_retention_matrix()

    # Summary manifest
    manifest = [
        ("segment_summary.csv",             len(seg_df),  "1 row / segment"),
        ("household_scored.csv",             len(hh_df),   "1 row / household"),
        ("segment_demographic_crosstab.csv", len(demo_df), "long-format demo"),
        ("campaign_results.csv",             len(camp_df), "segment × estimate_type"),
        ("retention_matrix.csv",             len(ret_df),  "cohort × quarter"),
    ]

    print("\n" + "=" * 70)
    print("EXPORT MANIFEST")
    print("=" * 70)
    print(f"  {'File':<42} {'Rows':>6}  {'Grain'}")
    print("  " + "-" * 66)
    for fname, rows, grain in manifest:
        print(f"  {fname:<42} {rows:>6}  {grain}")
    print()
    print(f"  All files written to outputs/extracts/")


if __name__ == "__main__":
    main()
