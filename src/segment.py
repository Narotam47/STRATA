"""
STRATA Phase 2 — K-Means Segmentation
Reads v_household_features, preprocesses, sweeps k=2..10, fits final model.
Outputs: outputs/models/kmeans_k{k}.pkl, outputs/extracts/cluster_assignments.csv
"""
from __future__ import annotations
import pickle, warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "processed" / "strata.duckdb"
MODELS_DIR   = PROJECT_ROOT / "outputs" / "models"
EXTRACTS_DIR = PROJECT_ROOT / "outputs" / "extracts"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# CHOSEN_K is set explicitly because k=4 is a deliberate business decision,
# not the metric optimum (which is k=2, trivially "active vs inactive").
# The automated sweep recommended k=3; k=4 is chosen because it recovers
# the Declining/Truly Lapsed split (bootstrap ARI=0.927, stable at k=3
# and k=5), which is the central analytical finding of this project.
# See outputs/reports/phase2_segmentation.md for the full justification.
CHOSEN_K = 4

# =============================================================================
# FEATURE SELECTION
# =============================================================================
# RFM argument — raw log-transformed values vs. NTILE scores:
#   The 1–5 NTILE scores are designed for human labelling, not Euclidean
#   distance. Using them directly means:
#     (1) within-quintile variation is discarded (f_score=5 spans 95–778 trips)
#     (2) tie-boundary artefacts affect ~94 HHs (3.8% of panel)
#     (3) Euclidean distance between score integers is not proportional to
#         the real behavioural difference (a step from f_score 4→5 may be
#         95→96 trips or 95→778 trips with equal k-means weight)
#   Log-transformed raw values preserve within-quintile variation after
#   heavy-tail correction and produce continuous features better suited to
#   Euclidean distance. NTILE scores are retained as labels for segment naming.
EXCLUDED = {
    "r_score":               "NTILE artefact; using log(recency_days) instead",
    "f_score":               "NTILE artefact; using log1p(frequency) instead",
    "m_score":               "NTILE artefact; using log(monetary) instead",
    "f_decile":              "redundant with log(frequency)",
    "rfm_code":              "string label",
    "coupon_disc_rate":      "marketing treatment received, not behaviour",
    "redemption_propensity": "marketing treatment; 1785/2469 NULLs",
    "top_category":          "categorical label — 303 levels, not a numeric input",
    "trend_score":           "NTILE distorted by 290 NULLs (11.7%)",
    "gap_decile":            "redundant with gap_score (same concept, finer resolution)",
    "mom_spend_ratio_dec_nov": "both months are H2; too noisy for low-frequency HHs",
    "has_demographics":      "coverage flag, not a behavioural feature",
    "is_concurrent_arm":     "campaign structure, not customer behaviour",
    "n_campaigns_exposed":   "campaign structure, not customer behaviour",
}

# Features that enter the model (pre-correlation filter)
CANDIDATE_COLS = [
    "frequency",               # → log1p
    "monetary",                # → log
    "recency_days",            # → log
    "tenure_days",             # left-skewed but bounded [0,365]; no transform
    "weekend_trip_share",      # [0,1], skew=0.88; no transform
    "top_store_trip_share",    # [0,1], slight left skew; no transform
    "retail_disc_rate",        # [0,1], skew=1.07; no transform
    "weight_sold_spend_share", # zero-inflated, skew=3.24 → log1p
    "mean_gap_days",           # heavy tail, skew=5.52 → log
    "std_gap_days",            # heavy tail, skew=3.27 → log1p (post-impute)
    "gap_score",               # ordinal 1–5; likely redundant with log(mean_gap)
    "basket_trend_direction",  # ordinal 1–3
    "private_label_spend_share", # [0,1], skew=0.87; no transform
    "top_category_spend_share",  # heavy tail, skew=3.68 → log1p
]

def div(char="─", n=72): print(char * n)


def load_data() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM v_household_features").df()
    con.close()
    print(f"Loaded {len(df):,} households, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (model_df, single_trip_df, full_df_with_flags).
    model_df has all imputed+transformed features for the 2,435 HHs that
    enter k-means. single_trip_df holds the 34 excluded HHs.
    """
    div()
    print("STEP 1 — NULL HANDLING")
    div()

    # 34 single-trip HHs: NULL mean_gap_days / gap_score.
    # These are structurally different customers (no observable cadence) and
    # cannot have a meaningful mean_gap imputed. Exclude from k-means and
    # report as a separate "Single-Trip" micro-segment.
    single_trip_mask = df["mean_gap_days"].isna()
    print(f"  Single-trip HHs (NULL mean_gap_days): {single_trip_mask.sum()}"
          " → excluded from k-means, reported as separate segment")
    single_trip_df = df[single_trip_mask].copy()
    df = df[~single_trip_mask].copy()

    # 33 two-trip HHs: NULL std_gap_days (have mean_gap but only one interval
    # → STDDEV of a single observation is undefined). Impute 0.0: one observed
    # interval means no measurable cadence variability.
    n_std_null = df["std_gap_days"].isna().sum()
    df["std_gap_days"] = df["std_gap_days"].fillna(0.0)
    print(f"  Two-trip HHs (NULL std_gap_days): {n_std_null} → imputed 0.0")

    # 2 HHs with NULL top_category_spend_share: all spend in COUPON/MISC
    # ITEMS (excluded from category ranking). Their concentration in any real
    # category is genuinely 0. Impute 0.0.
    n_tcs_null = df["top_category_spend_share"].isna().sum()
    df["top_category_spend_share"] = df["top_category_spend_share"].fillna(0.0)
    print(f"  COUPON-only HHs (NULL top_category_spend_share): {n_tcs_null} → imputed 0.0")

    # 290 HHs with NULL basket_trend_direction: fewer than 3 baskets in one
    # half-year. Mode = 2 (stable). Imputing stable is conservative and avoids
    # fabricating a trend signal where none is measurable.
    n_btd_null = df["basket_trend_direction"].isna().sum()
    df["basket_trend_direction"] = df["basket_trend_direction"].fillna(2.0)
    print(f"  Low-basket HHs (NULL basket_trend_direction): {n_btd_null} → imputed 2 (stable)")

    return df, single_trip_df


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    div()
    print("STEP 2 — TRANSFORMS (log / log1p for heavy-tailed features)")
    div()

    LOG1P_COLS = ["frequency", "weight_sold_spend_share", "std_gap_days"]
    # top_category_spend_share: [0,1] bounded; log1p still skewed (2.73)
    # Use sqrt (gentler) which handles the zero-heavy distribution better
    SQRT_COLS  = ["top_category_spend_share"]
    LOG_COLS   = ["monetary", "recency_days", "mean_gap_days"]
    PASSTHRU   = ["tenure_days", "weekend_trip_share", "top_store_trip_share",
                  "retail_disc_rate", "gap_score", "basket_trend_direction",
                  "private_label_spend_share"]

    # Before/after comparison for 4 representative features
    showcase = [
        ("frequency",            "log1p", LOG1P_COLS),
        ("monetary",             "log",   LOG_COLS),
        ("mean_gap_days",        "log",   LOG_COLS),
        ("top_category_spend_share", "sqrt",  SQRT_COLS),
    ]
    print(f"\n  {'Feature':<30} {'Transform':<8}  {'skew_before':>11}  {'skew_after':>10}")
    print(f"  {'─'*30} {'─'*8}  {'─'*11}  {'─'*10}")

    out = df[["household_id"] + CANDIDATE_COLS].copy()

    for col in LOG1P_COLS:
        before = out[col].values
        out[col] = np.log1p(before)
        after  = out[col].values
        row = next((r for r in showcase if r[0] == col), None)
        if row:
            print(f"  {col:<30} {'log1p':<8}  {stats.skew(before):>11.2f}  {stats.skew(after):>10.2f}")

    for col in LOG_COLS:
        before = out[col].values
        out[col] = np.log(before)
        after  = out[col].values
        row = next((r for r in showcase if r[0] == col), None)
        if row:
            print(f"  {col:<30} {'log':<8}  {stats.skew(before):>11.2f}  {stats.skew(after):>10.2f}")

    for col in SQRT_COLS:
        before = out[col].values
        out[col] = np.sqrt(before)
        after  = out[col].values
        print(f"  {col:<30} {'sqrt':<8}  {stats.skew(before):>11.2f}  {stats.skew(after):>10.2f}")

    # Quantile snapshots for three features
    print()
    for col, tr, _ in showcase[:3]:
        raw  = df[col].values
        xfrm = out[col].values
        print(f"  {col} ({tr}):")
        print(f"    raw  p5={np.percentile(raw,5):.1f}  p25={np.percentile(raw,25):.1f}  "
              f"p50={np.percentile(raw,50):.1f}  p75={np.percentile(raw,75):.1f}  "
              f"p95={np.percentile(raw,95):.1f}  max={raw.max():.1f}")
        print(f"    xfrm p5={np.percentile(xfrm,5):.2f}  p25={np.percentile(xfrm,25):.2f}  "
              f"p50={np.percentile(xfrm,50):.2f}  p75={np.percentile(xfrm,75):.2f}  "
              f"p95={np.percentile(xfrm,95):.2f}  max={xfrm.max():.2f}")

    return out


def check_collinearity(out: pd.DataFrame) -> list[str]:
    """
    Compute pairwise correlations among CANDIDATE_COLS.
    Flag pairs with |r| > 0.70; recommend which to drop.
    Returns final feature list.
    """
    div()
    print("STEP 3 — MULTICOLLINEARITY (Pearson |r| > 0.70)")
    div()

    feat_cols = CANDIDATE_COLS
    # pd.DataFrame.corr() skips NaN pairs; robust on numpy 2.x
    corr_df = out[feat_cols].corr(method='pearson')
    corr = corr_df.values

    high_pairs = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            r = corr[i, j]
            if abs(r) > 0.70:
                high_pairs.append((abs(r), feat_cols[i], feat_cols[j]))
    high_pairs.sort(reverse=True)

    print(f"\n  {'Feature A':<30} {'Feature B':<30} {'|r|':>5}")
    print(f"  {'─'*30} {'─'*30} {'─'*5}")
    for r, a, b in high_pairs:
        print(f"  {a:<30} {b:<30} {r:>5.3f}")

    # Decision: drop one from each high-correlation pair
    # Principle: prefer the feature that carries information BEYOND what the
    # correlated partner already encodes.
    DROP = []
    # gap_score is NTILE(5) of mean_gap_days — if both survive the threshold,
    # gap_score adds no information on top of the log-transformed mean_gap.
    # Drop gap_score; keep log(mean_gap_days) (continuous, more discriminating).
    if any(b in ("gap_score", "mean_gap_days") and a in ("gap_score", "mean_gap_days")
           for _, a, b in high_pairs):
        DROP.append("gap_score")
        print("\n  → Dropping gap_score: NTILE(5) of mean_gap_days — entirely redundant"
              " with log(mean_gap_days) which carries within-quintile variation.")

    # If log(frequency) and log(mean_gap_days) are highly correlated (because
    # frequency ≈ tenure / gap), drop mean_gap and retain frequency + tenure.
    # The conceptual difference: frequency is the outcome (how many trips); gap is
    # the cadence (how regularly spaced). Both are useful if orthogonal enough.
    freq_gap = next((r for r, a, b in high_pairs
                     if {a, b} == {"frequency", "mean_gap_days"}), None)
    if freq_gap and freq_gap > 0.80:
        DROP.append("mean_gap_days")
        print(f"  → Dropping mean_gap_days (|r|={freq_gap:.3f} with frequency): log(frequency)"
              " + tenure_days already encodes cadence; retaining both double-weights it.")
    elif freq_gap:
        print(f"\n  Note: frequency vs mean_gap_days |r|={freq_gap:.3f} — below 0.80 threshold."
              " Retaining both: frequency=outcome, mean_gap=cadence, orthogonal enough.")

    final = [c for c in CANDIDATE_COLS if c not in DROP]
    print(f"\n  Final feature count: {len(final)} (dropped: {DROP if DROP else 'none'})")
    return final


def sweep_k(X_scaled: np.ndarray, ks: range) -> pd.DataFrame:
    """Run k-means for each k; collect silhouette, inertia, CH."""
    div()
    print("STEP 4 — K-MEANS SWEEP k=2..10")
    div()
    print(f"\n  {'k':>3}  {'silhouette':>12}  {'inertia':>14}  {'calinski_harabasz':>18}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*14}  {'─'*18}")

    results = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=SEED)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        ch  = calinski_harabasz_score(X_scaled, labels)
        results.append({"k": k, "silhouette": sil, "inertia": km.inertia_, "ch": ch})
        print(f"  {k:>3}  {sil:>12.4f}  {km.inertia_:>14,.1f}  {ch:>18.1f}")

    return pd.DataFrame(results)


def choose_k(sweep: pd.DataFrame) -> int:
    """Return chosen k with reasoning printed."""
    div()
    print("STEP 5 — CHOOSING k")
    div()

    # Silhouette peak
    sil_k   = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    sil_max = sweep["silhouette"].max()

    # CH peak
    ch_k = int(sweep.loc[sweep["ch"].idxmax(), "k"])

    # Elbow in inertia: largest second derivative
    inertias = sweep["inertia"].values
    d2 = np.diff(np.diff(inertias))   # second derivative of inertia
    elbow_k = int(sweep["k"].values[1:-1][np.argmax(d2)])  # skip k=2,k=10 boundary

    print(f"\n  Silhouette peak:     k={sil_k}  (score={sil_max:.4f})")
    print(f"  CH peak:             k={ch_k}")
    print(f"  Inertia elbow:       k={elbow_k}")

    # Print silhouette at k=4,5,6 for context
    print("\n  Silhouette at k=4,5,6:")
    for k in [4, 5, 6]:
        row = sweep[sweep["k"] == k].iloc[0]
        print(f"    k={k}: {row['silhouette']:.4f}")

    # Reasoning
    print("""
  Reasoning:
    Three metrics rarely agree at the same k for grocery panel data because
    the true segment structure is hierarchical (Champions splits into
    "daily-shopper" vs "high-basket, weekly" at finer k).

    The silhouette peak flags the k where clusters are most internally
    cohesive AND well-separated. CH inflates with more small tight clusters
    so it tends to peak earlier. The inertia elbow marks where additional k
    adds diminishing compactness return.

    The winning k must also be *interpretable*: each segment needs a
    distinct behaviour profile that a marketing team can act on. k=2 is
    trivially "active vs lapsed"; k>7 produces segments that differ only
    in continuous-variable slices with no clear action boundary.

    Decision rule: pick the lowest k where silhouette ≥ 0.90 × peak AND
    the segments have qualitatively distinct profiles. If metrics disagree,
    prefer the k where silhouette and the inertia elbow agree, since CH is
    known to overfit to compact geometry.
    """)

    # Apply decision rule
    sil_threshold = 0.90 * sil_max
    candidates = sweep[sweep["silhouette"] >= sil_threshold]["k"].values
    # Among candidates, prefer the k closest to the elbow
    chosen = int(candidates[np.argmin(np.abs(candidates - elbow_k))])
    # Never go below 3 (trivial split) or above 7 (uninterpretable)
    chosen = int(np.clip(chosen, 3, 7))

    print(f"  → Chosen k = {chosen}  (silhouette={sweep[sweep['k']==chosen]['silhouette'].values[0]:.4f})")
    return chosen


def rfm_segment_label(r: int, f: int, m: int) -> str:
    if r >= 4 and f >= 4 and m >= 4: return "Champions"
    if f >= 4:                        return "Loyal"
    if r >= 4 and f <= 2:             return "New/Returning"
    if r >= 3 and f >= 3:             return "Mid-tier"
    if r <= 2 and f >= 3 and m >= 3:  return "At-Risk"
    if r <= 2 and f <= 2:             return "Lost"
    return "Mid-tier"


def profile_and_crosstab(df: pd.DataFrame, labels: np.ndarray,
                          final_cols: list[str], k: int,
                          cluster_names: dict | None = None) -> None:
    """Print cluster profiles and cross-tab with RFM segments.

    cluster_names: pre-computed {cluster_id: name} dict. When supplied the
    canonical names are used for display; otherwise a heuristic fallback runs.
    """
    df = df.copy()
    df["cluster"] = labels
    df["rfm_seg"] = df.apply(
        lambda r: rfm_segment_label(r["r_score"], r["f_score"], r["m_score"]),
        axis=1,
    )

    div()
    print(f"STEP 6 — CLUSTER PROFILES (k={k})")
    div()

    # Raw (untransformed) summary stats per cluster
    profile_cols = {
        "frequency":               "trips/yr",
        "monetary":                "net_rev $",
        "recency_days":            "recency d",
        "mean_gap_days":           "gap d",
        "tenure_days":             "tenure d",
        "retail_disc_rate":        "ret_disc%",
        "private_label_spend_share": "pvt_lbl%",
        "top_store_trip_share":    "loyalty%",
        "weekend_trip_share":      "wknd%",
        "top_category_spend_share": "top_cat%",
    }
    header = f"\n  {'Cluster':<9}" + "".join(f"{v:>10}" for v in profile_cols.values())
    print(header)
    print("  " + "─" * (9 + 10 * len(profile_cols)))
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        row = f"  {c:<9}"
        for col in profile_cols:
            row += f"{sub[col].median():>10.1f}"
        print(row + f"   n={len(sub)}")

    print()

    # Use pre-computed canonical names when provided; heuristic fallback otherwise
    if cluster_names is None:
        cluster_names = {}
        for c in sorted(df["cluster"].unique()):
            sub = df[df["cluster"] == c]
            med_f = sub["frequency"].median()
            med_r = sub["recency_days"].median()
            med_m = sub["monetary"].median()
            med_g = sub["mean_gap_days"].median()
            if med_r <= 14 and med_f >= 80 and med_m >= 2500:
                name = "Champions"
            elif med_r <= 30 and med_f >= 30 and med_m >= 1000:
                name = "Loyal Regulars"
            elif med_r <= 20 and med_f < 30:
                name = "New/Returning"
            elif med_g >= 20 or med_r >= 60:
                name = "Lapsed / Lost"
            elif med_f < 30 and med_m < 800:
                name = "Occasional"
            else:
                name = "Mid-tier"
            cluster_names[c] = name

    for c in sorted(cluster_names.keys()):
        name = cluster_names[c]
        sub  = df[df["cluster"] == c]
        med_f = sub["frequency"].median()
        med_r = sub["recency_days"].median()
        med_m = sub["monetary"].median()
        print(f"  Cluster {c} → '{name}'  (n={len(sub)}, "
              f"median freq={med_f:.0f}, recency={med_r:.0f}d, monetary=${med_m:.0f})")

    # Cross-tab
    div()
    print(f"STEP 7 — CROSS-TAB: k-means clusters vs. RFM segments")
    div()
    ct = pd.crosstab(df["cluster"].map(lambda c: f"C{c}:{cluster_names[c]}"),
                     df["rfm_seg"],
                     margins=True)
    # Print nicely
    rfm_order = ["Champions", "Loyal", "Mid-tier", "New/Returning", "At-Risk", "Lost", "All"]
    rfm_order = [x for x in rfm_order if x in ct.columns]
    ct = ct[rfm_order] if all(c in ct.columns for c in rfm_order[:-1]) else ct
    print("\n" + ct.to_string())

    print("""
  Interpretation guide:
    • If each k-means cluster maps cleanly to ONE RFM segment, k-means
      rediscovered the manual cuts — no additional signal.
    • If one RFM segment splits across multiple k-means clusters, k-means
      has refined the segmentation using the behavioural features not in RFM
      (cadence regularity, store loyalty, category concentration, etc.).
    • If k-means clusters cross multiple RFM segments, k-means has found
      structure orthogonal to R/F/M — e.g., a "high-frequency, low-monetary
      / low-loyalty" cluster that the RFM cuts miss entirely.
    """)


def main():
    div("═")
    print("STRATA — Phase 2: K-Means Segmentation")
    div("═")

    df_all   = load_data()
    df, single_trip_df = preprocess(df_all)

    out = transform_features(df)
    final_cols = check_collinearity(out)

    # Scale
    div()
    print("STEP 3b — SCALING (RobustScaler)")
    div()
    print("  Using RobustScaler (median + IQR) rather than StandardScaler")
    print("  because confirmed heavy tails in mean_gap_days, frequency, monetary")
    print("  inflate the standard deviation even after log transform — the log")
    print("  reduces but does not eliminate the long right tail. RobustScaler's")
    print("  IQR-based normalization is insensitive to the remaining outliers.")

    X = out[final_cols].values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n  Feature matrix: {X_scaled.shape[0]} HHs × {X_scaled.shape[1]} features")

    # Sweep — still runs so all metrics are visible; CHOSEN_K overrides the result
    sweep = sweep_k(X_scaled, range(2, 11))
    _ = choose_k(sweep)
    chosen_k = CHOSEN_K
    print(f"\n  NOTE: automated sweep recommended a different k; "
          f"CHOSEN_K={CHOSEN_K} hard-pinned (see module-level comment).")

    # Fit final model
    div()
    print(f"STEP 5b — FITTING FINAL MODEL (k={chosen_k})")
    div()
    km_final = KMeans(n_clusters=chosen_k, n_init=20, max_iter=500, random_state=SEED)
    labels = km_final.fit_predict(X_scaled)
    df["cluster"] = labels

    # Assign canonical names by median monetary descending so the mapping is
    # deterministic regardless of which cluster ID the optimizer assigns to which
    # centroid across runs.
    _K4_NAMES = ["Champions", "Loyal Actives", "Declining", "Truly Lapsed"]
    _monetary_rank = (
        df.groupby("cluster")["monetary"].median()
        .sort_values(ascending=False)
        .index
    )
    cluster_id_to_name = {cid: _K4_NAMES[rank] for rank, cid in enumerate(_monetary_rank)}
    df["cluster_name"] = df["cluster"].map(cluster_id_to_name)

    # Save model — explicit name so downstream code never needs to guess which file
    model_path = MODELS_DIR / "kmeans_k4.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": km_final, "scaler": scaler, "features": final_cols}, f)
    print(f"\n  Model saved → {model_path.relative_to(PROJECT_ROOT)}")

    # Save cluster assignments (joinable on household_id); cluster_name populated above
    assign = df[["household_id", "cluster", "cluster_name",
                  "r_score", "f_score", "m_score",
                  "rfm_code", "has_demographics"]].copy()
    assign_path = EXTRACTS_DIR / "cluster_assignments.csv"
    assign.to_csv(assign_path, index=False)
    print(f"  Assignments saved → {assign_path.relative_to(PROJECT_ROOT)}")

    # Single-trip segment
    print(f"\n  Single-trip segment (excluded from k-means): {len(single_trip_df)} HHs")
    print(f"    median frequency={single_trip_df['frequency'].median():.0f}, "
          f"median monetary=${single_trip_df['monetary'].median():.0f}, "
          f"median recency={single_trip_df['recency_days'].median():.0f}d")
    print(f"    Assign to cluster ID = {chosen_k} (separate bucket) in downstream analysis")

    profile_and_crosstab(df, labels, final_cols, chosen_k, cluster_names=cluster_id_to_name)

    div("═")
    print("Done.")
    div("═")


if __name__ == "__main__":
    main()
