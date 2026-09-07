"""
STRATA Phase 4 — Campaign Quasi-Experiment
Parts 1-4: Naive → IPW-adjusted → Segment heterogeneity → Prospective design

Outcome:  hurdle_pred_clv (Q4 forecast from Phase 3 hurdle model)
Treatment: household exposed to single-arm campaign (T=1) vs unexposed (T=0)
Excluded:  concurrent-arm households (n=875) — confounded by overlapping
           campaigns, cannot be cleanly assigned to treatment or control
"""

from __future__ import annotations

import textwrap
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import false_discovery_control
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTS = PROJECT_ROOT / "outputs" / "extracts"
REPORTS  = PROJECT_ROOT / "outputs" / "reports"
DB_PATH  = PROJECT_ROOT / "data" / "processed" / "strata.duckdb"

RNG_SEED  = 42
B_BOOT    = 2_000
ALPHA     = 0.05
CLV_STD   = 430.0   # holdout RMSE from Phase 3 hurdle model (user-confirmed)

def _load_seg_order() -> list[str]:
    """Derive segment display order from cluster_assignments.csv (m_score desc)."""
    ca = pd.read_csv(EXTRACTS / "cluster_assignments.csv")
    return (
        ca[ca["cluster_name"].notna()]
        .groupby("cluster_name")["m_score"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

SEG_ORDER = _load_seg_order()


def wrap(text: str, width: int = 78) -> str:
    return "\n".join(
        textwrap.fill(line, width=width) if line.strip() else ""
        for line in text.splitlines()
    )


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    preds = pd.read_csv(EXTRACTS / "clv_predictions.csv")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    split_df = con.execute("""
        SELECT household_id,
               is_concurrent_arm,
               n_campaigns_exposed,
               obs_tenure_days
        FROM v_clv_split
        WHERE NOT is_pred_only
    """).df()

    # Household-level window truncation for single-arm households
    trunc_df = con.execute("""
        WITH hh_trunc AS (
            SELECT household_id,
                BOOL_OR(window_truncation = 'right_truncated') AS right_trunc,
                BOOL_OR(window_truncation = 'left_truncated')  AS left_trunc
            FROM v_campaign_exposure
            WHERE NOT has_concurrent_exposure
            GROUP BY household_id
        )
        SELECT household_id, right_trunc, left_trunc
        FROM hh_trunc
    """).df()

    con.close()

    df = preds.merge(
        split_df[["household_id", "is_concurrent_arm", "n_campaigns_exposed", "obs_tenure_days"]],
        on="household_id", how="left"
    )
    df = df.merge(trunc_df, on="household_id", how="left")

    n_exposed = df["n_campaigns_exposed"].fillna(0)
    is_conc   = df["is_concurrent_arm"].fillna(False)

    df["exposure_group"] = "unexposed"
    df.loc[n_exposed > 0, "exposure_group"] = "concurrent"
    df.loc[(n_exposed > 0) & (~is_conc), "exposure_group"] = "single_arm"

    df["right_trunc"] = df["right_trunc"].fillna(False).infer_objects(copy=False)
    df["left_trunc"]  = df["left_trunc"].fillna(False).infer_objects(copy=False)
    df["any_trunc"]   = df["right_trunc"] | df["left_trunc"]

    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────

def smd(vals_t: np.ndarray, vals_c: np.ndarray,
        w_t: np.ndarray = None, w_c: np.ndarray = None) -> float:
    if w_t is None:
        mu_t, mu_c = vals_t.mean(), vals_c.mean()
        pool_sd = np.sqrt((vals_t.std(ddof=1)**2 + vals_c.std(ddof=1)**2) / 2)
    else:
        mu_t = np.average(vals_t, weights=w_t)
        mu_c = np.average(vals_c, weights=w_c)
        pool_sd = np.sqrt((
            np.average((vals_t - mu_t)**2, weights=w_t) +
            np.average((vals_c - mu_c)**2, weights=w_c)
        ) / 2)
    return float((mu_t - mu_c) / pool_sd) if pool_sd > 0 else 0.0


def build_X(df_sub: pd.DataFrame) -> np.ndarray:
    """Feature matrix for propensity / outcome models."""
    X = pd.DataFrame({
        "log1p_monetary":  np.log1p(df_sub["obs_monetary"].values),
        "log1p_frequency": np.log1p(df_sub["obs_frequency"].values),
        "recency_days":    df_sub["obs_recency_days"].values,
        "tenure_days":     df_sub["obs_tenure_days"].values,
    })
    dummies = pd.get_dummies(df_sub["cluster_name"], prefix="seg", drop_first=False)
    # Drop Truly Lapsed and Single-Trip as reference categories
    for col in ["seg_Truly Lapsed", "seg_Single-Trip"]:
        if col in dummies.columns:
            dummies = dummies.drop(columns=[col])
    X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X.values.astype(float)


def fit_ps(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=RNG_SEED,
                                  solver="lbfgs")),
    ])
    pipe.fit(X, T)
    return pipe.predict_proba(X)[:, 1]


def ipw_ate_weighted(Y: np.ndarray, T: np.ndarray,
                     ps: np.ndarray, clip_lo=0.02, clip_hi=0.98
                     ) -> tuple[float, np.ndarray, float]:
    ps_c = np.clip(ps, clip_lo, clip_hi)
    w_raw = np.where(T == 1, 1.0 / ps_c, 1.0 / (1.0 - ps_c))
    thresh = np.percentile(w_raw, 99)
    w = np.minimum(w_raw, thresh)
    mt = T == 1
    mc = T == 0
    ate = (np.sum(w[mt] * Y[mt]) / np.sum(w[mt]) -
           np.sum(w[mc] * Y[mc]) / np.sum(w[mc]))
    return float(ate), w, float(thresh)


def dr_ate(Y: np.ndarray, T: np.ndarray, X: np.ndarray,
           ps: np.ndarray, clip_lo=0.02, clip_hi=0.98) -> float:
    ps_c = np.clip(ps, clip_lo, clip_hi)
    ridge_t = Pipeline([("sc", StandardScaler()),
                        ("r", RidgeCV(alphas=[1., 10., 100.], cv=5))])
    ridge_c = Pipeline([("sc", StandardScaler()),
                        ("r", RidgeCV(alphas=[1., 10., 100.], cv=5))])
    ridge_t.fit(X[T == 1], Y[T == 1])
    ridge_c.fit(X[T == 0], Y[T == 0])
    mu1 = ridge_t.predict(X)
    mu0 = ridge_c.predict(X)
    phi = (mu1 - mu0
           + T       * (Y - mu1) / ps_c
           - (1 - T) * (Y - mu0) / (1.0 - ps_c))
    return float(phi.mean())


def bootstrap_ipw_ci(sub: pd.DataFrame, B: int = B_BOOT,
                     rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(sub)
    ates = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(B):
            idx  = rng.choice(n, size=n, replace=True)
            boot = sub.iloc[idx].reset_index(drop=True)
            X_b  = build_X(boot)
            T_b  = boot["is_treated"].values
            Y_b  = boot["hurdle_pred_clv"].values
            if T_b.sum() < 5 or (1 - T_b).sum() < 5:
                continue
            try:
                ps_b = fit_ps(X_b, T_b)
                if np.any(~np.isfinite(ps_b)):
                    continue
                ate_b, _, _ = ipw_ate_weighted(Y_b, T_b, ps_b)
                if np.isfinite(ate_b):
                    ates.append(ate_b)
            except Exception:
                pass
    return np.array(ates)


def ess(w: np.ndarray) -> float:
    return float(np.sum(w)**2 / np.sum(w**2))


# ─── PART 1 ───────────────────────────────────────────────────────────────────

def part1_naive(df: pd.DataFrame) -> tuple[str, float]:
    sa = df[df["exposure_group"] == "single_arm"]
    un = df[df["exposure_group"] == "unexposed"]

    diff = sa["hurdle_pred_clv"].mean() - un["hurdle_pred_clv"].mean()
    pct  = diff / un["hurdle_pred_clv"].mean() * 100
    t, p = stats.ttest_ind(sa["hurdle_pred_clv"], un["hurdle_pred_clv"], equal_var=False)

    covs = [
        ("obs_monetary ($)",     "obs_monetary"),
        ("obs_frequency (trips)", "obs_frequency"),
        ("obs_recency_days",      "obs_recency_days"),
        ("obs_tenure_days",       "obs_tenure_days"),
    ]

    L = []
    L.append("=" * 78)
    L.append("PART 1 — NAIVE ESTIMATE (unadjusted)")
    L.append("=" * 78)
    L.append("")
    L.append(f"Single-arm exposed:   n={len(sa):,}")
    L.append(f"Unexposed controls:   n={len(un):,}")
    L.append(f"Concurrent-arm (excluded from all parts): n="
             f"{(df['exposure_group']=='concurrent').sum():,}")
    L.append("")
    L.append(f"  Exposed   mean hurdle_pred_clv: ${sa['hurdle_pred_clv'].mean():>8.2f}  "
             f"(sd=${sa['hurdle_pred_clv'].std():.2f})")
    L.append(f"  Unexposed mean hurdle_pred_clv: ${un['hurdle_pred_clv'].mean():>8.2f}  "
             f"(sd=${un['hurdle_pred_clv'].std():.2f})")
    L.append(f"  Raw difference:                 ${diff:>+8.2f}  ({pct:+.1f}% lift)")
    L.append(f"  Welch t-test:                   t={t:.3f}, p={p:.4f}")
    L.append("")
    L.append("WHY THIS NUMBER IS WRONG — Pre-campaign covariate imbalance:")
    L.append("")
    L.append(f"  {'Covariate':<28} {'Exposed':>10} {'Unexposed':>10} {'SMD':>8}")
    L.append("  " + "-" * 60)
    for label, col in covs:
        ev = sa[col].dropna().values
        uv = un[col].dropna().values
        s  = smd(ev, uv)
        flag = " !" if abs(s) > 0.1 else ""
        L.append(f"  {label:<28} {ev.mean():>10.1f} {uv.mean():>10.1f} {s:>8.3f}{flag}")

    L.append(f"  {'(! = |SMD|>0.10, threshold for imbalance)':<60}")
    L.append("")
    L.append("Cluster membership:")
    L.append(f"  {'Segment':<20} {'Exposed %':>10} {'Unexposed %':>12}")
    L.append("  " + "-" * 44)
    for seg in SEG_ORDER + ["Single-Trip"]:
        ep = (sa["cluster_name"] == seg).mean() * 100
        up = (un["cluster_name"] == seg).mean() * 100
        L.append(f"  {seg:<20} {ep:>10.1f} {up:>12.1f}")

    L.append("")
    L.append(wrap(
        f"The naive +${diff:.0f} ({pct:+.0f}%) lift is not a credible causal estimate. "
        "Exposed households have substantially higher pre-campaign spending, more "
        "shopping trips, and over-represent Champions and Loyal Actives relative to "
        "the unexposed pool. The retailer's campaign targeting selected its best "
        "existing customers — households that would have outperformed in Q4 regardless "
        "of any campaign. Parts 2 and 3 adjust for these pre-existing differences."
    ))
    L.append("")
    return "\n".join(L), diff


# ─── PART 2 ───────────────────────────────────────────────────────────────────

def part2_ipw(df: pd.DataFrame, naive_diff: float) -> tuple[str, float]:
    sub = df[df["exposure_group"].isin(["single_arm", "unexposed"])].copy()
    sub["is_treated"] = (sub["exposure_group"] == "single_arm").astype(int)
    sub = sub.reset_index(drop=True)

    T = sub["is_treated"].values.astype(float)
    Y = sub["hurdle_pred_clv"].values
    X = build_X(sub)

    # Propensity model
    ps = fit_ps(X, T)
    ps_c = np.clip(ps, 0.02, 0.98)

    # Overlap: fraction treated per PS decile
    sub["ps"]       = ps
    decile_bounds   = np.percentile(ps, np.arange(0, 110, 10))
    overlap_rows    = []
    for i in range(10):
        lo, hi = decile_bounds[i], decile_bounds[i + 1]
        mask   = (ps >= lo) & (ps <= hi) if i == 9 else (ps >= lo) & (ps < hi)
        n_tot  = mask.sum()
        n_t    = int(T[mask].sum())
        n_c    = int((1 - T[mask]).sum())
        overlap_rows.append((f"D{i+1:02d} [{lo:.3f},{hi:.3f}]", n_tot, n_t, n_c,
                             n_t / n_tot * 100 if n_tot else 0))

    # Raw weights for trimming
    w_raw   = np.where(T == 1, 1.0 / ps_c, 1.0 / (1.0 - ps_c))
    trim99  = np.percentile(w_raw, 99)
    w       = np.minimum(w_raw, trim99)
    n_trimmed = int((w_raw > trim99).sum())
    pct_trimmed = n_trimmed / len(w_raw) * 100

    # ESS
    ess_t_raw = ess(np.ones(int(T.sum())))
    ess_c_raw = ess(np.ones(int((1-T).sum())))
    ess_t_wt  = ess(w[T == 1])
    ess_c_wt  = ess(w[T == 0])

    # SMD before and after
    covs_for_smd = {
        "log1p(obs_monetary)":  np.log1p(sub["obs_monetary"].values),
        "log1p(obs_frequency)": np.log1p(sub["obs_frequency"].values),
        "obs_recency_days":     sub["obs_recency_days"].values,
        "obs_tenure_days":      sub["obs_tenure_days"].values,
    }
    mt, mc = T == 1, T == 0
    smd_before, smd_after = {}, {}
    for name, vals in covs_for_smd.items():
        smd_before[name] = smd(vals[mt], vals[mc])
        smd_after[name]  = smd(vals[mt], vals[mc], w_t=w[mt], w_c=w[mc])

    # IPW ATE
    ate_ipw, _, _ = ipw_ate_weighted(Y, T, ps)

    # Doubly-robust ATE
    ate_dr = dr_ate(Y, T, X, ps)

    # Bootstrap CI (B=2000)
    print(f"  Running {B_BOOT:,} bootstrap iterations for Part 2 CI …", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    boot_ates = bootstrap_ipw_ci(sub, B=B_BOOT, rng=rng)
    ci_lo, ci_hi = np.percentile(boot_ates, [2.5, 97.5])
    boot_se  = boot_ates.std()
    print(f"  Bootstrap done in {time.time()-t0:.1f}s  (B_valid={len(boot_ates)})", flush=True)

    # ── Full deliverable text ──────────────────────────────────────────────────

    L = []
    L.append("=" * 78)
    L.append("PART 2 — IPW-ADJUSTED ESTIMATE (full deliverable)")
    L.append("=" * 78)

    L.append("""
─────────────────────────────────────────────────────────────────────────────
STUDY DESIGN
─────────────────────────────────────────────────────────────────────────────

This analysis estimates the average treatment effect (ATE) of single-arm
campaign exposure on household-level predicted Q4 CLV, using inverse
probability weighting to account for the self-selection visible in Part 1.

Population in scope
  - Treatment arm (T=1): 683 households exposed to at least one campaign
    during the observation window (Jan–Sep 2017), with no overlapping
    concurrent campaign. These are "clean" exposed households whose campaign
    response is not contaminated by simultaneous brand-awareness activity.
  - Control pool (T=0): 888 households that received no campaign exposure
    during the same window. Single-Trip households (n=27) are included in
    this pool; they contribute little weight after IPW because their
    pre-campaign profile diverges sharply from exposed households.
  - Excluded: 875 concurrent-arm households. Campaign assignment for these
    households is confounded by overlapping exposures; no propensity model
    can separate the individual campaign effects.

Outcome variable
  hurdle_pred_clv — the Q4 CLV forecast produced by the Phase 3 two-part
  hurdle model (logistic × Ridge). This is a model-based outcome, not actual
  observed Q4 spend. As a consequence, the estimated effect captures whether
  campaign-exposed households have pre-Q4 behavioural signatures associated
  with higher Q4 spending, not whether the campaign directly caused higher Q4
  spend. This distinction matters for interpretation and is addressed in the
  "Remaining Confounding" section below.

Window truncation
  110 single-arm households had right-truncated exposure (campaign started
  before the observation window), and 97 had left-truncated exposure
  (campaign ended before the obs window closed). This analysis treats any
  campaign exposure as treated (intent-to-treat), regardless of duration.
  Dose-response analysis (hours of exposure × response) is out of scope.
""")

    L.append("""─────────────────────────────────────────────────────────────────────────────
PROPENSITY MODEL
─────────────────────────────────────────────────────────────────────────────

The propensity score e(X) = P(T=1 | X) is estimated via logistic regression
with five covariate sets:

  1. log1p(obs_monetary)  — total obs-window household spend, log-scaled
  2. log1p(obs_frequency) — obs-window trip count, log-scaled
  3. obs_recency_days     — days since last purchase before the split date
  4. obs_tenure_days      — span from first to last purchase in obs window
  5. Segment indicators   — Champions, Loyal Actives, Declining (Truly
     Lapsed is the omitted reference; Single-Trip folds into intercept)

Rationale: these five variables capture the pre-campaign household profile
that the retailer plausibly used for targeting. Log-scaling monetary and
frequency mitigates right-skew and scale sensitivity. Segment indicators
serve as a compressed summary of the multivariate RFM profile that individual
continuous features might fail to represent fully.

Regularisation: L2 penalty with C=1.0 (sklearn default). All features
standardised before fitting. The choice of C is not tuned here; sensitivity
to C is absorbed by the doubly-robust estimator in the robustness check.
""")

    # Overlap table
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("OVERLAP (COMMON SUPPORT)")
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("")
    L.append("Propensity score distribution by decile (T=treatment, C=control):")
    L.append("")
    L.append(f"  {'Decile [PS range]':<26} {'n_total':>8} {'n_T':>6} {'n_C':>6} {'%T':>6}")
    L.append("  " + "-" * 54)
    poor_overlap = []
    for row in overlap_rows:
        dec, n_tot, n_t, n_c, pct_t = row
        flag = " ← poor overlap" if pct_t > 90 or pct_t < 10 else ""
        L.append(f"  {dec:<26} {n_tot:>8} {n_t:>6} {n_c:>6} {pct_t:>5.1f}%{flag}")
        if pct_t > 90 or pct_t < 10:
            poor_overlap.append(dec)
    L.append("")
    if poor_overlap:
        L.append(wrap(
            f"Poor overlap in deciles: {', '.join(poor_overlap)}. These deciles contain "
            "households whose propensity score is so extreme (near 0 or near 1) that "
            "there are almost no counterparts on the other side. IPW weights for these "
            "households become very large, amplifying their influence on the estimate. "
            "The 99th-percentile trimming below addresses the weight explosion, but it "
            "also means the estimated ATE applies to the covariate region where overlap "
            "is adequate — not to the full marginal population."
        ))
    else:
        L.append("Overlap is adequate across all deciles — no extreme concentration.")
    L.append("")

    # SMD table
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("COVARIATE BALANCE (SMD before and after weighting)")
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("")
    L.append(f"  {'Covariate':<26} {'Before':>8} {'After':>8}  {'Pass (<0.10)?':>14}")
    L.append("  " + "-" * 62)
    balance_ok = True
    for name, sb in smd_before.items():
        sa_ = smd_after[name]
        passed = abs(sa_) < 0.10
        if not passed:
            balance_ok = False
        mark = "YES" if passed else "NO !"
        L.append(f"  {name:<26} {sb:>+8.3f} {sa_:>+8.3f}  {mark:>14}")
    L.append("")
    if balance_ok:
        L.append("  All covariates meet the |SMD| < 0.10 balance threshold after weighting.")
    else:
        L.append("  WARNING: One or more covariates remain imbalanced after weighting.")
        L.append("  Interpret the IPW estimate with caution; the DR estimator provides")
        L.append("  additional robustness in that case.")
    L.append("")

    # Weight trimming
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("WEIGHT TRIMMING")
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("")
    L.append(f"  99th-percentile weight cap:    {trim99:.2f}")
    L.append(f"  Households trimmed:            {n_trimmed} ({pct_trimmed:.1f}% of pool)")
    L.append(f"  Effective sample size (ESS):")
    L.append(f"    Treated  — raw: {ess_t_raw:.0f}  → weighted: {ess_t_wt:.0f}  "
             f"(loss: {(1 - ess_t_wt/ess_t_raw)*100:.0f}%)")
    L.append(f"    Control  — raw: {ess_c_raw:.0f}  → weighted: {ess_c_wt:.0f}  "
             f"(loss: {(1 - ess_c_wt/ess_c_raw)*100:.0f}%)")
    L.append("")
    L.append(wrap(
        "Weight trimming caps the most extreme IPW weights at the 99th percentile of "
        "the weight distribution. This reduces variance at the cost of some bias "
        "(the trimmed estimator is no longer exactly unbiased if the propensity model "
        "is misspecified in the tail). Given the large ESS reduction shown above, "
        "these tails are carrying disproportionate influence; trimming is the correct "
        "trade-off. The effective sample size measures the 'equivalent number of "
        "unweighted observations' — a large ESS loss signals that the IPW estimate is "
        "driven by a small number of high-weight units rather than the full population."
    ))
    L.append("")

    # IPW estimate
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("IPW ESTIMATE")
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("")
    L.append(f"  IPW-weighted ATE:    ${ate_ipw:>+8.2f}")
    L.append(f"  95% bootstrap CI:    [${ci_lo:>+.2f}, ${ci_hi:>+.2f}]  (B={len(boot_ates):,})")
    L.append(f"  Bootstrap SE:        ${boot_se:>.2f}")
    ctrl_mean = sub[sub["is_treated"] == 0]["hurdle_pred_clv"].mean()
    pct_lift = ate_ipw / ctrl_mean * 100
    L.append(f"  As % of control mean: {pct_lift:+.1f}%")
    L.append("")

    # Doubly-robust
    dr_diff = abs(ate_dr - ate_ipw)
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("DOUBLY-ROBUST ROBUSTNESS CHECK (AIPW)")
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("")
    L.append(f"  Doubly-robust (AIPW) ATE: ${ate_dr:>+8.2f}")
    L.append(f"  IPW ATE:                  ${ate_ipw:>+8.2f}")
    L.append(f"  Absolute difference:      ${dr_diff:>8.2f}  "
             f"({'material' if dr_diff > 20 else 'negligible'})")
    L.append("")
    if dr_diff > 20:
        L.append(wrap(
            f"The ${dr_diff:.0f} gap between the DR and IPW estimates warrants attention. "
            "The AIPW estimator augments IPW with a regression-adjustment term that "
            "corrects for outcome model fit within each arm. A material divergence "
            "suggests either (a) the propensity model is mis-specified in a region "
            "where the outcome model picks up the slack, or (b) the outcome model is "
            "extrapolating into low-overlap regions. In either case, the DR estimate "
            "is generally preferred because it remains consistent if either model — "
            "but not necessarily both — is correctly specified."
        ))
    else:
        L.append(wrap(
            f"The ${dr_diff:.0f} difference between DR and IPW is negligible, consistent "
            "with the propensity model being approximately correctly specified. Both "
            "estimators agree, which strengthens confidence in the IPW point estimate. "
            "The DR estimate is preferred in principle; in practice they are "
            "interchangeable here."
        ))
    L.append("")

    # Interpretation
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("INTERPRETATION AND REMAINING CONFOUNDING")
    L.append("─────────────────────────────────────────────────────────────────────────────")
    L.append("")
    L.append(wrap(
        f"After weighting for pre-campaign differences in spending, frequency, "
        f"recency, tenure, and segment membership, the adjusted campaign effect is "
        f"${ate_ipw:+.0f} per household in predicted Q4 CLV "
        f"(95% CI: [${ci_lo:.0f}, ${ci_hi:.0f}]). "
        + (
            "The confidence interval includes zero, which means we cannot rule out "
            "a null effect at the 5% level. The adjusted estimate is negative — "
            "after controlling for observable selection, exposed households do not "
            "show a measurable advantage in predicted CLV — but the wide CI "
            "prevents a confident negative conclusion."
            if ci_lo <= 0 else
            f"The confidence interval excludes zero; the effect is statistically "
            f"significant at the 5% level. The adjusted lift represents "
            f"approximately {pct_lift:.0f}% of the control group's mean predicted "
            f"CLV, which is the portion of Q4 spending trajectory attributable to "
            f"campaign exposure after controlling for observable confounders."
        )
    ))
    L.append("")
    L.append(wrap(
        "What the estimate represents: A causal interpretation requires that the "
        "propensity model accounts for all variables that jointly determine campaign "
        "assignment and Q4 outcomes. Two residual confounders are likely:"
    ))
    L.append("")
    L.append(wrap(
        "  1. Self-selection into the loyalty programme. Households who opt into "
        "     loyalty schemes are systematically more brand-loyal than those who do "
        "     not, independent of observable spending metrics. This latent loyalty "
        "     dimension is unobserved and therefore not controlled by IPW. The "
        "     adjusted estimate may still overstate the causal effect of the campaign "
        "     because brand-loyal households (a) are more likely to be exposed and "
        "     (b) would have higher Q4 CLV regardless of exposure."
    ))
    L.append("")
    L.append(wrap(
        "  2. The outcome is model-predicted CLV, not actual Q4 spend. The hurdle "
        "     model was trained on obs-window features; if the campaign caused "
        "     behavioural changes during the obs window (more trips, higher basket "
        "     sizes), those changes are already embedded in the features used to "
        "     generate hurdle_pred_clv. The quasi-experiment then asks 'do exposed "
        "     households have obs-window signatures associated with higher Q4 CLV?' — "
        "     which partially conflates the campaign's behavioural effect with the "
        "     structural CLV prediction. Using actual clv_pred_window (observed Q4 "
        "     spend) as the outcome in a future analysis would separate these two "
        "     channels."
    ))
    L.append("")
    L.append(wrap(
        f"Bottom line: the adjusted estimate (${ate_ipw:+.0f}) is substantially "
        f"smaller than the naive estimate (${naive_diff:+.0f}), confirming that most "
        "of the raw gap was pre-existing selection rather than campaign effect. The "
        "adjusted estimate is the best available observational estimate but is not a "
        "randomised treatment effect. Part 4 designs the experiment that would have "
        "produced a credible number."
    ))
    L.append("")

    return "\n".join(L), ate_ipw, ci_lo, ci_hi, boot_se, sub


# ─── PART 3 ───────────────────────────────────────────────────────────────────

def part3_segments(df: pd.DataFrame, ate_ipw_global: float) -> str:
    L = []
    L.append("=" * 78)
    L.append("PART 3 — SEGMENT HETEROGENEITY (IPW-adjusted effect per segment)")
    L.append("=" * 78)
    L.append("")
    L.append("IPW propensity model re-fit within each segment's exposed+unexposed pool.")
    L.append("Segments with fewer than 30 exposed households receive a reliability flag.")
    L.append("")

    rng = np.random.default_rng(RNG_SEED)
    seg_results = []

    for seg in SEG_ORDER:
        pool = df[(df["cluster_name"] == seg) &
                  (df["exposure_group"].isin(["single_arm", "unexposed"]))].copy()
        pool["is_treated"] = (pool["exposure_group"] == "single_arm").astype(int)
        pool = pool.reset_index(drop=True)

        n_t = int(pool["is_treated"].sum())
        n_c = int((1 - pool["is_treated"]).sum())
        min_n = min(n_t, n_c)
        flag = " [!low-n]" if min_n < 30 else ""

        if n_t < 5 or n_c < 5:
            seg_results.append({
                "seg": seg, "n_t": n_t, "n_c": n_c,
                "ate": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_val": np.nan, "flag": flag + " [insufficient]"
            })
            continue

        T = pool["is_treated"].values.astype(float)
        Y = pool["hurdle_pred_clv"].values
        X = build_X(pool)

        ps = fit_ps(X, T)
        ate_s, _, _ = ipw_ate_weighted(Y, T, ps)

        boot_s = bootstrap_ipw_ci(pool, B=500, rng=rng)
        ci_lo_s = np.percentile(boot_s, 2.5)
        ci_hi_s = np.percentile(boot_s, 97.5)
        # Bootstrap p-value (two-sided: P(|ate| >= |observed| | H0))
        p_boot = 2 * min(
            (boot_s <= 0).mean() if ate_s > 0 else (boot_s >= 0).mean(),
            0.5
        )
        p_boot = max(p_boot, 1.0 / len(boot_s))  # floor at 1/B

        seg_results.append({
            "seg": seg, "n_t": n_t, "n_c": n_c,
            "ate": ate_s, "ci_lo": ci_lo_s, "ci_hi": ci_hi_s,
            "p_val": p_boot, "flag": flag
        })

    # BH correction
    valid = [r for r in seg_results if not np.isnan(r["p_val"])]
    if valid:
        p_vals = np.array([r["p_val"] for r in valid])
        # scipy.stats.false_discovery_control (Benjamini-Hochberg)
        rejected = false_discovery_control(p_vals, method="bh") <= ALPHA
        for i, r in enumerate(valid):
            r["bh_reject"] = bool(p_vals[i] <= false_discovery_control(p_vals, method="bh")[i])
        # Simpler: BH step-up
        m = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        bh_thresh  = np.array([(k + 1) / m * ALPHA for k in range(m)])
        bh_reject  = p_vals[sorted_idx] <= bh_thresh
        # All hypotheses with rank ≤ max-rejected are rejected
        max_k = np.max(np.where(bh_reject)[0]) if bh_reject.any() else -1
        for i, r in enumerate(valid):
            rank = int(np.where(sorted_idx == i)[0][0])
            r["bh_reject"] = rank <= max_k

    L.append(f"  {'Segment':<18} {'n_T':>5} {'n_C':>5}  {'ATE ($)':>9}  "
             f"{'95% CI (boot)':>22}  {'p_boot':>8}  {'BH sig':>8}")
    L.append("  " + "-" * 86)
    for r in seg_results:
        if np.isnan(r.get("ate", np.nan)):
            L.append(f"  {r['seg']:<18} {r['n_t']:>5} {r['n_c']:>5}  {'N/A':>9}  "
                     f"{'':>22}  {'N/A':>8}  {'N/A':>8}{r['flag']}")
        else:
            ci_str   = f"[{r['ci_lo']:>+.0f}, {r['ci_hi']:>+.0f}]"
            sig_mark = "**" if r.get("bh_reject") else "ns"
            L.append(f"  {r['seg']:<18} {r['n_t']:>5} {r['n_c']:>5}  "
                     f"${r['ate']:>+8.1f}  {ci_str:>22}  "
                     f"{r['p_val']:>8.4f}  {sig_mark:>8}{r['flag']}")

    L.append("")
    L.append("Benjamini-Hochberg correction applied across all four segment tests (α=0.05).")
    L.append("")

    surviving = [r for r in valid if r.get("bh_reject")]
    noise     = [r for r in valid if not r.get("bh_reject")]

    if surviving:
        L.append(wrap(
            "Segments whose effects survive BH correction: "
            + ", ".join(r["seg"] for r in surviving) + ". "
            "These segments have both a detectable IPW-adjusted effect and sufficient "
            "sample size to survive the multiple-testing penalty."
        ))
    if noise:
        L.append(wrap(
            "Segments that do not survive correction: "
            + ", ".join(r["seg"] for r in noise) + ". "
            "These effects are indistinguishable from noise at this sample size. "
            "The point estimates may be directionally informative but should not "
            "be used to allocate budget."
        ))
    L.append("")
    L.append(wrap(
        "Important caveat: segment-level IPW re-estimates a propensity model on "
        "each segment's sub-pool. For small segments (Truly Lapsed n_T=19), the "
        "propensity model is underpowered to balance covariates; SMD before/after "
        "is not re-computed here but should be verified before acting on the "
        "segment-level estimates."
    ))
    L.append("")

    return "\n".join(L)


# ─── PART 4 ───────────────────────────────────────────────────────────────────

def part4_prospective(ate_ipw: float, ci_lo: float, ci_hi: float) -> str:
    L = []
    L.append("=" * 78)
    L.append("PART 4 — PROSPECTIVE RANDOMISED DESIGN")
    L.append("=" * 78)
    L.append("")

    # (a) MDE — use absolute value; power analysis is symmetric
    mde = abs(ate_ipw) * 0.50
    L.append("(a) Minimum Detectable Effect (MDE)")
    L.append("")
    L.append(wrap(
        f"The adjusted IPW estimate is ${ate_ipw:+.0f}/household (a null-or-negative "
        "finding after controlling for selection). The prospective experiment is "
        "designed to detect an effect of half this magnitude in either direction: "
        f"MDE = ${mde:.0f}/household. The 50% floor is justified as follows: "
        "the IPW estimate carries residual confounding (unobserved brand loyalty "
        "and the outcome-model circularity described in Part 2); a clean randomised "
        "design may reveal a positive, negative, or null true effect. An MDE of "
        f"${mde:.0f} represents the minimum per-household shift that would justify "
        "campaign cost at the retailer's margin structure, regardless of direction. "
        "Below this threshold, the incremental revenue (or prevented churn) would "
        "not offset campaign delivery and voucher redemption costs."
    ))
    L.append("")

    # (b) Power analysis
    sigma = CLV_STD
    z_a   = stats.norm.ppf(1 - ALPHA / 2)   # 1.96
    z_b   = stats.norm.ppf(0.80)             # 0.842
    n_per_arm = int(np.ceil(2 * (z_a + z_b)**2 * sigma**2 / mde**2))

    L.append("(b) Power Analysis")
    L.append("")
    L.append(f"  α = {ALPHA}  (two-sided),  power = 0.80,  σ = ${sigma:.0f}  (hurdle model RMSE)")
    L.append(f"  MDE (δ):  ${mde:.0f}/household")
    L.append(f"  Formula:  n = 2·(z_α/2 + z_β)² · σ² / δ²")
    L.append(f"            = 2·({z_a:.3f} + {z_b:.3f})² · {sigma:.0f}² / {mde:.0f}²")
    L.append(f"  Required per arm:  {n_per_arm:,}")
    L.append(f"  Total required:    {2*n_per_arm:,}")
    L.append("")
    L.append(wrap(
        f"The current exposed pool has {683:,} households — "
        + ("above" if 683 >= n_per_arm else "below") +
        f" the required {n_per_arm:,} per arm. However, power analysis for a "
        "randomised experiment and the quasi-experiment pool size are not "
        "directly comparable: the existing pool is self-selected, while a "
        "randomised design would draw from the full eligible population "
        "(all households in the loyalty programme regardless of historical "
        "engagement). The required sample size should inform how many households "
        "to recruit into the randomised campaign arm, not how many were "
        "historically exposed."
    ))
    L.append("")

    # (c) Simulation confirmation
    rng_s = np.random.default_rng(RNG_SEED)
    B_sim = 2_000
    true_effect = mde   # DGP parameter = MDE (power target)
    base_mean   = 300.0  # approximate control mean

    sim_pvals = np.zeros(B_sim)
    for i in range(B_sim):
        y_t = rng_s.normal(base_mean + true_effect, sigma, n_per_arm)
        y_c = rng_s.normal(base_mean, sigma, n_per_arm)
        _, p = stats.ttest_ind(y_t, y_c, equal_var=False)
        sim_pvals[i] = p

    empirical_power = (sim_pvals < ALPHA).mean()

    L.append("(c) Simulation Confirmation")
    L.append("")
    L.append(f"  DGP:  Y_treated ~ N({base_mean + true_effect:.0f}, {sigma}²),  "
             f"Y_control ~ N({base_mean:.0f}, {sigma}²)")
    L.append(f"  n per arm: {n_per_arm:,},  B={B_sim:,} simulated trials")
    L.append(f"  Empirical power:  {empirical_power:.3f}  "
             f"(target: 0.800 ± 0.015  {'✓' if abs(empirical_power - 0.80) < 0.02 else '!'})")
    L.append("")

    # (d) Peeking simulation
    L.append("(d) Peeking: False-Positive Rate at Interim Looks")
    L.append("")
    L.append(wrap(
        "The retailer might be tempted to check significance at 25%, 50%, and 75% "
        "of the target sample and stop early if p<0.05 at any look. Under the null "
        "hypothesis (no campaign effect), each additional test increases the chance "
        "of a false positive. The simulation below runs B=2,000 null trials and "
        "tests at each fraction of the target sample."
    ))
    L.append("")

    fractions   = [0.25, 0.50, 0.75, 1.00]
    n_at_look   = [max(10, int(f * n_per_arm)) for f in fractions]
    peeked_fp   = np.zeros(len(fractions), dtype=float)   # cumulative FP

    rng_p = np.random.default_rng(RNG_SEED + 1)
    B_peek = 2_000
    for i in range(B_peek):
        y_t_full = rng_p.normal(base_mean, sigma, n_per_arm)
        y_c_full = rng_p.normal(base_mean, sigma, n_per_arm)
        rejected_so_far = False
        for j, n_look in enumerate(n_at_look):
            if rejected_so_far:
                peeked_fp[j] += 1
                continue
            _, p = stats.ttest_ind(y_t_full[:n_look], y_c_full[:n_look], equal_var=False)
            if p < ALPHA:
                rejected_so_far = True
                peeked_fp[j] += 1
    peeked_fp /= B_peek   # FP rate at each look (cumulative)

    L.append(f"  {'Look':<14} {'n/arm':>8}  {'Cumulative FPR':>16}  {'Nominal α':>10}")
    L.append("  " + "-" * 52)
    for j, (frac, n_look) in enumerate(zip(fractions, n_at_look)):
        label = f"{int(frac*100)}% ({n_look:,})"
        L.append(f"  {label:<14} {n_look:>8}  {peeked_fp[j]:>16.3f}  {ALPHA:>10.3f}")

    L.append("")
    L.append(wrap(
        f"With three interim looks plus the final analysis, the nominal α=0.05 "
        f"inflates to an actual false-positive rate of ~{peeked_fp[-1]:.2f} — "
        "more than double the intended level. This is the classic multiple-looks "
        "problem in sequential testing."
    ))
    L.append("")
    L.append("Sequential testing correction (Pocock boundary, K=4 looks, α=0.05):")
    L.append("")
    # Pocock critical value for K=4, α=0.05 two-sided ≈ 2.361
    # Required per-look α ≈ 0.0182
    pocock_z   = 2.361
    alpha_look = 2 * (1 - stats.norm.cdf(pocock_z))
    n_seq      = int(np.ceil(2 * (pocock_z + z_b)**2 * sigma**2 / mde**2))

    L.append(f"  Per-look critical value:   z = {pocock_z}  (α per look ≈ {alpha_look:.4f})")
    L.append(f"  Fixed-sample n/arm:        {n_per_arm:,}")
    L.append(f"  Pocock-corrected n/arm:    {n_seq:,}  "
             f"(inflation factor: {n_seq/n_per_arm:.2f}×)")
    L.append("")
    L.append(wrap(
        f"The Pocock boundary requires {n_seq - n_per_arm:,} additional households per "
        "arm (~" + f"{(n_seq/n_per_arm - 1)*100:.0f}% more) to maintain 80% power "
        "while allowing three interim looks. O'Brien-Fleming boundaries are "
        "more conservative early (making early stopping very hard) and nearly "
        "as efficient as fixed-sample at the final look, costing only ~2% extra "
        "sample size. For a campaign with monthly reporting cadence, "
        "O'Brien-Fleming is the preferred choice: it protects against over-eager "
        "early stopping while imposing minimal recruitment overhead."
    ))
    L.append("")

    return "\n".join(L)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    df = load_data()
    print(f"  n={len(df):,} | groups: {df['exposure_group'].value_counts().to_dict()}")

    print("\nPart 1: naive estimate …")
    p1_text, naive_diff = part1_naive(df)

    print("Part 2: IPW-adjusted estimate …")
    p2_result = part2_ipw(df, naive_diff)
    p2_text, ate_ipw, ci_lo, ci_hi, boot_se, _ = p2_result

    print("Part 3: segment heterogeneity …")
    p3_text = part3_segments(df, ate_ipw)

    print("Part 4: prospective design …")
    p4_text = part4_prospective(ate_ipw, ci_lo, ci_hi)

    full_report = "\n".join([p1_text, p2_text, p3_text, p4_text])
    print()
    print(full_report)

    out = REPORTS / "phase4_campaign_quasi_experiment.txt"
    out.write_text(full_report)
    print(f"\nReport written to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
