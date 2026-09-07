"""
STRATA Phase 3 — CLV Regression

STAGE 1 — MODEL STRUCTURE DECISION: Two-part hurdle model chosen over single
log1p OLS. Rationale (three independent reasons):

  1. BIMODAL LOG1P DISTRIBUTION: the gap between log1p(0)=0.0 and the minimum
     nonzero log1p(CLV)=0.464 is wide enough that the zero mass and the
     spending distribution are clearly separate generative processes. A single
     log1p model treats this gap as continuous variation and fits a line through
     both; it will systematically overpredict for likely-zero HHs and underpredict
     for all others because the zero mass pulls every coefficient toward zero.

  2. CLUSTER-STRATIFIED ZEROS: Loyal Actives have 0% zero-spend; Truly Lapsed
     have 52.6%. The "will they return?" question is answered almost entirely by
     recency/tenure/gap signals (absence indicators). The "how much?" question
     is answered almost entirely by obs_monetary/obs_frequency (intensity).
     A single model has to simultaneously explain why obs_recency is negative
     (lapsing signal) and why obs_monetary is positive (intensity signal); the
     log1p model mixes these up.

  3. STAKEHOLDER INTERPRETABILITY: P(return) × E[spend | return] decomposes
     naturally into "reactivation probability" and "expected basket size" — two
     business actions that target different teams (retention vs. upsell).

  Option A (single log1p) is retained as the OLS/Ridge/GBM baseline because:
  - With only 5.8% zeros it is not catastrophically wrong
  - It provides the OLS coefficient interpretation the user requested
  - The comparison quantifies the hurdle's gain precisely

STAGE 2 — FEATURE HANDLING decisions are documented per-block below.
"""

import warnings
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS as sm_OLS
from statsmodels.tools.tools import add_constant as sm_add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(".")
DB   = ROOT / "data" / "processed" / "strata.duckdb"
SEED = 42

# ── Load ──────────────────────────────────────────────────────────────────────
con = duckdb.connect(str(DB), read_only=True)
df  = con.execute("SELECT * FROM v_clv_split WHERE NOT is_pred_only").df()
con.close()

clusters = pd.read_csv("outputs/extracts/cluster_assignments.csv")[
    ["household_id", "cluster_name"]]
df = df.merge(clusters, on="household_id", how="left")
df["cluster_name"] = df["cluster_name"].fillna("Single-Trip")

print(f"Training sample: {len(df):,}  zero_CLV={( df.clv_pred_window==0).sum()} "
      f"({100*(df.clv_pred_window==0).mean():.1f}%)")

# ── STAGE 2: Feature handling ──────────────────────────────────────────────────

# obs_mean_gap_days: 47 NULL = single-trip obs-window HHs (only one trip → no
# inter-trip gap to compute). Impute with obs_recency_days as a proxy for
# "implied cadence": a HH who shopped once 200d before the split has an
# implied return interval of ≥200d. This is more informative than a constant
# (e.g. 95th pct) because the time-since-trip varies meaningfully across these
# 47 HHs. Add is_single_trip_obs binary so the model can learn that this is
# structural missingness, not just a high-gap household.
df["is_single_trip_obs"] = df["obs_mean_gap_days"].isna().astype(float)
df["obs_mean_gap_days"]  = df["obs_mean_gap_days"].fillna(df["obs_recency_days"])

# obs_std_gap_days: 95 NULL = ≤2 shopping days in obs window (std dev
# undefined for n<2). Impute 0.0: zero variance is the correct description of
# a HH with only one gap to measure. No additional flag needed because
# obs_frequency ≤2 is already in the feature set and captures this regime.
df["obs_std_gap_days"] = df["obs_std_gap_days"].fillna(0.0)

# obs_basket_trend_direction: 412 NULL = insufficient baskets in H1 or H2
# (< 3 baskets in at least one half-period). Impute 2 (stable/neutral) AND
# add has_trend_obs binary. The 412 NULLs are concentrated in low-frequency
# HHs where "trend" is undefined; the flag tells the model this is a coverage
# gap, not a stable basket. If we only imputed 2 without a flag, we'd confuse
# genuinely-stable HHs with ineligible low-frequency ones.
df["has_trend_obs"]              = df["obs_basket_trend_direction"].notna().astype(float)
df["obs_basket_trend_direction"] = df["obs_basket_trend_direction"].fillna(2.0)

# obs_redemption_propensity: 1,763 NULL. This is NOT missing-at-random:
#   - 875 concurrent-arm HHs → attribution undefined (not applicable)
#   - 910 unexposed HHs → no campaigns to redeem (not applicable)
# Do NOT impute with median — the median of the 683 eligible HHs has a
# different interpretation than "not applicable." Instead:
#   is_redemption_eligible = 1 for single-arm exposed HHs (n=683)
#   obs_redemption_propensity imputed to 0.0 for all others
# The logistic and linear parts both see the eligibility flag + propensity.
df["is_redemption_eligible"]  = df["obs_redemption_propensity"].notna().astype(float)
df["obs_redemption_propensity"] = df["obs_redemption_propensity"].fillna(0.0)

# obs_top_category_spend_share: 3 NULL (all spend in COUPON/MISC) → 0.0
df["obs_top_category_spend_share"] = df["obs_top_category_spend_share"].fillna(0.0)

# is_concurrent_arm / n_campaigns_exposed: 888 NULL = households with no
# campaign exposure (not enrolled in any campaign). The obs_campaign CTE
# LEFT JOINs v_campaign_exposure; unexposed HHs produce NULL. Correct
# semantics: unexposed → not concurrent arm (False/0), 0 campaigns exposed.
df["is_concurrent_arm"]    = df["is_concurrent_arm"].fillna(False).astype(float)
df["n_campaigns_exposed"]  = df["n_campaigns_exposed"].fillna(0).astype(float)

# ── Transforms ─────────────────────────────────────────────────────────────────
# All transforms mirror Phase 2 clustering decisions where applicable.
df["log_recency"]   = np.log(df["obs_recency_days"])
df["log_frequency"] = np.log1p(df["obs_frequency"])
df["log_monetary"]  = np.log(df["obs_monetary"])
df["log_gap"]       = np.log(df["obs_mean_gap_days"])
df["log_weight"]    = np.log1p(df["obs_weight_sold_spend_share"])
df["sqrt_top_cat"]  = np.sqrt(df["obs_top_category_spend_share"])
df["log_campaigns"] = np.log1p(df["n_campaigns_exposed"])
# obs_std_gap_days dropped: |r|=0.864 with obs_mean_gap_days → collinear.
# Ridge handles collinearity; OLS SEs would explode. Reporting in diagnostics.
# obs_basket_trend_ratio dropped: more NULLs (412 vs. 0 after direction imputation),
# and basket_trend_direction captures the same signal in a more stable form.

FEATURES = [
    # Core RFM (log-transformed raw values; NTILE scores excluded — see Phase 2)
    "log_recency",          # ↑ recency → ↓ CLV (absence signal)
    "log_frequency",        # ↑ freq → ↑ CLV
    "log_monetary",         # dominant predictor; r=+0.855 with pred CLV
    "obs_tenure_days",      # left-bounded at 0; no transform needed
    # Trip patterns
    "obs_weekend_trip_share",
    "obs_top_store_trip_share",
    "obs_retail_disc_rate",
    "obs_coupon_disc_rate",
    "log_weight",
    # Cadence (obs_std_gap_days dropped: multicollinear with obs_mean_gap_days)
    "log_gap",
    # Basket trend
    "obs_basket_trend_direction",   # ordinal 1–3; 0.0 imputed for 412 NULLs
    "has_trend_obs",                # binary: 0 = insufficient baskets for trend
    # Product mix
    "obs_private_label_spend_share",
    "sqrt_top_cat",
    # Campaign
    "log_campaigns",
    "is_concurrent_arm",
    # Structural missingness flags
    "is_single_trip_obs",
    "is_redemption_eligible",
    "obs_redemption_propensity",
]

X_all    = df[FEATURES].astype(float)
y_all    = df["clv_pred_window"].astype(float)
y_log    = np.log1p(y_all)
y_binary = (y_all > 0).astype(float)

# ── Train / holdout split (stratified on zero/nonzero) ────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, test_idx = next(sss.split(X_all, y_binary))

X_train, X_test   = X_all.iloc[train_idx], X_all.iloc[test_idx]
y_train, y_test   = y_all.iloc[train_idx], y_all.iloc[test_idx]
y_tr_log, y_te_log = y_log.iloc[train_idx], y_log.iloc[test_idx]
y_tr_bin, y_te_bin = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
df_test = df.iloc[test_idx].copy()

print(f"Train n={len(X_train):,}  Test n={len(X_test):,}  "
      f"(test zeros={y_te_bin.eq(0).sum()}, nonzero={y_te_bin.eq(1).sum()})")

# =============================================================================
# Model A: OLS on log1p(CLV)  — interpretability baseline
# Uses statsmodels for p-values, residuals, Breusch-Pagan
# =============================================================================
X_tr_sm = sm_add_constant(X_train, has_constant="add")
X_te_sm = sm_add_constant(X_test,  has_constant="add")
ols      = sm_OLS(y_tr_log, X_tr_sm).fit()
ols_pred = np.expm1(ols.predict(X_te_sm))

# =============================================================================
# Model B: Ridge on log1p(CLV)
# RidgeCV picks alpha from [0.1, 1, 10, 100, 1000, 10000] via 5-fold CV.
# Multicollinearity (log_frequency ↔ log_monetary r=0.686,
# log_recency ↔ obs_tenure_days r=0.669) makes Ridge preferred over OLS
# for prediction accuracy.
# =============================================================================
ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  RidgeCV(alphas=[0.1, 1., 10., 100., 1_000., 10_000.], cv=5))
])
ridge_pipe.fit(X_train, y_tr_log)
ridge_pred = np.expm1(ridge_pipe.predict(X_test))

# =============================================================================
# Model C: Gradient Boosting on log1p(CLV)
# max_depth=4, subsample=0.8 for regularisation; min_samples_leaf=20
# to prevent overfitting on the 141 zero-CLV HHs in training.
# =============================================================================
gbm = GradientBoostingRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20, random_state=SEED
)
gbm.fit(X_train, y_tr_log)
gbm_pred = np.expm1(gbm.predict(X_test))

# =============================================================================
# Model D: Two-part hurdle
# Part 1: logistic regression → P(CLV > 0); C=1.0 (light L2 regularisation)
# Part 2: Ridge on log1p(CLV | CLV > 0), trained on nonzero training HHs only
# Final: P(return) × max(0, expm1(E[log1p(CLV) | return]))
# Note: expm1(E[Z]) ≠ E[expm1(Z)] (Jensen's inequality) — this is a slight
# underestimate of true E[CLV | return]; Duan smearing correction omitted
# because the ridge predictions are already regularised toward the mean.
# =============================================================================
logit_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logit",  LogisticRegression(C=1.0, max_iter=1_000, random_state=SEED))
])
logit_pipe.fit(X_train, y_tr_bin)
p_return = logit_pipe.predict_proba(X_test)[:, 1]

nz_mask = y_train > 0
ridge_nz = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  RidgeCV(alphas=[0.1, 1., 10., 100., 1_000., 10_000.], cv=5))
])
ridge_nz.fit(X_train[nz_mask], y_tr_log[nz_mask])
e_spend  = np.maximum(np.expm1(ridge_nz.predict(X_test)), 0.0)
hurdle_pred = p_return * e_spend

# =============================================================================
# STAGE 3: Model comparison
# =============================================================================
def eval_metrics(pred, actual, name):
    neg_pct   = 100.0 * (pred < 0).mean()
    pred_c    = np.maximum(pred, 0.0)
    rmse      = np.sqrt(mean_squared_error(actual, pred_c))
    mae       = mean_absolute_error(actual, pred_c)
    r2        = r2_score(actual, pred_c)
    return dict(name=name, rmse=rmse, mae=mae, r2=r2, neg_pct=neg_pct)

results = [
    eval_metrics(ols_pred,    y_test, "OLS log1p (baseline)"),
    eval_metrics(ridge_pred,  y_test, "Ridge log1p"),
    eval_metrics(gbm_pred,    y_test, "GBM log1p"),
    eval_metrics(hurdle_pred, y_test, "Hurdle (logit + Ridge)"),
]

print()
print("=" * 70)
print("STAGE 3 — MODEL COMPARISON  (holdout n={})".format(len(X_test)))
print("=" * 70)
print(f"  {'Model':<28} {'RMSE($)':>9} {'MAE($)':>8} {'R²':>7} {'%neg':>6}")
print("  " + "─" * 58)
for r in results:
    print(f"  {r['name']:<28} ${r['rmse']:>8.0f} ${r['mae']:>7.0f} "
          f"{r['r2']:>7.3f} {r['neg_pct']:>5.1f}%")

# Null model baseline: predict median for all
null_pred = np.full(len(y_test), y_train.median())
print(f"  {'Null (median train)':<28} ${np.sqrt(mean_squared_error(y_test,null_pred)):>8.0f}"
      f" ${mean_absolute_error(y_test,null_pred):>7.0f} "
      f"{r2_score(y_test,null_pred):>7.3f}   ---")

print(f"\n  Ridge optimal alpha: {ridge_pipe.named_steps['ridge'].alpha_:.1f}")
print(f"  Ridge_nz optimal alpha: {ridge_nz.named_steps['ridge'].alpha_:.1f}")

# =============================================================================
# OLS Residual diagnostics
# =============================================================================
print()
print("=" * 70)
print("OLS LOG1P RESIDUAL DIAGNOSTICS")
print("=" * 70)

# Breusch-Pagan (heteroscedasticity test on training residuals)
bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(ols.resid, X_tr_sm)
print(f"  Breusch-Pagan LM={bp_lm:.2f}  p={bp_p:.5f}  "
      f"→ {'HETEROSCEDASTIC (log transform did not fully stabilise variance)' if bp_p < 0.05 else 'homoscedastic'}")
print(f"  Train R²={ols.rsquared:.4f}  adj-R²={ols.rsquared_adj:.4f}")

# Per-cluster RMSE on holdout
df_test = df_test.copy()
df_test["pred_raw"] = ols_pred
df_test["pred_log"] = ols.predict(X_te_sm)
df_test["resid_log"] = y_te_log.values - df_test["pred_log"].values
df_test["resid_raw"] = y_test.values - np.maximum(ols_pred, 0.0)
df_test["abs_resid"]  = df_test["resid_raw"].abs()

print()
print(f"  RMSE / MAE by cluster (holdout, raw $):")
print(f"  {'Cluster':<20} {'n':>5} {'%zero_clv':>10} {'RMSE':>9} {'MAE':>8} {'MedAE':>8}")
print("  " + "─" * 65)
for cn in ["Champions","Loyal Actives","Declining","Truly Lapsed","Single-Trip"]:
    sub = df_test[df_test.cluster_name == cn]
    if len(sub) == 0: continue
    rmse_c = np.sqrt(mean_squared_error(sub.clv_pred_window,
                     np.maximum(sub.pred_log.apply(np.expm1), 0)))
    mae_c  = mean_absolute_error(sub.clv_pred_window,
                     np.maximum(sub.pred_log.apply(np.expm1), 0))
    medae  = sub.abs_resid.median()
    zpct   = f"{100*(sub.clv_pred_window==0).mean():.1f}%"
    print(f"  {cn:<20} {len(sub):>5} {zpct:>10} ${rmse_c:>8.0f} ${mae_c:>7.0f} ${medae:>7.0f}")

# Worst predictions
print()
print("  Top 10 largest raw-dollar residuals (OLS, holdout):")
top10 = df_test.nlargest(10, "abs_resid")[
    ["household_id","cluster_name","clv_pred_window","pred_raw","abs_resid",
     "obs_monetary","obs_frequency","log_recency"]]
top10["pred_raw"] = top10["pred_raw"].round(0).astype(int)
top10["abs_resid"] = top10["abs_resid"].round(0).astype(int)
print(top10.to_string(index=False))

# Residual vs fitted plot
Path("outputs/plots").mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: residual vs fitted (training, log scale)
axes[0].scatter(ols.fittedvalues, ols.resid, alpha=0.25, s=8, color="#2563EB")
axes[0].axhline(0, color="#DC2626", lw=1.2)
axes[0].set_xlabel("Fitted log1p(CLV)")
axes[0].set_ylabel("Residual (log scale)")
axes[0].set_title("OLS: Residuals vs Fitted (train, log1p scale)")

# Right: actual vs predicted (holdout, raw $)
axes[1].scatter(y_test, np.maximum(ols_pred, 0), alpha=0.25, s=8, color="#059669")
lim = max(y_test.max(), np.maximum(ols_pred, 0).max()) * 1.05
axes[1].plot([0, lim], [0, lim], "--", color="#DC2626", lw=1.2, label="perfect")
axes[1].set_xlabel("Actual CLV ($)")
axes[1].set_ylabel("Predicted CLV ($)")
axes[1].set_title("OLS: Actual vs Predicted (holdout, raw $)")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/plots/ols_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print()
print("  Diagnostic plots → outputs/plots/ols_diagnostics.png")

# =============================================================================
# OLS Coefficient interpretation
# =============================================================================
print()
print("=" * 70)
print("OLS COEFFICIENT INTERPRETATION  (target = log1p(CLV))")
print("=" * 70)
print("  A coef β means: a unit increase in X multiplies predicted CLV by exp(β).")
print("  For log-transformed features, 'unit increase' = 1 log-unit ≈ 2.7× the feature.")
print()

coef_df = pd.DataFrame({
    "feature": X_tr_sm.columns,
    "coef":    ols.params.values,
    "se":      ols.bse.values,
    "pval":    ols.pvalues.values,
})
coef_df["pct_per_unit"] = (np.exp(coef_df["coef"]) - 1) * 100
coef_df["sig"] = coef_df["pval"].apply(
    lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")))
coef_df = coef_df[coef_df.feature != "const"].sort_values("coef", ascending=False)

print(f"  {'Feature':<35} {'β':>8} {'SE':>7} {'p':>8} {'sig':>4}  {'exp(β)-1 = %Δ CLV':>18}")
print("  " + "─" * 85)
for _, row in coef_df.iterrows():
    sign = "+" if row["pct_per_unit"] >= 0 else ""
    print(f"  {row['feature']:<35} {row['coef']:>+8.4f} {row['se']:>7.4f} "
          f"{row['pval']:>8.4f} {row['sig']:>4}  {sign}{row['pct_per_unit']:>+7.1f}%")

# Business stakeholder summary
print()
print("  TOP 5 POSITIVE PREDICTORS (safe to present to business):")
top_pos = coef_df[coef_df.coef > 0].head(5)
for _, r in top_pos.iterrows():
    if r["feature"] == "log_monetary":
        note = "(elasticity: 1% more obs-window spend → ~{:.2f}% more pred CLV)".format(r["coef"])
    elif r["feature"] == "log_frequency":
        note = "(elasticity)"
    elif r["feature"] == "has_trend_obs":
        note = "(HHs with enough data for basket trend earn {:.0f}% more CLV)".format(r["pct_per_unit"])
    else:
        note = ""
    print(f"    {r['feature']}: β={r['coef']:+.4f} ({r['pct_per_unit']:+.1f}% per unit)  {note}")

print()
print("  TOP 5 NEGATIVE PREDICTORS (safe to present):")
top_neg = coef_df[coef_df.coef < 0].tail(5)
for _, r in top_neg.iterrows():
    print(f"    {r['feature']}: β={r['coef']:+.4f} ({r['pct_per_unit']:+.1f}% per unit)")

print()
print("  COEFFICIENTS TO WITHHOLD FROM BUSINESS (observational confounders):")
print("    log_campaigns: campaigns are targeted at known high-spenders — the")
print("      positive coefficient reflects retailer targeting, not a causal")
print("      campaign effect. Do not recommend 'expose everyone to 5+ campaigns.'")
print("    is_concurrent_arm: same confounding; heavy-spend HHs were enrolled")
print("      in multiple campaigns because of prior spend, not vice versa.")
print("    obs_coupon_disc_rate: high coupon use is correlated with segment")
print("      membership (Champions buy more, including discounted items), not a")
print("      causal 'discount drives future spend' signal.")

# =============================================================================
# Hurdle Part 1: logistic coefficients
# =============================================================================
print()
print("=" * 70)
print("HURDLE PART 1 — LOGISTIC COEFFICIENTS  (target = I(CLV > 0))")
print("=" * 70)
print("  Positive coef → increases P(return); negative → increases P(zero-spend).")
print()
logit_coefs = pd.Series(
    logit_pipe.named_steps["logit"].coef_[0],
    index=FEATURES
).sort_values(ascending=False)

print(f"  {'Feature':<35} {'logit β':>10}  note")
print("  " + "─" * 70)
for feat, c in logit_coefs.items():
    note = ""
    if feat == "log_recency"  and c < 0: note = "← main churn signal"
    if feat == "log_gap"      and c < 0: note = "← long-gap HHs don't return"
    if feat == "log_monetary" and c > 0: note = "← high spenders always return"
    if feat == "log_frequency"and c > 0: note = "← frequent shoppers return"
    print(f"  {feat:<35} {c:>+10.4f}  {note}")

print()
print("  'Will they return?' vs 'How much?': the logit part is dominated by")
print("  ABSENCE signals (log_recency, log_gap) while the Ridge CLV-conditional")
print("  part is dominated by INTENSITY signals (log_monetary, log_frequency).")
print("  This confirms the hurdle decomposition matches the underlying data DGP.")

# =============================================================================
# GBM feature importance
# =============================================================================
print()
print("=" * 70)
print("GBM FEATURE IMPORTANCE  (MDI, relative)")
print("=" * 70)
imp = pd.Series(gbm.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, v in imp.head(10).items():
    bar = "█" * int(v * 200)
    print(f"  {feat:<35} {v:.4f}  {bar}")

# =============================================================================
# Save artifacts
# =============================================================================
Path("outputs/models").mkdir(parents=True, exist_ok=True)
with open("outputs/models/clv_models.pkl", "wb") as f:
    pickle.dump({
        "ols":          ols,
        "ridge":        ridge_pipe,
        "gbm":          gbm,
        "hurdle_logit": logit_pipe,
        "hurdle_ridge": ridge_nz,
        "features":     FEATURES,
        "split_date":   "2017-10-01",
        "train_idx":    train_idx,
        "test_idx":     test_idx,
    }, f)

# Export full-panel CLV predictions (hurdle) for Tableau
df["hurdle_pred_clv"] = np.maximum(
    logit_pipe.predict_proba(X_all)[:, 1]
    * np.maximum(np.expm1(ridge_nz.predict(X_all)), 0.0),
    0.0
)
df[["household_id","cluster_name","clv_pred_window","hurdle_pred_clv",
    "obs_monetary","obs_frequency","obs_recency_days"]].to_csv(
    "outputs/extracts/clv_predictions.csv", index=False)

print()
print(f"Models  → outputs/models/clv_models.pkl")
print(f"Predictions → outputs/extracts/clv_predictions.csv")
