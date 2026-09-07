# STRATA — Customer Segmentation & Lifetime Value Analytics

## Overview

STRATA is a four-phase analytics system that answers a single business question for a grocery retailer's marketing team: which customer segments exist, how much will each household spend in the next quarter, and whether the retailer's campaign budget is going to the households who need it most. It takes raw point-of-sale transactions and household identifiers, segments the panel into behaviorally distinct groups, predicts individual-household Q4 CLV using a hurdle model, and tests whether the retailer's campaign exposure actually caused a spending increase after controlling for the fact that campaigns were sent to the retailer's best customers to begin with. The outputs are five flat CSVs feeding a Tableau Public dashboard, two statistical inference reports, and two serialised models — all rebuildable from raw data in a single `make` command.

## Dataset

The 84.51° Complete Journey study is one of the few public datasets that combines transaction-level detail (household ID, product ID, price, quantity, week) with repeat-purchase history, campaign exposure records, coupon redemption, and household demographics in a single panel — the combination needed to build both a CLV model and a quasi-experimental campaign estimate without synthetic augmentation. The distribution used here is [cunningjames/completejourney_py](https://github.com/cunningjames/completejourney_py), pinned to commit `8c22498`, which provides pre-validated Parquet files with consistent column names. Run `make data` to fetch the eight source tables (~35 MB) into `data/raw/`.

## Methodology

**Phase 1 — SQL Data Engineering.** Eight DuckDB views form a dependency chain from raw tables to analysis-ready features. The net revenue formula `(sales_value - retail_disc - coupon_disc - coupon_match_disc)` corrects for manufacturer and retailer discounts that inflate gross sales by roughly 9% on average. The cleaning layer identified two systematic data quality issues: 4,836 transaction rows reference product IDs with no entry in the products table (an upstream gap pinned in `src/schema.py` and asserted by the data contract suite), and 97 transactions record `units_sold = 0` with non-zero `quantity` on weight-sold items, requiring a filter to prevent division-by-zero in unit-price computations. The final clean transaction table contains 1,464,471 rows across 2,469 unique households.

**Phase 2 — K-Means Segmentation.** Four segments (k=4) were chosen over the geometric optimum of k=2 that silhouette and Calinski-Harabasz criteria both favoured. The k=2 solution collapsed the panel into a single high-frequency mass and a single low-frequency mass — two groups for which no differentiated campaign strategy exists. K=4 was the smallest k at which each cluster carried a distinct combination of recency, frequency, and monetary profile that maps to a different intervention type. Cluster stability was validated with bootstrap ARI of 0.927 across 200 resamples with 80% subsampling, confirming the partition is reproducible. The resulting four segments are Champions (high-value, high-frequency regulars), Loyal Actives (solid mid-tier repeat buyers), Declining (formerly active households whose inter-visit gap has widened), and Truly Lapsed (households with no activity in the final months of the observation window). The key finding in segmentation is that Declining and Truly Lapsed are statistically indistinguishable on predicted CLV — their bootstrap CIs overlap at [$125, $140] and [$87, $131] respectively — meaning any targeting decision that treats them identically at the CLV level is defensible. Achieved silhouette coefficient: 0.19. The 12 features entering the model after a collinearity filter (|r| > 0.80 drops `mean_gap_days` in favour of `frequency + tenure_days`) are: `frequency`, `monetary`, `recency_days`, `tenure_days`, `weekend_trip_share`, `top_store_trip_share`, `retail_disc_rate`, `weight_sold_spend_share`, `std_gap_days`, `basket_trend_direction`, `private_label_spend_share`, `top_category_spend_share`.

**Phase 3 — CLV Regression.** A two-part hurdle model predicts Q4 (Oct–Dec 2017) household spend using nine months of observation-window features: log-recency, log-frequency, log-monetary, tenure in days, basket size, store visit rate, and cluster membership dummies. The hurdle's first stage is a logistic classifier predicting whether a household spends any amount in Q4; the second stage is OLS predicting log-spend conditional on spending. Observation-window features were computed from Jan–Sep 2017 only, with no Q4 information in the feature matrix, providing a strict temporal holdout. Holdout RMSE is $299 at the household level (R² = 0.631). The hardest households to predict are Champions and Loyal Actives, with per-segment holdout RMSEs of $380 and $356 — worse than a naive mean prediction for those groups — because nine months of regular grocery shopping does not predict Q4 holiday spend. The finding with the most practical implication is a negative coefficient on log-frequency: conditional on recency and monetary, households that visit more frequently spend less per Q4 quarter than expected, likely because high-frequency buyers are shopping for replenishment and are less susceptible to holiday incremental spend.

**Phase 4 — Statistical Inference.** Four hypothesis tests establish that the segments are statistically, not just geometrically, distinct: Welch's ANOVA rejects equal CLV means across segments (F=21.4, η²_p = 0.280, p < 0.001), and Games-Howell post-hoc tests confirm all pairwise CLV gaps except Declining vs. Truly Lapsed (p = 0.166). M-score quintile is associated with segment membership at Cramér's V = 0.448 — strong enough to be informative but below V = 0.5, confirming that segmentation adds information beyond what the retailer's existing scoring already captures. The campaign quasi-experiment finds a naive ATE of +$173.69 (unadjusted), which collapses to −$84.26 under IPW weighting (95% CI: [−208, +2]) after matching on observable covariates. The 61% ESS collapse (from 684 to 267 effective treated households) reflects extreme self-selection: the retailer's campaigns were targeted at its already-best customers. The adjusted estimate is inconclusive — the CI includes zero — but the direction reversal from +$174 to −$84 confirms that the naive figure is almost entirely a targeting artefact. A prospective randomised design to detect a half-ATE effect (MDE = $42) would require n = 3,272 total (1,636 per arm) with an O'Brien-Fleming sequential design to control peeking-induced false positive rate inflation.

## Limitations

**One-year observation window.** CLV is predicted over a 90-day Q4 horizon rather than a true lifetime, because the dataset covers a single calendar year. This understates segment-level value differences for Champions, who would compound their advantage over multiple years, and overstates the Truly Lapsed deficit, because some lapsed households may reactivate after a longer dormancy period that falls outside the data.

**Demographic coverage concentrated in high-value households.** Demographics are available for 801 of 2,469 households (32%). The 801 households with demographic records spend 3.7× the panel median in the observation window, meaning any demographic analysis is a portrait of the retailer's best customers, not a representative cross-section. Declining and Truly Lapsed segments have demographic coverage below 5% — any demographic breakdown for those segments is based on fewer than 35 households and should not guide targeting decisions.

**Residual confounding in the campaign analysis.** The IPW propensity model uses only four observable covariates (log-monetary, log-frequency, recency, tenure, and cluster dummies). Unobserved drivers of campaign targeting — store manager discretion, geographic targeting, prior campaign history — remain as residual confounders. The effective sample size after weighting collapsed to approximately 267 treated and 300 control households, driving the wide CI and preventing a confident conclusion in either direction.

**Champions and Loyal Actives generate the largest CLV prediction errors.** Per-segment holdout RMSEs of $380 (Champions) and $356 (Loyal Actives) exceed the panel-average RMSE of $299 and exceed a naive mean prediction for those groups. The Q4 holiday quarter introduces structural demand shifts — gift purchasing, pantry loading, seasonal promotions — that are not visible in the nine-month observation window's regular shopping patterns. Predictions for these segments should be used for ranking rather than as absolute CLV estimates.

**Segment counts differ between cluster_assignments.csv and segment_summary.csv for Truly Lapsed (191 vs 175).** The 16-household gap is correct behavior: these households were active in the Jan–Sep 2017 observation window (and were assigned to Truly Lapsed by k-means) but had no Q4 activity at all, so `v_clv_split` contains no record for them and the CLV model produced no prediction. They appear in `household_scored.csv` as `is_pred_only=True` with `hurdle_pred_clv=NaN`. Segment-level statistics derived from `clv_predictions.csv` (including `segment_summary.csv`) will always show 175 for this segment.

## Reproduction

```bash
git clone <repo>
make data      # Fetches the eight source Parquet tables (~35 MB) from the pinned upstream commit
make ingest    # Loads tables into data/processed/strata.duckdb and runs the 58-assertion data contract suite
make test      # Re-runs the data contract in isolation (no re-ingestion)
make extracts  # Regenerates the five Tableau CSVs in outputs/extracts/
```

Running `make` without a target runs the full pipeline through `ingest` (data → load → contract). The pipeline halts if any contract assertion fails. Additional scripts in `src/` run independently: `python -m src.segment`, `python -m src.clv_regression`, `python -m src.hypothesis_tests`, `python -m src.campaign_quasi_experiment`, `python -m src.export_extracts`.

## Outputs

**Tableau extracts** (`outputs/extracts/`):

| File | Rows | Contents |
|---|---:|---|
| `segment_summary.csv` | 5 | Per-segment CLV stats, revenue share, bootstrap CIs, zero-spend rate |
| `household_scored.csv` | 2,469 | One row per household — obs spend, predicted CLV, segment label, demographics |
| `segment_demographic_crosstab.csv` | 97 | Long-format: segment × demographic variable × value with coverage rates |
| `campaign_results.csv` | 11 | IPW, AIPW, and naive effect estimates by segment |
| `retention_matrix.csv` | 10 | Cohort quarter × quarters-since-cohort retention rates |
| `cluster_assignments.csv` | 2,435 | K-means assignments for obs-window households (household_id, cluster_name, RFM scores) |
| `clv_predictions.csv` | 2,446 | Per-household CLV predictions with obs-window features and cluster labels |

**Reports** (`outputs/reports/`):
- `phase2_segmentation.md` — K-selection sweep table, k=4 rationale, cluster profiles, bootstrap ARI, central finding
- `phase4_hypothesis_tests.txt` — Welch ANOVA, Games-Howell, chi-square, and bootstrap CI results
- `phase4_campaign_quasi_experiment.txt` — Naive, IPW, AIPW estimates with SMD balance table and prospective design spec

**Models** (`outputs/models/`):
- `kmeans_k4.pkl` — Final k=4 cluster model (keys: `model`, `scaler`, `features`)
- `clv_models.pkl` — Serialised hurdle model (keys: `hurdle_logit`, `hurdle_ridge`, `ols`, `ridge`, `gbm`, `features`, `split_date`)

**Dashboard:** [STRATA on Tableau Public]([your published dashboard link])

---

### Project layout

```
sql/          Eight SQL view definitions. Portable to Snowflake/BigQuery without Python changes.
src/          Importable package. schema.py is the single source of truth for the data contract.
tests/        Data contract suite (58 assertions). Asserts row counts, types, null keys, FK integrity.
outputs/
  extracts/   Aggregated CSVs for Tableau. Committed — see outputs/extracts/README.md for why.
  models/     Serialised estimators. Binaries gitignored; rebuildable from src/.
  reports/    Written inference deliverables.
data/raw/     Immutable upstream input. Gitignored. Fetched by make data.
data/processed/  strata.duckdb. Gitignored. Rebuildable from make ingest.
```
