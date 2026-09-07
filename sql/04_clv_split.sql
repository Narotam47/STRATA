-- =============================================================================
-- STRATA Phase 3 — CLV train/test split
-- =============================================================================
-- Grain: household_id (one row per household)
-- Split date: 2017-10-01  (timestamp boundary, exclusive for obs, inclusive for pred)
-- Observation window: [2017-01-01, 2017-10-01)  — 39 full weeks + partial wk 40
-- Prediction window:  [2017-10-01, 2018-01-02)  — Q4 + Jan 1 blip (weeks 40–53)
--
-- SPLIT RATIONALE
-- ───────────────
-- The full panel spans exactly one year: 2017-01-01 → 2018-01-01 (365 days).
-- A 2017-10-01 split gives:
--   Obs window  9 months (273 days): enough for tenure, cadence, basket trend
--   Pred window 3 months (92 days):  captures Q4 / holiday season; long enough
--   that "zero spend" (5.7% of HHs, n=141) reflects genuine lapsing, not
--   truncation — median obs-window profile of zero-spend HHs (tenure 149d,
--   frequency 5 trips) is structurally consistent with the Truly Lapsed
--   segment rather than a random early-panel entrant.
--
-- Alternative splits tested:
--   2017-07-01  zero-pct= 2.1%  — 6-month pred window, zeros not meaningful
--   2017-09-01  zero-pct= 4.1%  — 4-month pred, acceptable but loses Q4
--   2017-11-01  zero-pct= 8.7%  — only 2-month pred window, too noisy
--
-- The 2017-10-01 split maximises prediction-window revenue ($1.19M vs $0.80M
-- at Nov split) while maintaining the clearest zero-spend interpretation.
--
-- FEATURE LEAK AUDIT (relative to 2017-10-01 split)
-- ───────────────────────────────────────────────────
-- Features from v_household_features that cannot be used directly as training
-- inputs because they encode data from [2017-10-01, 2018-01-02):
--
--   DROPPED (cannot be recomputed without prediction-window data):
--   ┌────────────────────────────┬───────────────────────────────────────────┐
--   │ mom_spend_ratio_dec_nov    │ Uses both November AND December spend;     │
--   │                            │ both months are entirely in the prediction │
--   │                            │ window. Cannot be approximated; drop.      │
--   ├────────────────────────────┼───────────────────────────────────────────┤
--   │ gap_score, gap_decile,     │ NTILE() percentiles computed over the full │
--   │ trend_score                │ 2,469-HH panel using full-period feature   │
--   │                            │ values. Recomputing NTILE over an obs-     │
--   │                            │ window subset produces different buckets.  │
--   │                            │ Use raw obs_mean_gap_days and              │
--   │                            │ obs_basket_trend_ratio in regression.      │
--   └────────────────────────────┴───────────────────────────────────────────┘
--
--   RECOMPUTED (all CTEs below use WHERE transaction_timestamp < '2017-10-01'):
--   ┌────────────────────────────┬───────────────────────────────────────────┐
--   │ recency_days               │ Was relative to 2018-01-08.               │
--   │                            │ Recomputed as days to split date.         │
--   ├────────────────────────────┼───────────────────────────────────────────┤
--   │ frequency, monetary        │ Full-period counts/sums.                  │
--   │ tenure_days                │ Full-period max – min date.               │
--   │ weekend_trip_share         │ All computed over obs window only.        │
--   │ top_store_trip_share       │                                           │
--   │ retail_disc_rate           │                                           │
--   │ coupon_disc_rate           │                                           │
--   │ weight_sold_spend_share    │                                           │
--   │ mean_gap_days, std_gap     │                                           │
--   │ private_label_spend_share  │                                           │
--   │ top_category_spend_share   │                                           │
--   ├────────────────────────────┼───────────────────────────────────────────┤
--   │ basket_trend_ratio /       │ v_household_features uses H1=weeks 1–26,  │
--   │ basket_trend_direction     │ H2=weeks 27–53. Weeks 40–53 fall in the   │
--   │                            │ prediction window → LEAK.                 │
--   │                            │ Recomputed with H1=weeks 1–19,            │
--   │                            │ H2=weeks 20–39 (excludes partial week 40  │
--   │                            │ which straddles the Oct 1 boundary).      │
--   ├────────────────────────────┼───────────────────────────────────────────┤
--   │ redemption_propensity      │ Full-period redemptions include Nov/Dec    │
--   │                            │ (41% of all redemptions fall post-split). │
--   │                            │ Recomputed joining on                     │
--   │                            │ redemption_date < '2017-10-01'.           │
--   └────────────────────────────┴───────────────────────────────────────────┘
--
--   SAFE (no obs/pred distinction needed):
--   ┌────────────────────────────┬───────────────────────────────────────────┐
--   │ is_concurrent_arm          │ Campaign enrollment is pre-assigned by     │
--   │ n_campaigns_exposed        │ retailer before observation; structural,   │
--   │                            │ not behavioural.                           │
--   └────────────────────────────┴───────────────────────────────────────────┘
--
-- REGRESSION SAMPLE
-- ─────────────────
-- Panel:   2,469 HHs total
-- Pred-only (no obs history): 23 HHs → EXCLUDED (NULL features; untreatable)
-- Obs-only (zero pred spend): 141 HHs → RETAINED with clv_pred_window = 0
-- Both windows:             2,305 HHs → RETAINED
-- Regression training n:    2,446 HHs (2,305 + 141)
--
-- Single-trip HHs: included. They were excluded from k-means because they have
-- no inter-trip gap, but they have real future spend to predict and their
-- obs-window features (frequency=1, tenure=0, large recency) are valid inputs.
--
-- Demographic bias: NOT a sampling decision for this regression. The 801-HH
-- demographic subsample is skewed 3.7× toward high spenders; using it as the
-- regression population would systematically mispredict 68% of the panel.
-- Demographics (age, income, household_size, kids_count) are carried as optional
-- columns and used only in a secondary model on the 801-HH subsample.
-- =============================================================================

CREATE OR REPLACE VIEW v_clv_split AS

WITH

split_cfg AS (
    SELECT
        TIMESTAMP '2017-10-01' AS split_ts,
        DATE '2017-10-01'      AS split_date
),

-- =============================================================================
-- Block A: Observation-window RFM
-- recency = days from last obs-window trip to the split date (not 2018-01-08)
-- =============================================================================
obs_rfm AS (
    SELECT
        t.household_id,
        DATEDIFF('day', MAX(t.transaction_timestamp)::DATE, c.split_date)
                                                           AS obs_recency_days,
        COUNT(DISTINCT t.basket_id)                        AS obs_frequency,
        ROUND(SUM(t.net_revenue), 2)                       AS obs_monetary,
        DATEDIFF('day',
            MIN(t.transaction_timestamp)::DATE,
            MAX(t.transaction_timestamp)::DATE)            AS obs_tenure_days
    FROM v_transactions_clean t
    CROSS JOIN split_cfg c
    WHERE t.transaction_timestamp < c.split_ts
    GROUP BY t.household_id, c.split_date
),

-- =============================================================================
-- Block B: Observation-window transaction features
-- =============================================================================
obs_txn AS (
    SELECT
        t.household_id,
        ROUND(
            COUNT(DISTINCT t.transaction_timestamp::DATE)
                FILTER(WHERE DAYOFWEEK(t.transaction_timestamp) IN (0, 6))::DOUBLE
            / NULLIF(COUNT(DISTINCT t.transaction_timestamp::DATE), 0)
        , 4) AS obs_weekend_trip_share,
        ROUND(SUM(t.retail_disc) / NULLIF(SUM(t.shelf_price), 0), 4)
             AS obs_retail_disc_rate,
        ROUND(
            (SUM(t.coupon_disc) + SUM(t.coupon_match_disc))
            / NULLIF(SUM(t.sales_value), 0)
        , 4) AS obs_coupon_disc_rate,
        ROUND(
            COALESCE(SUM(t.sales_value) FILTER(WHERE t.is_weight_sold), 0)
            / NULLIF(SUM(t.sales_value), 0)
        , 4) AS obs_weight_sold_spend_share
    FROM v_transactions_clean t
    CROSS JOIN split_cfg c
    WHERE t.transaction_timestamp < c.split_ts
    GROUP BY t.household_id
),

-- =============================================================================
-- Block C: Observation-window store loyalty
-- =============================================================================
obs_store_trips AS (
    SELECT household_id, store_id,
           COUNT(DISTINCT basket_id) AS store_basket_count
    FROM v_transactions_clean t
    CROSS JOIN split_cfg c
    WHERE t.transaction_timestamp < c.split_ts
    GROUP BY household_id, store_id
),
obs_store AS (
    SELECT household_id,
        ROUND(MAX(store_basket_count)::DOUBLE
              / NULLIF(SUM(store_basket_count), 0), 4) AS obs_top_store_trip_share
    FROM obs_store_trips GROUP BY household_id
),

-- =============================================================================
-- Block D: Observation-window inter-trip intervals
-- Collapse to (household_id, trip_date) grain before LAG — same same-day
-- basket issue applies within the obs window.
-- =============================================================================
obs_trip_days AS (
    SELECT household_id, transaction_timestamp::DATE AS trip_date
    FROM v_transactions_clean t
    CROSS JOIN split_cfg c
    WHERE t.transaction_timestamp < c.split_ts
    GROUP BY household_id, transaction_timestamp::DATE
),
obs_trip_lagged AS (
    SELECT household_id, trip_date,
        LAG(trip_date) OVER (PARTITION BY household_id ORDER BY trip_date) AS prev_trip_date
    FROM obs_trip_days
),
obs_intervals AS (
    SELECT household_id,
        ROUND(AVG(DATEDIFF('day', prev_trip_date, trip_date))
              FILTER(WHERE prev_trip_date IS NOT NULL), 2) AS obs_mean_gap_days,
        ROUND(STDDEV(DATEDIFF('day', prev_trip_date, trip_date))
              FILTER(WHERE prev_trip_date IS NOT NULL), 2) AS obs_std_gap_days
    FROM obs_trip_lagged GROUP BY household_id
),

-- =============================================================================
-- Block E: Observation-window basket trend
-- H1 = weeks 1–19  (Jan 1 – ~May 13)
-- H2 = weeks 20–39 (~May 14 – ~Sep 24)
-- Week 40 is excluded from both halves: it straddles the Oct 1 boundary
-- (week 40 spans 2017-09-25 to 2017-10-02) and its obs-window portion
-- (Sep 25–30) is too short to include cleanly in either half.
-- H1 ≈ 19 weeks, H2 ≈ 20 weeks — symmetric enough.
-- =============================================================================
obs_basket AS (
    SELECT basket_id, household_id, week,
           SUM(net_revenue) AS basket_revenue
    FROM v_transactions_clean t
    CROSS JOIN split_cfg c
    WHERE t.transaction_timestamp < c.split_ts
    GROUP BY basket_id, household_id, week
),
obs_trend_raw AS (
    SELECT household_id,
        AVG(basket_revenue) FILTER(WHERE week <= 19) AS obs_h1_avg,
        AVG(basket_revenue) FILTER(WHERE week BETWEEN 20 AND 39) AS obs_h2_avg
    FROM obs_basket GROUP BY household_id
    HAVING COUNT(*) FILTER(WHERE week <= 19) >= 3
       AND COUNT(*) FILTER(WHERE week BETWEEN 20 AND 39) >= 3
),
obs_trend AS (
    SELECT household_id,
        ROUND((obs_h2_avg - obs_h1_avg) / NULLIF(obs_h1_avg, 0), 4)
            AS obs_basket_trend_ratio,
        CASE
            WHEN (obs_h2_avg - obs_h1_avg) / NULLIF(obs_h1_avg, 0) IS NULL THEN NULL
            WHEN (obs_h2_avg - obs_h1_avg) / NULLIF(obs_h1_avg, 0) < -0.10 THEN 1
            WHEN (obs_h2_avg - obs_h1_avg) / NULLIF(obs_h1_avg, 0) >  0.10 THEN 3
            ELSE 2
        END AS obs_basket_trend_direction
    FROM obs_trend_raw
),

-- =============================================================================
-- Block F: Observation-window product features
-- COUPON/MISC ITEMS excluded from top_category ranking (same rule as
-- v_household_features).
-- =============================================================================
obs_cat_spend AS (
    SELECT t.household_id, p.product_category, p.brand,
           SUM(t.sales_value) AS cat_brand_sales
    FROM v_transactions_clean t
    JOIN products p USING (product_id)
    CROSS JOIN split_cfg c
    WHERE t.transaction_timestamp < c.split_ts
    GROUP BY t.household_id, p.product_category, p.brand
),
obs_cat_agg AS (
    SELECT household_id, product_category, SUM(cat_brand_sales) AS cat_sales
    FROM obs_cat_spend GROUP BY household_id, product_category
),
obs_top_cat AS (
    SELECT household_id, product_category AS top_category, cat_sales AS top_cat_sales
    FROM (
        SELECT household_id, product_category, cat_sales,
            ROW_NUMBER() OVER (PARTITION BY household_id ORDER BY cat_sales DESC) AS rn
        FROM obs_cat_agg
        WHERE product_category != 'COUPON/MISC ITEMS'
    ) WHERE rn = 1
),
obs_product AS (
    SELECT cs.household_id,
        ROUND(
            COALESCE(SUM(cs.cat_brand_sales) FILTER(WHERE cs.brand = 'Private'), 0)
            / NULLIF(SUM(cs.cat_brand_sales), 0)
        , 4) AS obs_private_label_spend_share,
        ROUND(tc.top_cat_sales
              / NULLIF(SUM(cs.cat_brand_sales)
                       FILTER(WHERE cs.product_category != 'COUPON/MISC ITEMS'), 0)
        , 4) AS obs_top_category_spend_share,
        tc.top_category AS obs_top_category
    FROM obs_cat_spend cs
    LEFT JOIN obs_top_cat tc USING (household_id)
    GROUP BY cs.household_id, tc.top_cat_sales, tc.top_category
),

-- =============================================================================
-- Block G: Campaign features
-- is_concurrent_arm and n_campaigns_exposed are safe — campaign enrollment is
-- a pre-assigned treatment, not a function of obs/pred window behavior.
-- redemption_propensity restricted to redemption_date < split_date:
-- 41% of all redemptions (n=870) fall in the prediction window (Nov 526, Dec 274)
-- and must be excluded. Obs-window redemptions: 1,232 across 316 HHs.
-- =============================================================================
obs_campaign AS (
    SELECT
        ce.household_id,
        BOOL_OR(ce.has_concurrent_exposure)   AS is_concurrent_arm,
        COUNT(DISTINCT ce.campaign_id)         AS n_campaigns_exposed,
        CASE WHEN NOT BOOL_OR(ce.has_concurrent_exposure)
             THEN ROUND(
                 COUNT(DISTINCT cr.campaign_id)::DOUBLE
                 / NULLIF(COUNT(DISTINCT ce.campaign_id), 0)
             , 4)
             ELSE NULL END                     AS obs_redemption_propensity
    FROM v_campaign_exposure ce
    LEFT JOIN coupon_redemptions cr
        ON  cr.household_id  = ce.household_id
        AND cr.campaign_id   = ce.campaign_id
        AND cr.redemption_date < (SELECT split_date FROM split_cfg)
    GROUP BY ce.household_id
),

-- =============================================================================
-- Prediction-window target
-- CLV = SUM(net_revenue) over [split_ts, end of data).
-- HHs not appearing here get clv_pred_window = 0 in the final SELECT.
-- =============================================================================
pred_labels AS (
    SELECT household_id,
           ROUND(SUM(net_revenue), 2) AS clv_pred_window
    FROM v_transactions_clean t
    CROSS JOIN split_cfg c
    WHERE t.transaction_timestamp >= c.split_ts
    GROUP BY household_id
),

-- =============================================================================
-- Assembly
-- Anchor: all 2,469 HHs that appear anywhere in the panel.
-- LEFT JOIN to obs blocks (23 pred-only HHs will have NULL obs features —
-- filter these out in Python/analysis before fitting the regression model).
-- COALESCE pred_labels to 0.0 for the 141 obs-only HHs (true zero CLV).
-- =============================================================================
all_hh AS (
    SELECT DISTINCT household_id FROM v_transactions_clean
)

SELECT
    a.household_id,

    -- ── Obs-window RFM ────────────────────────────────────────────────────────
    r.obs_recency_days,
    r.obs_frequency,
    r.obs_monetary,
    r.obs_tenure_days,

    -- ── Obs-window behavioral ─────────────────────────────────────────────────
    tx.obs_weekend_trip_share,
    st.obs_top_store_trip_share,
    tx.obs_retail_disc_rate,
    tx.obs_coupon_disc_rate,
    tx.obs_weight_sold_spend_share,
    iv.obs_mean_gap_days,
    iv.obs_std_gap_days,
    tr.obs_basket_trend_ratio,
    tr.obs_basket_trend_direction,
    pr.obs_private_label_spend_share,
    pr.obs_top_category_spend_share,
    pr.obs_top_category,

    -- ── Campaign (structural) ─────────────────────────────────────────────────
    ca.is_concurrent_arm,
    ca.n_campaigns_exposed,
    ca.obs_redemption_propensity,

    -- ── Demographics (secondary model only; NULL for 1,668 HHs) ──────────────
    h.age,
    h.income,
    h.household_size,
    h.kids_count,
    h.marital_status,

    -- ── Target ───────────────────────────────────────────────────────────────
    -- COALESCE: obs-only HHs (141) get 0.0 (true zero, not missing).
    -- Pred-only HHs (23) also get 0.0 here — exclude them in Python because
    -- their obs features are all NULL and they cannot be trained on.
    COALESCE(p.clv_pred_window, 0.0) AS clv_pred_window,

    -- Convenience: flag pred-only HHs so Python can filter them in one step
    (r.obs_recency_days IS NULL) AS is_pred_only

FROM all_hh a
LEFT JOIN obs_rfm        r  USING (household_id)
LEFT JOIN obs_txn        tx USING (household_id)
LEFT JOIN obs_store      st USING (household_id)
LEFT JOIN obs_intervals  iv USING (household_id)
LEFT JOIN obs_trend      tr USING (household_id)
LEFT JOIN obs_product    pr USING (household_id)
LEFT JOIN obs_campaign   ca USING (household_id)
LEFT JOIN pred_labels    p  USING (household_id)
LEFT JOIN v_households   h  USING (household_id);
