-- =============================================================================
-- STRATA Household Feature Table
-- =============================================================================
-- Grain: household_id (one row per household, n = 2,469)
-- Upstream: v_transactions_clean, v_rfm_features, v_campaign_exposure,
--           products, coupon_redemptions
--
-- Each feature is annotated:
--   [C]  safe for Phase 2 clustering input
--   [R]  safe for Phase 3 CLV regression
--   [L]  leaks future information under a within-period temporal split
--        (e.g., train H1 weeks 1–26, predict H2 weeks 27–53)
--   [!]  closer to "marketing treatment received" than customer behaviour;
--        do not use in clustering — contaminates demographic validation
--
-- =============================================================================
-- TEMPORAL LEAKAGE NOTE
-- =============================================================================
-- Context A — Predict post-observation period (CLV, churn):
--   All features below are "past" relative to the target. No leakage.
--
-- Context B — Within-period split (train H1, predict H2):
--   Features that touch H2 data are leaks [L]. To adapt, add a WHERE clause
--   to the relevant CTE that confines it to the training window:
--     WHERE transaction_timestamp < TIMESTAMP '2017-07-01'   -- H1 cutoff
-- =============================================================================

CREATE OR REPLACE VIEW v_household_features AS

WITH

-- ── Analysis date (must match v_rfm_features) ────────────────────────────────
analysis_date AS (SELECT DATE '2018-01-08' AS dt),

-- =============================================================================
-- Block A: Store loyalty
-- COUNT(DISTINCT basket_id) OVER (...) is not supported in DuckDB; compute
-- store-level trip counts in a separate CTE before aggregating to household.
-- =============================================================================

store_trips AS (
    SELECT household_id, store_id,
           COUNT(DISTINCT basket_id) AS store_basket_count
    FROM v_transactions_clean
    GROUP BY household_id, store_id
),

hh_store AS (
    SELECT
        household_id,
        -- [C] [R] Share of trips at the single most-visited store.
        -- p25=0.562, p50=0.759, p95=1.0. Most households strongly store-loyal.
        ROUND(MAX(store_basket_count)::DOUBLE
              / NULLIF(SUM(store_basket_count), 0), 4) AS top_store_trip_share
    FROM store_trips
    GROUP BY household_id
),

-- =============================================================================
-- Block B: Transaction-grain household features
-- No products join needed here — all signals from v_transactions_clean cols.
-- [L Context B] Full-period aggregates; add WHERE transaction_timestamp < split
-- to confine to training window.
-- =============================================================================

hh_txn AS (
    SELECT
        t.household_id,

        -- [C] [R] [L] Days from first to last shopping day.
        -- Tenure=0 means exactly one shopping day (not NULL).
        DATEDIFF('day',
            MIN(t.transaction_timestamp)::DATE,
            MAX(t.transaction_timestamp)::DATE)            AS tenure_days,

        -- [C] [R] Share of shopping days on Saturday (DOW=6) or Sunday (DOW=0).
        -- Panel p50=0.321. DuckDB DAYOFWEEK: 0=Sunday, 6=Saturday (verified).
        ROUND(
            COUNT(DISTINCT t.transaction_timestamp::DATE)
                FILTER(WHERE DAYOFWEEK(t.transaction_timestamp) IN (0, 6))::DOUBLE
            / NULLIF(COUNT(DISTINCT t.transaction_timestamp::DATE), 0)
        , 4)                                               AS weekend_trip_share,

        -- [C] [R] Retailer-applied discount rate: retail_disc / shelf_price.
        -- Reflects product-category mix more than active deal-seeking.
        -- p5=7.3%, p50=14.6%, p95=25.3%.
        ROUND(SUM(t.retail_disc) / NULLIF(SUM(t.shelf_price), 0), 4)
                                                           AS retail_disc_rate,

        -- [R] [!] Coupon savings as share of sales_value.
        -- 41% of households have rate=0 (never used a coupon). Reflects who
        -- got targeted and responded — CLV regression only, not clustering.
        ROUND(
            (SUM(t.coupon_disc) + SUM(t.coupon_match_disc))
            / NULLIF(SUM(t.sales_value), 0)
        , 4)                                               AS coupon_disc_rate,

        -- [R] [L] [!] December vs November 2017 spend ratio minus 1.
        -- Both months fall in H2; NULL for ~603 households absent from either.
        -- Noisy for low-frequency households. CLV regression only.
        ROUND(
            SUM(t.net_revenue) FILTER(
                WHERE t.transaction_timestamp >= TIMESTAMP '2017-12-01'
                  AND t.transaction_timestamp <  TIMESTAMP '2018-01-01')
            / NULLIF(
                SUM(t.net_revenue) FILTER(
                    WHERE t.transaction_timestamp >= TIMESTAMP '2017-11-01'
                      AND t.transaction_timestamp <  TIMESTAMP '2017-12-01')
              , 0) - 1
        , 4)                                               AS mom_spend_ratio_dec_nov,

        -- [C] [R] Spend share on sold-by-weight products (is_weight_sold flag
        -- set in cleaning layer for product_ids 6534178/6534166/6544236).
        -- 1,342 of 2,469 HHs have zero; p75=7.2%, max=90.7%.
        -- High values flag HHs whose per-unit price signals are unreliable.
        ROUND(
            COALESCE(SUM(t.sales_value) FILTER(WHERE t.is_weight_sold), 0)
            / NULLIF(SUM(t.sales_value), 0)
        , 4)                                               AS weight_sold_spend_share

    FROM v_transactions_clean t
    GROUP BY t.household_id
),

-- =============================================================================
-- Block C: Inter-trip interval
-- Collapsed to unique shopping days per household before computing LAG because
-- 18.9% of consecutive basket pairs share a calendar date, which would
-- produce 0-day gaps that corrupt mean_gap and std_gap at basket grain.
-- After collapsing: p25=4.9d, p50=8.7d, p90=31.1d, mean=14.8d.
-- [L Context B] Add WHERE transaction_timestamp < split_date to trip_days.
-- =============================================================================

trip_days AS (
    SELECT household_id,
           transaction_timestamp::DATE AS trip_date
    FROM v_transactions_clean
    GROUP BY household_id, transaction_timestamp::DATE
),

trip_day_lagged AS (
    SELECT
        household_id,
        trip_date,
        LAG(trip_date) OVER (PARTITION BY household_id ORDER BY trip_date) AS prev_trip_date
    FROM trip_days
),

hh_intervals AS (
    SELECT
        household_id,

        -- [C] [R] [L] Mean days between consecutive shopping days.
        -- 34 single-day HHs → NULL (no gap to compute).
        -- [Gap-tail compression] NTILE(5) Q1 spans 19.8–272 days (14× range);
        -- gap_decile (NTILE 10) splits this tail for CLV models that need
        -- to distinguish near-lapsed from monthly shoppers. See final SELECT.
        ROUND(AVG(DATEDIFF('day', prev_trip_date, trip_date))
              FILTER(WHERE prev_trip_date IS NOT NULL), 2) AS mean_gap_days,

        -- [C] [R] [L] Std dev of inter-trip gap.
        -- Low = habitual cadence; high = erratic shopper. Orthogonal to mean.
        ROUND(STDDEV(DATEDIFF('day', prev_trip_date, trip_date))
              FILTER(WHERE prev_trip_date IS NOT NULL), 2) AS std_gap_days

    FROM trip_day_lagged
    GROUP BY household_id
),

-- =============================================================================
-- Block D: Basket-size trend
-- [L Context B] Both H1 and H2 baskets included; add WHERE week <= 26
-- to hh_basket_raw for an H1-only training-window version.
-- =============================================================================

hh_basket_raw AS (
    SELECT basket_id, household_id, week,
           SUM(net_revenue) AS basket_net_revenue
    FROM v_transactions_clean
    GROUP BY basket_id, household_id, week
),

hh_trend_raw AS (
    SELECT
        household_id,
        AVG(basket_net_revenue) FILTER(WHERE week <= 26) AS h1_avg,
        AVG(basket_net_revenue) FILTER(WHERE week >  26) AS h2_avg
    FROM hh_basket_raw
    GROUP BY household_id
    -- Require ≥3 baskets in each half for a meaningful trend.
    -- 290 of 2,469 HHs below threshold → NULL in final output.
    HAVING COUNT(*) FILTER(WHERE week <= 26) >= 3
       AND COUNT(*) FILTER(WHERE week >  26) >= 3
),

hh_trend AS (
    -- Two-CTE approach lets basket_trend_direction reference the computed ratio
    -- without repeating the expression (DuckDB cannot alias in the same SELECT).
    SELECT
        household_id,

        -- [R] [L] (H2_avg − H1_avg) / H1_avg.
        -- Panel p50 ≈ −0.002 (flat at the median).
        -- [Positive-tail compression] max=5.24 vs p95=0.985.
        -- See trend_score (NTILE 5) in final SELECT.
        ROUND((h2_avg - h1_avg) / NULLIF(h1_avg, 0), 4) AS basket_trend_ratio,

        -- [C] Ordinal: 1=declining (<−10%), 2=stable (±10%), 3=growing (>+10%).
        -- ±10% ≈ one SD of weekly basket-value variation; avoids flagging noise.
        -- Use this for clustering; trend_score for regression.
        CASE
            WHEN (h2_avg - h1_avg) / NULLIF(h1_avg, 0) IS NULL THEN NULL
            WHEN (h2_avg - h1_avg) / NULLIF(h1_avg, 0) < -0.10 THEN 1
            WHEN (h2_avg - h1_avg) / NULLIF(h1_avg, 0) >  0.10 THEN 3
            ELSE 2
        END                                              AS basket_trend_direction

    FROM hh_trend_raw
),

-- =============================================================================
-- Block E: Product-based features (requires join to products dimension)
-- COUPON/MISC ITEMS (105 products, $385,972 = 8.4% of total revenue) is
-- excluded from top_category ranking — gift-card / misc spend is not a
-- product-preference signal. It IS included in private-label denominator
-- (real spend that matters for share computation).
-- =============================================================================

category_spend AS (
    SELECT
        t.household_id,
        p.product_category,
        p.brand,
        SUM(t.sales_value) AS cat_brand_sales
    FROM v_transactions_clean t
    JOIN products p USING (product_id)
    GROUP BY t.household_id, p.product_category, p.brand
),

-- Collapse brand dimension for per-category ranking
cat_spend_agg AS (
    SELECT household_id, product_category,
           SUM(cat_brand_sales) AS cat_sales
    FROM category_spend
    GROUP BY household_id, product_category
),

-- Single top category per household, COUPON/MISC ITEMS excluded from ranking
top_category_cte AS (
    SELECT household_id, product_category AS top_category, cat_sales AS top_cat_sales
    FROM (
        SELECT
            household_id, product_category, cat_sales,
            ROW_NUMBER() OVER (PARTITION BY household_id ORDER BY cat_sales DESC) AS rn
        FROM cat_spend_agg
        WHERE product_category != 'COUPON/MISC ITEMS'
    ) ranked
    WHERE rn = 1
),

hh_product AS (
    SELECT
        cs.household_id,

        -- [C] [R] Private-label spend share. Two brand values in dataset:
        -- 'National' (85.04%) and 'Private' (14.96%).
        -- High share → price-sensitive, less brand-loyal.
        ROUND(
            COALESCE(SUM(cs.cat_brand_sales) FILTER(WHERE cs.brand = 'Private'), 0)
            / NULLIF(SUM(cs.cat_brand_sales), 0)
        , 4)                                               AS private_label_spend_share,

        -- [C] [R] Top-category spend / total non-COUPON spend.
        -- Low = diversified basket; high = concentrated buyer.
        ROUND(tc.top_cat_sales
              / NULLIF(SUM(cs.cat_brand_sales)
                       FILTER(WHERE cs.product_category != 'COUPON/MISC ITEMS'), 0)
        , 4)                                               AS top_category_spend_share,

        -- Top category name (label only; do not one-hot-encode for clustering —
        -- 303 categories produces a highly sparse input matrix).
        tc.top_category

    FROM category_spend cs
    LEFT JOIN top_category_cte tc USING (household_id)
    GROUP BY cs.household_id, tc.top_cat_sales, tc.top_category
),

-- =============================================================================
-- Block F: Campaign and coupon features
-- =============================================================================

hh_campaign AS (
    SELECT
        ce.household_id,
        BOOL_OR(ce.has_concurrent_exposure)  AS is_concurrent_arm,
        COUNT(DISTINCT ce.campaign_id)        AS n_campaigns_exposed,

        -- [R] [!] Redemptions per campaign exposed, single-arm HHs only.
        -- NULL for 875 concurrent-arm HHs — attribution impossible without Phase
        -- 11 IPW. Do not use in clustering:
        --   (1) 875 NULLs require imputation before clustering;
        --   (2) propensity = who got targeted and responded, not who they are;
        --       clustering on it creates marketing-assignment segments that
        --       contaminate demographic validation.
        -- Among single-arm HHs (n=1,405): 1,051 have propensity=0.
        CASE WHEN NOT BOOL_OR(ce.has_concurrent_exposure)
             THEN ROUND(
                 COUNT(DISTINCT cr.campaign_id)::DOUBLE
                 / NULLIF(COUNT(DISTINCT ce.campaign_id), 0)
             , 4)
             ELSE NULL
        END                                               AS redemption_propensity

    FROM v_campaign_exposure ce
    LEFT JOIN coupon_redemptions cr
        ON  cr.household_id = ce.household_id
        AND cr.campaign_id  = ce.campaign_id
    GROUP BY ce.household_id
),

-- =============================================================================
-- Block G: Assemble all CTEs at household grain
-- Base is v_rfm_features (2,469 HHs); all others are LEFT JOINed so that
-- households with no campaign exposure, no H2 baskets, etc., are retained.
-- =============================================================================

all_hh AS (
    SELECT
        r.household_id,
        r.has_demographics,
        r.recency_days,
        r.frequency,
        r.monetary,
        r.r_score,
        r.f_score,
        r.f_decile,
        r.m_score,
        r.rfm_code,
        tx.tenure_days,
        tx.weekend_trip_share,
        hs.top_store_trip_share,
        tx.retail_disc_rate,
        tx.coupon_disc_rate,
        tx.mom_spend_ratio_dec_nov,
        tx.weight_sold_spend_share,
        iv.mean_gap_days,
        iv.std_gap_days,
        bt.basket_trend_ratio,
        bt.basket_trend_direction,
        pr.private_label_spend_share,
        pr.top_category_spend_share,
        pr.top_category,
        hc.is_concurrent_arm,
        hc.n_campaigns_exposed,
        hc.redemption_propensity
    FROM v_rfm_features r
    LEFT JOIN hh_txn       tx USING (household_id)
    LEFT JOIN hh_store     hs USING (household_id)
    LEFT JOIN hh_intervals iv USING (household_id)
    LEFT JOIN hh_trend     bt USING (household_id)
    LEFT JOIN hh_product   pr USING (household_id)
    LEFT JOIN hh_campaign  hc USING (household_id)
)

-- =============================================================================
-- Final output: NTILE scores appended over the fully assembled household table.
-- These windows must run after aggregation is complete — they cannot be placed
-- inside individual CTEs because the OVER () sees only that CTE's output rows,
-- not the full 2,469-HH panel.
-- =============================================================================
SELECT
    a.*,

    -- ── mean_gap quintile / decile ────────────────────────────────────────
    -- gap_score=5 → shortest mean gap (most frequent). gap_score=1 → longest.
    -- [Gap-tail compression] NTILE(5) Q1 spans 19.8–272 days (14× range).
    -- gap_decile decile 1 = 31–272d (near-lapsed); decile 2 = 20–31d (monthly).
    -- Use gap_decile in CLV models where distinguishing these groups matters.
    CASE WHEN mean_gap_days IS NULL THEN NULL
         ELSE NTILE(5)  OVER (ORDER BY mean_gap_days DESC NULLS LAST) END AS gap_score,
    CASE WHEN mean_gap_days IS NULL THEN NULL
         ELSE NTILE(10) OVER (ORDER BY mean_gap_days DESC NULLS LAST) END AS gap_decile,

    -- ── basket-trend quintile ─────────────────────────────────────────────
    -- trend_score=5 → strongest H1→H2 basket-size growth.
    -- [Positive-tail compression] max=5.24 vs p95=0.985.
    -- Use basket_trend_direction for clustering; trend_score for regression.
    -- NULL for 290 HHs with fewer than 3 baskets in either half.
    CASE WHEN basket_trend_ratio IS NULL THEN NULL
         ELSE NTILE(5)  OVER (ORDER BY basket_trend_ratio NULLS LAST) END AS trend_score

FROM all_hh a;
