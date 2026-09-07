-- =============================================================================
-- STRATA RFM Feature Table
-- =============================================================================
-- Grain: household_id (one row per household, n = 2,469)
-- Upstream: v_transactions_clean from sql/01_cleaning_views.sql
--
-- This view does NOT recompute net_revenue. It calls SUM(net_revenue)
-- from the cleaning layer where the formula (sales_value - coupon_disc -
-- coupon_match_disc) and its coupon-adjustment-line treatment are already
-- encoded. Any change to the revenue definition belongs in 01_cleaning_views.sql.
-- =============================================================================

CREATE OR REPLACE VIEW v_rfm_features AS

WITH

-- ── Analysis date ────────────────────────────────────────────────────────────
-- Pinned to 2018-01-08: exactly 7 days after the last observed transaction
-- (2018-01-01, the final date in the pinned dataset at SOURCE_SHA 8c224984).
--
-- Why not CURRENT_DATE: RFM recency scores must be reproducible. CURRENT_DATE
-- would make every household's recency grow by one day per day, causing scores
-- to drift and making historical comparisons meaningless.
--
-- Why +7 days: standard convention for closed observation windows. It gives the
-- most recent household a non-zero baseline recency (7 days rather than 0),
-- which matters because recency=0 is undefined for the CLV models in Phase 3.
-- 7 days also represents one grocery shopping cycle for a weekly shopper,
-- which is the median trip interval in this panel.
analysis_date AS (
    SELECT DATE '2018-01-08' AS dt
),

-- ── Household-grain aggregates ───────────────────────────────────────────────
household_rfm_raw AS (
    SELECT
        t.household_id,

        -- Recency: calendar days between each household's most recent trip and
        -- the analysis date. Lower values = more recent = higher R score.
        -- Computed over all rows including coupon-adjustment lines (quantity=0):
        -- their transaction_timestamp matches the basket's purchase timestamp,
        -- so MAX(transaction_timestamp) is unaffected by including them.
        DATEDIFF('day', MAX(t.transaction_timestamp)::DATE, a.dt) AS recency_days,

        -- Frequency: distinct trip count. basket_id is the trip identifier —
        -- confirmed single-household (zero baskets shared across households)
        -- and single-point-in-time (zero baskets spanning multiple hours).
        -- Counting DISTINCT basket_id regardless of quantity=0 adjustment lines
        -- because those lines always co-occur with real purchase lines in the
        -- same basket; no basket consists solely of adjustment lines.
        COUNT(DISTINCT t.basket_id)                                AS frequency,

        -- Monetary: total net revenue per household over the observation window.
        --
        -- is_weight_sold rows (product_ids 6534178, 6534166, 6544236) are
        -- intentionally included. Basket-level revenue for sold-by-weight
        -- products is valid and accounts for 7.15% of total revenue across
        -- 1,127 households (avg $291 per affected household, up to 90.7% of one
        -- household's total spend). Excluding them would systematically understate
        -- monetary for heavy produce/deli buyers. Only per-unit price features
        -- (not net_revenue aggregates) should filter WHERE NOT is_weight_sold.
        --
        -- Coupon-adjustment lines (quantity=0) are also included: their negative
        -- net_revenue values (-coupon_disc) correctly offset the positive
        -- sales_value on the items they discount. SUM(net_revenue) at household
        -- grain equals true post-coupon spend.
        ROUND(SUM(t.net_revenue), 2)                               AS monetary

    FROM v_transactions_clean t
    CROSS JOIN analysis_date a
    GROUP BY t.household_id, a.dt
),

-- ── Quintile scoring ─────────────────────────────────────────────────────────
-- NTILE(5) applied independently to each dimension.
--
-- R scoring: ORDER BY recency_days DESC so that lower recency (more recent)
-- maps to higher score. R=5 = last shopped 7–8 days ago; R=1 = lapsed 33–367.
--
-- F scoring: ORDER BY frequency ASC. F=5 = 95–778 trips/year (2+/week);
-- F=1 = 1–16 trips/year (occasional shopper).
--
-- F NOTE — NTILE(5) is appropriate for segmentation but has two known issues:
--   (1) Tie-boundary ambiguity: ~94 households fall exactly at a quintile cut
--       (33 at the Q1/Q2 cut of 16 trips, 31 at Q2/Q3 of 32 trips, 18 at
--       Q3/Q4 of 55 trips, 12 at Q4/Q5 of 95 trips). NTILE() splits ties
--       arbitrarily by position in the sort order, meaning two households
--       with identical trip counts can receive different f_score values.
--       This affects ~3.8% of the panel. Use f_score for segmentation;
--       use f_decile (NTILE(10)) for precision ranking.
--   (2) Q5 compression: f_score=5 spans 95–778 trips/year. A household with
--       100 trips (2/week, committed regular) and one with 778 (15/week,
--       institutional-scale buyer) are not meaningfully equivalent for CLV
--       modelling. f_decile splits Q5 into two deciles (decile 9: 95–137,
--       decile 10: 137–778) for use when intra-heavy-shopper distinctions
--       matter. Decile 10 households (avg 220 trips/year) are extreme outliers
--       worth identifying separately in the Phase 3 CLV model.
--
-- M scoring: ORDER BY monetary ASC. M=5 = $2,986–$24,857; M=1 = $2–$354.
scored AS (
    SELECT
        household_id,
        recency_days,
        frequency,
        monetary,

        -- Recency: DESC so smaller days -> higher score
        NTILE(5)  OVER (ORDER BY recency_days DESC) AS r_score,

        -- Frequency quintile (primary; use for RFM code and segmentation)
        NTILE(5)  OVER (ORDER BY frequency)         AS f_score,

        -- Frequency decile (supplementary; use when intra-Q5 granularity matters)
        NTILE(10) OVER (ORDER BY frequency)         AS f_decile,

        -- Monetary
        NTILE(5)  OVER (ORDER BY monetary)          AS m_score
    FROM household_rfm_raw
)

-- ── Final output ─────────────────────────────────────────────────────────────
SELECT
    s.household_id,

    -- Demographic coverage flag from the cleaning layer.
    -- Carried through explicitly — never filter or inner-join on demographics
    -- without conditioning on this column first. Demographics cover only 801 of
    -- 2,469 households (32.4%), and that subset has 3.7× higher median spend
    -- than the unobserved majority.
    h.has_demographics,

    -- Raw RFM values
    s.recency_days,
    s.frequency,
    s.monetary,

    -- Scores (1–5, higher = better for all three)
    s.r_score,
    s.f_score,
    s.m_score,

    -- Supplementary: frequency decile (1–10) for intra-heavy-shopper analysis
    s.f_decile,

    -- RFM composite code (e.g. '533', '111', '555')
    -- Concatenation order is R-F-M by convention. Do not sort on this string
    -- directly — use r_score + f_score + m_score as separate integer columns
    -- for any numeric ranking.
    s.r_score::VARCHAR || s.f_score::VARCHAR || s.m_score::VARCHAR AS rfm_code

FROM scored s
JOIN v_households h USING (household_id);
