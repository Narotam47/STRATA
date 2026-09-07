-- =============================================================================
-- STRATA Cleaning Layer — DuckDB views
-- =============================================================================
-- Chain: raw tables
--   -> v_transactions_base    (net_revenue + row-level flags, no rows dropped)
--   -> v_transactions_clean   (orphan product_ids excluded; safe to JOIN products)
--   -> v_households           (one row per household; has_demographics boolean)
--   -> v_campaign_exposure    (one row per household×campaign; concurrent flag)
--   -> v_coupon_redemptions   (passthrough — documented clean)
--
-- Every rule below is sourced from the confirmed data audit, not re-derived.
-- Dollar and row amounts are measured at SOURCE_SHA = 8c224984 and pinned in
-- src/schema.py. If those counts drift, the contract suite will fail first.
-- =============================================================================


-- =============================================================================
-- Stage 1: v_transactions_base
--
-- Adds derived columns to every transaction row. No rows are dropped here:
-- the zero-quantity coupon-adjustment lines must survive into this view so
-- that their coupon_disc amounts flow into basket-level net_revenue correctly.
-- =============================================================================

CREATE OR REPLACE VIEW v_transactions_base AS

WITH orphan_products AS (
    -- 17 product_ids that appear in transactions but have no row in products.
    -- Identified at pinned commit 8c224984; pinned in src/schema.py as
    -- known_orphan_keys = 17 with known_orphan_rows = 4,836.
    --
    -- Breakdown:
    --   Revenue-bearing (3 ids):  1093587 ($124.63), 926775 ($31.62),
    --                             316011 ($16.32)  =>  $172.57 total (0.004%)
    --   Zero-revenue   (14 ids):  5977100, 5978649, 5978648, 5978650, 5978656,
    --                             5978657, 5978659, 5126087, 5126088, 5126106,
    --                             5126107, 5993051, 5993054, 5993055
    --   The zero-revenue ids are coupon-adjustment lines (quantity=0, sales_value=0)
    --   whose orphan status is caused by those products never having been
    --   stocked; their coupon_disc amounts are still real and must be netted.
    SELECT product_id
    FROM transactions
    WHERE product_id NOT IN (SELECT product_id FROM products)
),

weight_sold_products AS (
    -- 3 product_ids where "quantity" is in grams or ounces, not units.
    -- Evidence: max quantities of 89,638 / 36,140 / 35,836 at ~$0.0025/unit.
    -- Basket-level revenue is valid (7.15% of total, $328,464).
    -- Per-unit price derived features (price-per-unit, volume discounts) are
    -- meaningless for these ids and must exclude them.
    SELECT unnest([6534178, 6534166, 6544236]) AS product_id
)

SELECT
    -- ── original columns ─────────────────────────────────────────────────────
    t.household_id,
    t.store_id,
    t.basket_id,
    t.product_id,
    t.quantity,
    t.sales_value,
    t.retail_disc,
    t.coupon_disc,
    t.coupon_match_disc,
    t.week,
    t.transaction_timestamp,

    -- ── Rule 1: net_revenue ──────────────────────────────────────────────────
    -- sales_value is already post-retail-discount (shelf_price = sales_value +
    -- retail_disc). Coupon discounts are NOT pre-netted: they are recorded as
    -- separate zero-quantity adjustment lines in the same basket.
    --
    -- Row-level net_revenue is negative on those adjustment lines (e.g., a
    -- line with sales_value=0, coupon_disc=2.00 produces net_revenue=-2.00).
    -- This is correct: SUM(net_revenue) at basket grain produces the right
    -- post-coupon basket value because the negative adjustment offsets the
    -- positive sales_value on the item the coupon was applied to.
    --
    -- Do NOT drop zero-quantity rows before aggregating net_revenue.
    t.sales_value - t.coupon_disc - t.coupon_match_disc AS net_revenue,

    -- Convenience: gross shelf price (informational, not used for CLV revenue).
    t.sales_value + t.retail_disc AS shelf_price,

    -- ── Rule 2: orphan product flag ──────────────────────────────────────────
    -- Rows flagged here are excluded in v_transactions_clean for any analysis
    -- that joins to the products dimension. They are kept here so that
    -- coupon_disc on the 14 zero-revenue orphan ids is still netted correctly
    -- at basket grain.
    (t.product_id IN (SELECT product_id FROM orphan_products)) AS is_orphan_product,

    -- ── Rule 3: sold-by-weight flag ──────────────────────────────────────────
    -- TRUE for the 3 product_ids where quantity is in grams/ounces.
    -- Basket-level revenue (sales_value, net_revenue) is valid regardless.
    -- Filter WHERE NOT is_weight_sold before computing price-per-unit, unit
    -- count metrics, or volume-discount features.
    (t.product_id IN (SELECT product_id FROM weight_sold_products)) AS is_weight_sold,

    -- ── Convenience flags (sourced from audit, not contract) ─────────────────

    -- Zero-quantity coupon adjustment line. These carry coupon_disc amounts
    -- that must be included in basket-level net_revenue. Exclude from:
    --   - item count metrics
    --   - trip-level quantity aggregations
    --   - per-unit analyses
    -- Include in:
    --   - basket-level net_revenue (the coupon_disc here is real spend reduction)
    --   - coupon modelling (these are where the discount is encoded)
    (t.quantity = 0) AS is_coupon_adjustment,

    -- Fully-discounted item: positive quantity but zero price at register.
    -- Revenue is zero; retain for unit-count metrics, exclude from price models.
    (t.quantity > 0 AND t.sales_value = 0) AS is_fully_discounted

FROM transactions t;


-- =============================================================================
-- Stage 2: v_transactions_clean
--
-- Drops the 17 orphan product_ids. Safe to LEFT/INNER JOIN to the products
-- dimension without producing missing-product nulls.
--
-- Excluded rows: 4,836 rows, $172.57 revenue (0.004% of $4,596,040 total).
-- The dollar amount is immaterial; the exclusion is for referential integrity,
-- not revenue materiality. The 14 zero-revenue orphan rows are coupon
-- adjustment lines whose coupon_disc amounts have already been subtracted from
-- their basket net_revenue in v_transactions_base before this filter is applied
-- to aggregate queries.
--
-- NOTE: is_weight_sold rows are NOT excluded here. Basket-level revenue for
-- weight-sold products ($328,464, 7.15% of total) is valid and required for
-- RFM and CLV. Filter WHERE NOT is_weight_sold only when computing per-unit
-- price features.
-- =============================================================================

CREATE OR REPLACE VIEW v_transactions_clean AS
SELECT *
FROM v_transactions_base
WHERE NOT is_orphan_product;


-- =============================================================================
-- Stage 3: v_households
--
-- One row per household that has appeared in transactions (n = 2,469).
-- The has_demographics boolean must travel through every downstream join.
-- Never inner-join the demographics table directly in analytical queries:
-- demographics covers only 801 of 2,469 transacting households (32.4%), and
-- that 32.4% skews heavily toward high-value customers (median spend $2,635
-- vs $711 for households without demographics — a 3.7× gap at the median).
-- An inner join silently discards 68% of households and produces an analysis
-- of the top third of the spend distribution, not the panel.
-- =============================================================================

CREATE OR REPLACE VIEW v_households AS
SELECT
    hh.household_id,

    -- ── Rule 4: has_demographics flag ────────────────────────────────────────
    -- Explicit boolean: every query that touches demographics must condition on
    -- this column, not silently subset via inner join.
    (d.household_id IS NOT NULL) AS has_demographics,

    -- Demographic columns — NULL for 1,668 households without demographics.
    -- All are categorical strings as loaded; ordinal encoding happens in Phase 2.
    d.age,
    d.income,
    d.home_ownership,
    d.marital_status,
    d.household_size,
    d.household_comp,
    d.kids_count

FROM (SELECT DISTINCT household_id FROM transactions) hh
LEFT JOIN demographics d USING (household_id);


-- =============================================================================
-- Stage 4: v_campaign_exposure
--
-- One row per (household_id, campaign_id) — the multi-arm structure, not a
-- binary exposed/unexposed flag.
--
-- Rationale: 56% of exposed households appear in more than one campaign
-- (1,282 of 1,559 distinct exposed households). 875 households are
-- concurrently exposed to multiple overlapping campaigns, generating 2,986
-- concurrent pairs. A binary treatment model misclassifies these households.
-- Phase 11's causal analysis must either:
--   (a) restrict to households in exactly one campaign with a never-exposed
--       control group, or
--   (b) model exposure as multi-arm and weight by propensity (IPW).
-- The n_concurrent_campaigns column supports both approaches.
--
-- Campaign window truncation:
--   3 campaigns started before the transaction window (24, 25, 26).
--   5 campaigns end after the transaction window  (15, 20, 21, 22, 23).
--   These are not excluded — use the window_truncation column to apply
--   censoring logic in Phase 11 rather than silently dropping them.
-- =============================================================================

CREATE OR REPLACE VIEW v_campaign_exposure AS

WITH exposure AS (
    SELECT
        c.household_id,
        c.campaign_id,
        cd.campaign_type,
        cd.start_date,
        cd.end_date,
        DATEDIFF('day', cd.start_date, cd.end_date) AS duration_days
    FROM campaigns c
    JOIN campaign_descriptions cd USING (campaign_id)
),

-- For each (household, campaign), count how many OTHER campaigns for the same
-- household have overlapping date windows.
-- Overlap condition: two intervals [s1,e1] and [s2,e2] overlap iff s1 <= e2
-- AND s2 <= e1. Self-join on campaign_id <> to exclude self-overlap.
concurrent_counts AS (
    SELECT
        a.household_id,
        a.campaign_id,
        COUNT(b.campaign_id) AS n_concurrent_campaigns
    FROM exposure a
    JOIN exposure b
        ON  a.household_id  = b.household_id
        AND a.campaign_id  <> b.campaign_id
        AND a.start_date   <= b.end_date
        AND b.start_date   <= a.end_date
    GROUP BY a.household_id, a.campaign_id
),

txn_window AS (
    SELECT
        MIN(transaction_timestamp) AS txn_start,
        MAX(transaction_timestamp) AS txn_end
    FROM transactions
)

SELECT
    e.household_id,
    e.campaign_id,
    e.campaign_type,
    e.start_date,
    e.end_date,
    e.duration_days,

    -- ── Rule 5: concurrent exposure flag ─────────────────────────────────────
    -- n_concurrent_campaigns > 0 means this household was enrolled in at least
    -- one other campaign with an overlapping date window at the same time.
    -- These households require IPW or restriction to a clean single-arm group;
    -- they cannot be treated as pure single-treatment units.
    COALESCE(cc.n_concurrent_campaigns, 0)          AS n_concurrent_campaigns,
    COALESCE(cc.n_concurrent_campaigns, 0) > 0      AS has_concurrent_exposure,

    -- ── Campaign window truncation ────────────────────────────────────────────
    -- Identifies campaigns that started before or end after the transaction
    -- window. Do not drop these rows; apply left/right censoring in Phase 11.
    CASE
        WHEN e.start_date < w.txn_start AND e.end_date > w.txn_end THEN 'both_truncated'
        WHEN e.start_date < w.txn_start                             THEN 'left_truncated'
        WHEN e.end_date   > w.txn_end                               THEN 'right_truncated'
        ELSE 'within_window'
    END AS window_truncation,

    -- Effective observation window for this campaign (clamped to txn window).
    GREATEST(e.start_date, w.txn_start)             AS effective_start,
    LEAST(e.end_date, w.txn_end)                    AS effective_end,
    DATEDIFF('day',
        GREATEST(e.start_date, w.txn_start),
        LEAST(e.end_date, w.txn_end))               AS effective_duration_days

FROM exposure e
LEFT JOIN concurrent_counts cc USING (household_id, campaign_id)
CROSS JOIN txn_window w;


-- =============================================================================
-- Stage 5: v_coupon_redemptions
--
-- Passthrough view — no cleaning applied.
--
-- The data audit confirmed:
--   (1) All 2,102 redemption rows have non-null household_id, coupon_upc,
--       campaign_id (established by the contract suite's NOT NULL assertions).
--   (2) All redeeming household_id × campaign_id pairs exist in the campaigns
--       table — zero orphans (0 rows with no matching campaign membership).
--   (3) All campaign_ids resolve to campaign_descriptions — zero orphans
--       (established by the contract suite's referential integrity assertions).
--   (4) All coupon_upcs resolve to the coupons table — zero orphans
--       (established by the contract suite).
--   (5) ALL 2,102 redemption_dates fall within their campaign's start_date /
--       end_date — zero out-of-window redemptions. Campaign dates can be used
--       as hard filters; no temporal buffer is needed.
--
-- Impact: the 27 duplicate (household, coupon_upc, campaign_id) groups
-- (identified in the contract suite) are genuine repeat redemptions by the
-- same household, not load artefacts. Count them as-is in redemption metrics.
-- =============================================================================

CREATE OR REPLACE VIEW v_coupon_redemptions AS
SELECT *
FROM coupon_redemptions;
