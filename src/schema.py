"""Single source of truth for the STRATA data contract.

Every other module derives from this file: the downloader reads FILENAME/BYTES,
the loader builds its DDL from COLUMNS, and the test suite asserts against
PRIMARY_KEY, FOREIGN_KEYS and EXPECTED_ROWS. Switching to a different upstream
variant of the Complete Journey data is a change to this file alone.

Source: the `completejourney_py` distribution of the 84.51 Complete Journey
study, pinned to a single commit (see SOURCE_SHA). Because the source is
pinned, row counts and known integrity exceptions below are exact, not
approximate -- any drift is a real change worth failing on.
"""

from dataclasses import dataclass

SOURCE_REPO = "cunningjames/completejourney_py"
SOURCE_SHA = "8c224984175492833e9d2f0bf5756a462ff0a4ba"
SOURCE_PATH = "completejourney_py/data"
SOURCE_URL = f"https://github.com/{SOURCE_REPO}/tree/{SOURCE_SHA}/{SOURCE_PATH}"

RAW_URL_TEMPLATE = (
    f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_SHA}/{SOURCE_PATH}/{{filename}}"
)


@dataclass(frozen=True)
class Table:
    """One physical table: where it comes from, and what it must look like."""

    name: str
    columns: dict[str, str]
    primary_key: tuple[str, ...] | None
    expected_rows: int
    expected_bytes: int

    @property
    def filename(self) -> str:
        return f"{self.name}.parquet"

    @property
    def url(self) -> str:
        return RAW_URL_TEMPLATE.format(filename=self.filename)


# Integer widths are chosen from the observed value ranges of the pinned
# snapshot, not inherited from Parquet's int64 default:
#   basket_id  max 41,481,282,915  -> BIGINT (required)
#   coupon_upc max 59,986,600,074  -> BIGINT (required)
#   product_id max 18,316,298      -> INTEGER
#   store_id   max 34,280          -> INTEGER
#   week       max 53              -> SMALLINT
#   campaign_id max 27             -> SMALLINT
# On the 20.9M-row promotions table this halves the width of every key column.
TABLES: tuple[Table, ...] = (
    Table(
        name="transactions",
        columns={
            "household_id": "INTEGER",
            "store_id": "INTEGER",
            "basket_id": "BIGINT",
            "product_id": "INTEGER",
            "quantity": "INTEGER",
            "sales_value": "DOUBLE",
            "retail_disc": "DOUBLE",
            "coupon_disc": "DOUBLE",
            "coupon_match_disc": "DOUBLE",
            "week": "SMALLINT",
            "transaction_timestamp": "TIMESTAMP",
        },
        primary_key=("basket_id", "product_id"),
        expected_rows=1_469_307,
        expected_bytes=13_192_645,
    ),
    Table(
        name="products",
        columns={
            "product_id": "INTEGER",
            "manufacturer_id": "INTEGER",
            "department": "VARCHAR",
            "brand": "VARCHAR",
            "product_category": "VARCHAR",
            "product_type": "VARCHAR",
            "package_size": "VARCHAR",
        },
        primary_key=("product_id",),
        expected_rows=92_331,
        expected_bytes=1_228_107,
    ),
    Table(
        name="demographics",
        columns={
            "household_id": "INTEGER",
            "age": "VARCHAR",
            "income": "VARCHAR",
            "home_ownership": "VARCHAR",
            "marital_status": "VARCHAR",
            "household_size": "VARCHAR",
            "household_comp": "VARCHAR",
            "kids_count": "VARCHAR",
        },
        primary_key=("household_id",),
        expected_rows=801,
        expected_bytes=9_479,
    ),
    Table(
        name="campaigns",
        columns={
            "campaign_id": "SMALLINT",
            "household_id": "INTEGER",
        },
        primary_key=("household_id", "campaign_id"),
        expected_rows=6_589,
        expected_bytes=16_544,
    ),
    Table(
        name="campaign_descriptions",
        columns={
            "campaign_id": "SMALLINT",
            "campaign_type": "VARCHAR",
            "start_date": "TIMESTAMP",
            "end_date": "TIMESTAMP",
        },
        primary_key=("campaign_id",),
        expected_rows=27,
        expected_bytes=1_955,
    ),
    Table(
        name="coupons",
        columns={
            "coupon_upc": "BIGINT",
            "product_id": "INTEGER",
            "campaign_id": "SMALLINT",
        },
        # A coupon UPC maps to many products within a campaign, and the same
        # (upc, product, campaign) triple recurs 1,623 times in the snapshot.
        # There is no natural key here, so none is asserted.
        primary_key=None,
        expected_rows=116_204,
        expected_bytes=424_871,
    ),
    Table(
        name="coupon_redemptions",
        columns={
            "household_id": "INTEGER",
            "coupon_upc": "BIGINT",
            "campaign_id": "SMALLINT",
            "redemption_date": "TIMESTAMP",
        },
        # 27 households redeemed the same coupon in the same campaign more than
        # once; redemption_date does not disambiguate them all.
        primary_key=None,
        expected_rows=2_102,
        expected_bytes=13_589,
    ),
    Table(
        name="promotions",
        columns={
            "product_id": "INTEGER",
            "store_id": "INTEGER",
            "display_location": "VARCHAR",
            "mailer_location": "VARCHAR",
            "week": "SMALLINT",
        },
        # A product can hold several display/mailer placements in one
        # store-week, so (product, store, week) is not unique.
        primary_key=None,
        expected_rows=20_940_529,
        expected_bytes=21_567_722,
    ),
)

TABLES_BY_NAME: dict[str, Table] = {t.name: t for t in TABLES}


@dataclass(frozen=True)
class ForeignKey:
    """A key relationship, plus the exceptions the source is known to contain.

    `known_orphan_rows` / `known_orphan_keys` pin the violations that exist in
    the pinned snapshot. Pinning rather than tolerating means the suite still
    fails if the violation set changes in either direction.
    """

    child: str
    child_column: str
    parent: str
    parent_column: str
    known_orphan_rows: int = 0
    known_orphan_keys: int = 0


FOREIGN_KEYS: tuple[ForeignKey, ...] = (
    # 17 product_ids are referenced by transactions but absent from the product
    # dimension -- an upstream gap in the study data, not a load defect.
    ForeignKey("transactions", "product_id", "products", "product_id", 4_836, 17),
    ForeignKey("coupons", "product_id", "products", "product_id", 16, 6),
    ForeignKey("promotions", "product_id", "products", "product_id"),
    ForeignKey("campaigns", "campaign_id", "campaign_descriptions", "campaign_id"),
    ForeignKey("coupon_redemptions", "campaign_id", "campaign_descriptions", "campaign_id"),
    ForeignKey("coupon_redemptions", "coupon_upc", "coupons", "coupon_upc"),
)

# Columns that must never be null, because every downstream join depends on them.
NOT_NULL_KEYS: dict[str, tuple[str, ...]] = {
    "transactions": ("household_id", "basket_id", "product_id", "store_id"),
    "products": ("product_id",),
    "demographics": ("household_id",),
    "campaigns": ("campaign_id", "household_id"),
    "campaign_descriptions": ("campaign_id",),
    "coupons": ("coupon_upc", "product_id", "campaign_id"),
    "coupon_redemptions": ("household_id", "coupon_upc", "campaign_id"),
    "promotions": ("product_id", "store_id"),
}

# Households appearing in transactions; only a subset carry demographics.
# This is a property of the study design, not a data defect -- segmentation
# work that joins demographics must account for the 32% coverage.
EXPECTED_HOUSEHOLDS = 2_469
EXPECTED_HOUSEHOLDS_WITH_DEMOGRAPHICS = 801
