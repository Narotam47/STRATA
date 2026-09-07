"""Data contract for the ingested DuckDB database.

These are not unit tests of Python functions -- they assert properties of the
loaded data, and they are what stands between a successful-looking ingestion
and a quietly wrong one. `make ingest` runs them and fails if any break.

Every expected value is pinned in src/schema.py against a fixed upstream
commit, so a failure means something genuinely changed.
"""

import duckdb
import pytest

from src.schema import (
    EXPECTED_HOUSEHOLDS,
    EXPECTED_HOUSEHOLDS_WITH_DEMOGRAPHICS,
    FOREIGN_KEYS,
    NOT_NULL_KEYS,
    TABLES,
    TABLES_BY_NAME,
)

TABLE_IDS = [t.name for t in TABLES]


# --------------------------------------------------------------------------
# 1. Row counts survive the load
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table", TABLES, ids=TABLE_IDS)
def test_row_count_matches_source_file(con, raw_dir, table):
    """The table holds exactly as many rows as its source Parquet file.

    Counted straight off the file rather than trusting the loader, so a
    filtered or truncated CREATE TABLE is caught.
    """
    source = raw_dir / table.filename
    source_rows = duckdb.sql(
        f"SELECT COUNT(*) FROM read_parquet('{source.as_posix()}')"
    ).fetchone()[0]
    loaded_rows = con.execute(f"SELECT COUNT(*) FROM {table.name}").fetchone()[0]
    assert loaded_rows == source_rows, (
        f"{table.name}: loaded {loaded_rows:,} rows but source file has {source_rows:,}"
    )


@pytest.mark.parametrize("table", TABLES, ids=TABLE_IDS)
def test_row_count_matches_pinned_manifest(con, table):
    """Row count matches the pinned snapshot, catching upstream drift."""
    actual = con.execute(f"SELECT COUNT(*) FROM {table.name}").fetchone()[0]
    assert actual == table.expected_rows, (
        f"{table.name}: {actual:,} rows, manifest pins {table.expected_rows:,}. "
        "If the upstream data legitimately changed, update src/schema.py."
    )


# --------------------------------------------------------------------------
# 2. Declared types actually landed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table", TABLES, ids=TABLE_IDS)
def test_column_types_match_declaration(con, table):
    """Column names and types match src/schema.py exactly.

    Guards the failure mode where an integer key arrives as DOUBLE (which is
    what happens the moment a nullable int column is routed through pandas) or
    a numeric column arrives as VARCHAR. Either silently breaks joins and
    aggregations rather than raising.
    """
    described = con.execute(f"DESCRIBE {table.name}").fetchall()
    actual = {row[0]: row[1] for row in described}
    assert actual == dict(table.columns), (
        f"{table.name} schema drift.\n"
        f"  expected: {dict(table.columns)}\n"
        f"  actual:   {actual}"
    )


@pytest.mark.parametrize("table", TABLES, ids=TABLE_IDS)
def test_no_numeric_column_stored_as_text(con, table):
    """No column declared numeric is sitting in the database as VARCHAR."""
    numeric_types = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "DOUBLE", "DECIMAL"}
    declared_numeric = {c for c, t in table.columns.items() if t in numeric_types}
    described = {row[0]: row[1] for row in con.execute(f"DESCRIBE {table.name}").fetchall()}
    text_columns = {c for c in declared_numeric if described.get(c) == "VARCHAR"}
    assert not text_columns, f"{table.name}: numeric columns stored as text: {sorted(text_columns)}"


# --------------------------------------------------------------------------
# 3. Join keys are never null
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table_name", sorted(NOT_NULL_KEYS), ids=sorted(NOT_NULL_KEYS))
def test_join_keys_have_no_nulls(con, table_name):
    """Every key column used in a downstream join is fully populated."""
    offenders = {}
    for column in NOT_NULL_KEYS[table_name]:
        nulls = con.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE {column} IS NULL"
        ).fetchone()[0]
        if nulls:
            offenders[column] = nulls
    assert not offenders, f"{table_name}: null join keys {offenders}"


# --------------------------------------------------------------------------
# 4. Primary keys are unique where a natural key exists
# --------------------------------------------------------------------------

PK_TABLES = [t for t in TABLES if t.primary_key]


@pytest.mark.parametrize("table", PK_TABLES, ids=[t.name for t in PK_TABLES])
def test_primary_key_is_unique(con, table):
    """No duplicate primary keys.

    Only asserted for the five tables with a genuine natural key. coupons,
    coupon_redemptions and promotions legitimately contain repeated key
    combinations -- see the notes in src/schema.py -- so asserting uniqueness
    there would encode a false expectation.
    """
    key = ", ".join(table.primary_key)
    duplicate_groups = con.execute(
        f"SELECT COUNT(*) FROM (SELECT {key} FROM {table.name} "
        f"GROUP BY {key} HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert duplicate_groups == 0, (
        f"{table.name}: {duplicate_groups:,} duplicated values of ({key})"
    )


def test_tables_without_primary_key_are_deliberate():
    """The three key-less tables are exactly the ones documented as such.

    Stops someone quietly dropping a primary_key declaration to make a
    uniqueness failure go away.
    """
    keyless = {t.name for t in TABLES if not t.primary_key}
    assert keyless == {"coupons", "coupon_redemptions", "promotions"}


# --------------------------------------------------------------------------
# 5. Referential integrity, with the source's known gaps pinned
# --------------------------------------------------------------------------

FK_IDS = [f"{fk.child}.{fk.child_column}->{fk.parent}" for fk in FOREIGN_KEYS]


@pytest.mark.parametrize("fk", FOREIGN_KEYS, ids=FK_IDS)
def test_referential_integrity(con, fk):
    """Child keys resolve against the parent table.

    Three of these relationships are clean. Two are not: the study data
    references 17 product_ids from transactions and 6 from coupons that never
    appear in the product dimension. Those counts are pinned rather than
    tolerated, so the suite still fails if the gap grows, shrinks, or moves.
    """
    orphan_rows, orphan_keys = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT c.{fk.child_column}) "
        f"FROM {fk.child} c "
        f"WHERE c.{fk.child_column} NOT IN (SELECT {fk.parent_column} FROM {fk.parent})"
    ).fetchone()

    assert (orphan_rows, orphan_keys) == (fk.known_orphan_rows, fk.known_orphan_keys), (
        f"{fk.child}.{fk.child_column} -> {fk.parent}.{fk.parent_column}: "
        f"found {orphan_rows:,} orphan rows across {orphan_keys:,} distinct keys, "
        f"expected {fk.known_orphan_rows:,} / {fk.known_orphan_keys:,}"
    )


# --------------------------------------------------------------------------
# 6. Coverage facts that downstream modelling depends on
# --------------------------------------------------------------------------


def test_household_coverage(con):
    """Household counts match the study design.

    Demographics cover only 801 of 2,469 transacting households. Any
    segmentation that inner-joins demographics silently discards two thirds of
    the panel, so the ratio is asserted rather than assumed.
    """
    transacting = con.execute("SELECT COUNT(DISTINCT household_id) FROM transactions").fetchone()[0]
    with_demographics = con.execute("SELECT COUNT(*) FROM demographics").fetchone()[0]
    assert transacting == EXPECTED_HOUSEHOLDS
    assert with_demographics == EXPECTED_HOUSEHOLDS_WITH_DEMOGRAPHICS


def test_demographics_are_a_subset_of_transacting_households(con):
    """No demographic record describes a household that never shopped."""
    orphans = con.execute(
        "SELECT COUNT(*) FROM demographics d "
        "WHERE d.household_id NOT IN (SELECT household_id FROM transactions)"
    ).fetchone()[0]
    assert orphans == 0


def test_transactions_span_one_year(con):
    """The transaction window is the single year the study documents."""
    lo, hi = con.execute(
        "SELECT MIN(transaction_timestamp), MAX(transaction_timestamp) FROM transactions"
    ).fetchone()
    assert lo.year == 2017 and hi.year == 2018
    assert (hi - lo).days <= 366


def test_sales_values_are_finite_and_mostly_positive(con):
    """Money columns are usable: no NULLs, and returns stay a small minority."""
    nulls, negatives, total = con.execute(
        "SELECT COUNT(*) FILTER (WHERE sales_value IS NULL), "
        "       COUNT(*) FILTER (WHERE sales_value < 0), "
        "       COUNT(*) FROM transactions"
    ).fetchone()
    assert nulls == 0, f"{nulls:,} null sales_value rows"
    assert negatives / total < 0.01, f"{negatives:,}/{total:,} negative sales_value rows"


@pytest.mark.parametrize(
    "table_name,column",
    [("campaign_descriptions", "start_date"), ("transactions", "transaction_timestamp")],
)
def test_expected_tables_exist(con, table_name, column):
    """Sanity check that the named tables and date columns are present."""
    assert table_name in TABLES_BY_NAME
    con.execute(f"SELECT {column} FROM {table_name} LIMIT 1").fetchone()
