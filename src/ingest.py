"""Load the eight raw Complete Journey tables into a single DuckDB database.

Types are declared in src/schema.py and applied here with explicit CASTs rather
than inherited from the Parquet footer. That matters for two reasons: it pins
integer widths to the observed value ranges (halving key-column width on the
20.9M-row promotions table), and it makes the load fail loudly if an upstream
column is renamed or retyped instead of silently producing a differently-shaped
table.
"""

import sys
from pathlib import Path

import duckdb

from src.schema import TABLES, Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "strata.duckdb"


def _create_table_sql(table: Table, source: Path) -> str:
    """Build an explicit projection: every column named, every type asserted.

    Selecting columns by name (rather than SELECT *) means an upstream rename
    raises a Binder error here, at load time, instead of surfacing as a null
    column three notebooks later.
    """
    projection = ",\n        ".join(
        f'CAST("{col}" AS {dtype}) AS "{col}"' for col, dtype in table.columns.items()
    )
    return f"""
    CREATE OR REPLACE TABLE {table.name} AS
    SELECT
        {projection}
    FROM read_parquet('{source.as_posix()}')
    """


def ingest(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> dict[str, int]:
    missing = [t.filename for t in TABLES if not (raw_dir / t.filename).exists()]
    if missing:
        sys.exit(
            "Cannot ingest — missing source files:\n  "
            + "\n  ".join(missing)
            + "\n\nRun `make data` first."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    counts: dict[str, int] = {}

    print(f"Loading {len(TABLES)} tables into {db_path.relative_to(PROJECT_ROOT)}\n")
    for table in TABLES:
        con.execute(_create_table_sql(table, raw_dir / table.filename))
        count = con.execute(f"SELECT COUNT(*) FROM {table.name}").fetchone()[0]
        counts[table.name] = count
        flag = " " if count == table.expected_rows else "!"
        print(f"  {flag} {table.name:24s} {count:>12,} rows  {len(table.columns):>2d} cols")

    con.close()

    drift = {n: c for n, c in counts.items() if c != _expected(n)}
    print(f"\n  {'total':24s} {sum(counts.values()):>12,} rows")
    if drift:
        print("\n  ! row counts differ from the pinned manifest:")
        for name, count in drift.items():
            print(f"      {name}: got {count:,}, expected {_expected(name):,}")
    print(f"\nDatabase written to {db_path}")
    return counts


def _expected(name: str) -> int:
    return next(t.expected_rows for t in TABLES if t.name == name)


if __name__ == "__main__":
    ingest()
