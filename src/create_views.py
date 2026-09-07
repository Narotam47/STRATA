"""Load the cleaning layer SQL into the DuckDB database as persistent views."""

import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = PROJECT_ROOT / "sql"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "strata.duckdb"

VIEW_FILES = [
    SQL_DIR / "01_cleaning_views.sql",
    SQL_DIR / "02_rfm_features.sql",
    SQL_DIR / "03_household_features.sql",
    SQL_DIR / "04_clv_split.sql",
]

VIEW_NAMES = [
    "v_transactions_base",
    "v_transactions_clean",
    "v_households",
    "v_campaign_exposure",
    "v_coupon_redemptions",
    "v_rfm_features",
    "v_household_features",
    "v_clv_split",
]


def create_views(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        sys.exit(
            f"No database at {db_path}. Run `make ingest` before `make views`."
        )

    con = duckdb.connect(str(db_path))

    for sql_file in VIEW_FILES:
        print(f"Loading {sql_file.relative_to(PROJECT_ROOT)} ...")
        con.execute(sql_file.read_text())
        print(f"  OK")

    print()
    print("Registered views:")
    for name in VIEW_NAMES:
        row_count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        col_count = len(con.execute(f"DESCRIBE {name}").fetchall())
        print(f"  {name:30s} {row_count:>14,} rows  {col_count:>2d} cols")

    con.close()
    print(f"\nViews written to {db_path}")


if __name__ == "__main__":
    create_views()
