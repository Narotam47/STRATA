"""Shared fixtures. The suite runs read-only against the ingested database."""

from pathlib import Path

import duckdb
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "strata.duckdb"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def con():
    if not DB_PATH.exists():
        pytest.fail(
            f"No database at {DB_PATH}. Run `make ingest` before `make test`.",
            pytrace=False,
        )
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return RAW_DIR
