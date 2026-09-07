"""Fetch and verify the raw Complete Journey tables.

`python -m src.fetch_data`          downloads any missing tables (idempotent)
`python -m src.fetch_data --check`  verifies all eight are present, exits 1 if not

The source is pinned to a commit SHA in src/schema.py, so a download today and
a download next year produce byte-identical files.
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.schema import SOURCE_SHA, SOURCE_URL, TABLES

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Files whose size differs from the manifest are treated as corrupt rather than
# merely stale: a truncated download is the failure mode this guards against.
SIZE_TOLERANCE = 0


def _size_str(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024**2:.1f} MB"


def check(raw_dir: Path = RAW_DIR, quiet: bool = False) -> list[str]:
    """Return the names of tables that are missing or the wrong size."""
    problems: list[str] = []
    for table in TABLES:
        path = raw_dir / table.filename
        if not path.exists():
            problems.append(f"{table.filename} — MISSING")
            continue
        actual = path.stat().st_size
        if abs(actual - table.expected_bytes) > SIZE_TOLERANCE:
            problems.append(
                f"{table.filename} — size {actual:,} B, expected {table.expected_bytes:,} B "
                "(truncated or modified)"
            )
        elif not quiet:
            print(f"  ok      {table.filename:32s} {_size_str(actual):>10s}")
    return problems


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    total = sum(t.expected_bytes for t in TABLES)
    print(f"Source: {SOURCE_URL}")
    print(f"Pinned: {SOURCE_SHA[:12]}   Total: {_size_str(total)}\n")

    for table in TABLES:
        path = raw_dir / table.filename
        if path.exists() and not force and path.stat().st_size == table.expected_bytes:
            print(f"  cached  {table.filename:32s} {_size_str(table.expected_bytes):>10s}")
            continue
        try:
            urllib.request.urlretrieve(table.url, path)
        except urllib.error.URLError as exc:
            path.unlink(missing_ok=True)
            sys.exit(f"\nFAILED downloading {table.filename}: {exc}\nURL: {table.url}")
        actual = path.stat().st_size
        status = "ok" if actual == table.expected_bytes else "SIZE MISMATCH"
        print(f"  {status:8s}{table.filename:32s} {_size_str(actual):>10s}")

    problems = check(raw_dir, quiet=True)
    if problems:
        sys.exit("\nDownload finished but verification failed:\n  " + "\n  ".join(problems))
    print(f"\nAll {len(TABLES)} tables present in {raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only, do not download")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    if args.check:
        print(f"Checking {RAW_DIR} ...")
        problems = check()
        if problems:
            print("\nMissing or invalid source files:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            print(
                f"\nRun `make data` to fetch them, or download manually from:\n  {SOURCE_URL}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"\nAll {len(TABLES)} source tables present and correctly sized.")
    else:
        download(force=args.force)


if __name__ == "__main__":
    main()
