#!/usr/bin/env bash
# Verify the local Python environment and downloaded Binance aggTrades data.
# Usage: ./working_script.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.quant_work"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Virtual environment not found at $VENV_DIR."
    echo "Run ./setup_venv.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "Using Python: $(command -v python)"

cd "$PROJECT_DIR"
python -u - <<'PY'
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path("data/raw")
MONTHS = (
    "2020-02", "2020-03", "2020-04",
    "2021-04", "2021-05", "2021-06",
    "2022-04", "2022-05", "2022-06",
    "2022-10", "2022-11", "2022-12",
    "2023-06", "2023-07", "2023-08",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


print("\nData check")
print("Month    Download  Checksum  Rows")
print("-------  --------  --------  ------------")

missing_or_invalid = False
total_rows = 0
for month in MONTHS:
    archive = DATA_DIR / f"BTCUSDT-aggTrades-{month}.zip"
    checksum = archive.with_suffix(archive.suffix + ".CHECKSUM")
    downloaded = "yes" if archive.is_file() else "MISSING"
    checksum_status = "--"
    rows = "--"

    if not archive.is_file():
        missing_or_invalid = True
    else:
        if checksum.is_file():
            expected = checksum.read_text().split()[0].lower()
            checksum_status = "OK" if sha256(archive) == expected else "INVALID"
            if checksum_status == "INVALID":
                missing_or_invalid = True
        else:
            checksum_status = "MISSING"
            missing_or_invalid = True

        try:
            with zipfile.ZipFile(archive) as zip_file:
                csv_files = [name for name in zip_file.namelist() if name.endswith(".csv")]
                if len(csv_files) != 1:
                    raise ValueError(f"expected one CSV, found {len(csv_files)}")
                with zip_file.open(csv_files[0]) as csv_file:
                    count = sum(1 for _ in csv_file)
            rows = f"{count:,}"
            total_rows += count
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            rows = f"ERROR: {error}"
            missing_or_invalid = True

    print(f"{month}  {downloaded:<8}  {checksum_status:<8}  {rows}")

print("-------  --------  --------  ------------")
print(f"Total rows: {total_rows:,}")

if missing_or_invalid:
    print("\nSome data is missing or invalid. Run: python download_data.py")
    sys.exit(1)

print("\nAll expected files are downloaded, verified, and readable.")
PY
