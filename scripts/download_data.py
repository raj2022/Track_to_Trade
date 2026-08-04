"""
Download and verify BTCUSDT spot aggTrades monthly files from Binance public data.

Usage:
    python download_data.py

Pulls the five 3-month windows defined in the README, verifies each file's
SHA-256 checksum against Binance's published .CHECKSUM file, and refuses to
keep any file that fails verification.
"""

import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT"
SYMBOL = "BTCUSDT"
DATA_DIR = Path("data/raw")

# Contiguous 3-month windows around known regime transitions, plus one calm baseline.
WINDOWS = [
    ("2020-02", "2020-03", "2020-04"),  # COVID crash
    ("2021-04", "2021-05", "2021-06"),  # May 2021 crash
    ("2022-04", "2022-05", "2022-06"),  # LUNA / UST collapse
    ("2022-10", "2022-11", "2022-12"),  # FTX collapse
    ("2023-06", "2023-07", "2023-08"),  # calm baseline
]

MONTHS = sorted({m for window in WINDOWS for m in window})


def download_file(url: str, dest: Path) -> bool:
    """Stream-download a file with a progress bar. Returns False on HTTP failure."""
    resp = requests.get(url, stream=True, timeout=30)
    if resp.status_code != 200:
        print(f"  FAILED ({resp.status_code}): {url}")
        return False

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return True


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(data_path: Path, checksum_path: Path) -> bool:
    """Binance .CHECKSUM files are in the format: '<hash>  <filename>'."""
    expected = checksum_path.read_text().split()[0].strip().lower()
    actual = sha256sum(data_path).lower()
    return expected == actual


def fetch_month(month: str) -> None:
    filename = f"{SYMBOL}-aggTrades-{month}.zip"
    checksum_filename = f"{filename}.CHECKSUM"
    data_path = DATA_DIR / filename
    checksum_path = DATA_DIR / checksum_filename

    if data_path.exists() and checksum_path.exists() and verify_checksum(data_path, checksum_path):
        print(f"[{month}] already downloaded and verified, skipping")
        return

    print(f"[{month}] downloading {filename}")
    if not download_file(f"{BASE_URL}/{filename}", data_path):
        return
    if not download_file(f"{BASE_URL}/{checksum_filename}", checksum_path):
        return

    if verify_checksum(data_path, checksum_path):
        print(f"[{month}] checksum OK")
    else:
        print(f"[{month}] CHECKSUM MISMATCH — deleting corrupted file")
        data_path.unlink(missing_ok=True)
        sys.exit(1)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(MONTHS)} months into {DATA_DIR}/\n")
    for month in MONTHS:
        fetch_month(month)
    print("\nDone. Verify row counts per file before trusting anything downstream —")
    print("a known-volatile month should visibly have more rows than a calm one.")


if __name__ == "__main__":
    main()
