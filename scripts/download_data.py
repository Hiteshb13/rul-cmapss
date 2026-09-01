"""Download the NASA C-MAPSS Turbofan Engine Degradation Simulation data set.

Fetches the official ZIP from the NASA Prognostics Center of Excellence
repository (hosted on the NASA S3 bucket) and extracts the train/test/RUL text
files into ``data/raw/``.

Usage
-----
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# Official NASA PCoE download (verified against
# https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
DOWNLOAD_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)

EXPECTED_FILES = [
    f"{split}_FD00{fd}.txt"
    for split in ("train", "test", "RUL")
    for fd in (1, 2, 3, 4)
]


def _download(url: str, dest: Path, chunk_size: int = 1 << 16) -> None:
    """Download a file with a progress readout to ``dest``."""
    print(f"Downloading\n  {url}\nto\n  {dest}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)", end="")
        print()


def main() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if all((DATA_RAW / f).exists() for f in EXPECTED_FILES):
        print(f"All {len(EXPECTED_FILES)} files already present in {DATA_RAW}. Nothing to do.")
        return 0

    zip_path = DATA_RAW / "cmapss.zip"
    _download(DOWNLOAD_URL, zip_path)

    print("Extracting...")
    extracted: list[Path] = []

    def _extract_txts(zf: zipfile.ZipFile) -> list[str]:
        """Return the list of nested ``*.txt`` member names in ``zf``."""
        return [m for m in zf.namelist() if m.lower().endswith(".txt")]

    with zipfile.ZipFile(zip_path) as zf:
        txt_members = _extract_txts(zf)
        nested = [m for m in zf.namelist() if m.lower().endswith(".zip")]

        if not txt_members and nested:
            # The NASA archive wraps the data in an inner CMAPSSData.zip.
            inner_path = DATA_RAW / Path(nested[0]).name
            with zf.open(nested[0]) as src, open(inner_path, "wb") as dst:
                dst.write(src.read())
            print(f"  found nested archive {Path(nested[0]).name}, extracting it...")
            with zipfile.ZipFile(inner_path) as inner:
                for member in _extract_txts(inner):
                    target = DATA_RAW / Path(member).name
                    with inner.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted.append(target)
                    print(f"  extracted {target.name}")
            inner_path.unlink()
        else:
            for member in txt_members:
                target = DATA_RAW / Path(member).name
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted.append(target)
                print(f"  extracted {target.name}")

    zip_path.unlink()  # remove the intermediate archive

    missing = [f for f in EXPECTED_FILES if not (DATA_RAW / f).exists()]
    if missing:
        print(f"WARNING: could not find {missing} after extraction.", file=sys.stderr)
        return 1

    print(f"Done. Raw data files are in {DATA_RAW}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
