"""
Download daily EOH summary ZIPs for BTCUSDT options (2023-05-01~2023-10-31)
data source: https://data.binance.vision/data/option/daily/EOHSummary/BTCUSDT/
"""

import pathlib, datetime as dt, requests, zipfile, io, sys, time, pandas as pd, shutil
from pathlib import Path

BASE_URL = ("https://data.binance.vision/data/option/daily/EOHSummary/"
            "BTCUSDT/BTCUSDT-EOHSummary-{date}.zip")   # {YYYY-MM-DD}

OUT_DIR  = pathlib.Path("data/raw")
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "binance_eoh_summary.csv"

def daterange(start: str, end: str):
    """Yield datetime.date from start to end (inclusive)."""
    s = dt.datetime.strptime(start, "%Y-%m-%d").date()
    e = dt.datetime.strptime(end,   "%Y-%m-%d").date()
    step = dt.timedelta(days=1)
    while s <= e:
        yield s
        s += step

def build_url(day: dt.date) -> str:
    return BASE_URL.format(date=day.isoformat())

def download_zip(url: str, out_path: pathlib.Path, retry: int = 3) -> bool:
    """True on success, False if 404 (file absent)."""
    for n in range(retry):
        r = requests.get(url, stream=True, timeout=15)
        if r.status_code == 404:
            return False
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1 << 14):
                    f.write(chunk)
            return True
        print(f"retry {n+1} for {url} (HTTP {r.status_code})")
        time.sleep(1)
    return False

def unzip(path_zip: pathlib.Path, delete_zip: bool = False):
    with zipfile.ZipFile(path_zip) as zf:
        zf.extractall(path_zip.parent / path_zip.stem)  # e.g. eoh_data/2023-05-01/
    if delete_zip:
        path_zip.unlink()

def bulk_download(start="2023-05-18", end="2023-10-23", unzip_each=False):
    for day in daterange(start, end):
        url  = build_url(day)
        save = OUT_DIR / (day.isoformat() + ".zip")
        if save.exists():
            print(f"skip {save.name} (already)")
            continue
        ok = download_zip(url, save)
        if ok:
            print(f"{save.name} downloaded")
            if unzip_each:
                unzip(save, delete_zip=True)
        else:
            print(f"{save.name} not found")

def merge_eoh_csvs() -> Path:
    """Concatenate all CSVs under root/*/ and save one big CSV."""
    frames = []
    for day_folder in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):

        for csv in day_folder.glob("*.csv"):
            df = pd.read_csv(csv)
            df["date"] = pd.to_datetime(day_folder.name)   
            frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"✔  merged {len(merged):,} rows → {OUT_FILE}")
    return OUT_FILE

def remove_day_folders():
    """Delete every sub-directory under root (unzipped folders)."""
    removed = 0
    for p in OUT_DIR.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
            removed += 1
    print(f"removed {removed} date-folders")


if __name__ == "__main__":
    unzip_flag = len(sys.argv) > 1 and sys.argv[1] == "unzip"
    bulk_download(unzip_each=True)      # ZIP download + unzip
    merge_eoh_csvs()                    # to data/binance_eoh_summary.csv
    remove_day_folders()                # clean up the folders
    