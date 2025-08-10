import pathlib, datetime as dt, requests, zipfile, shutil, sys, time
import pandas as pd

OUT_DIR   = pathlib.Path("data/raw")        
OUT_FILE  = pathlib.Path("data/raw/binance_spot.csv")       
OUT_DIR.mkdir(parents=True, exist_ok=True)

def daterange(start: str, end: str):
    s = dt.datetime.strptime(start, "%Y-%m-%d").date()
    e = dt.datetime.strptime(end,   "%Y-%m-%d").date()
    while s <= e:
        yield s
        s += dt.timedelta(days=1)

def build_url(day: dt.date) -> str:
    return ("https://data.binance.vision/data/spot/daily/klines/"
            f"BTCUSDT/1h/BTCUSDT-1h-{day.isoformat()}.zip")

def download_zip(url: str, save_path: pathlib.Path, retry=3) -> bool:
    for n in range(retry):
        r = requests.get(url, stream=True, timeout=15)
        if r.status_code == 404:
            return False
        if r.ok:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(1 << 13):
                    f.write(chunk)
            return True
        time.sleep(1)
    return False

def unzip(path_zip: pathlib.Path, delete_zip=False):
    tgt_dir = path_zip.with_suffix("")   # e.g.  .../2023-05-18
    tgt_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(path_zip) as zf:
        zf.extractall(tgt_dir)
    if delete_zip:
        path_zip.unlink()

def bulk_download(start="2023-05-18", end="2023-10-23", unzip_each=True):
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

def merge_spot_csvs() -> pathlib.Path:
    frames = []
    for day_folder in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):
        csv_files = list(day_folder.glob("*.csv"))
        if not csv_files:                 
            continue
        df = pd.read_csv(csv_files[0], header=None,
                         names=["open_time","open","high","low","close","volume",
                                "close_time","qav","trades","taker_buy_base",
                                "taker_buy_quote","ignore"])

        df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.date
        df["hour"]     = pd.to_datetime(df["open_time"], unit="ms").dt.hour
        df = df[["date","hour","close"]]        
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True).sort_values(["date","hour"])
    merged.to_csv(OUT_FILE, index=False)
    print(f"merged {len(merged):,} rows: {OUT_FILE}")
    return OUT_FILE

def remove_day_folders():
    removed = 0
    for p in OUT_DIR.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
            removed += 1
    print(f"removed {removed} date-folders")

# ---------------------------------------------------------------
if __name__ == "__main__":
    bulk_download()        # ZIP download + unzip
    merge_spot_csvs()      # to data/binance_spot_summary.csv
    remove_day_folders()   # clean up the folders