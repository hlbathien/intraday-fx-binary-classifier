# Data cleaning and NY session filtering for all raw FX CSVs.
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from lib.config import get_raw_data_path, get_cleaned_data_path
from lib.session import mask_ny_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_preprocessing")

def clean_and_align(df: pd.DataFrame) -> pd.DataFrame:
    # Forward-fill small gaps (max 2 consecutive missing bars), drop outages
    df = df.set_index("Time")
    # Detect missing bars (assume 1-min frequency)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="T", name="Time")
    df = df.reindex(full_idx)
    # Forward-fill up to 2 consecutive missing
    mask_gap = df["Open"].isna()
    gap_groups = mask_gap.ne(mask_gap.shift()).cumsum()
    gap_sizes = mask_gap.groupby(gap_groups).transform("sum")
    # Only ffill if gap <=2
    can_ffill = (~mask_gap) | (gap_sizes <= 2)
    df_ffilled = df[can_ffill].copy()
    df_ffilled = df_ffilled.ffill()
    # Drop any remaining NaN (outages)
    df_ffilled = df_ffilled.dropna()
    return df_ffilled.reset_index()

def process_all():
    raw_dir = get_raw_data_path()
    cleaned_dir = get_cleaned_data_path()
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    files = list(raw_dir.glob("*.csv"))
    logger.info(f"Found {len(files)} raw files in {raw_dir}")
    for fpath in tqdm(files, desc="Processing raw files"):
        symbol = fpath.stem
        df = pd.read_csv(fpath)
        df["Time"] = pd.to_datetime(df["Time"], utc=True)
        df = clean_and_align(df)
        mask = mask_ny_session(df["Time"])
        df_ny = df[mask].copy()
        out_path = cleaned_dir / f"{symbol}.csv"
        df_ny.to_csv(out_path, index=False)
        logger.info(f"Exported cleaned {symbol} to {out_path}")

if __name__ == "__main__":
    process_all()
