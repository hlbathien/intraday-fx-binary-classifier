"""
Session masking utilities for NY session with DST handling.
"""
import pandas as pd
import pytz
from pandas import Series

def mask_ny_session(times: pd.Series) -> Series:
    """
    Returns a boolean mask for bars inside NY session (12:00 PM - 8:59 PM NY local time).
    Handles DST boundaries.
    times: pd.Series of UTC timestamps (pd.Timestamp, tz-aware or naive assumed UTC)
    """
    # Ensure times are tz-aware UTC
    times_utc = pd.to_datetime(times, utc=True)
    # Convert to NY local time
    ny = pytz.timezone("America/New_York")
    times_ny = times_utc.dt.tz_convert(ny)
    # NY session: 12:00 PM - 8:59 PM (inclusive)
    hour = times_ny.dt.hour
    minute = times_ny.dt.minute
    # 12:00 (12) to 20:59 (20:59)
    in_session = ((hour > 12) | ((hour == 12) & (minute >= 0))) & \
                 ((hour < 21) | ((hour == 20) & (minute <= 59)))
    return in_session
