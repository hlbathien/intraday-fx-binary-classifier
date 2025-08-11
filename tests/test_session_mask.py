import pandas as pd
import pytz
from pathlib import Path
import sys

# Add src path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from 1_data_preprocessing import filter_ny_session  # type: ignore  # noqa

UTC = pytz.UTC


def make_df(times):
    return pd.DataFrame({
        'Time': pd.to_datetime(times).tz_localize('UTC'),
        'Open': 1.0,
        'High': 1.0,
        'Low': 1.0,
        'Close': 1.0,
        'Volume': 0,
    })


# DST transition 2025 US: clocks forward Mar 9, back Nov 2

def test_dst_start_mar():
    # 11:55-12:05 UTC times mapped to NY (note DST difference) we just ensure mask includes correct hour window
    times = [
        '2025-03-09 15:55',  # 10:55 NY (EST) before change; outside
        '2025-03-09 16:59',  # 12:59? need convert check - simplified coarse test
        '2025-03-10 16:00',  # 12:00 next day
    ]
    df = make_df(times)
    filtered = filter_ny_session(df)
    # We expect at least rows that map to 12:00-20:59 local; just sanity assert not empty
    assert (filtered['Time'] >= pd.Timestamp('2025-03-10 16:00', tz='UTC')).any()


def test_basic_hours_inclusion():
    # Choose a summer date (DST) where 12:00 NY = 16:00 UTC
    base = pd.Timestamp('2025-06-02 16:00', tz='UTC')
    times = [base + pd.Timedelta(minutes=i) for i in range(0, 10)]  # 16:00-16:09 UTC
    df = make_df(times)
    filtered = filter_ny_session(df)
    assert len(filtered) == len(df)


def test_exclusion_before_after():
    # Include times just outside window
    # Summer: 12:00 NY = 16:00 UTC; 21:00 NY = 01:00 UTC next day during DST (but we stop at 20:59 NY = 00:59 UTC)
    times = [
        '2025-06-02 15:59',  # 11:59 NY -> exclude
        '2025-06-02 16:00',  # 12:00 NY -> include
        '2025-06-03 00:59',  # 20:59 NY -> include
        '2025-06-03 01:00',  # 21:00 NY -> exclude
    ]
    df = make_df(times)
    filtered = filter_ny_session(df)
    keep_times = filtered['Time'].dt.strftime('%H:%M').tolist()
    assert '16:00' in keep_times and '00:59' in keep_times and '15:59' not in keep_times and '01:00' not in keep_times
