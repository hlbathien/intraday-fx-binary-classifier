# Unit test for NY session mask and DST boundaries
import unittest
import pandas as pd
from lib.session import mask_ny_session

class TestNYSessionMask(unittest.TestCase):
    def test_dst_boundaries(self):
        # NY session: 12:00 PM - 8:59 PM NY local time
        # Test around DST start (2nd Sunday in March) and end (1st Sunday in November)
        # 2025 DST start: March 9, 2025; end: Nov 2, 2025
        # All times UTC
        data = pd.DataFrame({
            'Time': [
                '2025-03-09 15:59',  # 10:59 NY (before DST starts)
                '2025-03-09 16:00',  # 12:00 NY (DST starts)
                '2025-11-02 16:59',  # 12:59 NY (DST ends)
                '2025-11-02 17:00',  # 12:00 NY (after DST ends)
            ]
        })
        data['Time'] = pd.to_datetime(data['Time'], utc=True)
        mask = mask_ny_session(data['Time'])
        # Only 16:00 and 17:00 UTC should be in session
        self.assertTrue(mask.iloc[1])
        self.assertTrue(mask.iloc[3])
        self.assertFalse(mask.iloc[0])
        self.assertFalse(mask.iloc[2])

if __name__ == "__main__":
    unittest.main()
