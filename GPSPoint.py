from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd

@dataclass
class GPSPoint:
    timestamp: datetime
    latitude: float
    longitude: float
    speed_knots: float
    heading: float
    altitude: float
    fix_quality: int
    num_satellites: int
    source: str # whether the point is RMC or GGA

    def speed_mph(self):
        """Convert speed from knots to miles per hour."""
        return self.speed_knots * 1.15078
