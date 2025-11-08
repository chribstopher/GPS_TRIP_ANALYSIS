from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import math

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
    dilution_of_precision: float
    source: str # whether the point is RMC or GGA

    def speed_mph(self):
        """Convert speed from knots to miles per hour."""
        return self.speed_knots * 1.15078
    
    def is_moving(self, threshold_knots=0.5):
        """Determine if the GPS point indicates movement based on speed threshold."""
        return self.speed_knots > threshold_knots
    
    def distance_to(self, other):
        """Calculate the Haversine distance to another GPSPoint in miles."""
        R = 6371000 # Earth radius in meters
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c # distance in meters from one point to another

