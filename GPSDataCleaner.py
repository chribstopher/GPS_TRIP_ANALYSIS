from datetime import datetime
from typing import List, Tuple

import math
from math import radians, sin, cos, sqrt, atan2
import pandas as pd

from GPSPoint import GPSPoint


class GPSDataCleaner:
    """
    A class to clean and preprocess GPS data.
    """

    # def __init__(self, points: List[GPSPoint]):
    #     self.points = points

    def remove_duplicates(df: pd.DataFrame, time_threshold_seconds=0.1) -> pd.DataFrame:
        """
        Remove duplicate GPS points that are within a certain time threshold.
        """
        # check if df is empty
        if df.empty:
            return df

        # make a deep copy of df to remove dups from
        df_cpy = df.copy()
        # make sure timestamp is a datetime object
        df_cpy['timestamp'] = pd.to_datetime(df_cpy['timestamp'])

        # get the first row of the data to begin cleaning
        cleaned = [df_cpy.iloc[0]]
        last = df_cpy.iloc[0]
        # for every row in the df
        for row_idx in range(1, len(df_cpy)):
            # get cur row
            cur_row = df_cpy.iloc[row_idx]
            # calculate for the difference btwn current time and last known time
            time_diff = abs((cur_row['timestamp'] - last['timestamp']).total_seconds())

            # check to see how far away points are in time
            # check to see how close the lat and long are as well
            if (time_diff < time_threshold_seconds and
                abs(cur_row['latitude'] - last['latitude']) < 0.0001 and
                abs(cur_row['longitude'] - last['longitude']) < 0.0001):
                # if points are super close together, they are likely duplicate
                # so skip!
                continue

            cleaned.append(cur_row)
            last = cur_row

        # create new dataframe and re-index to account for removed rows
        nodupe_df = pd.DataFrame(cleaned).reset_index(drop=True)
        print(f"Removed {len(df_cpy) - len(nodupe_df)} duplicate points.")
        return nodupe_df



    def remove_outliers(df: pd.DataFrame, max_speed_knots=100) -> pd.DataFrame:
        """
        Removes GPS points that represent impossible movements (Gps glitches)

        CHECK THE CHECKSUMS AT SOME POINT?
        """

        # check to make sure there are at least 2 points to work with
        if len(df) < 2:
            return df

        # make a deep copy of df to remove outliers from
        df_cpy = df.copy()

        # start with first row
        cleaned = [df_cpy.iloc[0]]
        for row_idx in range(1, len(df_cpy)):
            # get previous and current point
            prev = cleaned[-1]
            curr = df_cpy.iloc[row_idx]
            # calculate for the difference btwn current time and last known time
            time_diff = abs((curr['timestamp'] - prev['timestamp']).total_seconds())
            if time_diff <= 0:
                continue  # skip invalid time differences

            # given lat and long points with respect to time
            # use haversine dist to convert lat/long to speed in knots
            distance = haversine_distance(
                prev['latitude'], prev['longitude'],
                curr['latitude'], curr['longitude']
            )
            # calculate implied speed in knots
            implied_speed_knots = (distance / time_diff) * 1.94384

            # if the speed is impossibly high, skip the point
            if implied_speed_knots > max_speed_knots:
                continue

            # check for very large jumps in location that could signify errors
            if abs(curr['latitude'] - prev['latitude']) > 10 or abs(curr['longitude'] - prev['longitude']) > 10:
                continue # skip if jump is impossibly large with respect to time

            cleaned.append(curr)
            prev = curr

        # create new dataframe and re-index to account for removed rows
        nooutlier_df = pd.DataFrame(cleaned).reset_index(drop=True)

        print(f"Removed {len(df_cpy) - len(nooutlier_df)} outlier points.")
        return nooutlier_df

    def trim_stationary_endpoints(self, points: List[GPSPoint], 
        movement_threshold_knots=0.5) -> Tuple[List[GPSPoint], int, int]:
        
        """ Removes stationary points from the start and end of the trip"""
        # find first moving point
        start_idx = 0
        for i, point in enumerate(points):
            if point.speed_knots > movement_threshold_knots:
                start_idx = i
                break
        # find the last moving points
        end_idx = len(points) - 1
        for i in range(len(points) - 1, -1, -1):
            if points[i].speed_knots > movement_threshold_knots:
                end_idx = i
                break
        trimmed = points[start_idx:end_idx + 1]
        print(f"Trimmed {start_idx} stationary points from start and {len(points) - 1 - end_idx} from end.")
        return trimmed, start_idx, end_idx

    def simplify_straight_segments(self, points: List[GPSPoint], 
        angle_threshold_degrees=5, min_distance=10) -> List[GPSPoint]:
        """ Remove redundant points along paths that are straight lines"""
        if len(points) < 3:
            return points
        simplified = [points[0]]
        for i in range(1, len(points) - 1):
            prev = simplified[-1]
            curr = points[i]
            next_pt = points[i + 1]

            # calculate breaing from prev to curr and curr to next
            bearing1 = self._calculate_bearing(prev, curr)
            bearing2 = self._calculate_bearing(curr, next_pt)

            # calculate the angle difference
            angle_diff = abs(bearing2 - bearing1)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            # keep point if direction changes very significantly
            if angle_diff > angle_threshold_degrees:
                simplified.append(curr)
            elif prev.distance_to(curr) >= min_distance:
                simplified.append(curr)

        simplified.append(points[-1])
        print(f"Simplified from {len(points)} to {len(simplified)} points.")
        return simplified

    def _calculate_bearing(self, point1: GPSPoint, point2: GPSPoint) -> float:
        """ calculate bearing from point 1 to point 2 in degrees"""
        lat1 = math.radians(point1.latitude)
        lat2 = math.radians(point2.latitude)
        delta_lon = math.radians(point2.longitude - point1.longitude)
        y = math.sin(delta_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360

    def clean_data(self) -> List[GPSPoint]:
        """ performs all of the data cleaning steps"""
        print("\nStarting GPS data cleaning...")
        # removing duplicates
        cleaned = self.remove_duplicates()
        # removing outliers
        cleaned, _, _ = self.remove_outliers(cleaned)
        # trimming stationary endpoints
        cleaned, _, _ = self.trim_stationary_endpoints(cleaned)
        # simplifying straight segments
        cleaned = self.simplify_straight_segments(cleaned)
        print("GPS data cleaning completed.\n")
        return cleaned


# get dist between two gps points given current lat/long and previous lat/long
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

        