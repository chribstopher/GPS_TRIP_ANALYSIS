from datetime import datetime
from itertools import groupby
from typing import List, Tuple

import math
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
from pyproj import Transformer
from GPSPoint import GPSPoint
from pykalman import KalmanFilter


class GPSDataCleaner:
    """
    A class to clean and preprocess GPS data.
    """

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


    def trim_stationary_endpoints(df: pd.DataFrame,
        movement_threshold_knots=0.5) -> pd.DataFrame:
        
        """ Removes stationary points from the start and end of the trip"""
        # get all the speeds from speed col
        speeds = df['speed_knots'].to_numpy()
        # create logical mask of all speeds over the thr
        moving = speeds > movement_threshold_knots
        # if there are no speeds above the thr,
        if not moving.any():
            # return the df
            return df.copy()

        # get the first index of the car moving
        start_idx = np.argmax(moving)
        # use the logical mask to get the last index of the car moving
        # reverse the array so that argmax will return the first idx of non movement
        # which would be the last in the actual list
        # convert back to original idx by subtracting it from len
        end_idx = len(speeds) - np.argmax(moving[::-1]) - 1
        # trip the df to only contain points within start and end
        trimmed_df = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)

        print(f"Trimmed {start_idx} stationary points from start and {len(df) - 1 - end_idx} from end.")
        return trimmed_df


    def kalman_filtering(df: pd.DataFrame) -> pd.DataFrame:
        """ use the kalman filter to smooth GPS data for better readings """
        # define pyproj converter that takes cord points from lat/long to xy
        # this is necessary for proper distance metrix calculations
        transformer = Transformer.from_crs("epsg:4326", "epsg:32617", always_xy=True)

        df_cpy = df.copy()
        # apply the transformer to the copied df to get x,y values
        df_cpy['x'], df_cpy['y'] = transformer.transform(
            df['longitude'].values,
            df['latitude'].values
        )

        # get only the x,y values to pass into kalman filter
        measurements = df_cpy[['x', 'y']].values

        kf = KalmanFilter(
            # defines how the state of gps data evolves over time
            transition_matrices=[[1, 0, 1, 0],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]],
            # defines how the x,y values relate to the state (ignoring velocity)
            observation_matrices=[[1, 0, 0, 0],
                                  [0, 1, 0, 0]],
            # define level of noise in GPS readings
            observation_covariance=5 ** 2 * np.eye(2),
            # define assumed level of uncertainty in data
            # since car is stopping / starting frequently, use a large number (5^2)
            transition_covariance=1 ** 2 * np.eye(4),
        )

        smoothed_state_means, _ = kf.smooth(measurements)

        # get the smoothed values
        df_cpy['x_smooth'] = smoothed_state_means[:, 0]
        df_cpy['y_smooth'] = smoothed_state_means[:, 1]

        # convert back to origional values
        df['longitude'], df['latitude'] = transformer.transform(
            df_cpy['x_smooth'].values,
            df_cpy['y_smooth'].values,
            direction='INVERSE'
        )

        return df


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

def get_curdirection(lat1, lon1, lat2, lon2) -> float:
    """ gets the current direction of the car, can be used to track turns"""
    # python funct. need radians, convert from degrees to rad
    # convert to radians (this works on Series if lat1 etc. are Series)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # get the change in longitude of car
    delta_lon = lon2 - lon1
    # calculate for turn angle using spherical trig
    # y gets the east west direction
    y = np.sin(delta_lon) * np.cos(lat2)
    # x gets the north / south direction
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(delta_lon)
    # get the current dir angle in radians
    cur_dir = np.degrees(np.arctan2(y, x))
    # convert back to 0-360 degrees from radians
    return (cur_dir + 360) % 360


# get dist between two gps points given current lat/long and previous lat/long
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

        