import re
import pandas as pd
from datetime import datetime, timedelta
from GPSPoint import GPSPoint
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from typing import List, Optional, Tuple
import sys

class GPSParser:
    def __init__(self):
        # initialize an empty list to store GPS points
        self.points = []
        self.current_date = None  # to keep track of the current date for GGA sentences


    def parse_rmc(self, line):
        """Parser for RMC NMEA sentences.
            the RMC sentence is formatted as follow:

            $GPRMC,221237.000,A,4305.1457,N,07740.8141,W,0.33,58.46,110925,,,A*47
            
            """
        try:
            tokens = line.strip().split(',')
            # check if the data has amount of tokens that it should
            if len(tokens) < 12 or tokens[0] != '$GPRMC':
                return None
            if tokens[2] != 'A':  # Check if the data is valid
                # do something here. I forget what A means
                return None
            # create a gps point from the tokens
            time_str = tokens[1]
            lat_str = tokens[3]
            lat_dir = tokens[4]
            lon_str = tokens[5]
            lon_dir = tokens[6] # the direction the vehicle is moving relative to true north
            speed = float(tokens[7]) if tokens[7] else 0.0
            heading = float(tokens[8]) if tokens[8] else 0.0
            date_str = tokens[9]
            # convert latitude and longitude to coordinates
            lat = self.convert_to_decimal(lat_str, lat_dir)
            lon = self.convert_to_decimal(lon_str, lon_dir)
            # parse the datetime
            timestamp = self.parse_datetime(date_str, time_str)
            if timestamp:
                self.current_date = timestamp.date()
            # create GPSPoint
            return GPSPoint(
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                speed_knots=speed,
                heading=heading,
                altitude=0.0,  # RMC does not provide altitude
                fix_quality=1,  # RMC does not provide fix quality so we say it is working
                num_satellites=0,  # RMC does not provide number of satellites
                dilution_of_precision=0.0,  # RMC does not provide DOP
                source='RMC'
            )
        except (ValueError, IndexError) as e:
            print(f"Error parsing RMC line: {line}. Error: {e}")
            return None



    def parse_gga(self, line):
        """Parser for GGA NMEA sentences.
        the GGA sentence is formatted as follow:

        $GPGGA,221237.250,4305.1457,N,07740.8141,W,1,04,2.05,64.4,M,-34.4,M,,*68
        
        """

        # TODO: fix the checksum file *68
        # check the docs for the final col, the checksum * should differentiate a column
        try:
            tokens = line.strip().split(',')
            # check if the data has amount of tokens that it should
            if len(tokens) < 15 or tokens[0] != '$GPGGA':
                return None
            # check fix quality (0 = invalid)
            fix_quality = int(tokens[6]) if tokens[6] else 0
            if fix_quality == 0:
                return None
            # create a gps point from the tokens
            time_str = tokens[1]
            lat_str = tokens[2]
            lat_dir = tokens[3]
            lon_str = tokens[4]
            lon_dir = tokens[5]
            num_satellites = int(tokens[7]) if tokens[7] else 0
            dilution_of_precision = float(tokens[8]) if tokens[8] else 0.0
            altitude = float(tokens[9]) if tokens[9] else 0.0
            # convert latitude and longitude to coordinates
            lat = self.convert_to_decimal(lat_str, lat_dir)
            lon = self.convert_to_decimal(lon_str, lon_dir)
            # parse the time (GGA does not have date, so we use current_date)
            timestamp = self.parse_time_with_date(time_str)
            # create GPSPoint
            return GPSPoint(
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                speed_knots=0.0,  # GGA does not provide speed
                heading=0.0,      # GGA does not provide heading
                altitude=altitude,
                fix_quality=fix_quality,
                num_satellites=num_satellites,
                dilution_of_precision=dilution_of_precision,
                source='GGA'
            )
        except (ValueError, IndexError) as e:
            print(f"Error parsing GGA line: {line}. Error: {e}")
            return None


    def convert_to_decimal(self, coord_str, direction):
        """Convert NMEA coordinate format (DDMM.MMMM) to decimal degrees."""
        if not coord_str:
            return 0.0
        try:
            # for longitude, degrees are the first three digits: DDDMM.MMMM
            # for latitude, degrees are the first two digits: DDMM.MMMM
            if direction in ['E', 'W']: # means we are looking at longitude
                degrees = int(coord_str[0:3])
                minutes = float(coord_str[3:])
            else:  # latitude
                degrees = int(coord_str[0:2])
                minutes = float(coord_str[2:])
            decimal = degrees + (minutes / 60)
            # apply the direction
            if direction in ['S', 'W']:
                decimal = -decimal
            return decimal
        except (ValueError, IndexError):
            return 0.0

    
    def parse_datetime(self, date_str, time_str):
        """parses DDMMYY and HHMMSS.SSS into a datetime object."""
        try:
            # validate date string
            if not date_str or len(date_str) < 6:
                return None
            # parse information
            day = int(date_str[0:2])
            month = int(date_str[2:4])
            year = int(date_str[4:6]) + 2000  # assuming 21st century
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            second = int(float(time_str[4:]))  # handle fractional seconds
            return datetime(year, month, day, hour, minute, second)
        except (ValueError, IndexError):
            return None
        

    def parse_time_with_date(self, time_str):
        """parses HHMMSS.SSS using current date from previous RMC sentence."""
        try:
            # validating the input
            if not self.current_date or not time_str:
                return None
            # parse time information
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            second = int(float(time_str[4:]))  # handle fractional seconds
            return datetime(self.current_date.year, self.current_date.month, self.current_date.day, hour, minute, second)
        except (ValueError, IndexError):
            return None

    def parse_file(self, filename):
        """Parse a file containing NMEA sentences."""
        print(f"Beginning to parse file:{filename}", filename)
        with open(filename, 'r') as file:
            for line_num, line in enumerate(file, 1):
                # skip header lines
                if line.startswith('Vers') or line.startswith('USE') or line.startswith('DEVELOPMENT') or line.startswith('$GPGSA') or line.startswith('$GPVTG'):
                    continue
                # handle possible arduino "burps" which give two sentences in one line
                if line.count('$') > 1:
                    sentences = [s for s in line.split('$') if s.strip()]
                    for sentence in sentences:
                        self._process_sentence('$' + sentence)
                else:
                    self._process_sentence(line)
        print(f"Finished parsing file:{filename}, total points parsed: {len(self.points)}")
        return self.points

    def _process_sentence(self, line):
        """Process a single NMEA sentence and update current date if needed."""
        point = None
        if line.startswith('$GPRMC'):
            point = self.parse_rmc(line)
        elif line.startswith('$GPGGA'):
            point = self.parse_gga(line)

        if point and point.timestamp:
            self.points.append(point)
    
    def to_dataframe(self):
        """Convert the list of GPS points to a pandas DataFrame."""
        return pd.DataFrame([vars(point) for point in self.points])


def kalman_filtering(df):
    dt = 1.0
    process = 1e-3
    measurement = 1e-2

    A = np.array([[1, dt, 0.5 * dt ** 2],
                  [0, 1, dt],
                  [0, 0, 1]])
    # A = np.array([[1, 1],
    #               [0, 1]])

    H = np.array([[1, 0, 0]])
    # H = np.array([[1, 0]])

    # Process and measurement covariance
    Q = np.eye(3) * process  # model noise
    # Q = np.array([[1e-5, 0],
    #               [0, 1e-5]])
    R = np.array([[measurement]])  # measurement noise
    # R = np.array([[0.01]])

    # Initialize state and covariance
    x = np.zeros((3, 1))  # position, velocity, acceleration
    P = np.eye(3)

    estimates = []

    for z in df:
        z = np.array([[z]])

        # --- Prediction ---
        x = A @ x
        P = A @ P @ A.T + Q

        # --- Update ---
        y = z - H @ x  # residual
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(3) - K @ H) @ P

        estimates.append(float(x[0]))


    return np.array(estimates)


# test the parsing code on example file '2025_05_01__145019_gps_file.txt'
if __name__ == "__main__":
    parser = GPSParser()
    gps_points = parser.parse_file('gps_files/2025_05_01__145019_gps_file.txt')
    df = parser.to_dataframe()
    # print(df.head())
    lat_filtered = kalman_filtering(df['latitude'].values)
    lon_filtered = kalman_filtering(df['longitude'].values)

    filtered_df = df.copy()
    filtered_df['latitude'] = lat_filtered
    filtered_df['longitude'] = lon_filtered

    plot_data(df, filtered_df)