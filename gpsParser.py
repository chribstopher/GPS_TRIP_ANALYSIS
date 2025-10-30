import re
import pandas as pd
from datetime import datetime, timedelta
from GPSPoint import GPSPoint

class GPSParser:
    def __init__(self):
        # initialize an empty list to store GPS points
        self.points = []

    def parse_rmc(self, line):
        """Parser for RMC NMEA sentences.
            the RMC sentence is formatted as follow:

            $GPRMC,221237.000,A,4305.1457,N,07740.8141,W,0.33,58.46,110925,,,A*47
            
            """
        tokens = line.split(',')
        if tokens[2] != 'A':  # Check if the data is valid
            # do something here. I forget what A means
            return None, None
        # create a gps point from the tokens
        datatype = tokens[0][3:6]  # RMC or GGA
        time_str = tokens[1]
        latitude_str = tokens[3]
        latitude_dir = tokens[4]
        latitude = self.convert_to_decimal(latitude_str, latitude_dir)
        longitude_str = tokens[5]
        longitude_dir = tokens[6]
        longitude = self.convert_to_decimal(longitude_str, longitude_dir)
        speed_str_knots = tokens[7]
        speed_knots = float(speed_str_knots) if speed_str_knots else 0.0
        heading_str = tokens[8]
        date_str = tokens[9]
        check_sum = tokens[11]
        # build datetime object if possible
        timestamp = self.parse_datetime(date_str, time_str)
        # create GPSPoint
        point = GPSPoint(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            speed_knots=speed_knots,
            heading=float(heading_str) if heading_str else 0.0,
            altitude=0.0,  # RMC does not provide altitude
            fix_quality=0,  # RMC does not provide fix quality
            num_satellites=0,  # RMC does not provide number of satellites
        )
        return point, timestamp.date()



        
        


    def parse_gga(self, line):
        """Parser for GGA NMEA sentences.
        the GGA sentence is formatted as follow:

        $GPGGA,221237.250,4305.1457,N,07740.8141,W,1,04,2.05,64.4,M,-34.4,M,,*68
        
        """
        tokens = line.split(',')
        time_str = tokens[1]
        latitude_str = tokens[2]
        latitude_dir = tokens[3]
        latitude = self.convert_to_decimal(latitude_str, latitude_dir)
        longitude_str = tokens[4]
        longitude_dir = tokens[5]
        longitude = self.convert_to_decimal(longitude_str, longitude_dir)
        fix_quality_str = tokens[6]
        num_satellites_str = tokens[7]
        altitude_str = tokens[9]
        check_sum = tokens[14]
        # build datetime object if possible
        timestamp = self.parse_time_only(time_str)
        # create GPSPoint
        point = GPSPoint(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            speed_knots=0.0,  # GGA does not provide speed
            heading=0.0,  # GGA does not provide heading
            altitude=float(altitude_str) if altitude_str else 0.0,
            fix_quality=int(fix_quality_str) if fix_quality_str else 0,
            num_satellites=int(num_satellites_str) if num_satellites_str else 0,
            source='GGA'
        )
        return point


    def convert_to_decimal(self, coord_str, direction):
        """Convert NMEA coordinate format (DDMM.MMMM) to decimal degrees."""
        if not coord_str:
            return 0.0
        # Split into degrees and minutes
        if '.' in coord_str:
            point_index = coord_str.index('.')
            degrees_len = point_index - 2
        else:
            degrees_len = len(coord_str) - 2
        degrees = float(coord_str[:degrees_len])
        minutes = float(coord_str[degrees_len:])
        decimal_degrees = degrees + (minutes / 60.0)
        # Apply direction
        if direction in ['S', 'W']:
            decimal_degrees *= -1
        return decimal_degrees
    
    def parse_datetime(self, date_str, time_str):
        """parses DDMMYY and HHMMSS.SSS into a datetime object."""
        if not date_str or not time_str:
            return None
        day = int(date_str[0:2])
        month = int(date_str[2:4])
        year = int(date_str[4:6]) + 2000  # assuming 21st century
        

    def parse_time_only(self, time_str):
        """parses HHMMSS.SSS into a time object.(since gga doesn't have date)"""

    def parse_file(self, filename):
        """Parse a file containing NMEA sentences."""
        with open(filename, 'r') as file:
            current_date = None
            for line in file:
                line = line.strip()
                if line.startswith('$GPRMC'):
                    point, current_date = self.parse_rmc(line)
                    if point:
                        self.points.append(point)
                elif line.startswith('$GPGGA'):
                    point = self.parse_gga(line, current_date)
                    if point:
                        self.points.append(point)



    def _process_sentence(self, line, current_date):
        """Process a single NMEA sentence and update current date if needed."""
    
    def to_dataframe(self):
        """Convert the list of GPS points to a pandas DataFrame."""