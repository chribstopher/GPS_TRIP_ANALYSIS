import re

class GPSParser:
    def __init__(self):
        # initialize an empty list to store GPS points
        self.points = []

    def parse_rmc(self, line):
        """Parser for RMC NMEA sentences."""


    def parse_gga(self, line):
        """Parser for GGA NMEA sentences."""

    def convert_to_decimal(self, coord_str, direction):
    
    def parse_datetime(self, date_str, time_str):
        """parses DDMMYY and HHMMSS.SSS into a datetime object."""

    def parse_time_only(self, time_str):
        """parses HHMMSS.SSS into a time object.(since gga doesn't have date)"""

    def parse_file(self, filename):
        """Parse a file containing NMEA sentences."""

    def _process_sentence(self, line, current_date):
        """Process a single NMEA sentence and update current date if needed."""
    
    def to_dataframe(self):
        """Convert the list of GPS points to a pandas DataFrame."""